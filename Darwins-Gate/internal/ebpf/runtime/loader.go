package runtime

import (
	"errors"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/cilium/ebpf"
	"github.com/vishvananda/netlink"

	"github.com/tnsr-q/Questions/internal/ebpf/objects"
)

const (
	DefaultPinRoot = "/sys/fs/bpf/tensorq"
	ethPAll        = 0x0003
)

type AttachDirection string

const (
	Ingress AttachDirection = "ingress"
	Egress  AttachDirection = "egress"
)

type Config struct {
	Interface string
	Direction AttachDirection
	PinRoot   string
}

type Runtime struct {
	Config Config

	Objs *objects.RouterTCObjects

	Link   netlink.Link
	Filter *netlink.BpfFilter
}

func New(cfg Config) (*Runtime, error) {
	if cfg.Interface == "" {
		return nil, fmt.Errorf("interface is required")
	}
	if cfg.PinRoot == "" {
		cfg.PinRoot = DefaultPinRoot
	}
	switch cfg.Direction {
	case Ingress, Egress:
	default:
		return nil, fmt.Errorf("unsupported direction %q", cfg.Direction)
	}

	if err := os.MkdirAll(cfg.PinRoot, 0o755); err != nil {
		return nil, fmt.Errorf("create pin root %s: %w", cfg.PinRoot, err)
	}

	iface, err := net.InterfaceByName(cfg.Interface)
	if err != nil {
		return nil, fmt.Errorf("lookup interface %q: %w", cfg.Interface, err)
	}

	nlLink, err := netlink.LinkByIndex(iface.Index)
	if err != nil {
		return nil, fmt.Errorf("netlink link by index %d: %w", iface.Index, err)
	}

	var objs objects.RouterTCObjects
	opts := &ebpf.CollectionOptions{
		Maps: ebpf.MapOptions{
			PinPath: cfg.PinRoot,
		},
	}
	if err := objects.LoadRouterTCObjects(&objs, opts); err != nil {
		return nil, fmt.Errorf("load router tc objects: %w", err)
	}

	rt := &Runtime{
		Config: cfg,
		Objs:   &objs,
		Link:   nlLink,
	}
	return rt, nil
}

func (r *Runtime) Start() error {
	if r == nil || r.Objs == nil {
		return fmt.Errorf("runtime not initialized")
	}

	if err := ensureClsact(r.Link); err != nil {
		return fmt.Errorf("ensure clsact: %w", err)
	}

	if err := r.pinMaps(); err != nil {
		return fmt.Errorf("pin maps: %w", err)
	}

	if err := r.attachTC(); err != nil {
		return fmt.Errorf("attach tc: %w", err)
	}

	return nil
}

func (r *Runtime) Close() error {
	var errs []error

	if r.Filter != nil {
		errs = append(errs, netlink.FilterDel(r.Filter))
		r.Filter = nil
	}

	if r.Objs != nil {
		errs = append(errs, r.Objs.Close())
		r.Objs = nil
	}

	return errors.Join(errs...)
}

func (r *Runtime) Wait() error {
	sigCh := make(chan os.Signal, 2)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	return nil
}

func (r *Runtime) pinMaps() error {
	if r.Objs == nil {
		return fmt.Errorf("objects not loaded")
	}

	routingMapPin := filepath.Join(r.Config.PinRoot, "routing_map")
	statsMapPin := filepath.Join(r.Config.PinRoot, "stats_map")

	// Best-effort cleanup of stale pin paths. If the old file is not a map pin,
	// Pin() will fail later anyway, so removing here is simpler.
	_ = os.Remove(routingMapPin)
	_ = os.Remove(statsMapPin)

	if err := r.Objs.RoutingMap.Pin(routingMapPin); err != nil {
		return fmt.Errorf("pin routing_map: %w", err)
	}
	if err := r.Objs.StatsMap.Pin(statsMapPin); err != nil {
		return fmt.Errorf("pin stats_map: %w", err)
	}

	return nil
}

func (r *Runtime) attachTC() error {
	parent, err := tcParent(r.Config.Direction)
	if err != nil {
		return err
	}

	filter := &netlink.BpfFilter{
		FilterAttrs: netlink.FilterAttrs{
			LinkIndex: r.Link.Attrs().Index,
			Parent:    parent,
			Handle:    1,
			Priority:  1,
			Protocol:  ethPAll,
		},
		Fd:           r.Objs.RoutePackets.FD(),
		Name:         "tensorq-router-tc",
		DirectAction: true,
	}

	// Remove an existing matching filter if present, so restart is idempotent enough.
	_ = removeNamedBpfFilter(r.Link, parent, filter.Name)

	if err := netlink.FilterAdd(filter); err != nil {
		return fmt.Errorf("filter add: %w", err)
	}

	r.Filter = filter
	return nil
}

func tcParent(dir AttachDirection) (uint32, error) {
	switch dir {
	case Ingress:
		return netlink.HANDLE_MIN_INGRESS, nil
	case Egress:
		return netlink.HANDLE_MIN_EGRESS, nil
	default:
		return 0, fmt.Errorf("unsupported direction %q", dir)
	}
}

func ensureClsact(link netlink.Link) error {
	qdiscs, err := netlink.QdiscList(link)
	if err != nil {
		return fmt.Errorf("list qdiscs: %w", err)
	}

	for _, q := range qdiscs {
		gq, ok := q.(*netlink.GenericQdisc)
		if !ok {
			continue
		}
		attrs := gq.Attrs()
		if attrs == nil {
			continue
		}
		if attrs.Handle == netlink.MakeHandle(0xffff, 0) &&
			attrs.Parent == netlink.HANDLE_CLSACT {
			return nil
		}
	}

	qdisc := &netlink.GenericQdisc{
		QdiscAttrs: netlink.QdiscAttrs{
			LinkIndex: link.Attrs().Index,
			Handle:    netlink.MakeHandle(0xffff, 0),
			Parent:    netlink.HANDLE_CLSACT,
		},
		QdiscType: "clsact",
	}

	if err := netlink.QdiscAdd(qdisc); err != nil {
		if !errors.Is(err, netlink.ErrExist) {
			return fmt.Errorf("add clsact qdisc: %w", err)
		}
	}

	return nil
}

func removeNamedBpfFilter(link netlink.Link, parent uint32, name string) error {
	filters, err := netlink.FilterList(link, parent)
	if err != nil {
		return fmt.Errorf("list filters: %w", err)
	}

	for _, f := range filters {
		bf, ok := f.(*netlink.BpfFilter)
		if !ok {
			continue
		}
		if bf.Name == name {
			if err := netlink.FilterDel(bf); err != nil {
				return fmt.Errorf("delete existing filter %q: %w", name, err)
			}
		}
	}
	return nil
}