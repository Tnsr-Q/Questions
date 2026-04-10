package tcattach

import (
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"

	"github.com/cilium/ebpf"
	"github.com/vishvananda/netlink"

	"github.com/tnsr-q/QFT-Engine/internal/ebpf/objects"
)

const (
	ethPAll = 0x0003
)

// ── Shared helpers ──────────────────────────────────────────────────────────

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

// ClassicAttachedTC represents a TC attachment using netlink clsact + BPF filter.
type ClassicAttachedTC struct {
	Ifindex int
	Handle  uint32
	Parent  uint32
	Filter  netlink.BpfFilter
	PinRoot string
}

func (a *ClassicAttachedTC) Close() error {
	if a == nil {
		return nil
	}
	return netlink.FilterDel(&netlink.BpfFilter{
		FilterAttrs: netlink.FilterAttrs{
			LinkIndex: a.Ifindex,
			Parent:    a.Parent,
			Handle:    a.Handle,
		},
	})
}

// AttachRouterTCClassic loads, attaches via netlink clsact, and pins maps.
// This is the fallback path when link.AttachTCX is not available.
func AttachRouterTCClassic(ifaceName string, dir AttachDirection) (*objects.RouterTCObjects, *ClassicAttachedTC, error) {
	return AttachRouterTCClassicWithPin(ifaceName, dir, DefaultPinRoot)
}

// AttachRouterTCClassicWithPin is the full version with custom pin path.
func AttachRouterTCClassicWithPin(ifaceName string, dir AttachDirection, pinRoot string) (*objects.RouterTCObjects, *ClassicAttachedTC, error) {
	iface, err := net.InterfaceByName(ifaceName)
	if err != nil {
		return nil, nil, fmt.Errorf("lookup interface %q: %w", ifaceName, err)
	}

	nlLink, err := netlink.LinkByIndex(iface.Index)
	if err != nil {
		return nil, nil, fmt.Errorf("link by index %d: %w", iface.Index, err)
	}

	if err := ensureClsact(nlLink); err != nil {
		return nil, nil, fmt.Errorf("ensure clsact on %s: %w", ifaceName, err)
	}

	// Ensure pin directory exists
	if err := os.MkdirAll(pinRoot, 0o755); err != nil {
		return nil, nil, fmt.Errorf("create pin root %s: %w", pinRoot, err)
	}

	var objs objects.RouterTCObjects
	opts := &ebpf.CollectionOptions{
		Maps: ebpf.MapOptions{
			PinPath: pinRoot,
		},
	}
	if err := objects.LoadRouterTCObjects(&objs, opts); err != nil {
		return nil, nil, fmt.Errorf("load router tc objects: %w", err)
	}

	parent, err := tcParent(dir)
	if err != nil {
		_ = objs.Close()
		return nil, nil, err
	}

	filter := netlink.BpfFilter{
		FilterAttrs: netlink.FilterAttrs{
			LinkIndex: iface.Index,
			Parent:    parent,
			Handle:    1,
			Priority:  1,
			Protocol:  ethPAll,
		},
		Fd:           objs.RoutePackets.FD(),
		Name:         "tensorq-router-tc",
		DirectAction: true,
	}

	// Remove existing filter with the same name for idempotent restarts
	if err := removeNamedBpfFilter(nlLink, parent, filter.Name); err != nil {
		_ = objs.Close()
		return nil, nil, fmt.Errorf("remove existing filter: %w", err)
	}

	if err := netlink.FilterAdd(&filter); err != nil {
		_ = objs.Close()
		return nil, nil, fmt.Errorf("attach tc filter on %s (%s): %w", ifaceName, dir, err)
	}

	// Pin maps so other processes can access them
	routingMapPin := filepath.Join(pinRoot, "routing_map")
	statsMapPin := filepath.Join(pinRoot, "stats_map")
	_ = os.Remove(routingMapPin)
	_ = os.Remove(statsMapPin)

	if err := objs.RoutingMap.Pin(routingMapPin); err != nil {
		_ = netlink.FilterDel(&filter)
		_ = objs.Close()
		return nil, nil, fmt.Errorf("pin routing_map: %w", err)
	}
	if err := objs.StatsMap.Pin(statsMapPin); err != nil {
		_ = netlink.FilterDel(&filter)
		_ = objs.Close()
		return nil, nil, fmt.Errorf("pin stats_map: %w", err)
	}

	return &objs, &ClassicAttachedTC{
		Ifindex: iface.Index,
		Handle:  filter.FilterAttrs.Handle,
		Parent:  parent,
		Filter:  filter,
		PinRoot: pinRoot,
	}, nil
}

// DetachClassic removes the TC filter and closes eBPF objects.
func DetachClassic(attached *ClassicAttachedTC, objs *objects.RouterTCObjects) error {
	var errs []error
	if attached != nil {
		errs = append(errs, attached.Close())
		_ = os.Remove(filepath.Join(attached.PinRoot, "routing_map"))
		_ = os.Remove(filepath.Join(attached.PinRoot, "stats_map"))
	}
	if objs != nil {
		errs = append(errs, objs.Close())
	}
	return errors.Join(errs...)
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