package tcattach

import (
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"

	"github.com/tnsr-q/QFT-Engine/internal/ebpf/objects"
)

const DefaultPinRoot = "/sys/fs/bpf/tensorq"

type AttachDirection string

const (
	Ingress AttachDirection = "ingress"
	Egress  AttachDirection = "egress"
)

type AttachedTC struct {
	Link    link.Link
	PinRoot string
}

func (a *AttachedTC) Close() error {
	if a == nil {
		return nil
	}
	var errs []error
	if a.Link != nil {
		errs = append(errs, a.Link.Close())
	}
	return errors.Join(errs...)
}

func AttachRouterTC(ifaceName string, dir AttachDirection) (*objects.RouterTCObjects, *AttachedTC, error) {
	return AttachRouterTCWithPin(ifaceName, dir, DefaultPinRoot)
}

// AttachRouterTCWithPin loads, attaches, and pins maps so that gatewayd /
// mutation-firewalld can read them via LoadPinnedMap.
func AttachRouterTCWithPin(ifaceName string, dir AttachDirection, pinRoot string) (*objects.RouterTCObjects, *AttachedTC, error) {
	iface, err := net.InterfaceByName(ifaceName)
	if err != nil {
		return nil, nil, fmt.Errorf("lookup interface %q: %w", ifaceName, err)
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

	var prog *ebpf.Program
	switch dir {
	case Ingress, Egress:
		prog = objs.RoutePackets
	default:
		_ = objs.Close()
		return nil, nil, fmt.Errorf("unsupported attach direction %q", dir)
	}

	var lnk link.Link

	// Example for newer TCX-capable environments:
	switch dir {
	case Ingress:
		lnk, err = link.AttachTCX(link.TCXOptions{
			Program:   prog,
			Attach:    ebpf.AttachTCXIngress,
			Interface: iface.Index,
		})
	case Egress:
		lnk, err = link.AttachTCX(link.TCXOptions{
			Program:   prog,
			Attach:    ebpf.AttachTCXEgress,
			Interface: iface.Index,
		})
	}
	if err != nil {
		_ = objs.Close()
		return nil, nil, fmt.Errorf("attach tc program to %s (%s): %w", ifaceName, dir, err)
	}

	// Pin maps so other processes (gatewayd, mutation-firewalld) can access them
	routingMapPin := filepath.Join(pinRoot, "routing_map")
	statsMapPin := filepath.Join(pinRoot, "stats_map")

	// Best-effort cleanup of stale pins
	_ = os.Remove(routingMapPin)
	_ = os.Remove(statsMapPin)

	if err := objs.RoutingMap.Pin(routingMapPin); err != nil {
		_ = objs.Close()
		_ = lnk.Close()
		return nil, nil, fmt.Errorf("pin routing_map: %w", err)
	}
	if err := objs.StatsMap.Pin(statsMapPin); err != nil {
		_ = objs.Close()
		_ = lnk.Close()
		return nil, nil, fmt.Errorf("pin stats_map: %w", err)
	}

	return &objs, &AttachedTC{Link: lnk, PinRoot: pinRoot}, nil
}

func Detach(a *AttachedTC, objs *objects.RouterTCObjects) error {
	var errs []error
	if a != nil {
		errs = append(errs, a.Close())
		// Unpin maps on detach
		_ = os.Remove(filepath.Join(a.PinRoot, "routing_map"))
		_ = os.Remove(filepath.Join(a.PinRoot, "stats_map"))
	}
	if objs != nil {
		errs = append(errs, objs.Close())
	}
	return errors.Join(errs...)
}