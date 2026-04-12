package maps

import (
	"errors"
	"fmt"
	"net"
	"path/filepath"

	"github.com/cilium/ebpf"

	"github.com/tnsr-q/Questions/internal/ebpf/types"
)

const (
	DefaultBPFFSRoot   = "/sys/fs/bpf"
	DefaultRoutingPath = "/sys/fs/bpf/routing_map"
	DefaultStatsPath   = "/sys/fs/bpf/stats_map"
)

type RouteMaps struct {
	Routing *ebpf.Map
	Stats   *ebpf.Map
}

func OpenPinned(routingPath, statsPath string) (*RouteMaps, error) {
	routing, err := ebpf.LoadPinnedMap(routingPath, nil)
	if err != nil {
		return nil, fmt.Errorf("open routing map: %w", err)
	}

	var stats *ebpf.Map
	if statsPath != "" {
		stats, err = ebpf.LoadPinnedMap(statsPath, nil)
		if err != nil && !errors.Is(err, ebpf.ErrMapIncompatible) {
			_ = routing.Close()
			return nil, fmt.Errorf("open stats map: %w", err)
		}
	}

	return &RouteMaps{
		Routing: routing,
		Stats:   stats,
	}, nil
}

func (m *RouteMaps) Close() error {
	var errs []error
	if m.Routing != nil {
		errs = append(errs, m.Routing.Close())
	}
	if m.Stats != nil {
		errs = append(errs, m.Stats.Close())
	}
	return errors.Join(errs...)
}

// UpdateRoute creates a DSR (Direct Server Return) route entry.
// For containerized deployments, use UpdateRouteDSR instead which handles interface/MAC lookups.
func UpdateRoute(m *ebpf.Map, vip string, dstIP string, dstMAC string) error {
	key, err := types.IPv4ToBE32(vip)
	if err != nil {
		return fmt.Errorf("parse VIP: %w", err)
	}

	// TODO: For DSR, we need nextHopIP, ifindex, and srcMAC from the RouteMutation payload
	// Placeholder: use dstIP as nextHopIP, ifindex 0, empty srcMAC
	entry, err := types.NewRouteEntry(dstIP, dstIP, 0, dstMAC, "")
	if err != nil {
		return fmt.Errorf("build route entry: %w", err)
	}

	if err := m.Update(&key, &entry, ebpf.UpdateAny); err != nil {
		return fmt.Errorf("map update vip=%s dst=%s mac=%s: %w", vip, dstIP, dstMAC, err)
	}
	return nil
}

// UpdateRouteDSR creates a full DSR route entry with proper interface and MAC configuration.
// This is the production version for containerized swarms that need RPF bypass.
// Parameters come from the Python Overseer's RouteMutation protobuf:
// - nextHopIP: The Alpha node's container IP (from RouteMutation.targetPodIp)
// - ifname: The egress interface name (e.g., "docker0", "eth0")
// - srcMAC: The gateway's own MAC (typically from the egress interface)
func UpdateRouteDSR(m *ebpf.Map, vip string, nextHopIP string, dstMAC string, ifname string, srcMAC string) error {
	// Parse VIP as the key
	key, err := types.IPv4ToBE32(vip)
	if err != nil {
		return fmt.Errorf("parse VIP: %w", err)
	}

	// Look up interface index
	iface, err := net.InterfaceByName(ifname)
	if err != nil {
		return fmt.Errorf("lookup interface %q: %w", ifname, err)
	}
	ifindex := uint32(iface.Index)

	// If srcMAC not provided, use the interface's MAC
	if srcMAC == "" && iface.HardwareAddr != nil {
		srcMAC = iface.HardwareAddr.String()
	}

	// Create DSR route entry
	// dstIP and nextHopIP are the same in basic DSR (both point to Alpha node)
	entry, err := types.NewRouteEntry(nextHopIP, nextHopIP, ifindex, dstMAC, srcMAC)
	if err != nil {
		return fmt.Errorf("build DSR route entry: %w", err)
	}

	if err := m.Update(&key, &entry, ebpf.UpdateAny); err != nil {
		return fmt.Errorf("map update DSR vip=%s next_hop=%s ifname=%s: %w", vip, nextHopIP, ifname, err)
	}
	return nil
}

func DeleteRoute(m *ebpf.Map, vip string) error {
	key, err := types.IPv4ToBE32(vip)
	if err != nil {
		return fmt.Errorf("parse VIP: %w", err)
	}
	if err := m.Delete(&key); err != nil {
		return fmt.Errorf("delete route vip=%s: %w", vip, err)
	}
	return nil
}

func EnsureTensorQDir(root string) string {
	return filepath.Join(root, "tensorq")
}
