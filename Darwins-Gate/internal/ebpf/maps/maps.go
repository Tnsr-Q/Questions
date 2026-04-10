// Package maps provides helpers for reading and writing pinned eBPF maps
package maps

import (
	"encoding/binary"
	"fmt"
	"net"

	"github.com/cilium/ebpf"
)

const (
	// DefaultRoutingPath is the default bpffs pin path for the routing map.
	DefaultRoutingPath = "/sys/fs/bpf/questions/routing_map"
	// DefaultStatsPath is the default bpffs pin path for the stats map.
	DefaultStatsPath = "/sys/fs/bpf/questions/stats_map"
)

// PinnedMaps holds references to the pinned eBPF maps.
type PinnedMaps struct {
	Routing *ebpf.Map
	Stats   *ebpf.Map
}

// Close releases both map references.
func (p *PinnedMaps) Close() error {
	var firstErr error
	if p.Routing != nil {
		if err := p.Routing.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if p.Stats != nil {
		if err := p.Stats.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// OpenPinned loads the pinned routing and stats maps from the given paths.
func OpenPinned(routingPath, statsPath string) (*PinnedMaps, error) {
	rm, err := ebpf.LoadPinnedMap(routingPath, nil)
	if err != nil {
		return nil, fmt.Errorf("load routing map %s: %w", routingPath, err)
	}
	sm, err := ebpf.LoadPinnedMap(statsPath, nil)
	if err != nil {
		rm.Close()
		return nil, fmt.Errorf("load stats map %s: %w", statsPath, err)
	}
	return &PinnedMaps{Routing: rm, Stats: sm}, nil
}

// RouteEntry matches the C struct route_entry (24 bytes).
type RouteEntry struct {
	DstIP     uint32
	NextHopIP uint32
	Ifindex   uint32
	DstMAC    [6]byte
	SrcMAC    [6]byte
}

// UpdateRoute writes or overwrites a route entry in the routing map.
func UpdateRoute(m *ebpf.Map, vip, dstIP, dstMAC string) error {
	key, err := ipToKey(vip)
	if err != nil {
		return fmt.Errorf("parse vip: %w", err)
	}
	realIP, err := ipToKey(dstIP)
	if err != nil {
		return fmt.Errorf("parse dst-ip: %w", err)
	}
	mac, err := net.ParseMAC(dstMAC)
	if err != nil {
		return fmt.Errorf("parse dst-mac: %w", err)
	}
	var macBytes [6]byte
	copy(macBytes[:], mac)

	entry := RouteEntry{
		DstIP:  realIP,
		DstMAC: macBytes,
	}
	entry.NextHopIP = realIP

	return m.Update(key, entry, ebpf.UpdateAny)
}

// DeleteRoute removes a route entry from the routing map.
func DeleteRoute(m *ebpf.Map, vip string) error {
	key, err := ipToKey(vip)
	if err != nil {
		return fmt.Errorf("parse vip: %w", err)
	}
	return m.Delete(key)
}

func ipToKey(ipStr string) (uint32, error) {
	ip := net.ParseIP(ipStr).To4()
	if ip == nil {
		return 0, fmt.Errorf("invalid IPv4: %q", ipStr)
	}
	return binary.BigEndian.Uint32(ip), nil
}
