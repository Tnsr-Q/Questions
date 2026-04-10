package maps

import (
	"errors"
	"fmt"
	"path/filepath"

	"github.com/cilium/ebpf"

	"darwinian-gate/internal/ebpf/types")

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

func UpdateRoute(m *ebpf.Map, vip string, dstIP string, dstMAC string) error {
	key, err := types.IPv4ToBE32(vip)
	if err != nil {
		return fmt.Errorf("parse VIP: %w", err)
	}

	entry, err := types.NewRouteEntry(dstIP, dstMAC)
	if err != nil {
		return fmt.Errorf("build route entry: %w", err)
	}

	if err := m.Update(&key, &entry, ebpf.UpdateAny); err != nil {
		return fmt.Errorf("map update vip=%s dst=%s mac=%s: %w", vip, dstIP, dstMAC, err)
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
