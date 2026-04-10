package obs

import (
	"encoding/binary"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/cilium/ebpf"
)

// RouteCounters mirrors the C struct in bpf/include/maps.h.
// 24 bytes: 3 × uint64 (hits, misses, errors).
type RouteCounters struct {
	Hits   uint64
	Misses uint64
	Errors uint64
}

// StatsReader reads the pinned eBPF stats_map periodically.
type StatsReader struct {
	mu       sync.Mutex
	mapPath  string
	statsMap *ebpf.Map
	last     RouteCounters
	lastRead time.Time
}

// NewStatsReader creates a reader for the pinned stats_map.
func NewStatsReader(pinPath string) *StatsReader {
	return &StatsReader{mapPath: pinPath}
}

// Read fetches the latest counters from the pinned eBPF map.
// Safe to call concurrently. Returns cached data on map error.
func (r *StatsReader) Read() (RouteCounters, time.Time, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.statsMap == nil {
		m, err := ebpf.LoadPinnedMap(r.mapPath, nil)
		if err != nil {
			return r.last, r.lastRead, fmt.Errorf("load pinned stats_map: %w", err)
		}
		r.statsMap = m
	}

	var values [3]uint64
	key := uint32(0)
	if err := r.statsMap.Lookup(&key, &values); err != nil {
		return r.last, r.lastRead, fmt.Errorf("map lookup: %w", err)
	}

	r.last = RouteCounters{
		Hits:   values[0],
		Misses: values[1],
		Errors: values[2],
	}
	r.lastRead = time.Now()
	return r.last, r.lastRead, nil
}

// ReadFromFile reads the stats_map directly from the pinned file path
// without keeping a reference. Useful for one-shot reads from processes
// that don't hold the map open.
func ReadFromFile(pinPath string) (RouteCounters, error) {
	data, err := os.ReadFile(pinPath)
	if err != nil {
		return RouteCounters{}, fmt.Errorf("read pinned file: %w", err)
	}

	// BPF map value layout: 3 × uint64 little-endian (kernel internal)
	// The pinned file contains the raw value for key 0.
	if len(data) < 24 {
		return RouteCounters{}, fmt.Errorf("stats_map too small: %d bytes", len(data))
	}

	return RouteCounters{
		Hits:   binary.LittleEndian.Uint64(data[0:8]),
		Misses: binary.LittleEndian.Uint64(data[8:16]),
		Errors: binary.LittleEndian.Uint64(data[16:24]),
	}, nil
}

// Close releases the cached map reference.
func (r *StatsReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.statsMap != nil {
		return r.statsMap.Close()
	}
	return nil
}
