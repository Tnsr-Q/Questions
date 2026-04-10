// Package metrics provides Prometheus-compatible metric collectors
// for the Darwinian Gateway data plane and control plane.
//
// All collectors implement http.Handler that renders plain text
// Prometheus exposition format.
package metrics

import (
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// ── StreamMetrics — data plane (StreamSimulation RPC) ──────────────────────

type StreamMetrics struct {
	SessionsStarted atomic.Int64
	SessionsFailed  atomic.Int64
	SessionsActive  atomic.Int64
	ChunksEmitted   atomic.Int64
	BytesEmitted    atomic.Int64
	LastChunkTs     atomic.Int64 // unix nanos
}

func (m *StreamMetrics) IncSessionsStarted() {
	m.SessionsStarted.Add(1)
	m.SessionsActive.Add(1)
}

func (m *StreamMetrics) IncSessionsFailed() {
	m.SessionsFailed.Add(1)
}

func (m *StreamMetrics) DecSessionsActive() {
	m.SessionsActive.Add(-1)
}

func (m *StreamMetrics) AddChunksEmitted(n int) {
	m.ChunksEmitted.Add(int64(n))
	m.LastChunkTs.Store(time.Now().UnixNano())
}

func (m *StreamMetrics) AddBytesEmitted(n int) {
	m.BytesEmitted.Add(int64(n))
}

// ── EBPFMapMetrics — control plane (UpdateAlphaRoute RPC) ──────────────────

type EBPFMapMetrics struct {
	MutationsReceived   atomic.Int64
	MutationsApplied    atomic.Int64
	MutationsRejected   atomic.Int64
	MutationsFailed     atomic.Int64
	DryRuns             atomic.Int64
	LastMutationTs      atomic.Int64 // unix nanos
	MapLookupErrors     atomic.Int64
	MapUpdateErrors     atomic.Int64
	MapEntries          atomic.Int64 // current routing_map size
	MapMaxEntries       int64        // constant: 1024
}

func (m *EBPFMapMetrics) IncMutationsReceived() {
	m.MutationsReceived.Add(1)
	m.LastMutationTs.Store(time.Now().UnixNano())
}

func (m *EBPFMapMetrics) IncMutationsApplied() {
	m.MutationsApplied.Add(1)
}

func (m *EBPFMapMetrics) IncMutationsRejected() {
	m.MutationsRejected.Add(1)
}

func (m *EBPFMapMetrics) IncMutationsFailed() {
	m.MutationsFailed.Add(1)
}

func (m *EBPFMapMetrics) IncDryRuns() {
	m.DryRuns.Add(1)
}

func (m *EBPFMapMetrics) IncMapLookupErrors() {
	m.MapLookupErrors.Add(1)
}

func (m *EBPFMapMetrics) IncMapUpdateErrors() {
	m.MapUpdateErrors.Add(1)
}

// ── SwarmMetrics — overseer / Pyro5 bridge ─────────────────────────────────

type SwarmMetrics struct {
	LeaderFetches      atomic.Int64
	LeaderFetchErrors  atomic.Int64
	TelemetryBatches   atomic.Int64
	TelemetryErrors     atomic.Int64
	ProposalsFetched   atomic.Int64
	ProposalsForwarded atomic.Int64
	ProposalsSkipped   atomic.Int64
	OutcomesReported   atomic.Int64
	OutcomeErrors      atomic.Int64
	LastLeaderFetchTs  atomic.Int64 // unix nanos
	QuorumHealthy      atomic.Int64 // 1 or 0
	CurrentEpoch       atomic.Int64
}

func (m *SwarmMetrics) IncLeaderFetches() {
	m.LeaderFetches.Add(1)
	m.LastLeaderFetchTs.Store(time.Now().UnixNano())
}

func (m *SwarmMetrics) IncLeaderFetchErrors() {
	m.LeaderFetchErrors.Add(1)
}

func (m *SwarmMetrics) IncTelemetryBatches() {
	m.TelemetryBatches.Add(1)
}

func (m *SwarmMetrics) IncTelemetryErrors() {
	m.TelemetryErrors.Add(1)
}

func (m *SwarmMetrics) IncProposalsFetched() {
	m.ProposalsFetched.Add(1)
}

func (m *SwarmMetrics) IncProposalsForwarded() {
	m.ProposalsForwarded.Add(1)
}

func (m *SwarmMetrics) IncProposalsSkipped() {
	m.ProposalsSkipped.Add(1)
}

func (m *SwarmMetrics) IncOutcomesReported() {
	m.OutcomesReported.Add(1)
}

func (m *SwarmMetrics) IncOutcomeErrors() {
	m.OutcomeErrors.Add(1)
}

// ── Prometheus Exposition Handler ──────────────────────────────────────────

// Collector groups all metric sinks for unified HTTP exposition.
type Collector struct {
	mu     sync.Mutex
	stream *StreamMetrics
	ebpf   *EBPFMapMetrics
	swarm  *SwarmMetrics
}

func NewCollector(stream *StreamMetrics, ebpf *EBPFMapMetrics, swarm *SwarmMetrics) *Collector {
	return &Collector{stream: stream, ebpf: ebpf, swarm: swarm}
}

func (c *Collector) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	c.mu.Lock()
	defer c.mu.Unlock()

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")

	wf := func(name, help string, value float64) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n%s %g\n\n", name, help, name, name, value)
	}
	wfCounter := func(name, help string, value float64) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s counter\n%s %g\n\n", name, help, name, name, value)
	}

	// Stream metrics
	if c.stream != nil {
		s := c.stream
		wfCounter("tensorq_stream_sessions_started_total", "Total StreamSimulation sessions initiated", float64(s.SessionsStarted.Load()))
		wfCounter("tensorq_stream_sessions_failed_total", "Total StreamSimulation sessions that failed", float64(s.SessionsFailed.Load()))
		wf("tensorq_stream_sessions_active", "Currently active StreamSimulation sessions", float64(s.SessionsActive.Load()))
		wfCounter("tensorq_stream_chunks_emitted_total", "Total CodeChunks sent to clients", float64(s.ChunksEmitted.Load()))
		wfCounter("tensorq_stream_bytes_emitted_total", "Total bytes sent to streaming clients", float64(s.BytesEmitted.Load()))
	}

	// eBPF map metrics
	if c.ebpf != nil {
		e := c.ebpf
		wfCounter("tensorq_ebpf_mutations_received_total", "Total UpdateAlphaRoute requests received", float64(e.MutationsReceived.Load()))
		wfCounter("tensorq_ebpf_mutations_applied_total", "Total route mutations written to kernel map", float64(e.MutationsApplied.Load()))
		wfCounter("tensorq_ebpf_mutations_rejected_total", "Total route mutations rejected by admission", float64(e.MutationsRejected.Load()))
		wfCounter("tensorq_ebpf_mutations_failed_total", "Total route mutations that failed kernel write", float64(e.MutationsFailed.Load()))
		wfCounter("tensorq_ebpf_dry_runs_total", "Total dry-run route mutations", float64(e.DryRuns.Load()))
		wfCounter("tensorq_ebpf_map_lookup_errors_total", "Total eBPF map lookup failures", float64(e.MapLookupErrors.Load()))
		wfCounter("tensorq_ebpf_map_update_errors_total", "Total eBPF map write failures", float64(e.MapUpdateErrors.Load()))
		wf("tensorq_ebpf_map_entries", "Current number of entries in routing_map", float64(e.MapEntries.Load()))
		wf("tensorq_ebpf_map_max_entries", "Maximum capacity of routing_map", float64(e.MapMaxEntries))
	}

	// Swarm bridge metrics
	if c.swarm != nil {
		s := c.swarm
		wfCounter("tensorq_swarm_leader_fetches_total", "Total GET /v1/leader calls", float64(s.LeaderFetches.Load()))
		wfCounter("tensorq_swarm_leader_fetch_errors_total", "Total leader fetch failures", float64(s.LeaderFetchErrors.Load()))
		wfCounter("tensorq_swarm_telemetry_batches_total", "Total telemetry batches sent", float64(s.TelemetryBatches.Load()))
		wfCounter("tensorq_swarm_telemetry_errors_total", "Total telemetry POST failures", float64(s.TelemetryErrors.Load()))
		wfCounter("tensorq_swarm_proposals_fetched_total", "Total GET /v1/proposals/alpha calls", float64(s.ProposalsFetched.Load()))
		wfCounter("tensorq_swarm_proposals_forwarded_total", "Total proposals forwarded to mutation-firewalld", float64(s.ProposalsForwarded.Load()))
		wfCounter("tensorq_swarm_proposals_skipped_total", "Total proposals skipped (unhealthy quorum / stale epoch)", float64(s.ProposalsSkipped.Load()))
		wfCounter("tensorq_swarm_outcomes_reported_total", "Total mutation outcomes sent back to swarm", float64(s.OutcomesReported.Load()))
		wfCounter("tensorq_swarm_outcome_errors_total", "Total outcome POST failures", float64(s.OutcomeErrors.Load()))
		wf("tensorq_swarm_quorum_healthy", "1 if current quorum is HEALTHY, 0 otherwise", float64(s.QuorumHealthy.Load()))
		wf("tensorq_swarm_current_epoch", "Current swarm leader epoch", float64(s.CurrentEpoch.Load()))
	}
}

// DefaultCollector returns a Collector wired to the global metric singletons.
func DefaultCollector() *Collector {
	return NewCollector(GatewayMetrics, EBPFMapMetrics, SwarmMetrics)
}

// Global instances — exported so packages can increment directly.
var (
	GatewayMetrics  = &StreamMetrics{}
	EBPFMapMetrics  = &EBPFMapMetrics{MapMaxEntries: 1024}
	SwarmMetrics    = &SwarmMetrics{}
)
