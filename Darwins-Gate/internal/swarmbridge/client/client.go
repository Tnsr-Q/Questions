// Package swarmbridge implements the Go-side client for the TensorQ
// overseer-coordinator bridge API.
//
// Transport: Unix Domain Socket (preferred) or localhost TCP on :8888.
// Auth: X-TensorQ-Bridge-Token header, loaded from /var/run/tensorq/bridge.token.
//
// Startup sequence (per bridge contract):
//
//	1. GET /v1/meta           — validate version/features
//	2. GET /v1/health         — poll until READY (exponential backoff)
//	3. GET /v1/leader         — cache leader/epoch state
//	4. POST /v1/telemetry     — periodic metric push
//	5. GET /v1/proposals/alpha — poll for route proposals, forward to firewall
package swarmbridge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"sync"
	"time"
)

// Default paths and timeouts.
const (
	DefaultTokenPath  = "/var/run/tensorq/bridge.token"
	DefaultSidecarURL = "http://localhost:8888"

	ConnectTimeout  = 250 * time.Millisecond
	LeaderTimeout   = 1 * time.Second
	ProposalTimeout = 1 * time.Second
	TelemetryTimeout = 2 * time.Second
	OutcomeTimeout  = 2 * time.Second

	HealthPollInterval = 500 * time.Millisecond
	HealthPollMaxWait  = 30 * time.Second

	TelemetryInterval = 5 * time.Second
	ProposalInterval  = 2 * time.Second
)

// ── Bridge API Schemas (from docs/architecture/31-bridge-contract.md) ──────

type BridgeMeta struct {
	BridgeProtocolVersion string `json:"bridge_protocol_version"`
	NodeID                string `json:"node_id"`
	Features              struct {
		LeaderFencing    bool `json:"leader_fencing"`
		WeightedRouting  bool `json:"weighted_routing"`
		FreezeProposals  bool `json:"freeze_proposals"`
		OutcomeFeedback  bool `json:"outcome_feedback"`
	} `json:"features"`
	Build struct {
		GitSHA           string `json:"git_sha"`
		BuildTimeUnixMs int64  `json:"build_time_unix_ms"`
	} `json:"build"`
}

type HealthStatus struct {
	Status            string `json:"status"` // "READY" | "NOT_READY"
	Reason            string `json:"reason,omitempty"`
	ObservedAtUnixMs  int64  `json:"observed_at_unix_ms"`
}

type LeaderState struct {
	LeaderID            *string             `json:"leader_id"`
	Epoch               int64               `json:"epoch"`
	LeaseExpiresAtMs    int64               `json:"lease_expires_at_unix_ms"`
	QuorumStatus        string              `json:"quorum_status"` // HEALTHY | DEGRADED | LOST
	Members             struct {
		Size   int      `json:"size"`
		KnownIDs []string `json:"known_ids"`
	} `json:"members"`
	ObservedAtUnixMs int64 `json:"observed_at_unix_ms"`
}

type LatencyStats struct {
	P50 float64 `json:"p50"`
	P95 float64 `json:"p95"`
	P99 float64 `json:"p99"`
}

type ErrorStats struct {
	Rate  float64            `json:"rate"`
	Codes map[string]int64   `json:"codes,omitempty"`
}

type DropStats struct {
	Rate    float64            `json:"rate"`
	Reasons map[string]int64   `json:"reasons,omitempty"`
}

type TelemetrySample struct {
	VIP           string       `json:"vip"`
	BackendHint   *struct {
		PodIP      string `json:"pod_ip"`
		EndpointID string `json:"endpoint_id,omitempty"`
	} `json:"backend_hint,omitempty"`
	LatencyMs     *LatencyStats `json:"latency_ms,omitempty"`
	Errors        *ErrorStats   `json:"errors,omitempty"`
	Drops         *DropStats    `json:"drops,omitempty"`
	ThroughputRPS float64       `json:"throughput_rps,omitempty"`
	CPUUtil       float64       `json:"cpu_util,omitempty"`
	MemUtil       float64       `json:"mem_util,omitempty"`
}

type TelemetryBatch struct {
	BatchID         string            `json:"batch_id"`
	NodeID          string            `json:"node_id"`
	SentAtUnixMs    int64             `json:"sent_at_unix_ms"`
	Samples         []TelemetrySample `json:"samples"`
	WindowMs        int64             `json:"window_ms"`
}

type TelemetryAck struct {
	Accepted        bool   `json:"accepted"`
	IngestedSamples int    `json:"ingested_samples"`
	BatchID         string `json:"batch_id"`
	ObservedAtUnixMs int64 `json:"observed_at_unix_ms"`
}

type Target struct {
	Type   string `json:"type"` // "POD_IP"
	PodIP  string `json:"pod_ip"`
}

type FitnessSummary struct {
	WindowMs int64              `json:"window_ms"`
	Metrics  map[string]float64 `json:"metrics"`
}

type ProposalConstraints struct {
	RequireCanary    bool    `json:"require_canary"`
	MaxStepChange    float64 `json:"max_step_change"`
	MinImprovement   float64 `json:"min_improvement"`
	StabilityWindowMs int64  `json:"stability_window_ms"`
}

type AlphaRouteProposal struct {
	ProposalID         string              `json:"proposal_id"`
	LeaderID           string              `json:"leader_id"`
	Epoch              int64               `json:"epoch"`
	LeaseExpiresAtMs   int64               `json:"lease_expires_at_unix_ms"`
	VIP                string              `json:"vip"`
	Target             Target              `json:"target"`
	TTLms              int64               `json:"ttl_ms"`
	Confidence         float64             `json:"confidence"`
	FitnessSummary     *FitnessSummary     `json:"fitness_summary,omitempty"`
	Constraints        *ProposalConstraints `json:"constraints,omitempty"`
	Reason             string              `json:"reason,omitempty"`
	CreatedAtUnixMs    int64               `json:"created_at_unix_ms"`
}

type ProposalResponse struct {
	Proposal   *AlphaRouteProposal `json:"proposal"`
	NoneReason *string             `json:"none_reason"` // NO_LEADER | NO_QUORUM | NO_DATA | COOLDOWN | null
}

type FreezeScope struct {
	Type         string `json:"type"` // "VIP"
	VIP          string `json:"vip"`
	DurationMs   int64  `json:"duration_ms"`
}

type FreezeProposal struct {
	ProposalID         string      `json:"proposal_id"`
	LeaderID           string      `json:"leader_id"`
	Epoch              int64       `json:"epoch"`
	LeaseExpiresAtMs   int64       `json:"lease_expires_at_unix_ms"`
	Scope              FreezeScope `json:"scope"`
	Reason             string      `json:"reason"`
	CreatedAtUnixMs    int64       `json:"created_at_unix_ms"`
}

type FreezeAck struct {
	Accepted         bool   `json:"accepted"`
	ProposalID       string `json:"proposal_id"`
	ObservedAtUnixMs int64  `json:"observed_at_unix_ms"`
}

type ProposalOutcome struct {
	ProposalID          string  `json:"proposal_id"`
	VIP                 string  `json:"vip"`
	Status              string  `json:"status"` // REJECTED | STAGED | APPLIED | PROBE_FAILED | ROLLED_BACK
	AppliedRouteVersion int64   `json:"applied_route_version,omitempty"`
	Rejection           *struct {
		Code    string `json:"code,omitempty"`
		Message string `json:"message,omitempty"`
	} `json:"rejection,omitempty"`
	Probe *struct {
		PassRatio float64 `json:"pass_ratio"`
		WindowMs  int64   `json:"window_ms"`
	} `json:"probe,omitempty"`
	Rollback *struct {
		Reason string `json:"reason,omitempty"`
	} `json:"rollback,omitempty"`
	MetricsDelta *struct {
		LatencyP95Ms float64 `json:"latency_p95_ms,omitempty"`
		DropRate     float64 `json:"drop_rate,omitempty"`
	} `json:"metrics_delta,omitempty"`
	ObservedAtUnixMs int64 `json:"observed_at_unix_ms"`
}

type OutcomeAck struct {
	Accepted         bool   `json:"accepted"`
	ProposalID       string `json:"proposal_id"`
	ObservedAtUnixMs int64  `json:"observed_at_unix_ms"`
}

// ── Client ─────────────────────────────────────────────────────────────────

// Client talks to the local Python swarm sidecar via UDS or TCP.
type Client struct {
	mu         sync.RWMutex
	httpClient *http.Client
	token      string
	sidecarURL string

	// Cached state
	meta   *BridgeMeta
	leader *LeaderState
}

// Config configures the swarmbridge client.
type Config struct {
	// SidecarURL overrides the default. Use "unix:///var/run/tensorq/bridge.sock"
	// for UDS or "http://localhost:8888" for TCP. Empty = auto-detect.
	SidecarURL string
	// TokenPath overrides the default token file path. Empty = auto-detect.
	TokenPath string
	// Token overrides TokenPath — use this if you already have the token.
	Token string
}

// New creates a swarmbridge client. Call Start() to perform the startup
// sequence (meta → health → leader).
func New(cfg Config) (*Client, error) {
	c := &Client{
		sidecarURL: DefaultSidecarURL,
	}

	if cfg.SidecarURL != "" {
		c.sidecarURL = cfg.SidecarURL
	}

	// Load token
	if cfg.Token != "" {
		c.token = cfg.Token
	} else {
		tokenPath := cfg.TokenPath
		if tokenPath == "" {
			tokenPath = DefaultTokenPath
		}
		if data, err := os.ReadFile(tokenPath); err == nil {
			c.token = string(data)
		}
		// If token missing, we continue — requests will fail with UNAUTHORIZED
	}

	// Build HTTP client with appropriate transport
	if err := c.buildHTTPClient(); err != nil {
		return nil, fmt.Errorf("build http client: %w", err)
	}

	return c, nil
}

func (c *Client) buildHTTPClient() error {
	transport := &http.Transport{
		DisableKeepAlives: false,
	}

	// UDS support
	if len(c.sidecarURL) >= 5 && c.sidecarURL[:5] == "unix:" {
		socketPath := c.sidecarURL[5:]
		transport.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			d := net.Dialer{Timeout: ConnectTimeout}
			return d.DialContext(ctx, "unix", socketPath)
		}
		// For UDS, the host part of the URL is ignored by net/http,
		// so we set a fake URL that http.ParseURL accepts.
		c.sidecarURL = "http://unix"
	}

	c.httpClient = &http.Client{
		Transport: transport,
		Timeout:   5 * time.Second,
	}

	return nil
}

// ── Startup Sequence ───────────────────────────────────────────────────────

// Start performs the bridge startup sequence: meta → health → leader.
// Returns an error if any step fails.
func (c *Client) Start(ctx context.Context) error {
	// 1. Validate meta
	meta, err := c.GetMeta(ctx)
	if err != nil {
		return fmt.Errorf("get meta: %w", err)
	}
	c.meta = meta

	// 2. Wait for healthy
	if err := c.WaitUntilReady(ctx); err != nil {
		return fmt.Errorf("wait ready: %w", err)
	}

	// 3. Fetch initial leader state
	leader, err := c.GetLeader(ctx)
	if err != nil {
		return fmt.Errorf("get leader: %w", err)
	}
	c.leader = leader

	return nil
}

// WaitUntilReady polls /v1/health with exponential backoff until READY or context cancels.
func (c *Client) WaitUntilReady(ctx context.Context) error {
	delay := HealthPollInterval
	deadline := time.Now().Add(HealthPollMaxWait)

	for time.Now().Before(deadline) {
		h, err := c.GetHealth(ctx)
		if err == nil && h.Status == "READY" {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}

		delay = min64(delay*2, 5*time.Second)
	}

	return fmt.Errorf("bridge not ready after %s", HealthPollMaxWait)
}

func min64(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

// ── Bridge Endpoints ───────────────────────────────────────────────────────

// GetMeta fetches protocol discovery info.
func (c *Client) GetMeta(ctx context.Context) (*BridgeMeta, error) {
	var out BridgeMeta
	if err := c.doJSON(ctx, http.MethodGet, "/v1/meta", nil, &out, ConnectTimeout); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetHealth checks bridge readiness.
func (c *Client) GetHealth(ctx context.Context) (*HealthStatus, error) {
	var out HealthStatus
	if err := c.doJSON(ctx, http.MethodGet, "/v1/health", nil, &out, ConnectTimeout); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetLeader fetches current leader/epoch/quorum state.
func (c *Client) GetLeader(ctx context.Context) (*LeaderState, error) {
	var out LeaderState
	if err := c.doJSON(ctx, http.MethodGet, "/v1/leader", nil, &out, LeaderTimeout); err != nil {
		return nil, err
	}
	c.mu.Lock()
	c.leader = &out
	c.mu.Unlock()
	return &out, nil
}

// PostTelemetry pushes a batch of samples to the swarm.
func (c *Client) PostTelemetry(ctx context.Context, batch TelemetryBatch) (*TelemetryAck, error) {
	var out TelemetryAck
	if err := c.doJSON(ctx, http.MethodPost, "/v1/telemetry", batch, &out, TelemetryTimeout); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetAlphaProposal fetches the current alpha proposal for a VIP.
// Returns nil proposal if quorum is unhealthy or no data available.
func (c *Client) GetAlphaProposal(ctx context.Context, vip string) (*ProposalResponse, error) {
	url := "/v1/proposals/alpha?vip=" + vip
	var out ProposalResponse
	if err := c.doJSON(ctx, http.MethodGet, url, nil, &out, ProposalTimeout); err != nil {
		return nil, err
	}
	return &out, nil
}

// PostFreeze submits a freeze proposal request.
func (c *Client) PostFreeze(ctx context.Context, req FreezeProposal) (*FreezeAck, error) {
	var out FreezeAck
	if err := c.doJSON(ctx, http.MethodPost, "/v1/proposals/freeze", req, &out, TelemetryTimeout); err != nil {
		return nil, err
	}
	return &out, nil
}

// PostOutcome reports a mutation outcome back to the swarm.
func (c *Client) PostOutcome(ctx context.Context, outcome ProposalOutcome) (*OutcomeAck, error) {
	var out OutcomeAck
	if err := c.doJSON(ctx, http.MethodPost, "/v1/outcomes", outcome, &out, OutcomeTimeout); err != nil {
		return nil, err
	}
	return &out, nil
}

// ── State Accessors ────────────────────────────────────────────────────────

// Leader returns the cached leader state (may be stale — call GetLeader to refresh).
func (c *Client) Leader() *LeaderState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.leader == nil {
		return nil
	}
	cp := *c.leader
	return &cp
}

// IsQuorumHealthy returns true if the last known quorum status is HEALTHY.
func (c *Client) IsQuorumHealthy() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.leader != nil && c.leader.QuorumStatus == "HEALTHY"
}

// IsLeaderLeaseValid returns true if the current time is within the leader's
// lease window and the leader ID is non-nil.
func (c *Client) IsLeaderLeaseValid() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.leader == nil || c.leader.LeaderID == nil {
		return false
	}
	leaseExpiry := time.UnixMilli(c.leader.LeaseExpiresAtMs)
	return time.Now().Before(leaseExpiry)
}

// GetConsensus fetches the current consensus decision from the Python overseer.
// TODO: Implement actual consensus fetching from the bridge sidecar.
func (c *Client) GetConsensus(ctx context.Context) (string, error) {
	return "", fmt.Errorf("GetConsensus not yet implemented")
}

// CallSolvePhysics invokes the JAX QFT-Engine for physics model solving.
// TODO: Implement actual RPC to JAX QFT-Engine via the bridge sidecar.
func (c *Client) CallSolvePhysics(alphaURI string, modelID string) (string, error) {
	return "", fmt.Errorf("CallSolvePhysics not yet implemented")
}

// ── HTTP Helper ────────────────────────────────────────────────────────────

func (c *Client) doJSON(ctx context.Context, method, path string, body, result interface{}, timeout time.Duration) error {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("marshal request: %w", err)
		}
		bodyReader = io.NopCloser(bytes.NewReader(data))
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, method, c.sidecarURL+path, bodyReader)
	if err != nil {
		return fmt.Errorf("create request %s %s: %w", method, path, err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.token != "" {
		req.Header.Set("X-TensorQ-Bridge-Token", c.token)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response %s %s: %w", method, path, err)
	}

	if resp.StatusCode >= 400 {
		return fmt.Errorf("bridge error %d %s %s: %s", resp.StatusCode, method, path, truncate(string(respBody), 512))
	}

	if result != nil {
		if err := json.Unmarshal(respBody, result); err != nil {
			return fmt.Errorf("unmarshal response %s %s: %w (body: %s)", method, path, err, truncate(string(respBody), 256))
		}
	}

	return nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
