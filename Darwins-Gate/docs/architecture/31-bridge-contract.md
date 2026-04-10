docs/architecture/31-bridge-contract.md

Bridge Contract (Python Swarm ↔ Go Coordinator)

This document defines the local-only, versioned, narrow interface (“Bridge”) used by the Go overseer-coordinatord to interact with the Python Pyro5 swarm node. The bridge exists to avoid coupling Go to Pyro5 internals and to enforce a safe trust boundary between “decision” and “actuation.”

Key principle: The bridge exchanges intent and telemetry, never kernel-ready actuation.

⸻

1) Scope

In scope
	•	Local Go ↔ Python IPC API for:
	•	leader/epoch discovery
	•	telemetry ingestion
	•	proposal retrieval
	•	outcome reporting
	•	health/readiness

Out of scope
	•	Pyro5 swarm internal RPC protocol
	•	Direct access to eBPF maps or Cilium internals
	•	External network exposure (bridge is not a cluster service)

⸻

2) Transport Requirements

2.1 Allowed transports (choose one; both allowed)

Preferred: Unix Domain Socket (UDS) + HTTP/1.1 JSON
Alternative: localhost TCP (127.0.0.1) + HTTP/1.1 JSON
(Optional future: local gRPC over UDS)

2.2 Security requirements
	•	Bridge server MUST bind to UDS or 127.0.0.1 only.
	•	If UDS:
	•	socket file permissions MUST restrict access to the coordinator user/group.
	•	socket path MUST be in a shared emptyDir volume when running as sidecar.
	•	No Kubernetes Service object may expose the bridge.
	•	Requests MUST include a capability token (see §6), unless file permissions alone are deemed sufficient for your threat model.

2.3 Timeouts and retries
	•	Coordinator MUST enforce:
	•	connect timeout: ≤ 250ms
	•	request timeout: ≤ 1s (leader/proposal), ≤ 2s (submit metrics), ≤ 2s (report outcome)
	•	Coordinator MUST implement exponential backoff with jitter when bridge is unavailable.

⸻

3) Versioning and Compatibility

3.1 Protocol version

Bridge protocol version is a semantic version: MAJOR.MINOR.PATCH
	•	MAJOR: breaking changes to endpoints or required fields
	•	MINOR: additive fields/endpoints, backward compatible
	•	PATCH: bug fixes, no schema changes

3.2 Negotiation
	•	Coordinator calls GET /v1/meta at startup.
	•	Coordinator MUST refuse to operate if:
	•	bridge_protocol_major differs
	•	required features are missing

3.3 Forward/backward compatibility rules
	•	Additive fields are allowed.
	•	Unknown fields MUST be ignored.
	•	Required field removal requires MAJOR bump.
	•	If a request contains unknown fields, Python MUST ignore them (do not reject).

⸻

4) Identity, Epoch, and Lease Concepts

4.1 Terms
	•	node_id: stable identifier for a swarm node (e.g., StatefulSet ordinal)
	•	leader_id: node_id of current leader
	•	epoch: monotonically increasing integer term (leader fencing)
	•	lease_expires_at_unix_ms: leader authority validity; proposals beyond lease are invalid
	•	quorum_status: HEALTHY | DEGRADED | LOST

4.2 Invariants
	1.	epoch MUST NOT decrease over time.
	2.	Proposals MUST include (leader_id, epoch, lease_expires_at_unix_ms).
	3.	Coordinator MUST NOT forward proposals when:
	•	quorum_status != HEALTHY
	•	lease is expired
	4.	Bridge MUST NOT return proposals with expired lease.

⸻

5) Data Model (JSON Schemas)

All requests/responses are JSON with Content-Type: application/json.

5.1 Common envelope

Responses MAY include an envelope:
	•	ok: boolean
	•	error: BridgeError | null
	•	data: object | null

BridgeError
	•	code: string (see §9)
	•	message: string (human-readable)
	•	retryable: boolean
	•	details: object (optional)

5.2 Meta

BridgeMeta
	•	bridge_protocol_version: string (e.g., “1.0.0”)
	•	node_id: string
	•	features: object
	•	leader_fencing: boolean
	•	weighted_routing: boolean
	•	freeze_proposals: boolean
	•	outcome_feedback: boolean
	•	build:
	•	git_sha: string
	•	build_time_unix_ms: int64

5.3 Leader view

LeaderState
	•	leader_id: string | null
	•	epoch: int64
	•	lease_expires_at_unix_ms: int64
	•	quorum_status: “HEALTHY” | “DEGRADED” | “LOST”
	•	members:
	•	size: int32
	•	known_ids: string[] (optional)
	•	observed_at_unix_ms: int64

5.4 Proposals

AlphaRouteProposal
	•	proposal_id: string
	•	leader_id: string
	•	epoch: int64
	•	lease_expires_at_unix_ms: int64
	•	vip: string
	•	target:
	•	type: “POD_IP” | “ENDPOINT_ID”
	•	pod_ip: string (if type=POD_IP)
	•	endpoint_id: string (if type=ENDPOINT_ID)
	•	ttl_ms: int64
	•	confidence: float (0..1)
	•	fitness_summary:
	•	window_ms: int64
	•	metrics:
	•	latency_p95_ms: float
	•	latency_p99_ms: float
	•	error_rate: float
	•	drop_rate: float
	•	throughput_rps: float
	•	constraints:
	•	require_canary: boolean
	•	max_step_change: float (0..1)  # e.g., max 25% shift per step
	•	min_improvement: float
	•	stability_window_ms: int64
	•	reason: string (optional)
	•	created_at_unix_ms: int64

FreezeProposal
	•	proposal_id: string
	•	leader_id: string
	•	epoch: int64
	•	lease_expires_at_unix_ms: int64
	•	scope:
	•	type: “VIP” | “GLOBAL”
	•	vip: string (if VIP)
	•	duration_ms: int64
	•	reason: string
	•	created_at_unix_ms: int64

5.5 Telemetry

TelemetryBatch
	•	batch_id: string
	•	node_id: string
	•	sent_at_unix_ms: int64
	•	samples: TelemetrySample[]
	•	window_ms: int64

TelemetrySample
	•	vip: string
	•	backend_hint:
	•	pod_ip: string (optional)
	•	endpoint_id: string (optional)
	•	latency_ms:
	•	p50: float
	•	p95: float
	•	p99: float
	•	errors:
	•	rate: float
	•	codes: object<string,int64> (optional)
	•	drops:
	•	rate: float
	•	reasons: object<string,int64> (optional)
	•	throughput_rps: float
	•	cpu_util: float (optional)
	•	mem_util: float (optional)

5.6 Outcome feedback

ProposalOutcome
	•	proposal_id: string
	•	vip: string
	•	status: “REJECTED” | “STAGED” | “APPLIED” | “PROBE_FAILED” | “ROLLED_BACK”
	•	applied_route_version: int64 (optional)
	•	rejection:
	•	code: string (optional)
	•	message: string (optional)
	•	probe:
	•	pass_ratio: float (optional)
	•	window_ms: int64 (optional)
	•	rollback:
	•	reason: string (optional)
	•	metrics_delta:
	•	latency_p95_ms: float (optional)
	•	drop_rate: float (optional)
	•	observed_at_unix_ms: int64

⸻

6) Authentication (Capability Token)

6.1 Token format
	•	Coordinator includes header: X-TensorQ-Bridge-Token: <token>
	•	Token is a random secret generated per pod startup, written to a shared volume path:
	•	e.g., /var/run/tensorq/bridge.token

6.2 Token rules
	•	Token is required for TCP-localhost mode.
	•	For UDS mode, token may be optional if file permissions are strict; recommended to keep token anyway.
	•	Token rotation is allowed on pod restart; no long-lived tokens.

⸻

7) Endpoints

All endpoints are under /v1.

7.1 GET /v1/meta

Purpose: protocol discovery and feature negotiation.
	•	Response: BridgeMeta

Coordinator behavior
	•	Must call at startup.
	•	Must verify protocol major and required features.

7.2 GET /v1/health

Purpose: liveness/readiness.
	•	Response:
	•	status: “READY” | “NOT_READY”
	•	reason: string (optional)
	•	observed_at_unix_ms: int64

7.3 GET /v1/leader

Purpose: fetch current leader/epoch/quorum state.
	•	Response: LeaderState

7.4 POST /v1/telemetry

Purpose: coordinator pushes metrics batches to swarm node.
	•	Request: TelemetryBatch
	•	Response:
	•	accepted: boolean
	•	ingested_samples: int32
	•	batch_id: string
	•	observed_at_unix_ms: int64

Rules:
	•	Bridge may accept telemetry even if quorum is degraded, but must record quorum status.

7.5 GET /v1/proposals/alpha?vip=<vip>

Purpose: fetch latest alpha proposal for a VIP.
	•	Query:
	•	vip (required)
	•	Response:
	•	proposal: AlphaRouteProposal | null
	•	none_reason: “NO_LEADER” | “NO_QUORUM” | “NO_DATA” | “COOLDOWN” | null

Rules:
	•	Must return proposal=null if:
	•	quorum not healthy
	•	lease expired
	•	no proposal available

7.6 POST /v1/proposals/freeze

Purpose: coordinator requests a freeze (optional; usually leader emits it).
	•	Request: FreezeProposal
	•	Response:
	•	accepted: boolean
	•	proposal_id: string
	•	observed_at_unix_ms: int64

7.7 POST /v1/outcomes

Purpose: coordinator reports mutation results back to swarm for GA learning.
	•	Request: ProposalOutcome
	•	Response:
	•	accepted: boolean
	•	proposal_id: string
	•	observed_at_unix_ms: int64

Rules:
	•	Swarm must incorporate outcomes into fitness.
	•	Outcomes must be idempotent by proposal_id.

⸻

8) Coordinator Behavioral Requirements

8.1 Startup sequence
	1.	GET /v1/meta → validate version/features
	2.	GET /v1/health until READY (with backoff)
	3.	GET /v1/leader → cache leader state
	4.	Start telemetry loop
	5.	Start proposal loop

8.2 Proposal forwarding rules

Coordinator MUST NOT forward a proposal to the firewall unless:
	•	LeaderState.quorum_status == HEALTHY
	•	now < lease_expires_at_unix_ms
	•	proposal.epoch == leader.epoch and proposal.leader_id == leader.leader_id

8.3 Backpressure

Coordinator MUST reduce telemetry/proposal frequency if:
	•	bridge returns OVERLOADED retryable errors
	•	latency spikes in bridge responses

⸻

9) Error Codes

BridgeError.code values:
	•	UNAUTHORIZED (token missing/invalid)
	•	UNSUPPORTED_VERSION
	•	NOT_READY
	•	NO_LEADER
	•	NO_QUORUM
	•	LEASE_EXPIRED
	•	INVALID_ARGUMENT
	•	OVERLOADED (retryable)
	•	INTERNAL (retryable depending on details)

Coordinator MUST treat as retryable only when retryable=true.

⸻

10) Logging and Observability Requirements

Bridge MUST emit structured logs for:
	•	leader changes (epoch bump)
	•	quorum status changes
	•	proposal generation events (proposal_id, vip)
	•	telemetry ingestion (batch_id, sample count)
	•	outcomes ingestion (proposal_id, status)

Bridge SHOULD expose metrics (implementation detail) for:
	•	request counts/latency per endpoint
	•	proposal production rate
	•	telemetry ingestion rate
	•	outcome ingestion rate
	•	quorum time-in-state

⸻

11) Security and Hardening Checklist
	•	Bridge binds to UDS or 127.0.0.1 only
	•	No Service exposes bridge
	•	Token enabled (especially for localhost TCP)
	•	Coordinator uses timeouts and backoff
	•	Bridge rejects large bodies (size limits)
	•	Schema validation on all requests
	•	All proposals include epoch + lease
	•	Bridge never returns proposals if quorum unhealthy

⸻

12) Size Limits and Resource Controls

Recommended defaults:
	•	Max request body:
	•	telemetry: 1–5 MB (cap with server limit)
	•	outcomes: 256 KB
	•	Max samples per telemetry batch: 1k–10k (depending on window)
	•	Rate limits:
	•	telemetry: ≤ 10 Hz per coordinator (prefer 1–2 Hz initially)
	•	proposals: ≤ 2 Hz per VIP (usually lower; GA windows should be seconds-minutes)

⸻

13) Appendix: “Intent vs Actuation” Contract (Non-negotiable)

Bridge outputs
	•	leader state
	•	proposal intent (VIP + target identity + constraints)
	•	never: MAC/ifindex/kernel paths/map names

Firewall outputs (back to swarm via outcomes)
	•	accepted/rejected reasons
	•	probe summary
	•	applied/rollback status

This ensures the swarm remains portable and the kernel remains protected.

⸻
