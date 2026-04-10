 docs/architecture/30-swarm-integration.md, aligned with the Route Mutation Firewall guardrails. This defines the Pyro5 swarm’s responsibilities, the narrow bridge contract to Go, and the leader/epoch mechanics that prevent split-brain from touching kernel state.

⸻

Swarm Integration Spec — Pyro5 “Modern Meta” Brain + Go/eBPF Muscle

0) Purpose

Define a safe, scalable integration between:
	•	Python Pyro5 Swarm (“Brain”): decentralized coordination, leader election, Darwinian/GA logic, proposal generation.
	•	Go Control/Actuation (“Muscle”): deterministic admission, map staging, probing, commit/rollback, observability, eBPF program lifecycle.

This spec ensures:
	•	swarm failures do not stop the datapath,
	•	swarm partitions do not cause split-brain kernel updates,
	•	AI proposals remain bounded and auditable.

⸻

1) Roles and Responsibilities

1.1 Swarm Brain (Python/Pyro5)

Responsibilities
	•	Membership tracking (who is in the swarm)
	•	Leader election and epoch management
	•	Aggregation of performance metrics from coordinators
	•	GA selection: compute candidate “alpha” routes
	•	Produce intent proposals (not kernel-ready actuation)

Non-responsibilities
	•	No eBPF map writes
	•	No MAC/next-hop authority
	•	No direct calls to mutation-firewalld unless explicitly routed through the coordinator with auth and fencing

1.2 Overseer Coordinator (Go)

Responsibilities
	•	Collect metrics from datapath and application layers
	•	Submit metrics to local swarm node
	•	Pull proposals from local swarm node
	•	Forward proposals to mutation-firewalld with correct identity and fencing metadata
	•	Track proposal lifecycle and outcomes (applied/rolled back) and report back to swarm

Non-responsibilities
	•	No direct eBPF map writes
	•	No bypass of firewall admission pipeline

1.3 Mutation Firewall (Go, privileged)

Responsibilities
	•	Sole authority to commit route mutations
	•	Deterministic admission, rate limiting, staging, probing, rollback
	•	Resolves MAC/next-hop, validates endpoint state
	•	Enforces leader epoch fencing and idempotency
	•	Produces audit logs and metrics

1.4 eBPF Loader (Go, privileged)

Responsibilities
	•	Load/attach programs (TC/XDP)
	•	Create/pin maps
	•	Version map schemas and program artifacts
	•	Provide compatibility matrix (kernel features)
	•	Controlled upgrades and restarts

⸻

2) Deployment Topology

2.1 Recommended “Sidecar Brain” Pod

Deploy swarm node and coordinator together to guarantee locality and reduce attack surface.

Pod composition
	•	Container A: swarm-node (Python Pyro5 + local bridge server)
	•	Container B: overseer-coordinatord (Go)
	•	Shared emptyDir volume for UDS sockets and readiness files

Rationale
	•	Bridge is local-only (UDS), non-routable.
	•	Coordinator survives swarm restart but freezes proposals until leader confirmed.
	•	Datapath is isolated: gatewayd and mutation-firewalld can run separately.

2.2 Swarm as StatefulSet

Swarm nodes should be deployed as a StatefulSet to ensure:
	•	stable node identity
	•	predictable membership naming
	•	optional persistent state (LKG proposals, epoch record)

⸻

3) Network and Security Boundaries

3.1 Local Bridge Requirements (Python ↔ Go)

Bridge must be:
	•	UDS or localhost-only port bound to 127.0.0.1
	•	not exposed via Service
	•	restricted via file permissions if UDS
	•	optionally protected by a capability token stored in the shared volume

Absolutely prohibited
	•	exposing Pyro daemon to cluster network without authentication and explicit policy
	•	direct swarm → firewall network calls without coordinator mediation

3.2 Identity and Auth (Coordinator → Firewall)
	•	mTLS required
	•	RBAC: per coordinator identity (or namespace) allowed VIP ranges
	•	Requests carry epoch and mutation_id for fencing/idempotency

⸻

4) Swarm Consensus and Epoch Fencing

4.1 Definitions
	•	Leader: single swarm node authorized to finalize proposals.
	•	Epoch: monotonically increasing integer representing the “term” of the leader.
	•	Lease: leader authority validity window.

4.2 Epoch Invariants
	1.	Epoch must never decrease.
	2.	Only the leader for the current epoch may emit proposals for commitment.
	3.	If a leader loses quorum, it must stop producing proposals immediately.

4.3 Fencing Contract with Firewall

Every proposal forwarded to the firewall must include:
	•	epoch
	•	leader_id
	•	lease_expires_at
	•	proposal_id (or mutation_id)
	•	expected_route_version (optional but strongly recommended)

Firewall acceptance rule
	•	Reject if epoch < last accepted epoch for that VIP scope
	•	Reject if lease expired
	•	Reject if leader_id doesn’t match expected leader identity (if configured)
	•	Reject if proposal replay without idempotency match

4.4 Split-Brain Handling

Swarm partition may yield two “leaders.” The firewall must ensure:
	•	only one epoch stream is accepted
	•	stale epoch stream is rejected deterministically
	•	repeated stale epoch proposals trigger throttling and an audit alert

⸻

5) Proposal Model (“Intent”, not Actuation)

5.1 Proposal Types

Swarm produces intent proposals, never direct kernel entries.

Minimum proposal types:
	•	AlphaRouteProposal (single VIP → chosen backend)
	•	WeightedRouteProposal (VIP → backend set + weights) [future]
	•	FreezeProposal (stop mutating VIP due to instability)
	•	RollbackRecommendation (advisory only; firewall decides)

5.2 Proposal Payload Fields

Each proposal must include:
	•	proposal_id (globally unique)
	•	epoch, leader_id, lease_expires_at
	•	vip
	•	target_identity (pod IP or endpoint identity)
	•	ttl_ms
	•	fitness_summary (metrics used)
	•	confidence (0..1)
	•	constraints:
	•	max allowed step change
	•	canary requirement (yes/no)
	•	stability window requirements

5.3 Explicitly excluded fields
	•	target_mac_address must be treated as a hint only, not required
	•	Any kernel offsets, map file paths, or interface indices are forbidden in proposals

⸻

6) Metrics Ingestion and Feedback Loop Control

6.1 Metrics sources

Coordinator sends to swarm:
	•	latency distributions (p50/p95/p99)
	•	error rates
	•	drop counters (from BPF/Hubble)
	•	throughput/load
	•	backend health signals (readiness, probes)
	•	optional: cost signals (CPU/memory)

6.2 Feedback control guardrails

To avoid oscillation (“GA thrash”), swarm must implement:
	•	sampling window: proposals must be computed over stable windows, not instantaneous spikes
	•	hysteresis: require minimum improvement to replace current alpha
	•	cooldowns: no more than N alpha changes per VIP per time window
	•	stability scoring: penalize frequent rollbacks and volatility

6.3 Outcome reporting (Firewall → Swarm)

Coordinator must report:
	•	accepted/rejected reason
	•	applied version
	•	probe pass/fail summary
	•	rollback reason (if any)
	•	post-commit SLO deltas

Swarm must incorporate these into fitness (e.g., penalize proposals that trigger rollback).

⸻

7) Local Bridge API Contract (Narrow, Versioned)

7.1 Required endpoints (conceptual)

The Go coordinator must be able to do:
	•	GetLeader() → {leader_id, epoch, lease_expires_at, quorum_status}
	•	SubmitMetrics() → ack + ingestion status
	•	GetProposal(vip) → latest intent proposal for VIP (or “none”)
	•	ReportOutcome(proposal_id) → used for GA fitness updates
	•	Health() → readiness/liveness

7.2 Versioning

Bridge must declare:
	•	bridge_protocol_version
	•	supported features (weighted routing, freeze, etc.)

Backward compatibility rules:
	•	additive fields allowed
	•	unknown fields ignored
	•	breaking changes require new version path or explicit version bump

⸻

8) Fault Handling Requirements

8.1 Swarm node crash
	•	Coordinator continues running but transitions to NO_PROPOSALS mode.
	•	Firewall continues serving; datapath continues with last committed routes.

8.2 Leader loss / quorum loss
	•	Leader stops proposals; followers do not “guess” proposals.
	•	Coordinator refuses to forward proposals if quorum status is unhealthy.

8.3 Bridge unresponsive
	•	Coordinator backs off with exponential retry
	•	After threshold, emits alert and stops proposing

8.4 Swarm partition
	•	Competing leaders may exist.
	•	Only proposals with newest epoch and valid lease are admissible at firewall.
	•	Stale epoch proposals trigger warnings and rate limiting.

⸻

9) Operational Controls

9.1 Manual override modes
	•	freeze VIP (block new proposals for VIP)
	•	freeze all mutations (global kill switch)
	•	require human approval (switch firewall into approval mode)
	•	force rollback to LKG (emergency)

9.2 Change management
	•	new swarm logic versions must be canaried
	•	swarm protocol version changes require compatibility tests
	•	coordinator ↔ bridge integration tests must run in CI

⸻

10) Testing Requirements

10.1 Unit tests (Python)
	•	leader election: epoch monotonicity
	•	quorum detection
	•	proposal generation stability
	•	cooldown/hysteresis enforcement
	•	serialization/version compatibility

10.2 Integration tests (Kind)

Scenarios:
	•	leader failover: epoch increments, firewall accepts new epoch
	•	split-brain: firewall rejects stale epoch proposals
	•	metrics spike: no immediate oscillation due to hysteresis
	•	rollback feedback: swarm penalizes unstable proposals

⸻

11) Non-Negotiable Invariants Checklist
	1.	Swarm never commits kernel changes directly.
	2.	Firewall is the only kernel mutation authority.
	3.	All proposals carry epoch + lease (fencing).
	4.	Coordinator forwards proposals only when leader/quorum is healthy.
	5.	Firewall rejects stale epochs deterministically.
	6.	Outcome feedback is mandatory; swarm learns from rollback events.
	7.	Bridge is local-only and versioned.

⸻

