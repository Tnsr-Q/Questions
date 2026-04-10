# Guardrails Spec — Route Mutation Firewall (AI/Swarm → Go → eBPF)

This is a **no-code** spec . It defines the deterministic safety envelope around `UpdateAlphaRoute` so Python/Pyro5 (or any AI) can propose changes without risking kernel/network stability.

---

## 0) Scope and Goals

### In scope

* Admission, staging, commit, probe, monitoring, rollback for **VIP → backend** routing mutations.
* Trust boundaries between:

  * Swarm/AI (Python Pyro5)
  * Overseer coordinator (Go)
  * Mutation Firewall (Go, privileged)
  * eBPF loader/map writer (Go, privileged)
  * Kernel datapath program(s) (TC/XDP)

### Out of scope (explicitly)

* The GA algorithm itself (selection, fitness computation).
* Policy engine semantics beyond mutation admission (CiliumNetworkPolicy, L7 policy).
* Full multi-cluster federation details (but hooks are defined).

### Primary goals

1. **Safety:** AI can’t brick networking or corrupt kernel state.
2. **Correctness:** Mutations map to reachable forwarding actions.
3. **Scalability:** Prevent thrash (map churn, route flapping).
4. **Observability:** Every mutation is attributable and measurable.
5. **Reversibility:** Rollback is automatic, fast, and reliable.

---

## 1) Trust Model & Threat Model

### Actors

* **Swarm Brain (Python/Pyro5):** proposes routes; may be faulty, partitioned, or compromised.
* **Overseer Coordinator (Go):** aggregates metrics, requests proposals, forwards to firewall.
* **Mutation Firewall (Go):** sole authority to admit/commit kernel changes.
* **Kernel datapath:** executes route rewriting; cannot “reason,” only enforce.

### Threats to defend against

* Malformed inputs (bad IP/MAC strings, IPv6/IPv4 confusion, zero values).
* Stale proposals (pod moved, MAC changed, endpoint deleted).
* Split-brain swarm leaders issuing conflicting “alpha” routes.
* Mutation storms (excessive rate), route flapping, oscillations.
* Resource exhaustion: BPF map full, CPU spikes, event floods.
* L2/L3 mismatch causing silent drops.
* Privilege escalation via bridge interface (non-local access, spoofed identity).

### Trust boundaries (must be explicit)

* **Boundary A:** Swarm → Local Bridge (untrusted input; treat as hostile).
* **Boundary B:** Coordinator → Firewall (mTLS, authz, quotas).
* **Boundary C:** Firewall → Kernel (privileged; only deterministic operations).

---

## 2) API Contract and Semantics

### Public API (current)

* `CortexGateway.UpdateAlphaRoute(RouteMutation) -> RouteAck`

### Internal semantics (guardrailed)

* `UpdateAlphaRoute` is interpreted as **“propose route mutation”**, not “force commit now.”
* Firewall may respond with:

  * **APPLIED:** committed and active
  * **STAGED:** staged but not yet active (waiting probes)
  * **REJECTED:** failed admission
  * **PROBE_FAILED:** staged but reverted
  * **ROLLED_BACK:** committed then reverted due to SLO breach

### Recommended proto extensions (optional but strongly recommended)

Add to `RouteMutation` (or embed via metadata headers if you must):

* `mutation_id` (idempotency key)
* `expected_route_version` (optimistic concurrency)
* `ttl_ms` (expiration / fail-safe)
* `dry_run` (admission only)
* `epoch` (leader fencing)
* `confidence` / `fitness_score` (for audit)
* `reason` (human-readable)

Add to `RouteAck`:

* `status` enum
* `applied_route_version`
* `checks_passed[]` / `checks_failed[]`
* `rollback_reason`
* `observability_ids` (trace/span IDs)

---

## 3) State Model

### Route Record (logical)

Each VIP has an authoritative record:

* `vip`
* `active_entry` (backend + resolved next-hop details)
* `active_version` (monotonic)
* `last_known_good_entry` (LKG)
* `staged_entry` (optional)
* `epoch` (from leader fencing)
* `applied_at`
* `expires_at` (if TTL used)
* `health_status` (OK / degraded / quarantined)

### Kernel Map Model (physical)

Preferred: **two-phase map commit**

* `routing_map_A`
* `routing_map_B`
* `active_map_idx` (0/1)
* `counters_map` (hit/miss/errors)

---

## 4) Admission Pipeline (Deterministic Validators)

Admission must be **pure and deterministic**. No “AI reasoning” here.

### 4.1 Input parsing and canonicalization

Reject if any fail:

* `virtual_ip` parses to valid IP
* `target_pod_ip` parses to valid IP
* Both are same IP family unless explicitly supporting dual-stack
* No unspecified/broadcast/multicast loopback unless explicitly allowed
* `target_mac_address` parses if present (but not authoritative)

Canonicalize:

* store IPs as network byte order `u32` (IPv4) or 16B for IPv6
* normalize MAC (lowercase, 6 bytes)

### 4.2 Policy constraints (static allowlists)

Reject if:

* VIP not in `AllowedVIPCIDRs`
* target IP not in `AllowedPodCIDRs`
* caller not authorized for that VIP scope
* requested TTL > `MaxTTL` or TTL missing when required
* mutation would violate “blast radius” limits (see §7)

### 4.3 Liveness / existence checks

Reject if:

* target pod IP is not currently a known endpoint (EndpointSlice/Cilium endpoint cache)
* endpoint is not “Ready” or fails minimum health criteria
* endpoint node is unschedulable / draining (optional policy)

### 4.4 Leader fencing (split-brain defense)

Reject if:

* request epoch < `vip.current_epoch`
* epoch missing when fencing enabled
* caller identity not matching current leader attestation (if used)

### 4.5 Idempotency and concurrency

* If `mutation_id` already applied: return the prior result (idempotent replay).
* If `expected_route_version` provided and != current: reject with conflict.

### 4.6 Rate limiting and budgets

Reject or defer if:

* per-VIP mutation rate exceeds `VIPMaxMutationsPerSecond`
* per-caller mutation rate exceeds `CallerMaxMPS`
* global mutation rate exceeds `GlobalMaxMPS`

---

## 5) Resolution (MAC/Next-Hop) Rules

### Key rule

**The firewall owns resolution.** The brain may provide MAC as a hint, not truth.

Resolution outputs:

* `next_hop_ip` (if routed)
* `next_hop_mac`
* `egress_ifindex` (optional)
* `l2_rewrite_mode` (none / mac-only / full L2+L3 rewrite)

Resolution must consider:

* Native routing vs encapsulation mode
* Whether the TC/XDP hook point sees post-decapsulation packets
* Whether the correct L2 neighbor is the target node, gateway, or router

Reject if:

* next hop cannot be resolved
* neighbor entry is incomplete and cannot be confirmed within timeout
* resolved MAC differs from provided MAC (if provided) AND `StrictMacMatch` enabled

---

## 6) Staging, Probing, and Commit

### 6.1 Staging

* Write candidate entry to **inactive** map (A/B).
* Mark VIP `staged_version = active_version + 1`.
* Do not flip active pointer yet.

### 6.2 Probe requirements (must pass before commit)

Define probes per VIP (configurable):

**Probe types**

* **Reachability probe:** a synthetic request from gateway to VIP must receive response.
* **Backend confirmation probe:** ensure traffic reaches the target (requires backend echo/ID or telemetry).
* **Latency probe:** p95/p99 within allowed delta from baseline.
* **Drop probe:** drop counters do not spike beyond threshold.

Probe windows:

* `ProbeWarmup` (e.g., 250ms–2s)
* `ProbeDuration` (e.g., 2–10s)
* `ProbeSuccessRatio` (e.g., ≥ 0.99)

### 6.3 Commit

Commit only if probes pass:

* Flip `active_map_idx`.
* Set VIP `active_version = staged_version`.
* Persist `last_known_good_entry = previous active_entry`.
* Start post-commit monitoring window (§6.4).

### 6.4 Post-commit monitoring window

For `MonitorWindow` (e.g., 30s–5m):

* Continuously evaluate SLO guardrails.
* If violated: rollback immediately.

---

## 7) Rollback Rules (Automatic)

Rollback must be **fast** and **unconditional** under defined triggers.

### 7.1 Immediate rollback triggers

* Probe failure (pre-commit)
* Post-commit SLO breach (within monitor window)
* Map update error
* Map pressure critical (near full) + update failures begin
* Detected route oscillation (flip-flop pattern)
* Kernel datapath error counters spike (checksum errors, store_bytes failures)

### 7.2 Rollback action

* Flip `active_map_idx` back (A/B pointer).
* Restore VIP `active_entry = last_known_good_entry`.
* Mark VIP state as `QUARANTINED` with cooldown.
* Emit audit event.

### 7.3 Cooldown / quarantine policy

If VIP rolls back N times in T minutes:

* Freeze mutations for that VIP for `CooldownDuration`.
* Require manual override or stricter admission for next attempt.

---

## 8) Anti-Flap and Stability Controls (Darwinian guardrails)

These prevent “GA thrash.”

### 8.1 Hysteresis

A new alpha must beat current by:

* `MinImprovement` (e.g., 5–10% latency improvement or error reduction)
* and sustain improvement for `StabilityWindow`

### 8.2 Step limits

Limit magnitude of change:

* weighted rollout only (if supported): start at 10%, then 25%, 50%, 100%
* or restrict to one backend switch per VIP per window

### 8.3 Mutation budgets

* Maximum unique backends tried per VIP per hour/day
* Maximum reversals per VIP per hour/day

---

## 9) Observability and Audit (Non-negotiable)

### 9.1 Audit log requirements

Every mutation attempt (accepted, staged, rejected, applied, rolled back) must record:

* timestamp (monotonic and wall)
* caller identity (mTLS subject, swarm node ID)
* mutation_id, epoch, expected_version
* before/after entries (VIP, target identity, resolved next hop)
* admission checks passed/failed
* probe summary
* commit result
* rollback reason (if any)
* correlation IDs (trace/span)

Audit storage must be:

* append-only (WORM semantics preferred)
* queryable by VIP, mutation_id, time window

### 9.2 Metrics requirements

Minimum metrics:

* `mutations_total{status,vip}`
* `mutation_latency_seconds{phase}`
* `probe_pass_ratio{vip}`
* `rollback_total{vip,reason}`
* `map_update_failures_total{map}`
* `map_occupancy_ratio{map}`
* `datapath_hit_total{vip}`, `datapath_miss_total{vip}`
* `drops_total{reason}` (from BPF counters and/or Hubble)

### 9.3 Tracing requirements

* Trace each mutation across:

  * coordinator request
  * firewall admission
  * staging
  * probe
  * commit/rollback

---

## 10) Failure Mode Requirements (Explicit Behaviors)

### 10.1 Swarm down

* No new route changes, but datapath continues.
* Firewall returns `REJECTED` with reason `NO_LEADER` (or similar).
* If an active route TTL expires, fallback to LKG or a safe default.

### 10.2 Swarm split-brain

* Only one epoch is accepted.
* Requests with stale epoch are rejected.

### 10.3 Kernel map full / pressure

* Firewall rejects mutations once `map_occupancy_ratio > HighWaterMark`.
* Autoscale or increase map capacity in controlled maintenance window.

### 10.4 Probe system unhealthy

* Do not commit if probes can’t run.
* Return `STAGED` or `REJECTED` depending on policy.

---

## 11) Security Controls

### 11.1 Local bridge hardening (Python ↔ Go)

* Local-only (UDS or localhost bound)
* filesystem permissions on UDS
* capability token if needed
* no remote network access to bridge port

### 11.2 AuthN/AuthZ

* Coordinator → firewall: mTLS required
* RBAC: caller allowed VIP ranges
* Optional: leader attestation signature (future)

### 11.3 Privilege separation

* `gatewayd` non-privileged
* `overseer-coordinatord` non-privileged
* `mutation-firewalld` privileged for map access only
* `ebpf-loaderd` privileged for attach/detach
* Never co-locate loader privilege with public ingress.

---

## 12) Test Requirements (CI and Integration)

### 12.1 CI gates (must pass)

* eBPF verifier acceptance for supported kernel versions
* struct layout compatibility test (C header ↔ Go struct)
* fuzz tests for IP/MAC parsing and admission rules
* idempotency/concurrency tests

### 12.2 Integration scenarios

* route switch success path
* route switch probe fail → rollback
* flapping prevention (rapid alternating alphas)
* swarm leader change (epoch bump)
* swarm partition (conflicting epochs)
* map pressure high-water behavior
* kernel attach/detach during live traffic (controlled)

---

## 13) Configuration Parameters (Recommended defaults)

Group configs:

### Admission

* `AllowedVIPCIDRs`
* `AllowedPodCIDRs`
* `VIPMaxMutationsPerSecond`
* `CallerMaxMPS`
* `GlobalMaxMPS`
* `MaxTTL`, `MinTTL`
* `StrictMacMatch` (default: false)
* `LeaderFencingEnabled` (default: true)

### Probe

* `ProbeWarmup`
* `ProbeDuration`
* `ProbeSuccessRatio`
* `MonitorWindow`
* `MaxLatencyDelta`
* `MaxDropSpike`

### Stability

* `MinImprovement`
* `StabilityWindow`
* `CooldownDuration`
* `MaxRollbacksPerHour`

### Maps

* `MaxEntriesRoutingMap`
* `HighWaterMark` / `CriticalWaterMark`
* `EnableABMaps` (default: true)

---

## 14) Invariants Checklist (the “do not violate” list)

1. Kernel mutation authority exists only in mutation-firewall + loader.
2. No commit without successful probes (unless manual override mode).
3. Every commit has an immediate rollback path.
4. The firewall never trusts MAC as authoritative; it resolves next hop.
5. Split-brain is fenced by epoch monotonicity.
6. Mutation rates are bounded globally, per VIP, and per caller.
7. Every action is audited and correlated.

---

If you want, I can also produce `docs/architecture/30-swarm-integration.md` as a matching spec: swarm leader/epoch rules, what the Python bridge must expose, and how metrics flow into the GA without creating feedback instability.
