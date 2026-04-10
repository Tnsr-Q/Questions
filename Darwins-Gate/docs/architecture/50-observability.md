Below is a draft for `docs/architecture/50-observability.md`.

---

# Observability

This document defines the observability model for the TensorQ Darwinian Gateway: what to measure, where to measure it, how to correlate signals across layers, and which alerts are required before AI-driven route mutation is allowed to operate unattended. The observability stack is intentionally **multi-layered** because Cilium/Hubble, the Python swarm, the Go bridge/firewall, and the custom eBPF datapath fail in different ways. Hubble provides distributed network visibility on top of Cilium and eBPF, with node-level and cluster-level views available through Hubble and Hubble Relay. Cilium and Hubble both expose Prometheus metrics independently, and Hubble can also export flows for later log consumption. ([docs.cilium.io][1])

## 1. Goals

The observability system must answer five questions quickly and unambiguously:

1. **Did the brain make a bad proposal?**
2. **Did the firewall reject or roll back for a good reason?**
3. **Did the kernel datapath apply the mutation correctly?**
4. **Did Cilium/Hubble observe the expected traffic shift?**
5. **Is the platform itself degrading under the cost of observability?**

This last point matters because Cilium documents that enabling Hubble can impose measurable performance overhead, with the tuning guide noting roughly 1–15% overhead depending on network workload and configuration. ([docs.cilium.io][2])

---

## 2. Observability layers

The stack is divided into six layers. Each layer has its own signals and its own operator questions.

### 2.1 Layer A — Swarm / brain observability

This covers the Python Pyro5 swarm: leader election, epoch churn, proposal generation, proposal suppression, and feedback incorporation.

**Primary questions**

* Do we have a healthy leader and quorum?
* Are proposals being generated too often?
* Is the swarm reacting to rollback outcomes, or is it oscillating?

### 2.2 Layer B — Bridge observability

This covers the local Go ↔ Python bridge. Its purpose is to prove that the coordinator is talking to the brain correctly and that protocol/version issues are not being mistaken for network failures.

**Primary questions**

* Is the bridge healthy and version-compatible?
* Are telemetry batches reaching the swarm?
* Are proposal fetches timing out or returning `NO_QUORUM` / `NO_LEADER`?

### 2.3 Layer C — Firewall observability

This is the most important control-plane layer. It decides admission, staging, probing, commit, rollback, and cooldown.

**Primary questions**

* Why was a proposal accepted or rejected?
* Did probes pass?
* Was rollback triggered by real service degradation, map pressure, or bad resolution?

### 2.4 Layer D — Custom eBPF datapath observability

This covers the route-rewrite program and its maps: hit/miss behavior, map occupancy, update failures, and kernel-side rewrite errors.

**Primary questions**

* Is the route map healthy and within capacity?
* Are packets actually taking the mutated path?
* Are checksum or rewrite errors occurring?

### 2.5 Layer E — Cilium/Hubble network observability

Hubble is Cilium’s observability layer and provides transparent visibility into service communication and network behavior, including cluster-wide visibility through Hubble Relay. Hubble’s Observer service exposes flow-related APIs, while Cilium and Hubble metrics are available to Prometheus independently. ([docs.cilium.io][1])

**Primary questions**

* Did traffic move to the expected backend or node?
* Did drops increase, and where?
* Is there evidence that the network path, rather than the proposal logic, failed?

### 2.6 Layer F — Runtime security/process observability

Tetragon is useful here if you want to observe process execution, runtime behavior, and policy-enforced events alongside network behavior. Tetragon exposes Prometheus metrics, JSON logs, and gRPC streams for execution and runtime events, and can also surface BPF program statistics. ([tetragon.io][3])

---

## 3. Golden signals by subsystem

## 3.1 Swarm signals

### Required metrics

* `swarm_leader_epoch`
* `swarm_quorum_status`
* `swarm_proposals_total{vip}`
* `swarm_proposal_suppressed_total{reason}`
* `swarm_feedback_ingested_total{status}`
* `swarm_proposal_interval_seconds{vip}`
* `swarm_stability_score{vip}`

### Required logs

* leader elected / leader lost
* epoch increment
* proposal emitted
* proposal suppressed for cooldown / insufficient improvement
* rollback outcome incorporated

### Required alerts

* **No leader for > N seconds**
* **Epoch churn too high**
* **Proposal rate above stability threshold**
* **Proposal generation continues while quorum degraded**

---

## 3.2 Bridge signals

### Required metrics

* `bridge_requests_total{endpoint,status}`
* `bridge_request_latency_seconds{endpoint}`
* `bridge_protocol_version_mismatch_total`
* `bridge_auth_failures_total`
* `bridge_timeouts_total{endpoint}`
* `bridge_overload_total`

### Required logs

* startup meta negotiation result
* health transitions
* authentication failures
* malformed request rejection
* protocol version incompatibility

### Required alerts

* **Bridge unavailable**
* **Bridge protocol mismatch**
* **Bridge request timeout rate above threshold**
* **Bridge overloaded**

---

## 3.3 Firewall signals

### Required metrics

* `mutations_total{vip,status}`
* `mutation_phase_latency_seconds{phase}`
* `admission_rejections_total{reason}`
* `probe_pass_ratio{vip}`
* `rollback_total{vip,reason}`
* `vip_quarantine_total{vip}`
* `leader_epoch_rejections_total`
* `idempotent_replays_total`
* `resolution_failures_total{reason}`

### Required logs

Every mutation attempt must log:

* mutation ID
* VIP
* target identity
* leader ID / epoch
* expected route version
* admission decisions
* probe summary
* commit result
* rollback reason

### Required alerts

* **Rollback rate above threshold**
* **Probe failures spike**
* **Epoch fencing rejects spike**
* **Mutation rate exceeds budget**
* **Resolution failures spike**

---

## 3.4 Custom eBPF datapath signals

### Required metrics

* `routing_map_occupancy_ratio`
* `routing_map_update_failures_total`
* `routing_map_hit_total{vip}`
* `routing_map_miss_total{vip}`
* `rewrite_errors_total{reason}`
* `checksum_fixup_errors_total`
* `active_map_index`
* `route_switch_total{vip}`

### Required logs

The datapath itself should log minimally, because high-volume kernel logging is dangerous. Instead:

* expose counters in BPF maps
* sample structured debug events only under controlled troubleshooting modes

### Required alerts

* **Map occupancy above high-water mark**
* **Map update failures > 0**
* **Hit rate unexpectedly collapses**
* **Rewrite error counter increments**

This aligns with Cilium’s own operational reality: BPF maps have fixed capacities and insertions can fail when limits are reached, so map pressure must be observed proactively. ([docs.cilium.io][4])

---

## 3.5 Cilium / Hubble signals

### Required metrics

At minimum, collect:

* Cilium agent health and scrape success
* Hubble metrics scrape success
* drop-related metrics
* policy-related verdict metrics
* flow volume by source/destination namespace / workload
* Hubble Relay health if cluster-wide view is enabled

Cilium’s metrics guide states that Cilium and Hubble can both expose Prometheus metrics and be scraped independently, while Hubble’s setup and internals docs show that Hubble Relay provides a cluster-wide API by aggregating Hubble server connections from all nodes. ([docs.cilium.io][4])

### Required logs / flow exports

Use three Hubble outputs:

1. **Live flow stream** for interactive debugging
2. **Prometheus metrics** for dashboards and alerts
3. **Exporter file output** for retained, queryable flow logs

Hubble Exporter supports writing flows to files with rotation, filters, and field masks, which is ideal for postmortem analysis without forcing every investigation through a live CLI session. ([docs.cilium.io][5])

### Required alerts

* **Hubble Relay unhealthy**
* **Hubble metrics scrape failures**
* **Flow drop rate increases for Darwinian VIPs**
* **Unexpected service/backend divergence between firewall intent and Hubble-observed traffic**

---

## 3.6 Runtime security / Tetragon signals

### Required metrics

* Tetragon health metrics
* process event rates
* policy event rates
* BPF program stats if enabled

Tetragon’s metrics docs state that it exposes Prometheus metrics both for Tetragon’s own health and for activity of observed processes, and its execution monitoring docs describe JSON log and gRPC event output for runtime events. ([tetragon.io][3])

### Required alerts

* **Unexpected process execution in privileged components**
* **Policy enforcement spikes**
* **BPF program stats indicate abnormal load or drops**

Tetragon also documents BPF program statistics as a source of performance data for loaded programs, which is useful when distinguishing “bad route” from “hot eBPF program under stress.” ([tetragon.io][6])

---

## 4. Correlation model

Observability is only useful if events across layers can be stitched together.

### 4.1 Correlation identifiers

Every route mutation must carry:

* `mutation_id`
* `proposal_id`
* `vip`
* `leader_id`
* `epoch`
* `route_version`
* trace/span ID

These IDs must appear in:

* bridge logs
* firewall logs
* coordinator logs
* audit records
* probe results
* optional Hubble flow annotations where possible
* exported outcome reports back to swarm

### 4.2 Event chain

The expected event chain is:

1. `swarm_proposal_emitted`
2. `coordinator_proposal_forwarded`
3. `firewall_admission_passed`
4. `firewall_staged`
5. `probe_started`
6. `probe_passed`
7. `route_committed`
8. `traffic_shift_observed` (Hubble)
9. `monitor_window_passed`

Rollback chain is:

1. `firewall_staged`
2. `probe_failed` or `post_commit_slo_breach`
3. `route_rolled_back`
4. `vip_quarantined`
5. `outcome_reported_to_swarm`

The system is not considered production-ready until these chains are visible and queryable end to end.

---

## 5. Dashboards

## 5.1 Executive dashboard

Purpose: answer “is the Darwinian system safe to run?”

Panels:

* active leader / epoch
* quorum status
* mutations attempted / applied / rolled back
* rollback rate
* map occupancy
* Hubble-observed drop rate
* bridge health
* mutation freeze state

## 5.2 Firewall operator dashboard

Purpose: explain individual mutation decisions.

Panels:

* admission reject reasons over time
* probe pass ratio per VIP
* rollback reasons per VIP
* route version timeline
* per-VIP quarantine state
* resolution failures

## 5.3 Datapath dashboard

Purpose: detect kernel/map issues before user-visible failures.

Panels:

* map occupancy and failures
* hit/miss ratio
* rewrite / checksum error counters
* program attach state
* active map index

## 5.4 Network dashboard

Purpose: validate that traffic behaved as intended.

Panels:

* Hubble flow volume by VIP/backend
* drop reason breakdown
* policy verdicts
* node-to-node and namespace-to-namespace traffic shifts
* Hubble Relay / Observer health

## 5.5 Runtime security dashboard

Purpose: detect compromise or misuse in privileged services.

Panels:

* privileged process executions
* Tetragon policy events
* unusual execution spikes in firewall/loader containers
* BPF program stats if enabled

---

## 6. Alerting policy

### 6.1 Paging alerts

Page immediately on:

* no swarm leader with mutations still attempted
* firewall rollback storm
* route map update failures
* map occupancy above critical threshold
* Hubble or Relay unavailable during active mutation window
* bridge version mismatch after deploy
* privileged component executing unexpected binaries

### 6.2 Ticket / non-paging alerts

Open a ticket on:

* sustained proposal suppression
* rising bridge latency
* Hubble exporter lag / file rotation problems
* elevated observability overhead
* degraded quorum without active mutation

Cilium’s tuning guide explicitly warns that Hubble observability has a performance cost, so “observability overhead” is not theoretical and deserves its own tracking and budgeting. ([docs.cilium.io][2])

---

## 7. Sampling and retention policy

### 7.1 Metrics

* scrape interval: 15s by default
* 5s for firewall and route-map health if mutation frequency is high
* retention based on your Prometheus tier, but at least enough for rollback forensics

### 7.2 Logs

* bridge/firewall audit logs: long retention
* coordinator logs: medium retention
* swarm debug logs: medium retention
* kernel/BPF debug logs: short retention, sampled only

### 7.3 Hubble flows

Use:

* live stream for immediate debugging
* exporter file rotation for retained analysis
* Prometheus metrics for alerting and dashboards

Hubble Exporter supports file rotation and filtering, so retained flow logs should be filtered to Darwinian VIPs and relevant namespaces rather than “everything forever.” ([docs.cilium.io][5])

---

## 8. Minimum readiness gate before autonomous actuation

Autonomous mutations must remain disabled until all conditions below are true:

1. Hubble metrics are enabled and scraping successfully. Cilium’s metrics docs make clear that Cilium and Hubble metrics are independently configurable; both must be confirmed live. ([docs.cilium.io][4])
2. Hubble Relay is healthy if cluster-wide visibility is required. ([docs.cilium.io][7])
3. Bridge meta/version negotiation is green.
4. Firewall metrics and audit logs are present.
5. Route-map occupancy and update-failure metrics are present.
6. Probe metrics are present and correlated to mutation IDs.
7. Outcome feedback path to swarm is working.
8. At least one synthetic rollback drill has been observed end to end.

---

## 9. Troubleshooting heuristics

### Symptom: proposal accepted, but traffic does not move

Check in order:

1. firewall commit log
2. route map counters
3. Hubble flows for expected backend
4. Cilium drop/policy signals
5. next-hop/MAC resolution history

### Symptom: repeated rollback after apparently good proposals

Check:

1. probe success ratio
2. post-commit latency deltas
3. Hubble flow asymmetry
4. map pressure
5. route flapping caused by swarm proposal frequency

### Symptom: degraded cluster performance during observability-heavy runs

Check:

1. Hubble enabled metrics + exporter volume
2. Relay health and throughput
3. Prometheus scrape load
4. Cilium tuning / observability overhead budget

Cilium’s tuning guide explicitly notes that Hubble observability can cost 1–15% performance depending on workload, so observability-induced degradation must be treated as a real failure mode. ([docs.cilium.io][2])

---

## 10. Recommended implementation order

1. Enable Cilium and Hubble metrics
2. Stand up Hubble Relay if cluster-wide visibility is needed
3. Add firewall and bridge metrics/logs
4. Add route-map health counters
5. Add Hubble exporter with filtered retained flows
6. Add Tetragon only where runtime/process observability is worth the operational cost
7. Run synthetic mutation and rollback drills before enabling autonomous actuation

Cilium’s docs show that Hubble setup, Relay-based cluster visibility, Prometheus/Grafana integration, and exporter-based retained flow logging are all supported building blocks for this plan. Tetragon complements that with process/runtime observability and Prometheus metrics. ([docs.cilium.io][8])

---

If you want, I can continue with `00-overview.md` so the architecture docs read as a coherent set rather than isolated spec files.

[1]: https://docs.cilium.io/en/stable/observability/hubble/index.html?utm_source=chatgpt.com "Network Observability with Hubble"
[2]: https://docs.cilium.io/en/stable/operations/performance/tuning.html?utm_source=chatgpt.com "Tuning Guide — Cilium 1.19.2 documentation"
[3]: https://tetragon.io/docs/installation/metrics/?utm_source=chatgpt.com "Metrics"
[4]: https://docs.cilium.io/en/stable/observability/metrics.html?utm_source=chatgpt.com "Monitoring & Metrics — Cilium 1.19.2 documentation"
[5]: https://docs.cilium.io/en/latest/observability/hubble/configuration/export.html?utm_source=chatgpt.com "Configuring Hubble exporter"
[6]: https://tetragon.io/docs/troubleshooting/bpf-progs-stats/?utm_source=chatgpt.com "BPF programs statistics"
[7]: https://docs.cilium.io/en/latest/overview/component-overview.html?utm_source=chatgpt.com "Component Overview — Cilium 1.20.0-dev documentation"
[8]: https://docs.cilium.io/en/stable/observability/hubble/setup.html?utm_source=chatgpt.com "Setting up Hubble Observability"
