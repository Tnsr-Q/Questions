Below is a draft for `docs/architecture/00-overview.md`.

---

# Overview

TensorQ Darwinian Gateway is a **two-plane system** built on top of Cilium’s eBPF-based networking model. The first plane is a **data plane** that streams simulation logic to a browser runtime using chunked code delivery. The second plane is a **control plane** that proposes, validates, stages, and commits route mutations for a custom VIP-to-backend datapath. Cilium is the substrate because it already provides flexible routing modes, eBPF service handling, and Hubble-based observability over the cluster network. ([docs.cilium.io][1])

The system is intentionally split into a **brain** and a **muscle**. The **brain** is a Python Pyro5 swarm that performs leader election, aggregates telemetry, and generates Darwinian route proposals. The **muscle** is a set of Go services that own admission control, map staging, probing, rollback, and eBPF program lifecycle. This separation is architectural, not stylistic: kernel mutation is safety-critical, while proposal generation is exploratory and failure-prone. The result is that AI or swarm logic can evolve rapidly without being allowed to write directly into kernel state. This division is an engineering inference from Cilium’s local-agent control model and Hubble’s embedded-per-node design, rather than something prescribed by Cilium itself. Cilium’s API and Hubble architecture both emphasize node-local control and observation surfaces, which fit this separation well. ([docs.cilium.io][2])

## System intent

The immediate goal is to build a **high-performance research fabric** for physics and math workloads that can do four things well:

1. stream code and parameters into browser or edge runtimes,
2. observe network and service behavior in real time,
3. evolve route selection or service preference based on measured outcomes,
4. enforce selected changes through deterministic, reversible kernel operations.

Cilium’s routing model matters here because native routing and encapsulation have different packet semantics, and those semantics determine whether a custom TC/XDP route mutator can safely infer next-hop behavior from local host state. In native routing mode, Cilium delegates non-local traffic to the Linux routing stack and requires the network to route PodCIDRs; in tunnel mode, more of the forwarding path is abstracted behind encapsulation. ([docs.cilium.io][1])

## Architectural thesis

The architecture is based on a simple rule:

**AI proposes. Deterministic infrastructure disposes.**

That means the swarm can generate candidate routes, weights, or freeze recommendations, but it cannot directly apply them. Every proposal must pass through a **Route Mutation Firewall** that performs validation, leader fencing, rate limiting, MAC/next-hop resolution, staging, probing, commit, and rollback. This is the core safety boundary of the system. It mirrors a broader truth in Cilium deployments: datapath behavior is programmable, but that programmability sits on top of fixed-capacity BPF maps, local agent state, and kernel-enforced execution constraints, so unsafe control inputs can have immediate operational consequences. ([docs.cilium.io][2])

## Core components

The system is composed of the following major subsystems.

### 1. Cilium networking substrate

Cilium provides the cluster CNI, routing mode, service path, and Hubble integration. Depending on environment, the cluster may run in native routing or encapsulation mode. If kube-proxy replacement is enabled later, Cilium’s eBPF datapath takes over service handling and that capability depends on socket-LB. Gateway API or Ingress traffic, when enabled, passes through a per-node Envoy that integrates with Cilium’s eBPF policy engine. ([docs.cilium.io][1])

### 2. Browser streaming plane

The browser-facing `StreamSimulation` RPC streams code in ordered chunks such as preamble, definition, and execution stages. That plane is intentionally recoverable: failures here should degrade a session, not the cluster network. The streaming transport should be treated separately from the route-mutation path, even if both are exposed through the same top-level service definition. Connect-style or HTTP streaming choices must therefore be made with infra behavior in mind, including idle timeouts and reverse-proxy support, but those are transport concerns rather than kernel-safety concerns. Cilium itself exposes HTTP stream timeout controls in the agent, which is relevant when planning long-lived non-gRPC streams in the surrounding platform. ([docs.cilium.io][3])

### 3. Swarm brain

The Python swarm runs leader election, maintains an epoch/lease view of authority, consumes telemetry, and emits proposals. It is deployed as a fault-tolerant coordination layer, ideally as a StatefulSet with stable node identities. It does not own kernel semantics, interface indices, pinned map paths, or next-hop resolution. Those are all intentionally excluded from the brain’s authority. This is a local architectural rule, not a Cilium requirement. It exists to keep the exploratory layer portable and the privileged layer minimal.

### 4. Overseer coordinator

The Go coordinator is the bridge between measured reality and the swarm. It pushes telemetry into the local swarm node, retrieves proposals, checks leader/epoch health, and forwards admissible proposals to the firewall. It also reports outcomes back to the swarm so the Darwinian fitness function can learn from rejection, rollback, or success.

### 5. Route Mutation Firewall

The firewall is the only component allowed to turn intent into kernel-relevant actions. It validates input, resolves actual next-hop information, stages route-map changes, runs synthetic probes, commits via atomic map switch or controlled update, and rolls back when probes or post-commit SLOs fail. This is where idempotency, epochs, cooldowns, and blast-radius budgets live.

### 6. eBPF loader and custom datapath

The loader manages custom TC/XDP programs and the pinned maps they depend on. Those maps are intentionally separate from Cilium’s internal service/load-balancing maps in the initial design, because deep coupling to internal LB map semantics increases upgrade risk. The custom datapath therefore lives in a **parallel VIP map** model first, with the option to integrate more deeply later if the semantics and operational burden justify it. This separation is a design choice based on Cilium’s documented local-agent model and routing flexibility, not a documented best practice from the project itself. ([docs.cilium.io][2])

## Trust boundaries

There are three critical trust boundaries in the system.

### Boundary A — Swarm to coordinator

Everything coming from the swarm is treated as **untrusted intent**. It may be stale, oscillatory, partitioned, or malicious. Proposals are therefore fenced by leader epoch and lease.

### Boundary B — Coordinator to firewall

This is the **authenticated control boundary**. The coordinator must present identity, proposal metadata, and mutation IDs. The firewall may still reject.

### Boundary C — Firewall to kernel

This is the **privileged mutation boundary**. Only deterministic, validated, rate-limited operations are allowed here. No AI or swarm code crosses this boundary.

This trust model is consistent with Cilium’s architecture, where local agent APIs affect local resources unless explicitly marked otherwise, and where Hubble’s per-node server is embedded in the agent and aggregated separately through Relay. ([docs.cilium.io][2])

## Deployment model

The preferred deployment model is:

* **Cilium agent** on every node
* **Hubble** enabled for visibility
* **Swarm node + coordinator** colocated in a pod using a local bridge
* **Firewall** and **loader** separated as privileged services
* **Gateway / browser streaming plane** kept logically separate from kernel actuation

This aligns well with Cilium’s per-node architecture and Hubble’s embedded server model. If cluster-wide observability is required, Hubble Relay should be added because it aggregates Hubble server connections from all nodes into a full cluster view. ([docs.cilium.io][4])

## Routing and service stance

The default platform stance for this project is:

* prefer **native routing** when the underlay can route PodCIDRs,
* delay **kube-proxy replacement** until explicit rollout gates are passed,
* enable **Gateway API / per-node Envoy** only when real L7 ingress requirements exist,
* keep the Darwinian route mutator on a **parallel VIP map** first.

This recommendation follows directly from Cilium’s documented mode behavior. Native routing uses the Linux routing subsystem and requires routable PodCIDRs. Kube-proxy replacement is a deliberate operating mode built on socket-LB. Gateway API traffic passes through a per-node Envoy proxy with hooks into the eBPF policy engine. Across ClusterMesh, all clusters must use the same datapath mode, which makes premature mode mixing a future scaling hazard. ([docs.cilium.io][1])

## Observability-first progression

The system is designed to become autonomous in phases.

### Phase 0 — visibility only

Install Cilium, enable Hubble, and ensure bridge/firewall metrics exist before any autonomous mutation is allowed. Hubble provides deep visibility into service communication and network behavior, including node-level and cluster-level views. ([docs.cilium.io][5])

### Phase 1 — dry-run proposals

Allow the swarm to generate proposals, but keep the firewall in dry-run or stage-only mode. This verifies the telemetry loop, leader fencing, and proposal economics without risking route churn.

### Phase 2 — supervised mutation

Enable controlled staging, probing, commit, and rollback for a bounded set of VIPs under mutation budgets and cooldowns.

### Phase 3 — autonomous mutation

Only after end-to-end observability, rollback drills, and map health signals are proven should the system allow unattended operation.

This progression is an internal operational policy, but it is informed by the fact that Cilium’s observability and service-path features are independently configurable and carry real datapath implications, especially when Hubble and kube-proxy replacement are enabled. ([docs.cilium.io][6])

## Why this architecture exists

The architecture exists because high-level adaptation and low-level packet mutation are fundamentally different kinds of software. The Python swarm is optimized for experimentation, optimization logic, and resilience patterns. The Go/eBPF layer is optimized for predictable latency, strict typing, and safe interaction with kernel state. Cilium sits beneath both as the cluster networking substrate, providing routing options, observability, and service-path primitives that make the whole system feasible. Hubble gives the system the feedback it needs to tell the difference between a bad proposal, a bad probe result, and a bad network. ([docs.cilium.io][1])

## Document map

This overview is the top-level index for the rest of the architecture set:

* `10-stream-simulation.md` — browser/runtime streaming plane
* `20-route-mutation-firewall.md` — deterministic safety envelope for route changes
* `30-swarm-integration.md` — swarm roles, epoch fencing, and coordinator behavior
* `31-bridge-contract.md` — local bridge API between Python and Go
* `40-cilium-mode-matrix.md` — routing/service/ingress mode decisions
* `50-observability.md` — metrics, traces, logs, and operator dashboards

Together, these documents define a system where **the brain is allowed to be inventive, but the muscle is only allowed to be correct**.

---



[1]: https://docs.cilium.io/en/stable/network/concepts/routing.html?utm_source=chatgpt.com "Routing — Cilium 1.19.2 documentation"
[2]: https://docs.cilium.io/en/stable/api.html?utm_source=chatgpt.com "API Reference — Cilium 1.19.2 documentation"
[3]: https://docs.cilium.io/en/stable/cmdref/cilium-agent_hive.html?utm_source=chatgpt.com "cilium-agent hive — Cilium 1.19.2 documentation"
[4]: https://docs.cilium.io/en/stable/observability/hubble/setup.html?utm_source=chatgpt.com "Setting up Hubble Observability"
[5]: https://docs.cilium.io/en/stable/observability/hubble/index.html?utm_source=chatgpt.com "Network Observability with Hubble"
[6]: https://docs.cilium.io/en/stable/network/kubernetes/kubeproxy-free.html?utm_source=chatgpt.com "Kubernetes Without kube-proxy"
