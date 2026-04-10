Below is a draft for `docs/architecture/40-cilium-mode-matrix.md`.

---

# Cilium Mode Matrix

This document maps the Cilium deployment modes that materially affect the Darwinian Gateway architecture: routing semantics, service/load-balancing path, ingress/Gateway placement, encryption overhead, and where custom TC/XDP hooks can safely live. Cilium supports both overlay and native-routing models, kube-proxy replacement, and a per-node Envoy path for Gateway API / Ingress. Those choices change the packet path enough that the Darwinian route-mutation layer must treat them as first-class configuration modes. ([docs.cilium.io][1])

## 1. Decision summary

For this project, the default recommendation is:

* **Routing:** native routing when the underlay can route PodCIDRs correctly.
* **Service path:** enable kube-proxy replacement only after explicit rollout gates.
* **Ingress/Gateway:** use Cilium Gateway API / per-node Envoy only when L7 ingress or policy-aware ingress is required.
* **Custom Darwinian datapath:** prefer a **parallel VIP map + dedicated TC/XDP hook** rather than mutating Cilium internal LB maps directly.
* **Encryption:** avoid tunnel mode + WireGuard together unless the security requirement clearly outweighs the extra encapsulation overhead.

Those defaults follow directly from Cilium’s current behavior: native routing delegates non-local traffic to the Linux routing stack and requires the network to route PodCIDRs; kube-proxy replacement depends on socket-LB; Gateway API traffic passes through a per-node Envoy proxy; and tunnel mode combined with WireGuard results in double encapsulation for pod-to-pod traffic. ([docs.cilium.io][1])

---

## 2. Mode matrix

### 2.1 Routing mode matrix

| Mode                                                    | What Cilium does                                                                                                                                                                                                  | When to use it                                                                           | Main upside                                                                    | Main downside                                                                         | Darwinian Gateway implication                                                                                |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Native routing**                                      | Delegates non-local endpoint traffic to the Linux routing subsystem; the network must be able to route PodCIDRs; Cilium enables IP forwarding when native routing is configured. ([docs.cilium.io][1])            | Bare metal, L3-capable fabric, BGP/L2-underlay environments, serious performance tuning  | Lower encapsulation overhead; better fit for high-throughput east-west traffic | Requires correct underlay routing and operational discipline                          | Best default for the Darwinian router; firewall should resolve next-hop/L2 details from actual routing state |
| **Encapsulation / tunnel routing**                      | Uses overlay tunneling such as VXLAN or Geneve for inter-node pod traffic. Cilium’s performance guide notes that the default deployment prioritizes compatibility over maximum performance. ([docs.cilium.io][2]) | Heterogeneous or simpler cloud/network setups where PodCIDRs are not routed natively     | Easiest compatibility story                                                    | More overhead; packet path less transparent; custom L2 rewrite semantics get trickier | Darwinian route mutation should avoid assuming a direct next-hop MAC model                                   |
| **Native routing + route advertisement / L2 discovery** | Cilium can automate route learning/advertisement using L2 neighbor discovery on shared L2 domains or BGP across L3 boundaries. ([docs.cilium.io][3])                                                              | Production native-routing clusters that need scale beyond flat single-subnet assumptions | Makes native routing operationally viable at larger scale                      | More moving parts in the network control plane                                        | Preferred scaling path if the platform grows beyond a simple L2 fabric                                       |

### 2.2 Service and ingress mode matrix

| Mode                              | Requirement / behavior                                                                                                                                                                                                                                          | Best use                                                                                | Risk / cost                                                                                                                                                                                      | Darwinian Gateway implication                                                                                     |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **kube-proxy present**            | Standard Kubernetes service path remains in place                                                                                                                                                                                                               | Lowest-risk initial Cilium adoption                                                     | More components in the datapath; not the cleanest high-performance stack                                                                                                                         | Good for initial bring-up and observability-first rollout                                                         |
| **kube-proxy replacement**        | Cilium fully replaces kube-proxy; this depends on socket-LB. ([docs.cilium.io][4])                                                                                                                                                                              | Mature clusters where service-path simplification and eBPF service handling are desired | Rollout risk; Cilium warns that kube-proxy migration needs `k8sServiceHost` and `k8sServicePort` set so the agent can still reach the API server after kube-proxy removal. ([docs.cilium.io][5]) | Recommended only after the Darwinian firewall and rollback logic are already working                              |
| **Ingress / Gateway API enabled** | Gateway / Ingress traffic passes through a per-node Envoy proxy, which integrates with the eBPF policy engine; enabling Gateway API auto-enables Envoy config support. Ingress prerequisites include kube-proxy replacement and L7 proxy. ([docs.cilium.io][6]) | North-south traffic, policy-aware ingress, HTTP-aware entry points                      | Extra hop through Envoy; more control-plane and proxy complexity                                                                                                                                 | Keep the Darwinian TC/XDP path out of the Envoy-managed ingress path unless there is a clear reason to merge them |

### 2.3 Encryption mode matrix

| Mode                              | Behavior                                                                                            | Upside                             | Downside                                               | Darwinian Gateway implication                                             |
| --------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| **No transparent encryption**     | Plain datapath                                                                                      | Lowest latency / lowest overhead   | No transport encryption at pod-to-pod level            | Best for controlled research fabrics when lower-layer trust is acceptable |
| **WireGuard with native routing** | WireGuard protects traffic without overlay-tunnel stacking                                          | Good security/performance balance  | Operational key management and validation still matter | Compatible with Darwinian routing if next-hop semantics remain clear      |
| **WireGuard with tunnel routing** | Pod-to-pod traffic is encapsulated twice: first VXLAN/Geneve, then WireGuard. ([docs.cilium.io][7]) | Maximum compatibility + encryption | Extra overhead and more opaque packet path             | Avoid unless security requirements dominate performance and debuggability |

---

## 3. Recommended architecture by environment

### 3.1 Bare metal or controlled research cluster

Use **native routing**, delay **kube-proxy replacement** until after baseline validation, and add **Gateway API / Envoy** only if L7 ingress is actually needed. Native routing is the best fit when the underlay can route PodCIDRs and when high-throughput, low-overhead east-west traffic matters. Cilium explicitly states that native routing requires the network to be capable of forwarding pod/workload addresses and that it delegates non-local traffic to the Linux routing subsystem. ([docs.cilium.io][1])

**Implication for the Darwinian layer:** this is the cleanest environment for a custom parallel VIP map. The firewall should resolve actual next-hop data from the host routing / neighbor state rather than trusting a MAC proposed by the Python brain.

### 3.2 Cloud / compatibility-first cluster

Use **encapsulation** first, retain **kube-proxy** initially, and postpone custom L2 rewrite assumptions. Cilium’s performance guide says the default installation aims for maximum compatibility rather than maximum performance, which is the right mental model for this mode. ([docs.cilium.io][2])

**Implication for the Darwinian layer:** keep the custom datapath conservative. Prefer proposal-only logic and do not assume that a podIP → nodeMAC mapping is sufficient to express the actual forwarding path.

### 3.3 L7 ingress-heavy cluster

If the platform needs Gateway API / Ingress, accept that traffic will traverse **per-node Envoy**, and that this Envoy is part of Cilium’s policy enforcement path. Cilium documents that Gateway / Ingress traffic bound to backend services passes through a per-node Envoy proxy which can interact with the eBPF policy engine. ([docs.cilium.io][6])

**Implication for the Darwinian layer:** keep ingress optimization separate from east-west route mutation. The Darwinian router should not attempt to co-own Envoy’s job unless the system is specifically evolving L7 gateway decisions.

---

## 4. Darwinian Gateway placement rules by mode

### Rule A — In native routing mode, custom route mutation can safely target Linux-routed east-west traffic

Because native routing hands non-local traffic to the Linux routing stack, this is the cleanest place for a custom VIP resolution layer. The firewall can treat the Python/Pyro5 swarm output as intent, then derive actual L2/L3 forwarding information from the node’s routing and neighbor state. ([docs.cilium.io][1])

### Rule B — In tunnel mode, the firewall must not assume “target node MAC” is the forwarding truth

Overlay routing makes direct L2 assumptions weaker, especially once encryption or other encapsulation is added. WireGuard on top of tunnel mode explicitly produces double encapsulation, which makes low-level packet-path reasoning more brittle. ([docs.cilium.io][7])

### Rule C — Gateway API and Ingress are Envoy territory

If Gateway API or Ingress is enabled, the packet path for north-south traffic includes per-node Envoy, and Cilium uses that Envoy as a policy enforcement point. ([docs.cilium.io][6])
The Darwinian router should therefore focus on:

* east-west service selection,
* backend promotion / canarying,
* kernel map safety,
  and not on replacing the gateway proxy itself.

### Rule D — kube-proxy replacement is a rollout event, not a default toggle

Cilium supports gradual node-by-node rollout of kube-proxy replacement through `CiliumNodeConfig`, and warns that the API server reachability settings must be correct before kube-proxy is removed. ([docs.cilium.io][5])
For this project, kube-proxy replacement should be enabled only after:

1. route-mutation firewall is stable,
2. rollback and map observability are in place,
3. the cluster has passed controlled canary migration.

---

## 5. Mode-specific recommendations

## 5.1 v0 / bring-up mode

* Routing: **encapsulation or native**, whichever gets the cluster healthy fastest
* Services: **keep kube-proxy**
* Ingress: **off unless needed**
* Darwinian router: **observe + dry-run only**

Use this mode to establish Hubble/metrics visibility and validate bridge/firewall contracts before any kernel mutation logic is trusted.

## 5.2 v1 / performance mode

* Routing: **native routing**
* Services: **still optional kube-proxy replacement**
* Ingress: **off unless required**
* Darwinian router: **active for east-west VIP selection**

This is the preferred first real production mode for the research hub.

## 5.3 v2 / full eBPF services mode

* Routing: **native routing**
* Services: **kube-proxy replacement enabled**
* Ingress: **Gateway API / Envoy if needed**
* Darwinian router: **active + staged + rollback-protected**

Use this only after node-by-node migration and datapath regression testing.

## 5.4 v3 / encrypted mode

* Routing: preferably **native + WireGuard**
* Avoid **tunnel + WireGuard** unless mandated
* Darwinian router: **active, but budget extra headroom for overhead and observability**

---

## 6. Practical decision tree

Start with these questions:

1. **Can the underlay route PodCIDRs correctly?**
   If yes, prefer **native routing**. If no, use **encapsulation**. ([docs.cilium.io][1])

2. **Do you need Cilium to replace kube-proxy now, or is that just an eventual optimization?**
   If it is not required now, delay it. Cilium treats kube-proxy replacement as a deliberate mode built on socket-LB and supports gradual migration. ([docs.cilium.io][4])

3. **Do you actually need Gateway API / Ingress?**
   If yes, accept per-node Envoy as part of the datapath. If no, keep the ingress layer out of scope. ([docs.cilium.io][6])

4. **Is transparent encryption required?**
   If yes, prefer the least compositionally complex path. Avoid tunnel + WireGuard unless required. ([docs.cilium.io][7])

---

## 7. Project-specific default stance

For the TensorQ Darwinian Gateway, the default platform stance should be:

* **Cilium native routing**
* **parallel Darwinian VIP map**
* **Go firewall owns next-hop resolution**
* **kube-proxy replacement delayed until post-canary**
* **Gateway API enabled only when a real L7 ingress requirement appears**
* **avoid double encapsulation paths**

That combination gives the cleanest packet semantics, the smallest number of stacked forwarding abstractions, and the most reliable place to insert an AI-supervised but deterministic route-mutation layer. The recommendation is an engineering inference from Cilium’s documented mode behavior, not a direct statement from Cilium itself. ([docs.cilium.io][1])

---

If you want, I can produce the next companion doc as `50-observability.md` with a concrete metric/trace/log schema for Hubble, bridge, firewall, and BPF map health.

[1]: https://docs.cilium.io/en/stable/network/concepts/routing.html "Routing — Cilium 1.19.2 documentation"
[2]: https://docs.cilium.io/en/stable/operations/performance/tuning.html "Tuning Guide — Cilium 1.19.2 documentation"
[3]: https://docs.cilium.io/en/latest/overview/intro.html "Introduction to Cilium & Hubble — Cilium 1.20.0-dev documentation"
[4]: https://docs.cilium.io/en/stable/network/kubernetes/kubeproxy-free.html "Kubernetes Without kube-proxy — Cilium 1.19.2 documentation"
[5]: https://docs.cilium.io/en/stable/configuration/per-node-config.html "Per-node configuration — Cilium 1.19.2 documentation"
[6]: https://docs.cilium.io/en/stable/network/servicemesh/gateway-api/gateway-api.html "Gateway API Support — Cilium 1.19.2 documentation"
[7]: https://docs.cilium.io/en/latest/security/network/encryption-wireguard.html "WireGuard Transparent Encryption — Cilium 1.20.0-dev documentation"
