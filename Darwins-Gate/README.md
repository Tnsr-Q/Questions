# Darwin's Gate — Tier-1 ConnectRPC Monolith

[![License](LICENSE)](LICENSE)

**Darwin's Gate** is a closed-loop research grid for evolutionary swarm intelligence, featuring real-time kernel-level route mutations driven by JAX-based genetic algorithms and rendered in-browser via Pyodide/WebGPU.

## Architecture Overview

### Unified ConnectRPC Gateway (Tier-1)

The system runs a **monolithic ConnectRPC switchboard** (`gatewayd`) that serves both control plane and data plane traffic over a single HTTP/2 port (8080):

| Plane | Function | RPC Method | Caller |
|-------|----------|------------|--------|
| **Control** | eBPF map mutations | `UpdateAlphaRoute` | JAX Swarm (Python/Pyro5) |
| **Data** | Pyodide/ONNX streaming | `StreamSimulation` | Browser JS (WebGL frontend) |

### Design Rationale

**Why Monolithic?** The dual-server architecture (separate `gatewayd` + `mutation-firewalld`) was evaluated and **explicitly deprecated** to optimize for:

- **Minimum latency** between Swarm consensus and WebGL rendering
- **Reduced operational complexity** (no inter-service RPC, no mTLS setup)
- **Simplified admission pipeline** (condensed from multi-stage to synchronous validation)

**Security Trade-off Accepted:** The unified gateway holds `CAP_BPF` privileges on a public-facing service. This is acceptable for our closed-loop research grid where kernel interaction is strictly hardcoded to a single eBPF map (`/sys/fs/bpf/tensorq/routing_map`).

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Gateway** | Go 1.22 + ConnectRPC | Unified HTTP/2 switchboard |
| **eBPF** | Cilium + TC hooks | Kernel-level route rewriting |
| **Swarm** | Python/JAX + Pyro5 | Genetic algorithm brain |
| **Frontend** | TypeScript + Pyodide + WebGPU | In-browser ONNX inference |
| **Orchestration** | Kubernetes + Helm | Deployment and scaling |

## Project Structure

```
Darwins-Gate/
├── cmd/
│   ├── gatewayd/              # Unified ConnectRPC monolith (Tier-1)
│   ├── ebpf-loaderd/          # eBPF program loader (privileged init)
│   └── overseer-coordinatord/ # Swarm ↔ Gateway coordinator
├── bpf/
│   ├── tc/                    # TC/XDP datapath programs
│   └── maps/                  # eBPF map definitions
├── proto/
│   └── Tnsr-Q/                # Protobuf + ConnectRPC definitions
├── gen/go/                    # Generated Go code (buf build)
├── swarm/                     # Python JAX swarm implementation
├── web/                       # Frontend (Pyodide + WebGPU)
├── deploy/k8s/                # Kubernetes manifests
│   └── services/
│       ├── gatewayd.yaml      # Main service deployment
│       └── mutation-firewalld.yaml  # DEPRECATED (historical reference)
└── docs/
    └── architecture/
        ├── 00-overview.md
        ├── 20-route-mutation-firewall.md  # DEPRECATED spec
        └── 30-swarm-integration.md
```

## Quick Start

### Prerequisites

- Go 1.22+
- Linux kernel 5.10+ with eBPF support
- Cilium installed (for eBPF pinning)
- Kubernetes cluster (kind/minikube/production)
- Node.js + npm (for frontend development)

### Build and Deploy

```bash
# 1. Generate protobuf code
cd proto && buf generate

# 2. Build gatewayd binary
go build -o bin/gatewayd ./cmd/gatewayd

# 3. Load eBPF programs (requires root/CAP_BPF)
sudo ./bin/ebpf-loaderd

# 4. Deploy to Kubernetes
kubectl apply -k deploy/k8s/services/

# 5. Verify connectivity
curl -v --http2-prior-knowledge \
  -H "Content-Type: application/json" \
  -d '{"modelId": "alpha-v1"}' \
  http://localhost:8080/tensorq.darwinian.v1.CortexGateway/StreamSimulation
```

## API Reference

### Control Plane: `UpdateAlphaRoute`

Mutates the kernel eBPF routing map to redirect VIP traffic to a new backend pod.

**Request:**
```protobuf
message RouteMutation {
  string mutation_id = 1;      // Idempotency key
  string virtual_ip = 2;       // VIP to mutate
  string target_pod_ip = 3;    // New backend destination
  string target_mac_address = 4;
  int64 ttl_ms = 5;            // Expiration timeout
  int64 epoch = 6;             // Leader fencing token
  bool dry_run = 7;            // Validate without applying
}
```

**Response:**
```protobuf
message RouteAck {
  RouteStatus status = 1;      // APPLIED, REJECTED, PROBE_FAILED, etc.
  string status_message = 2;
  repeated string checks_passed = 3;
  repeated string checks_failed = 4;
  int64 applied_at_unix_ns = 5;
  string observability_id = 6; // Trace correlation ID
}
```

### Data Plane: `StreamSimulation`

Streams Pyodide execution chunks and ONNX model weights to the browser.

**Request:**
```protobuf
message SimRequest {
  string model_id = 1;
  map<string, string> parameters = 2;
}
```

**Response Stream:**
```protobuf
message CodeChunk {
  string content = 1;          // Python code or metadata
  ChunkType type = 2;          // PREAMBLE, EXECUTION, WEIGHTS
  int64 sequence_id = 3;
  bytes weights = 4;           // Binary ONNX payload
}
```

## Admission Pipeline (Condensed)

The unified gateway implements synchronous validation within `UpdateAlphaRoute`:

1. **Input Parsing:** Strict Big Endian IP parsing, MAC length verification
2. **TTL Check:** Reject expired proposals based on `epoch` + `ttl_ms`
3. **Policy Constraints:** VIP/pod CIDR allowlists (via env vars)
4. **Leader Fencing:** Epoch monotonicity check (if enabled)
5. **Map Update:** Atomic eBPF map swap with error handling

**Dropped Features** (from original dual-server design):
- ❌ Staging → Probing → Commit → Rollback pipeline
- ❌ A/B map switching
- ❌ Synthetic reachability probes
- ❌ Automatic rollback on SLO breach

**Failure Mode:** If a route fails, the browser canvas safely drops the stream—acceptable for research iterations.

## Observability

### Metrics (Prometheus)

Exposed at `/metrics`:

- `ebpf_mutations_total{status}` — Mutation outcomes
- `ebpf_map_update_errors_total` — Kernel update failures
- `gateway_sessions_active` — Active frontend streams
- `swarm_outcomes_reported_total` — Fitness feedback to GA

### Tracing

Each mutation includes an `observability_id` for distributed tracing across:
- Swarm proposal → Gateway admission → Kernel update → Browser render

## Security Considerations

| Risk | Mitigation | Status |
|------|------------|--------|
| Privileged gateway exposed | Hardcoded to single map path | ✅ Accepted |
| Mutation storms | Per-VIP + global rate limiting | ⚠️ Future work |
| Split-brain swarm | Epoch-based leader fencing | ✅ Implemented |
| Map exhaustion | High-water mark rejection | ⚠️ Future work |

## Roadmap

- [ ] Add weighted rollouts (10% → 25% → 50% → 100%)
- [ ] Implement probe-based validation (optional flag)
- [ ] Add mutation budget quotas per VIP/hour
- [ ] Integrate Hubble for L7 telemetry
- [ ] Support IPv6 dual-stack routing

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

This is a closed-loop research project. External contributions require coordination with the core team.

---

**See Also:**
- [Architecture Overview](docs/architecture/00-overview.md)
- [Swarm Integration Guide](docs/architecture/30-swarm-integration.md)
- [Deprecation Notice: Dual-Server Design](docs/architecture/20-route-mutation-firewall.md)
