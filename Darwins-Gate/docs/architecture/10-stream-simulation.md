Below is a draft for `docs/architecture/10-stream-simulation.md`.

---

# Stream Simulation

This document defines the **browser/runtime streaming plane** of the TensorQ Darwinian Gateway. Its purpose is to deliver simulation logic incrementally to a remote execution environment, typically a browser-hosted Python runtime such as Pyodide, with low startup latency, bounded execution semantics, and strong separation from the kernel route-mutation control plane.

The core design assumption is simple:

**streaming code is recoverable; kernel mutation is not.**

Because of that, the simulation streaming plane is designed to be:

* latency-sensitive,
* integrity-aware,
* sandboxed,
* observable,
* and operationally independent from route mutation.

It may share transport infrastructure or a top-level service definition with the control plane, but it must never share trust assumptions with it.

---

## 1. Purpose

`StreamSimulation` exists to support high-velocity research workflows where simulation logic, parameterization, and execution hints are delivered dynamically rather than bundled into a static application artifact.

Typical use cases include:

* streaming parameterized numerical kernels to a browser runtime,
* warming a Python runtime before full logic is available,
* delivering partial simulation code while backend compilation or templating continues,
* rapidly iterating on models without rebuilding a frontend bundle.

This plane is optimized for:

* fast time-to-first-execution,
* chunked delivery,
* deterministic chunk semantics,
* resumable or restartable sessions,
* low coupling to cluster networking internals.

---

## 2. Scope

### In scope

* the `StreamSimulation(SimRequest) returns (stream CodeChunk)` RPC
* chunk ordering and execution semantics
* browser/runtime safety constraints
* integrity and versioning model for streamed code
* observability and operational guardrails for streaming sessions

### Out of scope

* route mutation and kernel actuation
* swarm leader election
* MAC / next-hop resolution
* eBPF map lifecycle
* L4/L7 policy enforcement semantics outside transport-level access control

---

## 3. Design goals

The streaming plane must satisfy five goals.

### 3.1 Fast startup

The client should be able to begin useful work before the entire simulation payload is available.

### 3.2 Deterministic chunk semantics

Each chunk type must have a clear execution contract so client runtimes behave consistently.

### 3.3 Runtime safety

The streamed code must execute inside a constrained environment with explicit boundaries around imports, memory, network access, and side effects.

### 3.4 Recoverability

If a stream fails, the session should degrade gracefully and be restartable without affecting cluster routing or service health.

### 3.5 Observability

Operators must be able to distinguish:

* backend generation delay,
* transport delay,
* client runtime failure,
* invalid chunk ordering,
* and integrity failure.

---

## 4. Service model

The streaming plane is defined by the `CortexGateway.StreamSimulation` RPC:

* request: `SimRequest`
* response: ordered stream of `CodeChunk`

### 4.1 Request intent

A `SimRequest` selects a model family and supplies parameter overrides or hyperparameters. The server may use this to:

* choose a template,
* select a precompiled fragment,
* generate code dynamically,
* or stitch together a hybrid artifact from cached and dynamic parts.

### 4.2 Response intent

A `CodeChunk` is not “just text.” It is an **execution-stage artifact**. The client must interpret each chunk according to its declared `ChunkType`.

---

## 5. Chunk semantics

The current chunk model is:

* `CHUNK_TYPE_PREAMBLE`
* `CHUNK_TYPE_DEFINITION`
* `CHUNK_TYPE_EXECUTION`

These types must be treated as **semantic phases**, not cosmetic hints.

## 5.1 Preamble

Preamble chunks are safe to execute immediately and are intended to reduce idle time while the rest of the stream is still being generated.

Typical contents:

* imports
* constants
* helper initialization
* runtime checks
* lightweight environment setup

### Rules

* must be side-effect minimal
* must not depend on future chunks
* must be idempotent if replayed
* must complete quickly

### Client behavior

* execute immediately upon receipt if integrity checks pass
* record execution success/failure per chunk

## 5.2 Definition

Definition chunks carry the bulk of the simulation logic.

Typical contents:

* function definitions
* class definitions
* generated kernels
* structured data tables
* model-specific logic

### Rules

* must not auto-execute the simulation
* may depend on prior preamble chunks
* should be bufferable and concatenatable
* must be valid in the target runtime once assembled

### Client behavior

* buffer in memory or append to a session code buffer
* do not execute until execution phase is received
* optionally perform syntax prevalidation if runtime permits

## 5.3 Execution

Execution chunks trigger the simulation run.

Typical contents:

* function call entrypoints
* orchestration logic
* final assembly trigger
* result emission hook

### Rules

* must be the final phase of the stream for a given session revision
* must not introduce large new definitions unexpectedly
* should assume all required preamble and definition chunks are already available

### Client behavior

* verify that prior required chunks were received and accepted
* execute only once per session revision unless explicitly retried
* emit structured results or errors

---

## 6. Ordering and sequence rules

Each chunk includes a `sequence_id`. This is required even when transport usually preserves order, because it gives the client a deterministic replay and validation model.

### 6.1 Ordering invariants

1. Sequence IDs must be strictly increasing within a stream.
2. Preamble must precede execution.
3. Execution must not arrive before all mandatory definitions.
4. A client must reject or quarantine out-of-contract chunk order.

### 6.2 Client handling of disorder

If chunks arrive out of expected order:

* buffer only if reordering policy allows it,
* otherwise fail the session with an ordering error,
* never execute execution chunks if required prior phases are missing.

### 6.3 End-of-stream

A stream is considered complete only when:

* execution chunk has been processed, or
* server terminates with explicit “no execution” semantics, or
* stream fails and client marks session incomplete.

---

## 7. Integrity model

The stream must be treated as executable content and therefore requires integrity protection.

## 7.1 Required integrity properties

At minimum, the system should support:

* per-chunk digest
* stream/session identifier
* stable sequence numbering

### Strongly recommended

* chained digests (`prev_hash`)
* signed manifest or signed final chunk
* server-issued session nonce

## 7.2 Integrity failure behavior

If a chunk fails integrity validation:

* do not execute it,
* mark the session invalid,
* emit a structured integrity error,
* do not “best effort” continue into execution.

## 7.3 Replay protection

To prevent stale or duplicated execution:

* each stream session should have a unique session ID,
* each execution chunk should be tied to that session ID,
* clients should ignore duplicate execution chunks unless an explicit retry mode is enabled.

---

## 8. Runtime sandbox requirements

The browser/runtime is an execution boundary and must be treated as hostile or fragile depending on context.

## 8.1 Sandbox goals

* prevent arbitrary privilege escalation
* constrain memory growth
* constrain network side effects
* constrain import surface
* keep execution deterministic enough for debugging

## 8.2 Required controls

The client runtime should support or emulate:

* restricted import allowlist
* memory/time budget per session
* isolated namespace per stream session
* explicit result channel
* explicit cancellation path

## 8.3 Forbidden assumptions

The streamed code must not assume:

* filesystem access,
* unrestricted network access,
* persistent global state across sessions,
* exact parity with server-side Python environment,
* availability of native extensions unless explicitly provisioned.

---

## 9. Session lifecycle

Each streaming request creates a **simulation session**.

### 9.1 Session states

Recommended logical states:

* `CREATED`
* `STREAMING`
* `PREAMBLE_READY`
* `DEFINITIONS_READY`
* `EXECUTABLE`
* `RUNNING`
* `COMPLETED`
* `FAILED`
* `CANCELLED`

### 9.2 State transitions

* request accepted → `CREATED`
* first chunk received → `STREAMING`
* preamble accepted/executed → `PREAMBLE_READY`
* required definitions accepted → `DEFINITIONS_READY`
* execution chunk validated → `EXECUTABLE`
* runtime starts execution → `RUNNING`
* execution succeeds → `COMPLETED`
* integrity/order/runtime error → `FAILED`
* user abort / timeout → `CANCELLED`

### 9.3 Cancellation

Client or server may cancel a session. Cancellation must:

* stop further chunk handling,
* prevent later execution,
* release runtime resources,
* leave no partial session state that can accidentally execute later.

---

## 10. Server responsibilities

The server is responsible for producing a stream that is both performant and contract-correct.

## 10.1 Generation responsibilities

The server must:

* validate `SimRequest`
* resolve model selection
* generate or fetch code fragments
* assign correct chunk types
* assign monotonic sequence IDs
* emit integrity metadata if enabled

## 10.2 Performance responsibilities

The server should:

* flush preamble as early as possible
* avoid holding the first chunk until the entire program is generated
* reuse cached fragments where possible
* separate heavy code generation from transport buffering

## 10.3 Safety responsibilities

The server must not:

* emit execution chunks before dependencies are ready
* emit unbounded payloads without chunking
* mix multiple logical sessions into one stream
* smuggle route-mutation or privileged control instructions through the streaming plane

---

## 11. Client responsibilities

The client is not a passive recipient; it enforces session safety.

## 11.1 Validation responsibilities

The client must:

* validate sequence ordering
* validate integrity metadata if present
* enforce chunk-type semantics
* reject invalid execution phase transitions

## 11.2 Runtime responsibilities

The client must:

* isolate each session namespace
* bound execution time and memory where possible
* surface runtime exceptions with session and sequence metadata
* provide user-visible session state

## 11.3 Retry behavior

Client retries must be explicit and bounded. Automatic replay of execution chunks is prohibited unless:

* the session is known to be idempotent,
* the prior run did not actually begin execution,
* and retry policy allows it.

---

## 12. Error model

The streaming plane needs a clear error taxonomy so operators do not confuse a runtime failure with a network or backend failure.

### 12.1 Error classes

Recommended categories:

* `REQUEST_INVALID`
* `MODEL_NOT_FOUND`
* `GENERATION_FAILED`
* `STREAM_ABORTED`
* `CHUNK_ORDER_INVALID`
* `INTEGRITY_FAILED`
* `RUNTIME_SETUP_FAILED`
* `RUNTIME_EXECUTION_FAILED`
* `SESSION_TIMEOUT`
* `SESSION_CANCELLED`

### 12.2 Error attribution

Each error must be attributable to one of:

* request validation
* backend generation
* transport
* client validation
* client runtime

### 12.3 Partial failures

If preamble succeeds but later definitions fail:

* the session must still be marked failed,
* previously executed preamble should not be treated as a successful run.

---

## 13. Performance model

The point of chunking is to reduce end-to-end wait time, not just to split strings.

## 13.1 Latency budget components

Track separately:

* request admission latency
* time to first chunk
* time to first preamble execution
* time to full definitions available
* time to execution start
* total execution duration

## 13.2 Chunk sizing

Chunk size should balance:

* early flush behavior,
* runtime parse overhead,
* transport overhead,
* and buffering pressure in proxies and browsers.

The `chunk_size_hint` in `SimRequest` is advisory only. The server may override it based on:

* model characteristics,
* transport behavior,
* configured maximums,
* client capability profile.

## 13.3 Caching

The server may cache:

* preamble fragments,
* model templates,
* compiled definitions,
* static helper code.

Caching policy must not compromise correctness when hyperparameters materially change the generated logic.

---

## 14. Security model

The streaming plane must assume executable content is security-sensitive.

## 14.1 Trust boundaries

* request parameters are untrusted input
* generated code is trusted only after server validation and integrity wrapping
* client runtime is isolated from cluster privilege boundaries

## 14.2 Required controls

* authentication/authorization on stream initiation
* per-session quota enforcement
* body size and stream duration limits
* model allowlists
* integrity metadata support
* explicit separation from route mutation APIs

## 14.3 Hard rule

The streaming plane must never be usable as an alternate control channel for:

* route updates,
* firewall commands,
* eBPF loader actions,
* or swarm control messages.

---

## 15. Observability requirements

Observability for the streaming plane must be independent from route-mutation observability, even if dashboards correlate them later.

## 15.1 Required metrics

* `stream_sessions_total{model,status}`
* `stream_request_latency_seconds`
* `stream_time_to_first_chunk_seconds`
* `stream_time_to_preamble_exec_seconds`
* `stream_time_to_execution_seconds`
* `stream_chunk_total{type}`
* `stream_bytes_total`
* `stream_failures_total{reason}`
* `stream_runtime_failures_total{reason}`
* `stream_integrity_failures_total`
* `stream_order_failures_total`

## 15.2 Required logs

Per session:

* session ID
* model ID
* chunk counts by type
* first/last sequence ID
* start/end timestamps
* runtime outcome
* failure reason if any

## 15.3 Required traces

At minimum:

* request accepted
* first chunk flushed
* preamble sent
* definitions complete
* execution chunk sent
* execution result received or timeout

---

## 16. Operational guardrails

### 16.1 Rate limiting

Limit:

* concurrent sessions per caller
* bytes streamed per session
* execution duration per session
* model-specific concurrency if some models are expensive

### 16.2 Timeouts

Define explicit budgets for:

* stream establishment
* inter-chunk idle time
* total stream duration
* runtime execution duration

### 16.3 Kill switches

Operators must be able to:

* disable execution but still allow dry-run streaming
* disable specific models
* disable all new sessions globally
* cancel stuck sessions

---

## 17. Separation from control plane

This is the most important architectural rule in this document.

The streaming plane and the route-mutation plane may coexist in the same service namespace, but they must remain **semantically and operationally isolated**.

That means:

* independent authz policies,
* independent quotas,
* independent observability,
* independent failure domains,
* no shared “hidden” control messages.

A stream failure must never:

* freeze route mutation,
* mutate kernel state,
* or alter swarm leader state.

A route mutation failure must never:

* corrupt a streaming session,
* or be reported as a browser runtime error.

---

## 18. Readiness criteria

The streaming plane is considered production-ready when:

1. chunk ordering is enforced client-side,
2. integrity checks are implemented or explicitly waived with documented risk,
3. session state transitions are observable,
4. runtime errors are classified and attributable,
5. stream time-to-first-chunk and time-to-execution are measured,
6. cancellation and timeout behavior are tested,
7. there is no path from streamed content to privileged control APIs.

---

## 19. Recommended evolution path

### Phase 0 — basic streaming

* chunking works
* preamble flush works
* runtime executes definitions + execution chunk
* basic metrics/logging exist

### Phase 1 — structured integrity

* add chunk digests and session IDs
* add replay protection
* add explicit ordering enforcement

### Phase 2 — hardened runtime

* import allowlists
* execution budgets
* session isolation guarantees
* operator kill switches

### Phase 3 — advanced optimization

* model-aware caching
* client capability negotiation
* resumable sessions or artifact reuse
* signed stream manifests

---

## 20. Summary

`StreamSimulation` is the **recoverable execution plane** of the Darwinian Gateway. It is built to reduce startup latency by streaming code in semantically meaningful phases, while preserving runtime safety, session integrity, and operational clarity.

Its job is to deliver simulation logic quickly and safely.

Its job is **not** to be clever about routing, kernel state, or swarm authority.

That distinction is what keeps the broader architecture coherent: the browser plane is allowed to be dynamic, but the kernel plane is only allowed to be correct.
