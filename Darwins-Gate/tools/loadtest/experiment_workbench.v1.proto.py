No “function” blocks in proto (proto can’t define functions; validation is an RPC).
Separate Kernel Registry from Experiment Runs (otherwise you paint yourself into a corner).
Make the “manifest” a stable, canonical ExperimentManifest and treat “knobs” as UI metadata derived from a schema (or embedded as UI hints).
Switch from float to double for scientific knobs.
Make VizFrame a oneof payload (Arrow IPC / JSON / small binary), not a single bytes blob with ad-hoc interpretation.
Add content-addressed hashes (freeze_hash) as bytes or hex string, not uint64 (uint64 truncates).
Use server streaming for logs/frames/results; keep final results as a URI or inline summary.






syntax = "proto3";

package experiment_workbench.v1;

import "google/protobuf/any.proto";
import "google/protobuf/duration.proto";
import "google/protobuf/struct.proto";
import "google/protobuf/timestamp.proto";

// -----------------------------
// 1) Registry: kernels + schemas
// -----------------------------

message KernelId {
  string name = 1;      // e.g. "ligo.echo_search"
  string version = 2;   // SemVer e.g. "1.2.0"
}

message KernelCapability {
  bool supports_cluster_compute = 1;
  bool supports_browser_viz = 2;
  bool supports_browser_preview_compute = 3; // lightweight subset, optional
}

message KernelDescriptor {
  KernelId id = 1;
  string title = 2;
  string description = 3;
  KernelCapability capability = 4;

  // Your UI builder can ingest either:
  // (A) JSON Schema (recommended) OR
  // (B) proto-derived schema + UI hints
  //
  // Keep it as bytes to avoid baking JSON schema into protobuf types.
  bytes manifest_json_schema = 10;   // Draft 2020-12 JSON Schema, UTF-8 bytes
  bytes ui_hints_json = 11;          // optional UI rendering hints, UTF-8 JSON

  // For output typing: schema for results & viz frames (optional)
  bytes results_json_schema = 12;
}

message ListKernelsRequest {}
message ListKernelsResponse {
  repeated KernelDescriptor kernels = 1;
}

message GetKernelRequest {
  KernelId id = 1;
}
message GetKernelResponse {
  KernelDescriptor kernel = 1;
}

service KernelRegistryService {
  rpc ListKernels(ListKernelsRequest) returns (ListKernelsResponse);
  rpc GetKernel(GetKernelRequest) returns (GetKernelResponse);
}


// -----------------------------
// 2) Experiment manifest: the contract
// -----------------------------

// A single, canonical “run spec” the UI edits.
// You can store this as JSON and send as bytes too; here it’s structured.
message ExperimentManifest {
  KernelId kernel = 1;                 // which kernel/plugin executes this
  string experiment_id = 2;            // human-friendly or user-supplied
  string title = 3;
  string owner = 4;

  // Parameters must be generic to support "any theory".
  // Use Struct so you’re not constantly regenerating proto per kernel.
  // The schema in KernelDescriptor constrains this.
  google.protobuf.Struct parameters = 10;

  // Input bindings: logical name -> URI/path + optional hash
  repeated InputBinding inputs = 11;

  // Target execution hint
  ExecutionTarget target = 12;

  Reproducibility reproducibility = 13;

  // Optional: any extra metadata the UI/orchestrator wants to preserve
  google.protobuf.Struct metadata = 14;
}

message InputBinding {
  string name = 1;            // "h1", "l1", "events_csv", etc.
  string uri = 2;             // file://..., s3://..., https://..., nix-store path, etc.
  string sha256_hex = 3;      // optional 64-hex; if set, must match
  uint64 size_bytes = 4;      // optional preflight
}

enum ExecutionTarget {
  EXECUTION_TARGET_UNSPECIFIED = 0;
  EXECUTION_TARGET_CLUSTER = 1;   // Nix/OCI/K8s job
  EXECUTION_TARGET_LOCAL = 2;     // local nix run
  EXECUTION_TARGET_BROWSER_VIZ = 3; // viz-only; compute done elsewhere
  EXECUTION_TARGET_BROWSER_PREVIEW = 4; // lightweight compute if kernel supports
}

message Reproducibility {
  uint64 seed = 1;
  bool require_hash_check = 2;
  // Record determinism knobs (thread counts, hashseed) for audits
  map<string, string> env = 3; // e.g. OMP_NUM_THREADS=1, PYTHONHASHSEED=0
}


// -----------------------------
// 3) Validation / Preflight
// -----------------------------

message ValidateManifestRequest {
  ExperimentManifest manifest = 1;
}

message ValidationIssue {
  enum Severity {
    SEVERITY_UNSPECIFIED = 0;
    INFO = 1;
    WARNING = 2;
    ERROR = 3;
  }
  Severity severity = 1;
  string path = 2;        // JSON pointer-like: "/parameters/hyperparameters/dt_step"
  string message = 3;
}

message ValidateManifestResponse {
  bool ok = 1;
  repeated ValidationIssue issues = 2;

  // helpful for UI: estimated resource cost & whether browser target is feasible
  google.protobuf.Duration estimated_runtime = 10;
  uint64 estimated_peak_memory_bytes = 11;
  bool feasible_in_browser = 12;
}

service ExperimentValidationService {
  rpc ValidateManifest(ValidateManifestRequest) returns (ValidateManifestResponse);
}


// -----------------------------
// 4) Run orchestration + streaming updates
// -----------------------------

message StartRunRequest {
  ExperimentManifest manifest = 1;
  // Optional: orchestrator hints (queue, priority, resource caps)
  google.protobuf.Struct run_hints = 2;
}

message RunId {
  string id = 1; // UUID/ULID
}

message StartRunResponse {
  RunId run_id = 1;

  // Content-addressed identity of the *intended* run:
  // sha256(canonical(manifest)) recommended.
  string freeze_hash_hex = 2;

  // Where results will be persisted (optional)
  string results_uri = 3;
}

message SubscribeRunRequest {
  RunId run_id = 1;
}

// One stream that carries logs, progress, frames, and final summary.
message RunEvent {
  oneof event {
    LogEvent log = 1;
    ProgressEvent progress = 2;
    VizFrameEvent viz = 3;
    ResultSummaryEvent summary = 4;
    RunEndedEvent ended = 5;
  }
}

message LogEvent {
  google.protobuf.Timestamp time = 1;
  string stage = 2;       // "preflight", "loading", "fft", "nulls", "persist"
  string message = 3;
}

message ProgressEvent {
  google.protobuf.Timestamp time = 1;
  string stage = 2;
  double fraction = 3;    // 0..1
}

message VizFrameEvent {
  google.protobuf.Timestamp time = 1;
  string freeze_hash_hex = 2;
  VizFrame frame = 3;
}

message ResultSummaryEvent {
  google.protobuf.Timestamp time = 1;
  string freeze_hash_hex = 2;

  // Small summary in-band for UI
  google.protobuf.Struct summary = 3;

  // Full results stored out-of-band
  string results_uri = 4;
  string run_manifest_out_uri = 5;
}

message RunEndedEvent {
  google.protobuf.Timestamp time = 1;
  enum Status {
    STATUS_UNSPECIFIED = 0;
    SUCCEEDED = 1;
    FAILED = 2;
    CANCELED = 3;
  }
  Status status = 2;
  string error_message = 3;
}

// Viz frames: typed payload + hints.
// Don’t force Arrow always; allow small JSON or image.
message VizFrame {
  string kind = 1;  // "timeseries", "spectrum", "histogram", "heatmap", "network_graph", ...
  google.protobuf.Struct spec = 2; // axis labels, units, log flags, styling hints, etc.

  oneof payload {
    ArrowIpc arrow_ipc = 10;
    JsonTable json_table = 11;
    ImageBlob image = 12;
  }
}

message ArrowIpc {
  bytes ipc = 1;           // Arrow IPC stream/file bytes
  string mime = 2;         // "application/vnd.apache.arrow.stream"
}

message JsonTable {
  // Small payloads only; for big data use Arrow.
  google.protobuf.Struct table = 1; // e.g. {columns:[...], data:[...]} or row-based
}

message ImageBlob {
  bytes data = 1;
  string mime = 2;         // "image/png", "image/webp"
}

service ExperimentRunService {
  rpc StartRun(StartRunRequest) returns (StartRunResponse);
  rpc SubscribeRun(SubscribeRunRequest) returns (stream RunEvent);
}