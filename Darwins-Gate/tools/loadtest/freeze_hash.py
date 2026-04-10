
5) Freeze-hash computation (reference algorithm)
Pseudocode
Load JSON as a native object.
Reject any non-finite numbers.
Serialize as canonical JSON bytes.
Compute sha256.




6) How this plugs into your protodocs-driven UI

UI flow

ListKernels() returns kernels + parameters_json_schema (+ UI hints)
UI builds the form from that schema
UI emits a manifest JSON file
UI computes and displays freeze_hash immediately (same canonicalization)
StartRun(manifest) returns run_id + server-computed freeze_hash (must match)
Subscribe stream for logs + viz frames + summary

Invariant: if UI + server disagree on the freeze hash, you have a canonicalization mismatch—fix it once, and everything becomes content-addressable.






import json, hashlib, math

def _check(x):
    if isinstance(x, float):
        if not math.isfinite(x): raise ValueError("non-finite float")
        if x == 0.0: return 0.0  # normalize -0.0 -> 0.0
        return x
    if x is None or isinstance(x, (bool, int, str)): return x
    if isinstance(x, list): return [_check(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _check(v) for k, v in x.items()}
    raise TypeError(type(x))

def canonical_bytes(obj) -> bytes:
    obj = _check(obj)
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return s.encode("utf-8")

def freeze_hash_hex(manifest_obj) -> str:
    return hashlib.sha256(canonical_bytes(manifest_obj)).hexdigest()