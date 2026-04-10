#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results

FREEZE_FLAG=""
AUDIT_VERIFY="false"
for arg in "$@"; do
  if [[ "$arg" == "--freeze" ]]; then
    FREEZE_FLAG="--freeze-tolerances"
  elif [[ "$arg" == "--audit-verify" ]]; then
    AUDIT_VERIFY="true"
  fi
done

echo "Running verification suite with 2 hour safety timeout..."
timeout 7200 pytest tests/ -v --junitxml=results/results.xml \
  --tolerance-ledger="configs/tolerance_priors.yaml" \
  $FREEZE_FLAG

if [[ "$AUDIT_VERIFY" == "true" ]]; then
  python -c "
from src.tolerance.dynamic_ledger import DynamicToleranceLedger
import hashlib
import json
ledger = DynamicToleranceLedger(base_config_path='configs/tolerance_priors.yaml', freeze_mode=True)
snap = json.dumps(ledger.export_snapshot(), sort_keys=True)
print('✅ Tolerance checksum:', hashlib.sha256(snap.encode()).hexdigest())
"
fi

echo "Results written to results/results.xml"
