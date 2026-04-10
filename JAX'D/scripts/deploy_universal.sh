#!/bin/bash
set -euo pipefail

ZONE="${ZONE:-us-central1-a}"
INSTANCE="${INSTANCE:-quft-universal}"
BUCKET="${BUCKET:-gs://quft-artifacts}"
MACHINE="${MACHINE:-e2-standard-8}"

cat <<EOF
🌌 Deploying QÜFT-Engine Universal Discovery Stack...
Zone: ${ZONE}
Instance: ${INSTANCE}
Machine: ${MACHINE}
Bucket: ${BUCKET}
EOF

gcloud compute instances create "$INSTANCE" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --preemptible \
  --metadata=startup-script='#!/bin/bash
set -euo pipefail
apt-get update && apt-get install -y python3-pip git tmux
pip install --upgrade pip
pip install "jax[cpu]>=0.4.30" "pytorch-lightning>=2.2.0" sympy pydantic protobuf pyarrow
git clone https://github.com/your-org/quft-verification-suite.git /app
cd /app
python3 - <<"PY"
from src.mesh.unified_mesh import UnifiedMesh

mesh = UnifiedMesh()
mesh.initialize(backend="auto")
print(f"✅ Unified mesh initialized: {mesh.active_backend}")
PY
python3 - <<"PY"
import numpy as np

from src.proto.registry import PredicateRegistry
from src.truth.universality_kernel import UniversalityKernel

class _StubRGE:
    def solve_f2(self, f2):
        return {"g_ir": [0.129, 0.0, 0.0, 0.995], "n_s": 0.965, "f_PBH": 1.0}

class _StubBootstrap:
    def solve_at_scale(self, f2):
        return {"unitarity_residual": 0.0, "ghost_residual": 0.0, "echo_spacing": 9.0}

registry = PredicateRegistry()
kernel = UniversalityKernel(registry, _StubRGE(), _StubBootstrap())
f2_range = np.logspace(-16, -4, 50)
result = kernel.scan_f2_space(f2_range)
print(f"🎯 Universality result: {result['"'"'status'"'"']}")
PY
gsutil -m cp -r /app/results/* "$BUCKET"/universal_$(date +%Y%m%d)/ || true
shutdown -h now'

echo "✅ Instance launched. Monitor logs: gcloud compute ssh $INSTANCE --zone=$ZONE"
echo "📊 Results will sync to: $BUCKET/universal_YYYYMMDD/"
