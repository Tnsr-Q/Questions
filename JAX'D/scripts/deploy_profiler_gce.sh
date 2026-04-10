#!/bin/bash
set -euo pipefail

# Configuration
ZONE="us-central1-a"
INSTANCE="quft-regge-profiler"
BUCKET="${BUCKET:-gs://your-quft-artifacts-bucket}"
MACHINE="e2-standard-4"  # 4 vCPU, 16GB RAM (credit efficient)
SHUTDOWN_HOURS="${SHUTDOWN_HOURS:-4}"
PROFILE_RUN_ID="run_$(date +%Y%m%d_%H%M)"
REPO_URL="${REPO_URL:-https://github.com/your-org/quft-verification-suite.git}"

echo "🚀 Creating preemptible GCE instance for Regge profiling..."
gcloud compute instances create "$INSTANCE" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --preemptible \
  --no-address \
  --tags=tensorboard-server \
  --metadata=startup-script="#!/bin/bash
    set -euo pipefail
    apt-get update && apt-get install -y python3-pip git tmux
    pip install --break-system-packages --upgrade pip
    pip install --break-system-packages 'jax[cpu]>=0.4.30' numpy scipy tensorboard pyyaml
    git clone '$REPO_URL' /app
    cd /app
    mkdir -p logs/regge_prof

    tmux new-session -d -s profiler \"python3 -c \\\"from src.regge_shard_map_profiler import ProfiledShardedReggeSolver; import jax.numpy as jnp; solver = ProfiledShardedReggeSolver(N_t=256, profile_dir='./logs/regge_prof'); delta = jnp.zeros(256) + 0.05; traj, meta = solver.scan_with_profiler(delta, run_id='$PROFILE_RUN_ID'); import json; print(json.dumps(meta))\\\" && echo '[✅ Profiling Complete]' && sleep 30\"

    tensorboard --logdir=./logs/regge_prof --host=0.0.0.0 --port=6006 --reload_interval=10 &
    echo 'TensorBoard running on :6006'

    (sleep $((SHUTDOWN_HOURS*3600)) && gsutil -m cp -r ./logs/regge_prof '$BUCKET' && shutdown -h now) &
  "

echo "📡 Instance created. Wait 2-3 minutes for startup."
echo "🔒 To proxy TensorBoard securely:"
echo "   gcloud compute ssh $INSTANCE --zone=$ZONE -- -N -L 6006:localhost:6006"
echo "🌐 Open http://localhost:6006 in your browser."
echo "⏱️  Auto-shutdown scheduled after $SHUTDOWN_HOURS hours. Logs will sync to $BUCKET."
