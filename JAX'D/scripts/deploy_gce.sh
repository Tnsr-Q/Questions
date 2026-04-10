#!/bin/bash
set -euo pipefail

# Cost-optimized deployment for GCP free-tier credits.
ZONE="us-central1-a"
MACHINE="e2-standard-4"
IMAGE_FAMILY="debian-11"
INSTANCE_NAME="quft-verify"
BUCKET="${BUCKET:-your-verify-bucket}"

STARTUP_SCRIPT=$(cat <<'EOF'
#!/bin/bash
set -euo pipefail
apt-get update
apt-get install -y docker.io python3-pip git

WORKDIR=/home/quft-verification-suite
if [ ! -d "$WORKDIR/.git" ]; then
  echo "Repository missing at $WORKDIR" >&2
  shutdown now
  exit 1
fi

cd "$WORKDIR"
pip install pytest numpy scipy sympy pyyaml

docker build -t quft-test -f docker/Dockerfile .
mkdir -p /home/results
docker run --rm -v /home/results:/results quft-test \
  pytest tests/ -v --junitxml=/results/results.xml

gsutil cp -r /home/results/* gs://__BUCKET__/$(date +%Y%m%d)_run/
shutdown now
EOF
)

STARTUP_SCRIPT="${STARTUP_SCRIPT/__BUCKET__/$BUCKET}"

gcloud compute instances create "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project=debian-cloud \
  --metadata=startup-script="$STARTUP_SCRIPT" \
  --preemptible \
  --no-address
