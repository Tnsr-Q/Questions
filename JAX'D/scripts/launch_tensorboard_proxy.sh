#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Scanning for shard layout traces..."
find . -name "shard_layouts.json" -exec cat {} \; | jq . || true

echo ""
echo "🌐 Launching TensorBoard with automatic profile visualization..."
tensorboard --logdir=./tb_regge_layout --bind_all --port=6006
