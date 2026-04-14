#!/usr/bin/env bash
set -euo pipefail

PROTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO_OUT="$PROTO_DIR/../gen/go"

echo "🔍 Checking prerequisites..."
if ! command -v buf >/dev/null 2>&1; then
    echo "❌ buf not found. Install with: go install github.com/bufbuild/buf/cmd/buf@latest"
    exit 1
fi

mkdir -p "$GO_OUT"

echo "📦 Generating Go + ConnectRPC stubs..."
cd "$PROTO_DIR"
buf generate

echo "✅ Generated files in $GO_OUT/"
ls -la "$GO_OUT"/darwinianv1*/ 2>/dev/null || true
