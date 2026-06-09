#!/usr/bin/env bash
set -euo pipefail

IMAGE_PATH="${1:?Usage: examples/curl_infer.sh /path/to/image.jpg}"
QUESTION="${2:-Is any worker wearing a safety helmet?}"

IMAGE_B64="$(base64 < "$IMAGE_PATH" | tr -d '\n')"

curl -s http://localhost:8000/api/infer \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"${IMAGE_B64}\",\"question\":\"${QUESTION}\"}"
