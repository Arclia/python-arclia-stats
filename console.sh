#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd -- "$SCRIPT_DIR"


IMAGE_NAME="$(basename "$SCRIPT_DIR")_dev"


docker buildx build \
  -f Dockerfile.dev \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  -t "$IMAGE_NAME" \
  .

docker run \
  -it \
  --rm \
  --mount "type=bind,source=$(pwd),target=/usr/src/app" \
  --env TWINE_USERNAME \
  --env TWINE_PASSWORD \
  "$@" \
  "$IMAGE_NAME" bash
