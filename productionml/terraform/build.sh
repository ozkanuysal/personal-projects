#!/usr/bin/env bash
set -euo pipefail
set -x

USERNAME="${USERNAME:-ozkan}"
VER="${VER:-latest}"
APP_DIR="${APP_DIR:-0}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"

# Build images for different architectures
docker build -t "${USERNAME}/${APP_DIR}-arm64:${VER}" --platform linux/arm64 -f "${DOCKERFILE}" "${APP_DIR}"
docker build -t "${USERNAME}/${APP_DIR}-amd64:${VER}" --platform linux/amd64 -f "${DOCKERFILE}" "${APP_DIR}"

# Push the images
docker push "${USERNAME}/${APP_DIR}-arm64:${VER}"
docker push "${USERNAME}/${APP_DIR}-amd64:${VER}"

# Create and push the manifest
docker manifest create "${USERNAME}/${APP_DIR}:${VER}" \
    "${USERNAME}/${APP_DIR}-arm64:${VER}" \
    "${USERNAME}/${APP_DIR}-amd64:${VER}"

docker manifest push "${USERNAME}/${APP_DIR}:${VER}"