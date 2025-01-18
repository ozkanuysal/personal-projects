#!/usr/bin/env bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -euo pipefail

if [ -x "$(command -v docker-compose)" ]; then
    dc=docker-compose
else
    dc="docker compose"
fi

export COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yml"
if [ $# -gt 0 ]; then
    exec $dc run --rm airflow_webserver "${@}"
else
    exec $dc run --rm airflow_webserver
fi