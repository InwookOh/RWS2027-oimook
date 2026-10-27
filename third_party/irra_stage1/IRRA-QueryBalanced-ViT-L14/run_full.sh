#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
exec "${UPAR_PYTHON:-python}" -B "$PROJECT_ROOT/scripts/launch_full.py" "$@"
