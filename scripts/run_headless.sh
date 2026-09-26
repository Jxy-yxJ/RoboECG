#!/usr/bin/env bash
# Run a python script with the Isaac Sim environment (headless by default).
#
# Usage:  ./scripts/run_headless.sh scripts/m0_check_ecg_scene.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

exec "$ISAACSIM_ENV/bin/python" "$@"
