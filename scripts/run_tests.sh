#!/usr/bin/env bash
# Pure-logic test runner.
#
# The host has ROS 2 Humble pytest plugins installed whose optional deps
# (lark/jinja2) are missing; disabling plugin autoload keeps collection clean.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 exec python3 -m pytest tests/ "$@"
