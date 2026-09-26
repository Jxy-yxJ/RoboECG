#!/usr/bin/env bash
# FARUS-inspired ECG electrode placement in Isaac Sim: local environment.
#
# Prefers the E: data partition (NTFS, /dev/nvme0n1p1) at /media/jxy/E for
# pip/tmp/shader caches so the small /home partition is not filled.  If E is
# not mounted or not writable (NTFS can go read-only after a Windows
# fast-startup shutdown), falls back to /tmp/roboecg_cache.
#
# Usage:  source scripts/env.sh

export ISAACSIM_ENV="${ISAACSIM_ENV:-/home/jxy/isaacsim-compat/env}"

# WattToolkit leaks PYTHONHOME/PYTHONPATH into the shell and breaks conda python.
unset PYTHONHOME
unset PYTHONPATH

CACHE_ROOT="/media/jxy/E/IsaacSimData"
if [ ! -d "$CACHE_ROOT" ] || ! touch "$CACHE_ROOT/.write_test" 2>/dev/null; then
  echo "[env.sh] WARNING: $CACHE_ROOT is not writable; using /tmp/roboecg_cache" >&2
  CACHE_ROOT="/tmp/roboecg_cache"
fi
rm -f "$CACHE_ROOT/.write_test" 2>/dev/null

export OMNI_KIT_ACCEPT_EULA=YES

# Isaac Sim assets are reachable directly; only set the proxy when it is up
# (a dead proxy makes the asset-root check fail).
if (exec 3<>/dev/tcp/127.0.0.1/7890) 2>/dev/null; then
  export http_proxy="http://127.0.0.1:7890"
  export https_proxy="$http_proxy"
  export HTTP_PROXY="$http_proxy"
  export HTTPS_PROXY="$https_proxy"
  exec 3<&- 2>/dev/null || true
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  echo "[env.sh] WARNING: proxy 127.0.0.1:7890 not listening; running without proxy" >&2
fi
export NO_PROXY="127.0.0.1,localhost,::1"

export PIP_CACHE_DIR="$CACHE_ROOT/pip-cache"
export TMPDIR="$CACHE_ROOT/tmp"
export __GL_SHADER_DISK_CACHE_PATH="$CACHE_ROOT/shader-cache"
export LD_LIBRARY_PATH="$ISAACSIM_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# The metrics extension calls socket.getfqdn(); this host's short name is not
# in /etc/hosts, so provide a user-level alias without touching the system.
export HOSTALIASES="$SCRIPT_DIR/isaacsim_hostaliases"

export FARUS_POSE_VENV="${FARUS_POSE_VENV:-/home/jxy/.venvs/farus_pose}"

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$__GL_SHADER_DISK_CACHE_PATH"
