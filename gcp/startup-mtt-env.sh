#!/bin/bash
# MTT-runner boot provisioning: builds shared Python 3.10 env with pokerstove + deps under /opt.
# Idempotent: skips if /opt/mttenv/.ready already exists (survives reboots on a kept disk).
# Repo itself is NOT synced here (configs change per experiment) — tar-sync it per run.
set -euo pipefail

READY=/opt/mttenv/.ready
LOG=/var/log/mtt-startup.log
exec > >(tee -a "$LOG") 2>&1
echo "=== mtt startup $(date -u) ==="

if [ -f "$READY" ]; then
  echo "env already provisioned, skipping"
  exit 0
fi

PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20240814/cpython-3.10.14+20240814-x86_64-unknown-linux-gnu-install_only.tar.gz"
WHEEL_GS="gs://mtt-results/setup/pokerstove-1.2-cp310-cp310-linux_x86_64.whl"

# 1. standalone CPython 3.10.14 (Debian 13 ships py3.13; pokerstove wheel is cp310)
if [ ! -x /opt/py310/bin/python3.10 ]; then
  echo "--- fetching standalone python 3.10.14"
  curl -fsSL "$PY_URL" -o /tmp/py310.tar.gz
  mkdir -p /opt/py310
  tar xzf /tmp/py310.tar.gz -C /opt --strip-components=0   # extracts to /opt/python
  rm -rf /opt/py310 && mv /opt/python /opt/py310
fi

# 2. shared venv + pinned deps (cp310 wheels exist for these exact versions)
echo "--- building venv /opt/mttenv"
/opt/py310/bin/python3.10 -m venv /opt/mttenv
/opt/mttenv/bin/pip install --upgrade pip
/opt/mttenv/bin/pip install treys==0.1.3 numpy==1.26.4 pandas==2.2.1

# 3. pokerstove native wheel — pip-installed into venv (loader imports it directly, line 21-26)
echo "--- installing pokerstove wheel"
gsutil cp "$WHEEL_GS" /tmp/pokerstove.whl
/opt/mttenv/bin/pip install /tmp/pokerstove.whl

# 4. sanity: import + pokerstove load
/opt/mttenv/bin/python - <<'PY'
import numpy, pandas, treys, pokerstove
assert hasattr(pokerstove, "CardSet") and hasattr(pokerstove, "PokerHandEvaluator")
print("ENV_OK", "np", numpy.__version__, "pd", pandas.__version__)
PY

# 5. gcsfuse (bucket mount convenience; report pull still works over ssh)
if ! command -v gcsfuse >/dev/null 2>&1; then
  echo "--- installing gcsfuse"
  export GCSFUSE_REPO=gcsfuse-$(. /etc/os-release; echo "$VERSION_CODENAME")
  echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" > /etc/apt/sources.list.d/gcsfuse.list
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg || true
  apt-get update -y && apt-get install -y gcsfuse || echo "gcsfuse install failed (non-fatal)"
fi

touch "$READY"
echo "=== mtt startup DONE $(date -u) — env at /opt/mttenv/bin/python ==="
