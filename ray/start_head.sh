#!/usr/bin/env bash
# Ray head node. Doesn't need a GPU -- can run on the RTX 5000 Ada box
# itself, a third small machine, or even your laptop as long as both GPU
# workers can reach it on the ports below.
set -euo pipefail

PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

python3 -m pip install -q "ray[default]>=2.9,<3"
ray start --head \
  --port="${PORT}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${DASHBOARD_PORT}"

cat <<EOF

Head node is up. Dashboard: http://$(hostname -I 2>/dev/null | awk '{print $1}' || hostname):${DASHBOARD_PORT}
Note this node's IP -- both start_worker.sh invocations need it as RAY_HEAD_IP.
Stop with: ray stop
EOF
