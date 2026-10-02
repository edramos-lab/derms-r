#!/usr/bin/env bash
# Run on the RTX 5090 box, AFTER install-k3s-server.sh has printed its
# K3S_URL and K3S_TOKEN values on the RTX 5000 Ada box.
#
# Usage:
#   K3S_URL=https://<server-ip>:6443 K3S_TOKEN=<token> ./install-k3s-agent.sh
set -euo pipefail

: "${K3S_URL:?Set K3S_URL to the value install-k3s-server.sh printed}"
: "${K3S_TOKEN:?Set K3S_TOKEN to the value install-k3s-server.sh printed}"

curl -sfL https://get.k3s.io | K3S_URL="${K3S_URL}" K3S_TOKEN="${K3S_TOKEN}" sh -

cat <<EOF

Agent joined. Verify from the SERVER node (not here -- this node has no
kubeconfig of its own by design):
  kubectl get nodes

Both nodes should now show Ready. Next: ../30-gpu-setup/README.md on BOTH
nodes (NVIDIA Container Toolkit + containerd runtime registration + GPU
labels) before any of ../60-jobs/ will actually schedule GPU work.
EOF
