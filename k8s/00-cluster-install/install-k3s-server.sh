#!/usr/bin/env bash
# Run this FIRST, on exactly one node, before anything else in k8s/.
# Everything else in this package (../30-gpu-setup/, ../60-jobs/, etc.)
# assumes a working Kubernetes API server already exists -- this script and
# install-k3s-agent.sh are what actually stand one up.
#
# Uses k3s (not kubeadm/full upstream k8s): single binary, embedded
# datastore, minimal footprint -- the right fit for a 2-node bare-metal GPU
# cluster where both nodes are otherwise busy running training, not for a
# multi-tenant production platform.
#
# Picks the RTX 5000 Ada box as the control-plane node by convention (k3s
# does NOT taint the server node, so it still runs training Jobs too, same
# as any other node) -- run this on that box.
set -euo pipefail

curl -sfL https://get.k3s.io | sh -s - \
  --write-kubeconfig-mode 644 \
  --disable=traefik \
  --disable=servicelb

echo
echo "== Waiting for the node to register =="
sudo /usr/local/bin/k3s kubectl wait --for=condition=Ready node --all --timeout=120s

mkdir -p "$HOME/.kube"
sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"

SERVER_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || hostname)"
TOKEN="$(sudo cat /var/lib/rancher/k3s/server/node-token)"

cat <<EOF

k3s server is up. kubectl is configured (\$KUBECONFIG not needed -- copied
to ~/.kube/config). Verify:
  kubectl get nodes

Note these two values -- install-k3s-agent.sh on the RTX 5090 box needs both:
  K3S_URL=https://${SERVER_IP}:6443
  K3S_TOKEN=${TOKEN}

Next: run install-k3s-agent.sh on the RTX 5090 box, then
../30-gpu-setup/README.md to register the NVIDIA runtime on BOTH nodes
(including this one) and apply the GPU labels.

If the two boxes aren't on a mutually-trusted private network, open
6443/tcp (API server), 8472/udp (flannel VXLAN), and 10250/tcp (kubelet)
between them before running install-k3s-agent.sh.
EOF
