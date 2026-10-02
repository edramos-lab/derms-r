# Step 0: install Kubernetes itself (k3s)

Everything else in `k8s/` assumes a working Kubernetes API server already
exists. This is the step that actually creates one, on the two bare-metal
boxes (RTX 5000 Ada + RTX 5090) — do this before `../30-gpu-setup/`.

## Why k3s, not kubeadm

This is a 2-node, single-purpose cluster (run the DERMS-R training matrix),
not a multi-tenant platform being folded into an existing fleet. k3s is a
fully conformant Kubernetes distribution — the NVIDIA device plugin,
kube-prometheus-stack, and every manifest in this repo work on it
unmodified — but as a single binary with an embedded datastore it's
dramatically less to install and operate than kubeadm (separate container
runtime setup, a CNI plugin, etcd) on boxes that are also meant to spend
their resources on training, not the control plane. The one real
difference: k3s bundles its own containerd at a non-default path, which
matters for exactly one later step (registering the NVIDIA runtime — see
`../30-gpu-setup/README.md`). If these nodes will eventually join a larger
cloud/enterprise Kubernetes fleet, kubeadm is the more conventional choice
instead; for this benchmark, k3s is the better fit.

## 1. Install the server (RTX 5000 Ada box)

```bash
./install-k3s-server.sh
```

Prints a `K3S_URL` and `K3S_TOKEN` — copy both for the next step. k3s does
not taint the server node, so it still runs training Jobs like any other
node; this isn't a dedicated control-plane-only machine.

## 2. Join the agent (RTX 5090 box)

```bash
K3S_URL=https://<server-ip>:6443 K3S_TOKEN=<token> ./install-k3s-agent.sh
```

## 3. Verify, from the server box

```bash
kubectl get nodes
```

Both should show `Ready`. If the agent never appears, check connectivity on
6443/tcp (API server), 8472/udp (flannel VXLAN), and 10250/tcp (kubelet)
between the two boxes.

## Next

`../30-gpu-setup/README.md` — NVIDIA Container Toolkit + containerd runtime
registration (the k3s-specific path) + the `gpu-arch` labels — on **both**
nodes. Nothing in `../60-jobs/` or `../80-monitoring/` can schedule GPU work
until that step is done.

## Uninstalling

```bash
# On the RTX 5090 box (agent):
/usr/local/bin/k3s-agent-uninstall.sh
# On the RTX 5000 Ada box (server):
/usr/local/bin/k3s-uninstall.sh
```
