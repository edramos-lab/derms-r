# GPU node setup

One-time per cluster. Do this before applying anything under `60-jobs/`.

## 1. NVIDIA Container Toolkit on every GPU node

Both the RTX 5000 Ada node and the RTX 5090 node need the NVIDIA Container
Toolkit registered as the container runtime (same thing the RTX 5000 Ada box
already needed for plain `docker run --gpus all`, per the existing
deployment notes for that host). Verify on each node:

```bash
nvidia-smi                 # driver + GPU visible
nvidia-ctk runtime configure --runtime=containerd   # or docker, matching your CRI
```

## 2. Install the NVIDIA device plugin (DaemonSet)

This is what turns `nvidia.com/gpu` into a schedulable resource. Install once
for the whole cluster (it runs on every node with the toolkit, auto-detects
the GPU count):

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm install nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin --create-namespace \
  --version 0.16.2
```

Confirm each node now advertises `nvidia.com/gpu: 1`:

```bash
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpu: .status.allocatable["nvidia.com/gpu"]}'
```

## 3. Label each node by GPU architecture

This is the field the training Jobs use to pick the matching image
(`derms-r-train:ada` vs `derms-r-train:blackwell`, see `../40-images/`) and
land on the right physical card — **required**, because the Ada image will
not run on the RTX 5090 (sm_120 / Blackwell) and vice versa; this is a CUDA
compatibility issue, not something Kubernetes scheduling can route around on
its own.

```bash
kubectl label node <rtx-5000-ada-node-name> gpu-arch=ada derms.gpu/model=rtx5000-ada
kubectl label node <rtx-5090-node-name>     gpu-arch=blackwell derms.gpu/model=rtx5090
```

Find node names with `kubectl get nodes`. If both GPUs are plugged into the
same physical host, use `nodeSelector` + `nvidia.com/gpu` resource requests
as normal — the device plugin and `CUDA_VISIBLE_DEVICES` handle picking the
specific card; the `gpu-arch` label still needs to resolve to whichever node
hosts each card (normally one node = one label value, since the image choice
is host-wide, not per-GPU).

## 4. Verify

```bash
kubectl get nodes -L gpu-arch,derms.gpu/model
```

Both nodes should show up with their label before moving on to
`../40-images/` (build the two training images) and `../50-prepare/`
(optional ALL-dataset prep job).
