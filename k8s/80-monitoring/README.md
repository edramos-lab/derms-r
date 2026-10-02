# GPU + system monitoring on Kubernetes

Uses the standard platform pattern rather than hand-rolling Prometheus/Grafana
Deployments from scratch: install **kube-prometheus-stack** (bundles
Prometheus, Grafana, node-exporter, and the Prometheus Operator CRDs) once per
cluster, then add the two pieces specific to this benchmark — the NVIDIA DCGM
exporter for GPU metrics, and the two dashboards also used by the Docker
deployment (`../../monitoring/docker/`).

## 1. Install kube-prometheus-stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set grafana.sidecar.dashboards.enabled=true \
  --set grafana.sidecar.dashboards.searchNamespace=ALL
```

This already gives you host CPU/memory/disk/network dashboards via its
bundled node-exporter — the `Host System` dashboard below duplicates the key
panels in one place alongside the GPU ones, but kube-prometheus-stack's own
node-exporter dashboards work too if you'd rather use those.

`searchNamespace=ALL` matters: the dashboard ConfigMaps below live in the
`derms-r` namespace, not `monitoring`, so the sidecar needs to look
cluster-wide (or add `derms-r` to its namespace list) to find them.

## 2. GPU metrics: DCGM exporter + ServiceMonitor

```bash
kubectl apply -f k8s/80-monitoring/dcgm-exporter.yaml
kubectl apply -f k8s/80-monitoring/servicemonitor-dcgm.yaml
```

`dcgm-exporter.yaml`'s DaemonSet schedules onto any node with the `gpu-arch`
label (both the RTX 5000 Ada and RTX 5090 pools from
`../30-gpu-setup/README.md` — node affinity checks the label key exists,
not a specific value, so one manifest covers both). The ServiceMonitor's
`release: kube-prometheus-stack` label must match whatever Helm release name
you used in step 1 — edit it if you installed under a different name.

## 3. Dashboards

```bash
kubectl apply -f k8s/80-monitoring/grafana-dashboards-configmap.yaml
```

Loaded automatically by the chart's Grafana sidecar (no manual import) —
same dashboard JSON as the Docker deployment, since DCGM and node-exporter
expose identical metric names either way.

## 4. Access Grafana

```bash
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80
```

Default login is `admin` / `prom-operator` (kube-prometheus-stack's chart
default) — change it. Open the **DERMS-R** folder, or search "GPU Overview"
/ "Host System".

## Verifying GPU metrics are flowing

```bash
kubectl exec -n derms-r -it daemonset/dcgm-exporter -- curl -s localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL
```

Should print one line per GPU on that node. If it's empty, check
`kubectl get pods -n derms-r -l app=dcgm-exporter -o wide` landed on the
expected nodes and that `nvidia-smi` works inside the pod
(`kubectl exec ... -- nvidia-smi`) — if that fails, it's the same
NVIDIA Container Toolkit / device plugin prerequisite from
`../30-gpu-setup/README.md`, not a DCGM-specific problem.

## Files

```
dcgm-exporter.yaml               DaemonSet + Service, GPU metrics
servicemonitor-dcgm.yaml         tells kube-prometheus-stack's Prometheus to scrape it
grafana-dashboards-configmap.yaml  the two dashboards, auto-loaded by the Grafana sidecar
```

Not included here on purpose: Prometheus/Grafana/Alertmanager core
Deployments. Re-implementing kube-prometheus-stack by hand would be a lot of
YAML to maintain for no benefit over the widely-used chart.
