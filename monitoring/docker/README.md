# GPU + host monitoring for the Docker deployment

Standalone Prometheus + Grafana stack for the RTX 5000 Ada box (or any Docker
host running the training image). Does not modify or depend on
`../../docker-compose.yml` / `../../docker-compose.gpu.yml` — the existing
orchestrator, MLflow, and training containers keep running untouched.

## What it collects

| Exporter | Port | Metrics |
|---|---|---|
| `dcgm-exporter` (NVIDIA DCGM) | 9400 | Per-GPU utilization, memory used/free, temperature, power draw, SM clock |
| `node-exporter` | 9100 | Host CPU, memory, disk, network, load average |
| `cadvisor` | 8081 | Per-container CPU/memory (including the training containers) |
| `prometheus` | 9090 | Scrapes the three exporters above, 15-day retention |
| `grafana` | 3000 | Two pre-provisioned dashboards: GPU Overview, Host System |

## Run it

On the GPU host (driver + NVIDIA Container Toolkit already set up, same
prerequisite as the training image):

```bash
docker-compose -f monitoring/docker/docker-compose.monitoring.yml up -d
```

Open http://localhost:3000, log in `admin` / `change-me-on-first-login`
(change it immediately — `GF_SECURITY_ADMIN_PASSWORD` in
`docker-compose.monitoring.yml`), and open the **DERMS-R** folder for the
two dashboards. No manual datasource or dashboard import needed — both are
provisioned from `grafana/provisioning/` and `grafana/dashboards/` at
container start.

On a CPU-only host (e.g. the Mac), comment out the `dcgm-exporter` service
first — without a GPU + toolkit it will fail to start, and the GPU
dashboard will just show "no data" for every panel if you skip it instead.

## Remote access

If running this on the headless RTX 5000 Ada box, reach Grafana over SSH
port-forwarding rather than opening 3000/9090 to the network:

```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 edramos@10.175.190.224
```

## Stopping it

```bash
docker-compose -f monitoring/docker/docker-compose.monitoring.yml down
```

This only stops the monitoring containers (`derms-prometheus`,
`derms-grafana`, `derms-dcgm-exporter`, `derms-node-exporter`,
`derms-cadvisor`) — it never touches the orchestrator/MLflow/training
containers from the main compose stack.
