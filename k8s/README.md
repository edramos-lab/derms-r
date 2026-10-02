# DERMS-R on Kubernetes (RTX 5000 Ada + RTX 5090)

Deployable package for running the full DERMS-R job matrix (README.md's
canonical benchmark: 12 architectures x 5 ratios x 5 folds x 5 datasets,
batched as 60 train Jobs + 5 lock Jobs) across a two-node Kubernetes pool
mixing an RTX 5000 Ada (Ada Lovelace, sm_89) and an RTX 5090 (Blackwell,
sm_120). This is the Kubernetes option from
`docs/benchmark-execution-plan.md` §5, made concrete.

**This is additive.** It doesn't touch the existing `docker-compose.yml` /
`docker-compose.gpu.yml` stack or any running containers from that setup —
this is a separate deployment path for a separate (Kubernetes) cluster.

## Why the RTX 5090 needs its own image

The repo's root `Dockerfile` is pinned to
`pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`. That build has no sm_120
kernels and will not run on an RTX 5090. `40-images/Dockerfile.blackwell`
swaps only the base image (CUDA 12.8+/PyTorch 2.7+) — everything else is
identical. Both images must exist and be scheduled onto the matching node;
see `30-gpu-setup/README.md` and `40-images/build-and-push.sh`.

## Why 65 Jobs, not 1,505

`train.py` has no `--ratios` flag — every invocation already runs all 5
canonical ratios x 5 folds internally for whichever `--models`/`--datasets`
it's given. So one Job per (model, dataset) pair covers 25 fits:
12 models x 5 datasets = 60 train Jobs, plus one lock/refit Job per dataset
(5 more) once that dataset's 12 models are done = **65 Jobs total**.

## Deploy order

```bash
# 1. GPU nodes: toolkit, device plugin, labels (one-time per cluster)
#    See 30-gpu-setup/README.md.

# 2. Edit placeholders, then apply the static scaffolding
#    (namespace, PVCs, MLflow, RBAC, the 20:00/08:00 scheduler):
#      - 10-storage/pvc-datasets.yaml, pvc-results.yaml:
#        REPLACE_ME_RWX_STORAGECLASS / REPLACE_ME_STORAGECLASS
#      - 70-scheduling/operating-window-cronjobs.yaml: REPLACE_ME_IANA_TZ
kubectl apply -k k8s/

# 3. Seed the datasets + hf-cache PVCs (out of band -- see "Seeding data" below)

# 4. Build and push both training images
REGISTRY=ghcr.io/you k8s/40-images/build-and-push.sh

# 5. Optional: prepare the "all" (Blood Cell Cancer) split, if using that dataset
#    Edit REPLACE_ME_REGISTRY in 50-prepare/prepare-all-dataset-job.yaml first.
kubectl apply -f k8s/50-prepare/prepare-all-dataset-job.yaml
kubectl wait --for=condition=complete job/prepare-all-dataset -n derms-r --timeout=30m

# 6. Generate and apply the Job matrix (epochs=10 dry run first; see
#    docs/benchmark-execution-plan.md for why)
python3 -m pip install pyyaml  # if not already available
python3 k8s/60-jobs/generate_manifests.py --registry ghcr.io/you --epochs 10
kubectl apply -k k8s/60-jobs/generated/

# 7. Watch it run
kubectl get jobs -n derms-r -L derms.gpu-arch -w
kubectl logs -n derms-r -l derms.role=train -f --max-log-requests 20

# 8. Optional but recommended: GPU + system monitoring (Grafana)
#    See 80-monitoring/README.md.
```

Once the epochs=10 dry run validates the pipeline end-to-end, re-run step 6
with `--epochs 50` into a fresh `--output-dir` (or clear the PVC's
`/workspace/results/<dataset>` subtree first) for the canonical benchmark
numbers.

## Seeding data onto the PVCs

The `derms-datasets` and `derms-hf-cache` PVCs start empty. Easiest path:
run a throwaway pod that mounts both PVCs and `kubectl cp` your local
`./datasets` and `./hf-cache` directories into it, e.g.:

```bash
kubectl run derms-seed --image=busybox -n derms-r --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"seed","image":"busybox","command":["sleep","3600"],
  "volumeMounts":[{"name":"d","mountPath":"/datasets"},{"name":"h","mountPath":"/hf-cache"}]}],
  "volumes":[{"name":"d","persistentVolumeClaim":{"claimName":"derms-datasets"}},
             {"name":"h","persistentVolumeClaim":{"claimName":"derms-hf-cache"}}]}}'
kubectl cp ./datasets derms-r/derms-seed:/datasets
kubectl cp ./hf-cache derms-r/derms-seed:/hf-cache
kubectl delete pod derms-seed -n derms-r
```

cifar10/cifar100/flowers102/food101 can also just be fetched by the training
Jobs themselves (`--download` is already in the generated train Job
commands) once the empty PVC exists — only the raw
`Blood cell Cancer [ALL]` folder needs to be copied in manually before
running `50-prepare/prepare-all-dataset-job.yaml`.

## The 20:00-08:00 operating window

`70-scheduling/operating-window-cronjobs.yaml` suspends every train/lock Job
at 08:00 and resumes them at 20:00, using Kubernetes Jobs' native
`spec.suspend` field. See that file's header comment for the one real
limitation: `train.py` has no SIGTERM handler, so the fold in progress at
suspend time restarts from its beginning on resume rather than from the
exact epoch — progress is durable only at the completed-(ratio,fold)
granularity, which is what `--resume` already checks for.

If the window shouldn't apply (e.g. you want continuous 24/7 operation),
skip applying `operating-window-cronjobs.yaml`, or `kubectl delete
cronjob derms-suspend-08h00 derms-resume-20h00 -n derms-r` after the fact.

## Load balancing across the two GPU types

`60-jobs/generate_manifests.py` assigns each (model, dataset) job to the Ada
or Blackwell pool using the same relative-throughput model as
`docs/benchmark-execution-plan.md`, via a longest-processing-time-first
(LPT) heuristic, so both nodes finish around the same time instead of one
idling. The script prints the resulting split and estimated relative load;
re-run it (it's deterministic and idempotent) if you change `--models` or
`--datasets`.

## Monitoring

Training progress:
- MLflow UI: `kubectl port-forward -n derms-r svc/mlflow 5001:5000` then open
  http://localhost:5001, or use the `mlflow-nodeport` Service
  (`20-mlflow/mlflow.yaml`) on port 30500 of any node.
- Per-dataset progress: `kubectl get jobs -n derms-r -l derms.dataset=cifar10`
- Overall: `kubectl get jobs -n derms-r -L derms.role,derms.dataset,derms.gpu-arch`

GPU and host system (utilization, memory, temperature, power, CPU, disk,
network): Grafana via kube-prometheus-stack + the NVIDIA DCGM exporter — see
**`80-monitoring/README.md`**. Same two dashboards (GPU Overview, Host
System) as the Docker deployment's `monitoring/docker/`, since both expose
identical Prometheus metric names.

## Files

```
00-namespace.yaml                  derms-r namespace
10-storage/                        PVCs for datasets, hf-cache, results, mlflow data
20-mlflow/mlflow.yaml              MLflow server (Deployment + Service)
30-gpu-setup/README.md             device plugin install + node labeling (imperative, one-time)
40-images/                         Dockerfile.blackwell + build-and-push.sh
50-prepare/                        optional one-off Job for the "all" dataset split
60-jobs/
  rbac.yaml                        ServiceAccount/Role for the lock Job's wait step
  generate_manifests.py            builds the 65-Job matrix (not applied by kustomization.yaml)
  generated/                       generate_manifests.py's output (gitignored; regenerate as needed)
70-scheduling/
  rbac.yaml                        ServiceAccount/Role for the suspend/resume CronJobs
  operating-window-cronjobs.yaml   20:00-08:00 suspend/resume
80-monitoring/                     DCGM exporter + ServiceMonitor + Grafana dashboards (kube-prometheus-stack)
kustomization.yaml                 ties together everything except 60-jobs/generated/
```
