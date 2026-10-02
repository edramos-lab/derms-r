# DERMS-R on Ray (RTX 5000 Ada + RTX 5090)

Lighter-weight alternative to `../k8s/` for the same two-GPU pool — no
Kubernetes control plane, no NVIDIA device plugin, no CronJob/RBAC machinery.
Same job unit (one task per (model, dataset) pair, 60 train + 5 lock = 65
tasks covering the 1,505 fits) and the same LPT load-balancing between the
Ada and Blackwell pools as `../k8s/60-jobs/generate_manifests.py`.

**This is additive**, like `../k8s/` — it doesn't touch the existing Docker
Compose stack or the Kubernetes package; pick whichever orchestration path
fits your situation (see `docs/benchmark-execution-plan.md` §5 vs §5.3).

## Prerequisites

- Both training images already built (`../k8s/40-images/build-and-push.sh`,
  or build locally without pushing anywhere since Ray workers run them
  directly via `docker run`, not a registry pull).
- Shared storage for `datasets/`, `results/`, and `hf-cache/` reachable at
  the **same path** on both GPU boxes (NFS mount, or rsync'd copies kept in
  sync) — `train.py --resume` needs to see the same `results/` tree
  regardless of which node a given task lands on next.
- MLflow reachable from both GPU boxes (reuse the existing
  `docker-compose.yml` MLflow service, or the `k8s/20-mlflow/` one).

## 1. Start the head node

On any reachable machine (doesn't need a GPU):

```bash
RAY_HEAD_IP=<note this node's IP> ./start_head.sh
```

Dashboard at `http://<head-ip>:8265`.

## 2. Start a worker on each GPU box

On the RTX 5000 Ada box:

```bash
RAY_HEAD_IP=<head-ip> GPU_ARCH=ada \
  DATASETS_DIR=/path/to/shared/datasets \
  RESULTS_DIR=/path/to/shared/results \
  ./start_worker.sh
```

On the RTX 5090 box:

```bash
RAY_HEAD_IP=<head-ip> GPU_ARCH=blackwell \
  DATASETS_DIR=/path/to/shared/datasets \
  RESULTS_DIR=/path/to/shared/results \
  ./start_worker.sh
```

Each worker runs **inside** the matching training image
(`derms-r-train:ada` / `derms-r-train:blackwell`, same two images as the
Kubernetes path — RTX 5090 is Blackwell/sm_120 and will not run the Ada
image) and tags itself with a Ray custom resource (`ada` or `blackwell`) so
`driver.py` can pin tasks to the right physical GPU. Confirm both joined:

```bash
docker exec derms-ray-worker-ada ray status   # from either worker, or the head
```

You should see both custom resources (`ada: 1.0`, `blackwell: 1.0`) listed.

## 3. Run the driver

From the head node (or anywhere that can reach it):

```bash
python3 -m pip install "ray[default]>=2.9,<3"
python3 driver.py --epochs 10 --mlflow-uri http://<mlflow-host>:5001
```

This submits all 60 train tasks (load-balanced across `ada`/`blackwell`),
then per dataset waits for its 12 train tasks and submits that dataset's
lock/refit task. Prints a `model/dataset: completed` line as each task
finishes. Once the epochs=10 dry run validates the pipeline, re-run with
`--epochs 50` for the canonical benchmark.

Add `--no-window` to ignore the 20:00-08:00 operating window and run
continuously instead (see `docs/benchmark-execution-plan.md` §3.1 for why
that window exists and what it costs in calendar time).

## The 20:00-08:00 operating window

Unlike the Kubernetes path's native `spec.suspend`, `driver.py` implements
the window itself: each task waits for the window to open before starting,
and a watchdog inside `run_within_window()` terminates the subprocess if the
window closes before it finishes, then waits for the next window and retries
with `--resume`. Same limitation as the Kubernetes CronJobs: `train.py` has
no SIGTERM handler, so the fold in progress at window-close restarts from
its beginning, not mid-epoch — progress is durable only at the
completed-(ratio,fold) granularity that `--resume` checks for.

## Stopping

```bash
# On each GPU box:
docker stop derms-ray-worker-ada      # or derms-ray-worker-blackwell
docker rm derms-ray-worker-ada
# On the head node:
ray stop
```

## Files

```
start_head.sh     Ray head node (no GPU needed)
start_worker.sh   GPU worker, runs inside the ada/blackwell training image
driver.py         submits the 65-task matrix with LPT load balancing + the operating window
```
