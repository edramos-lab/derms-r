#!/usr/bin/env python3
"""Ray driver for the DERMS-R job matrix across RTX 5000 Ada + RTX 5090 Ray
workers (see start_head.sh / start_worker.sh).

Same job unit and load-balancing as k8s/60-jobs/generate_manifests.py: one
task per (model, dataset) pair (60 train tasks, each internally running 25
ratio x fold fits via train.py -- it has no --ratios flag) plus 5 lock tasks,
assigned to the "ada" or "blackwell" Ray custom resource with a
longest-processing-time-first heuristic so both GPU pools finish around the
same time.

Each task wraps `python train.py ...` as a subprocess rather than importing
engine.py directly, so the worker's environment (CUDA, torch, timm) is
exactly what the training image already provides.

Run on the Ray head node (or any machine that can reach it):
    python driver.py --epochs 10 --mlflow-uri http://<mlflow-host>:5001
"""
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime

import ray

MODELS = [
    "resnet50", "efficientnet_b0", "efficientnetv2_s", "convnextv2_tiny",
    "xception41", "vit_base", "deit3_base", "swin_tiny", "maxvit_tiny",
    "nextvit_base", "efficientvit_b0", "mobilenetv4",
]
DATASETS = ["cifar10", "cifar100", "flowers102", "food101", "all"]

# Same figures as docs/benchmark-execution-plan.md / k8s/60-jobs/generate_manifests.py.
REF_IMG_S = {
    "mobilenetv4": 900, "efficientnet_b0": 550, "efficientvit_b0": 700,
    "resnet50": 500, "efficientnetv2_s": 380, "xception41": 350,
    "convnextv2_tiny": 300, "swin_tiny": 320, "maxvit_tiny": 220,
    "nextvit_base": 260, "deit3_base": 230, "vit_base": 230,
}
N_DEV = {
    "cifar10": 50_000, "cifar100": 50_000, "flowers102": 2_040,
    "food101": 75_750, "all": 2_918,
}
GPU_POOL = {"ada": 0.70, "blackwell": 1.50}  # relative_speed, matches the plan's §3


def job_cost(model: str, dataset: str) -> float:
    return N_DEV[dataset] / REF_IMG_S[model]


def assign_gpu_pools(jobs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    costed = sorted(jobs, key=lambda j: job_cost(*j), reverse=True)
    load = {arch: 0.0 for arch in GPU_POOL}
    assignment: dict[tuple[str, str], str] = {}
    for model, dataset in costed:
        cost = job_cost(model, dataset)
        best_arch = min(GPU_POOL, key=lambda a: load[a] + cost / GPU_POOL[a])
        load[best_arch] += cost / GPU_POOL[best_arch]
        assignment[(model, dataset)] = best_arch
    return assignment, load


# ---- 20:00-08:00 operating window (see docs/benchmark-execution-plan.md §3.1) ----
WINDOW_START_HOUR = 20
WINDOW_END_HOUR = 8


def in_window(now: datetime | None = None) -> bool:
    h = (now or datetime.now()).hour
    return h >= WINDOW_START_HOUR or h < WINDOW_END_HOUR


def wait_for_window() -> None:
    while not in_window():
        time.sleep(300)


def run_within_window(cmd: list[str], cwd: str) -> str:
    """Runs cmd until it finishes or the operating window closes. If the
    window closes first, terminates the subprocess and reports 'interrupted'
    -- same known limitation as the Kubernetes CronJob suspend (no SIGTERM
    handler in train.py, so the fold in progress restarts from its beginning
    next window; --resume already skips completed (ratio,fold) combinations)."""
    wait_for_window()
    proc = subprocess.Popen(cmd, cwd=cwd)
    while proc.poll() is None:
        if not in_window():
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            return "interrupted"
        time.sleep(30)
    return "completed" if proc.returncode == 0 else f"failed({proc.returncode})"


def train_command(model: str, dataset: str, epochs: int, mlflow_uri: str | None,
                   data_dir: str, output_dir: str) -> list[str]:
    cmd = [
        "python", "train.py",
        "--data", data_dir, "--output", output_dir,
        "--datasets", dataset, "--models", model,
        "--epochs", str(epochs), "--batch-size", "32", "--device", "cuda",
        "--resume",
    ]
    if dataset != "all":
        cmd.append("--download")
    if mlflow_uri:
        cmd += ["--mlflow", "--mlflow-tracking-uri", mlflow_uri, "--mlflow-experiment", "derms-r-benchmark"]
    return cmd


def lock_command(dataset: str, mlflow_uri: str | None, data_dir: str, output_dir: str) -> list[str]:
    cmd = [
        "python", "train.py",
        "--data", data_dir, "--output", output_dir,
        "--datasets", dataset, "--resume", "--lock", dataset, "--device", "cuda",
    ]
    if mlflow_uri:
        cmd += ["--mlflow", "--mlflow-tracking-uri", mlflow_uri, "--mlflow-experiment", "derms-r-benchmark"]
    return cmd


@ray.remote(num_gpus=1, max_retries=3)
def run_train(model: str, dataset: str, epochs: int, mlflow_uri: str | None,
              data_dir: str, output_dir: str, cwd: str) -> str:
    cmd = train_command(model, dataset, epochs, mlflow_uri, data_dir, output_dir)
    result = run_within_window(cmd, cwd)
    while result == "interrupted":
        result = run_within_window(cmd, cwd)  # --resume picks up where it left off
    return f"{model}/{dataset}: {result}"


@ray.remote(num_gpus=1, max_retries=1)
def run_lock(dataset: str, mlflow_uri: str | None, data_dir: str, output_dir: str, cwd: str) -> str:
    cmd = lock_command(dataset, mlflow_uri, data_dir, output_dir)
    result = run_within_window(cmd, cwd)
    while result == "interrupted":
        result = run_within_window(cmd, cwd)
    return f"lock/{dataset}: {result}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=10, help="10 for the dry run, 50 for the canonical benchmark")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--mlflow-uri", default=None, help="e.g. http://<mlflow-host>:5001")
    ap.add_argument("--data-dir", default="/workspace/datasets", help="path as seen INSIDE the worker container")
    ap.add_argument("--output-dir", default="/workspace/results", help="path as seen INSIDE the worker container")
    ap.add_argument("--cwd", default="/workspace", help="train.py's directory inside the worker container")
    ap.add_argument("--no-window", action="store_true", help="ignore the 20:00-08:00 window and run continuously")
    args = ap.parse_args()

    global WINDOW_START_HOUR, WINDOW_END_HOUR
    if args.no_window:
        WINDOW_START_HOUR, WINDOW_END_HOUR = 0, 24

    ray.init(address="auto")

    jobs = [(m, d) for d in args.datasets for m in args.models]
    assignment, load = assign_gpu_pools(jobs)
    print(f"Load balance: {({a: round(l, 1) for a, l in load.items()})}")

    train_futures = {
        (model, dataset): run_train.options(resources={arch: 1}).remote(
            model, dataset, args.epochs, args.mlflow_uri, args.data_dir, args.output_dir, args.cwd
        )
        for (model, dataset), arch in assignment.items()
    }

    for dataset in args.datasets:
        ds_futures = [train_futures[(m, dataset)] for m in args.models]
        results = ray.get(ds_futures)
        for r in results:
            print(r)
        lock_future = run_lock.options(resources={"ada": 1}).remote(
            dataset, args.mlflow_uri, args.data_dir, args.output_dir, args.cwd
        )
        print(ray.get(lock_future))

    print("All 65 tasks (60 train + 5 lock) complete.")


if __name__ == "__main__":
    main()
