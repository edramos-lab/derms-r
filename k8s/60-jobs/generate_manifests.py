#!/usr/bin/env python3
"""Generate the DERMS-R training/lock Job matrix for the RTX 5000 Ada +
RTX 5090 Kubernetes pool.

train.py has no --ratios flag: every invocation runs all 5 canonical ratios
x 5 folds internally for whichever --datasets/--models it's given (see
derms/config.py's RATIOS / FOLDS). So the natural unit of Kubernetes work is
one Job per (model, dataset) pair -- 12 models x 5 datasets = 60 train Jobs,
each covering 25 ratio x fold fits -- plus one lock/refit Job per dataset (5
more), for 65 Jobs total instead of 1,505 individual containers.

Each train Job is pinned to one GPU architecture (Ada or Blackwell) via
nodeSelector + a matching image tag, because the two CUDA/PyTorch builds are
not interchangeable (see ../40-images/Dockerfile.blackwell). Jobs are
load-balanced across the two GPU pools with a longest-processing-time-first
(LPT) heuristic using the same per-model/per-dataset throughput model as the
benchmark execution plan (docs/benchmark-execution-plan.md), so the Ada node
and the Blackwell node finish around the same time instead of one sitting
idle while the other works through a backlog.

Usage:
    python generate_manifests.py --registry ghcr.io/you --epochs 10
    kubectl apply -f generated/

Re-run with a different --epochs for the canonical 50-epoch pass once the
epochs=10 dry run (the default below) has validated the pipeline.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import yaml

MODELS = [
    "resnet50", "efficientnet_b0", "efficientnetv2_s", "convnextv2_tiny",
    "xception41", "vit_base", "deit3_base", "swin_tiny", "maxvit_tiny",
    "nextvit_base", "efficientvit_b0", "mobilenetv4",
]
DATASETS = ["cifar10", "cifar100", "flowers102", "food101", "all"]

# Reference throughput (img/s) used for load-balancing weight only -- same
# figures as docs/benchmark-execution-plan.md's throughput model. Not a
# performance guarantee; re-balance after the calibration run in that plan
# if measured numbers differ a lot from this ranking.
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
GPU_POOL = {
    "ada": {"node_selector": "ada", "image_tag": "ada", "relative_speed": 0.70},
    "blackwell": {"node_selector": "blackwell", "image_tag": "blackwell", "relative_speed": 1.50},
}

NAMESPACE = "derms-r"
MLFLOW_URI = f"http://mlflow.{NAMESPACE}.svc.cluster.local:5000"


def k8s_name(*parts: str) -> str:
    name = "-".join(parts).replace("_", "-").lower()
    return re.sub(r"[^a-z0-9-]", "-", name)[:253]


def job_cost(model: str, dataset: str) -> float:
    """Relative runtime proxy: dataset size / model reference throughput."""
    return N_DEV[dataset] / REF_IMG_S[model]


def assign_gpu_pools(jobs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """LPT: sort by cost desc, greedily assign to whichever pool finishes
    the whole batch earliest after taking the job, scaled by relative GPU speed."""
    costed = sorted(jobs, key=lambda j: job_cost(*j), reverse=True)
    load = {arch: 0.0 for arch in GPU_POOL}
    assignment: dict[tuple[str, str], str] = {}
    for model, dataset in costed:
        cost = job_cost(model, dataset)
        best_arch = min(
            GPU_POOL,
            key=lambda arch: load[arch] + cost / GPU_POOL[arch]["relative_speed"],
        )
        load[best_arch] += cost / GPU_POOL[best_arch]["relative_speed"]
        assignment[(model, dataset)] = best_arch
    return assignment, load


def base_env(hf_cache: bool = True) -> list[dict]:
    env = [
        {"name": "NO_ALBUMENTATIONS_UPDATE", "value": "1"},
    ]
    if hf_cache:
        env.append({"name": "HF_HOME", "value": "/root/.cache/huggingface"})
    return env


def base_volumes() -> list[dict]:
    return [
        {"name": "datasets", "persistentVolumeClaim": {"claimName": "derms-datasets"}},
        {"name": "results", "persistentVolumeClaim": {"claimName": "derms-results"}},
        {"name": "hf-cache", "persistentVolumeClaim": {"claimName": "derms-hf-cache"}},
        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}},
    ]


def base_volume_mounts() -> list[dict]:
    return [
        {"name": "datasets", "mountPath": "/workspace/datasets", "readOnly": True},
        {"name": "results", "mountPath": "/workspace/results"},
        {"name": "hf-cache", "mountPath": "/root/.cache/huggingface"},
        {"name": "shm", "mountPath": "/dev/shm"},
    ]


def train_job(model: str, dataset: str, arch: str, registry: str, epochs: int) -> dict:
    pool = GPU_POOL[arch]
    name = k8s_name("train", dataset, model)
    download_flag = [] if dataset == "all" else ["--download"]
    command = [
        "python", "train.py",
        "--data", "/workspace/datasets",
        "--output", "/workspace/results",
        "--datasets", dataset,
        "--models", model,
        "--epochs", str(epochs),
        "--batch-size", "32",
        "--device", "cuda",
        "--resume",
        *download_flag,
        "--mlflow",
        "--mlflow-tracking-uri", MLFLOW_URI,
        "--mlflow-experiment", "derms-r-benchmark",
    ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {
                "derms.role": "train",
                "derms.dataset": dataset,
                "derms.model": k8s_name(model),
                "derms.gpu-arch": arch,
            },
        },
        "spec": {
            "backoffLimit": 6,
            "suspend": False,
            "template": {
                "metadata": {
                    "labels": {
                        "derms.role": "train",
                        "derms.dataset": dataset,
                        "derms.model": k8s_name(model),
                        "derms.gpu-arch": arch,
                    }
                },
                "spec": {
                    "restartPolicy": "OnFailure",
                    "terminationGracePeriodSeconds": 60,
                    "runtimeClassName": "nvidia",
                    "nodeSelector": {"gpu-arch": pool["node_selector"]},
                    "tolerations": [
                        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
                    ],
                    "containers": [
                        {
                            "name": "train",
                            "image": f"{registry}/derms-r-train:{pool['image_tag']}",
                            "command": command,
                            "env": base_env(),
                            "resources": {
                                "requests": {"cpu": "4", "memory": "16Gi"},
                                "limits": {"cpu": "8", "memory": "24Gi", "nvidia.com/gpu": "1"},
                            },
                            "volumeMounts": base_volume_mounts(),
                        }
                    ],
                    "volumes": base_volumes(),
                },
            },
        },
    }


def lock_job(dataset: str, registry: str, num_models: int) -> dict:
    """Refits and locks the dataset's winning architecture x ratio. Must wait
    for all `num_models` train Jobs for this dataset to reach Complete first
    (train.py --lock needs the full candidate set present under --output)."""
    name = k8s_name("lock", dataset)
    wait_script = f"""
set -euo pipefail
echo "Waiting for {num_models} train Jobs (derms.dataset={dataset}) to complete..."
until [ "$(kubectl get jobs -n {NAMESPACE} -l derms.role=train,derms.dataset={dataset} \\
    -o jsonpath='{{.items[?(@.status.succeeded==1)].metadata.name}}' | wc -w)" -ge "{num_models}" ]; do
  sleep 30
done
echo "All {num_models} train Jobs for {dataset} are complete."
""".strip()
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {"derms.role": "lock", "derms.dataset": dataset},
        },
        "spec": {
            "backoffLimit": 2,
            "template": {
                "metadata": {"labels": {"derms.role": "lock", "derms.dataset": dataset}},
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": "derms-lock-waiter",
                    "runtimeClassName": "nvidia",
                    "nodeSelector": {"gpu-arch": "ada"},  # arbitrary; refit is a single run
                    "tolerations": [
                        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
                    ],
                    "initContainers": [
                        {
                            "name": "wait-for-train-jobs",
                            "image": "bitnami/kubectl:1.29",
                            "command": ["bash", "-c", wait_script],
                        }
                    ],
                    "containers": [
                        {
                            "name": "lock",
                            "image": f"{registry}/derms-r-train:ada",
                            "command": [
                                "python", "train.py",
                                "--data", "/workspace/datasets",
                                "--output", "/workspace/results",
                                "--datasets", dataset,
                                "--resume",
                                "--lock", dataset,
                                "--device", "cuda",
                                "--mlflow",
                                "--mlflow-tracking-uri", MLFLOW_URI,
                                "--mlflow-experiment", "derms-r-benchmark",
                            ],
                            "env": base_env(),
                            "resources": {
                                "requests": {"cpu": "4", "memory": "16Gi"},
                                "limits": {"cpu": "8", "memory": "24Gi", "nvidia.com/gpu": "1"},
                            },
                            "volumeMounts": base_volume_mounts(),
                        }
                    ],
                    "volumes": base_volumes(),
                },
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", required=True, help="e.g. ghcr.io/you (must match --registry used in ../40-images/build-and-push.sh)")
    ap.add_argument("--epochs", type=int, default=10, help="10 for the dry run, 50 for the canonical benchmark")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--output-dir", default="k8s/60-jobs/generated")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(m, d) for d in args.datasets for m in args.models]
    assignment, load = assign_gpu_pools(jobs)

    manifests = []
    for (model, dataset), arch in assignment.items():
        manifests.append(train_job(model, dataset, arch, args.registry, args.epochs))
    for dataset in args.datasets:
        manifests.append(lock_job(dataset, args.registry, num_models=len(args.models)))

    for doc in manifests:
        name = doc["metadata"]["name"]
        (out / f"{name}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))

    kustomization = {"resources": sorted(p.name for p in out.glob("*.yaml"))}
    (out / "kustomization.yaml").write_text(yaml.safe_dump(kustomization, sort_keys=False))

    print(f"Wrote {len(manifests)} Job manifests to {out}/ "
          f"({len(jobs)} train + {len(args.datasets)} lock).")
    print("Load balance (relative finish-time units, lower is better-matched):")
    for arch, total in load.items():
        n = sum(1 for a in assignment.values() if a == arch)
        print(f"  {arch:10s}: {n:3d} jobs assigned, relative load {total:.2f}")
    print(f"\nApply with: kubectl apply -k {out}")


if __name__ == "__main__":
    main()
