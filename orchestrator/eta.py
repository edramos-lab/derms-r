"""Estimated time-to-complete for running training jobs.

Each MLflow run covers exactly one (model, ratio, fold) combo (see
derms/engine.py's `benchmark()`), so a finished run's wall-clock duration
divided by its own `epochs` param directly estimates that architecture's
seconds-per-epoch on this hardware. We prefer that per-architecture history;
for an architecture with no finished runs yet, we fall back to scaling a
global seconds-per-epoch-per-million-parameters rate (derived from whatever
architectures we do have history for) by that architecture's own parameter
count — the "based on model size" half of the estimate.
"""
import json
import urllib.error
import urllib.request
from statistics import median

from derms.config import FOLDS, RATIOS

# One MLflow run per (ratio, fold) combo, per architecture per dataset.
RUN_COMBOS_PER_MODEL_DATASET = len(RATIOS) * FOLDS


def _post(tracking_uri, path, body, timeout=8):
    request = urllib.request.Request(
        f"{tracking_uri.rstrip('/')}{path}",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _run_params(run):
    return {p["key"]: p.get("value") for p in run.get("data", {}).get("params", [])}


def seconds_per_epoch_by_model(tracking_uri, max_runs=500):
    """{architecture: median seconds/epoch} from this architecture's own
    finished MLflow runs. Best-effort: any MLflow/network failure just means
    no history, not an error — the caller falls back to size-based scaling."""
    if not tracking_uri:
        return {}
    try:
        experiments = _post(tracking_uri, "/api/2.0/mlflow/experiments/search", {"max_results": 1000})
        experiment_ids = [e["experiment_id"] for e in experiments.get("experiments", [])]
        if not experiment_ids:
            return {}
        result = _post(tracking_uri, "/api/2.0/mlflow/runs/search", {
            "experiment_ids": experiment_ids,
            "filter": "attributes.status = 'FINISHED'",
            "max_results": max_runs,
            "order_by": ["attributes.start_time DESC"],
        })
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError):
        return {}
    samples = {}
    for run in result.get("runs", []):
        info = run.get("info", {})
        start, end = info.get("start_time"), info.get("end_time")
        if not start or not end or end <= start:
            continue
        params = _run_params(run)
        model = params.get("model")
        try:
            epochs = int(params.get("epochs", 0))
        except (TypeError, ValueError):
            epochs = 0
        if not model or epochs <= 0:
            continue
        samples.setdefault(model, []).append((end - start) / 1000 / epochs)
    return {model: median(values) for model, values in samples.items()}


def estimate_job_hours(job, history, params_by_model):
    """Total estimated hours for a train job: every selected architecture,
    for every selected dataset, scored at all canonical ratios across all
    folds. Returns None if there's neither run history nor a parameter count
    for any of the job's architectures — nothing to estimate from."""
    models = job.get("models") or []
    datasets = job.get("datasets") or []
    epochs = job.get("epochs") or 0
    if not models or not datasets or epochs <= 0:
        return None
    rates = [
        history[model] / (params_by_model[model] / 1_000_000)
        for model in history
        if params_by_model.get(model)
    ]
    global_rate = median(rates) if rates else None
    total_seconds = 0.0
    estimated_any = False
    for model in models:
        seconds_per_epoch = history.get(model)
        if seconds_per_epoch is None:
            params = params_by_model.get(model)
            if params and global_rate:
                seconds_per_epoch = global_rate * (params / 1_000_000)
        if seconds_per_epoch is None:
            continue
        estimated_any = True
        total_seconds += seconds_per_epoch * epochs * len(datasets) * RUN_COMBOS_PER_MODEL_DATASET
    return total_seconds / 3600 if estimated_any else None
