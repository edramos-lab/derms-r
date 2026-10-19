"""FastAPI inference server for the RIGAL-AI Inference tab.

Runs as the `inference-api` compose service, on the `derms-r-train:cuda`
image (same image training uses -- already has torch/timm, no new image to
build). Not exposed on a host port; the orchestrator (the one authenticated,
publicly reachable service) proxies browser requests to this container over
the internal `derms-r` Docker network. See `orchestrator/server.py`'s
`/api/inference/*` routes.

Two serving backends per locked model:
  - /predict        direct PyTorch, loads the checkpoint referenced by
                     lock.json (same checkpoint format test.py validates
                     against -- {'identity', 'model': state_dict, ...}).
  - /predict/triton  proxies to the triton-server container (poll mode: any
                     model export_models.py --deploy-triton has written into
                     the shared model repository is already loaded there,
                     no explicit load/unload call needed).

Both log every prediction to derms.inference_db for the dashboard.
"""
from __future__ import annotations

import io
import json
import os
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from PIL import Image
from torchvision import transforms

from .config import MEAN, STD
from .inference_db import get_records, get_stats, get_total_count, log_inference
from .models import model_for
from .utils import load_checkpoint

RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "/workspace/results"))
TRITON_MODEL_REPO = Path(os.getenv("TRITON_MODEL_REPO", "/workspace/triton-models"))
TRITON_URL = os.getenv("TRITON_URL", "http://triton-server:8000")
INPUT_SIZE = 224

app = FastAPI(title="RIGAL-AI Inference API", version="1.0.0")

# A handful of models resident at once is a non-issue on a 32GB workstation
# GPU (unlike the Jetson-constrained sibling project this pattern is ported
# from) -- no eviction policy needed for v1, just don't reload a model that's
# already loaded.
_model_cache: dict[str, torch.nn.Module] = {}
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(MEAN), std=list(STD)),
    ])


def _locked_dirs():
    """Yield (job_id, dataset, lock_dir) for every completed lock in RESULTS_ROOT."""
    for lock_path in sorted(RESULTS_ROOT.glob("runs/*/*/locked/lock.json")):
        lock_dir = lock_path.parent
        job_id = lock_dir.parents[1].name
        dataset = lock_dir.parents[0].name
        yield job_id, dataset, lock_dir


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _deployable_entry(job_id: str, dataset: str, lock_dir: Path) -> Optional[dict]:
    payload = _read_json(lock_dir / "lock.json")
    if not payload:
        return None
    selection = payload.get("selection") or {}
    winner = selection.get("winner") or {}
    architecture = winner.get("model") or winner.get("architecture")
    classes = payload.get("classes") or []
    if not architecture or not classes:
        return None
    test_payload = _read_json(lock_dir / "test_result.json")
    test_accuracy = (test_payload.get("metrics") or {}).get("accuracy")
    model_name = f"{dataset}_{architecture}"
    return {
        "job_id": job_id,
        "dataset": dataset,
        "architecture": architecture,
        "num_classes": len(classes),
        "classes": classes,
        "locked_at_utc": payload.get("locked_at_utc"),
        "tested": bool(test_payload),
        "test_accuracy": test_accuracy,
        "model_name": model_name,
        "deployed_triton": (TRITON_MODEL_REPO / model_name / "config.pbtxt").exists(),
    }


def _find_deployable(job_id: str, dataset: str) -> dict:
    lock_dir = RESULTS_ROOT / "runs" / job_id / dataset / "locked"
    entry = _deployable_entry(job_id, dataset, lock_dir)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No locked model for job={job_id} dataset={dataset}.")
    return entry


def _load_pytorch_model(job_id: str, dataset: str) -> tuple[torch.nn.Module, list[str], str]:
    entry = _find_deployable(job_id, dataset)
    cache_key = f"{job_id}:{dataset}"
    if cache_key not in _model_cache:
        lock_dir = RESULTS_ROOT / "runs" / job_id / dataset / "locked"
        model = model_for(entry["architecture"], entry["num_classes"], False)
        checkpoint = load_checkpoint(lock_dir / "best.pt")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state)
        model.to(_device)
        model.eval()
        _model_cache[cache_key] = model
    return _model_cache[cache_key], entry["classes"], entry["model_name"]


@app.get("/health")
async def health():
    return {"status": "ok", "device": _device, "cached_models": list(_model_cache.keys())}


@app.get("/deployable")
async def deployable():
    entries = []
    for job_id, dataset, lock_dir in _locked_dirs():
        entry = _deployable_entry(job_id, dataset, lock_dir)
        if entry is not None:
            entries.append(entry)
    return {"models": entries}


@app.post("/predict")
async def predict(job_id: str = Query(...), dataset: str = Query(...), file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid image: {error}")

    model, classes, model_name = _load_pytorch_model(job_id, dataset)
    tensor = _transform()(img).unsqueeze(0).to(_device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    elapsed_ms = (time.perf_counter() - start) * 1000

    class_idx = int(np.argmax(probs))
    prob_dict = {name: float(probs[i]) for i, name in enumerate(classes)}

    try:
        log_inference(backend="pytorch", dataset=dataset, model_name=model_name,
                      filename=file.filename, predicted_class=classes[class_idx],
                      class_index=class_idx, confidence=float(probs[class_idx]),
                      probabilities=prob_dict, inference_time_ms=round(elapsed_ms, 2),
                      device=_device)
    except Exception:
        pass  # logging failures must not break a prediction response

    return {
        "predicted_class": classes[class_idx],
        "class_index": class_idx,
        "confidence": float(probs[class_idx]),
        "probabilities": prob_dict,
        "inference_time_ms": round(elapsed_ms, 2),
        "backend": "pytorch",
    }


@app.post("/predict/triton")
async def predict_triton(job_id: str = Query(...), dataset: str = Query(...), file: UploadFile = File(...)):
    entry = _find_deployable(job_id, dataset)
    if not entry["deployed_triton"]:
        raise HTTPException(status_code=409,
                            detail=f"{entry['model_name']} is not deployed to Triton yet. Deploy it first.")
    classes = entry["classes"]
    triton_model = entry["model_name"]

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid image: {error}")

    tensor = _transform()(img).unsqueeze(0).numpy().astype(np.float32)
    tensor_bytes = tensor.tobytes()
    header = json.dumps({
        "inputs": [{
            "name": "input", "shape": list(tensor.shape), "datatype": "FP32",
            "parameters": {"binary_data_size": len(tensor_bytes)},
        }],
        "outputs": [{"name": "output", "parameters": {"binary_data": True}}],
    }).encode()
    body = header + tensor_bytes

    start = time.perf_counter()
    try:
        req = urllib.request.Request(
            f"{TRITON_URL}/v2/models/{triton_model}/infer",
            data=body,
            headers={
                "Content-Type": "application/octet-stream",
                "Inference-Header-Content-Length": str(len(header)),
            },
        )
        resp = urllib.request.urlopen(req, timeout=60)
        resp_body = resp.read()
    except urllib.error.URLError as error:
        raise HTTPException(status_code=502, detail=f"Triton request failed: {error}")
    elapsed_ms = (time.perf_counter() - start) * 1000

    resp_header_len = int(resp.headers.get("Inference-Header-Content-Length", len(resp_body)))
    resp_json = json.loads(resp_body[:resp_header_len])
    output_meta = resp_json["outputs"][0]
    if "binary_data_size" in output_meta.get("parameters", {}):
        binary_size = output_meta["parameters"]["binary_data_size"]
        logits_bytes = resp_body[resp_header_len:resp_header_len + binary_size]
        logits = np.array(struct.unpack(f"{len(logits_bytes) // 4}f", logits_bytes), dtype=np.float64)
    else:
        logits = np.array(output_meta["data"], dtype=np.float64)

    logits -= logits.max()
    exp_vals = np.exp(logits)
    probs = exp_vals / exp_vals.sum()
    class_idx = int(np.argmax(probs))
    prob_dict = {name: float(probs[i]) for i, name in enumerate(classes)}

    try:
        log_inference(backend="triton", dataset=dataset, model_name=triton_model,
                      filename=file.filename, predicted_class=classes[class_idx],
                      class_index=class_idx, confidence=float(probs[class_idx]),
                      probabilities=prob_dict, inference_time_ms=round(elapsed_ms, 2),
                      device="cuda")
    except Exception:
        pass

    return {
        "predicted_class": classes[class_idx],
        "class_index": class_idx,
        "confidence": float(probs[class_idx]),
        "probabilities": prob_dict,
        "inference_time_ms": round(elapsed_ms, 2),
        "backend": "triton",
    }


@app.get("/stats")
async def stats(backend: Optional[str] = None, dataset: Optional[str] = None):
    return get_stats(backend=backend, dataset=dataset)


@app.get("/records")
async def records(limit: int = 100, offset: int = 0, backend: Optional[str] = None, dataset: Optional[str] = None):
    rows = get_records(limit=limit, offset=offset, backend=backend, dataset=dataset)
    total = get_total_count(backend=backend, dataset=dataset)
    return {"records": rows, "total": total, "limit": limit, "offset": offset}
