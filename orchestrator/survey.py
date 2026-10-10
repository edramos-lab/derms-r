"""Dataset profiles and saved DERMS-R scores for the dashboard and EDA views."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_IMAGE_COUNT = 8000
MAX_CLASSES = 120

TORCHVISION_MARKERS = {
    "cifar10": ("cifar-10-batches-py", "cifar-10-python.tar.gz"),
    "cifar100": ("cifar-100-python", "cifar-100-python.tar.gz"),
    "flowers102": ("flowers-102", "flowers-102-categories"),
    "food101": ("food-101",),
}


def _image_count(root: Path):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for filename in filenames:
            if Path(filename).suffix.lower() in IMAGE_SUFFIXES:
                count += 1
                if count >= MAX_IMAGE_COUNT:
                    return count, True
    return count, False


def _class_folders(root: Path):
    classes = []
    for child in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        sample = next((item for item in child.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES), None)
        if sample is None:
            continue
        count, truncated = _image_count(child)
        classes.append({"name": child.name, "images": count, "truncated": truncated})
        if len(classes) >= MAX_CLASSES:
            break
    return classes


def profile_dataset(root: Path, name: str, expected_classes):
    path = root / name
    profile = {
        "id": name,
        "present": path.is_dir() and any(path.iterdir()) if path.is_dir() else False,
        "path": str(path),
        "layout": "missing",
        "expected_classes": expected_classes,
        "class_count": 0,
        "classes": [],
        "classes_truncated": False,
        "image_count": 0,
        "image_count_truncated": False,
        "entries": [],
        "note": "Not on disk. Prepare Dataset downloads the official split into the training container.",
    }
    if not profile["present"]:
        return profile
    entries = []
    for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
        if child.name.startswith("."):
            continue
        kind = "dir" if child.is_dir() else "file"
        entries.append({"name": child.name, "kind": kind})
        if len(entries) >= 12:
            break
    profile["entries"] = entries
    markers = TORCHVISION_MARKERS.get(name, ())
    if any((path / marker).exists() for marker in markers):
        profile["layout"] = "torchvision"
        profile["class_count"] = expected_classes
        profile["note"] = "Official torchvision archive is on disk. Class names are read when training prepares the split."
        return profile
    classes = _class_folders(path)
    if classes:
        profile["layout"] = "class-folders"
        profile["classes"] = classes
        profile["class_count"] = len(classes)
        profile["classes_truncated"] = len(classes) >= MAX_CLASSES
        profile["image_count"] = sum(row["images"] for row in classes)
        profile["image_count_truncated"] = any(row["truncated"] for row in classes) or profile["classes_truncated"]
        profile["note"] = "Images are grouped in class folders."
        return profile
    count, truncated = _image_count(path)
    profile["layout"] = "files" if count or entries else "empty"
    profile["image_count"] = count
    profile["image_count_truncated"] = truncated
    profile["note"] = "Directory is present. It does not use class folders or a recognized torchvision archive."
    return profile


def profile_datasets(root: Path, catalog):
    return [profile_dataset(root, row["id"], row.get("classes")) for row in catalog]


def collect_scores(results: Path, limit_files=30):
    rows = []
    if not results.is_dir():
        return rows
    paths = []
    for path in results.rglob("summary.json"):
        if "orchestrator" in path.parts:
            continue
        paths.append(path)
        if len(paths) >= limit_files:
            break
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        best = None
        for record in data:
            if not isinstance(record, dict):
                continue
            score = record.get("derms_r_raw")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                continue
            if best is None or score > best["derms_r"]:
                best = {
                    "dataset": record.get("dataset"),
                    "model": record.get("model"),
                    "ratio": record.get("ratio"),
                    "derms_r": score,
                    "mean_val_accuracy": record.get("mean_validation_accuracy"),
                    "std_val_accuracy": record.get("sd_validation_accuracy"),
                    "num_parameters": record.get("num_parameters"),
                    "source": str(path.parent),
                }
        if best:
            try:
                mtime = path.stat().st_mtime
                best["date"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:
                best["date"] = None
            # Run output lives under <visible_results>/runs/<job_id>/... — pull
            # job_id out so the caller can look up which job (and therefore
            # which MLflow experiment) produced this summary.json.
            parts = path.parts
            best["job_id"] = parts[parts.index("runs") + 1] if "runs" in parts else None
            rows.append(best)
    rows.sort(key=lambda row: row["derms_r"], reverse=True)
    return rows
