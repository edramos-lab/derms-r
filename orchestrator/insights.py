"""Exploratory Dataset Analysis (EDA) for the DERMS-R orchestrator.

Ports the metrics and plots from the standalone EDA4ImageClassification tool
(~/Documents/MLOPS/EDA4ImageClassification) onto locally prepared datasets:
class-balance statistics (entropy/Gini/CV/imbalance ratio), image-size
statistics, an RGB channel histogram, a sample-image grid, a K-Fold /
Stratified K-Fold split visualization, and PCA (2D) / t-SNE (3D) projections
of a per-image mean-RGB feature vector — the same feature EDA4ImageClassification
uses for its `/dimred` endpoint.

Two data sources feed this, kept separate for speed:
  - "full" listing: every image's class label, no pixel decoding. Cheap even
    for tens of thousands of images. Backs the class-balance metrics and the
    cross-validation fold visualization (both only need counts/order).
  - "sample" listing: up to PER_CLASS_SAMPLE images per class, pixels decoded.
    Backs the RGB histogram, sample grid, and PCA/t-SNE.
"""
import base64
import io
import pickle
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

# One consistent typeface and a publication-grade DPI across every plot, so
# figures pulled into a paper/report don't mismatch each other.
PLOT_FONT_FAMILY = "DejaVu Sans"
PLOT_DPI = 300
matplotlib.rcParams["font.family"] = PLOT_FONT_FAMILY
matplotlib.rcParams["savefig.dpi"] = PLOT_DPI

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PER_CLASS_SAMPLE = 40
MAX_TOTAL_SAMPLE = 600
THUMB = 24
MAX_SIZE_SCAN = 1000  # cap on real width/height/file-size scanning for class-folder datasets


# ---------------------------------------------------------------------------
# Full (label-only) listings — for accurate metrics and cross-validation.
# ---------------------------------------------------------------------------

def _full_cifar10(path: Path):
    root = path / "cifar-10-batches-py"
    meta_path = root / "batches.meta"
    if not meta_path.exists():
        return None
    with meta_path.open("rb") as fh:
        meta = pickle.load(fh, encoding="latin1")
    class_names = meta.get("label_names", [])
    train_labels, test_labels = [], []
    for batch_path in sorted(root.glob("data_batch_*")):
        with batch_path.open("rb") as fh:
            train_labels.extend(pickle.load(fh, encoding="latin1")["labels"])
    test_path = root / "test_batch"
    if test_path.exists():
        with test_path.open("rb") as fh:
            test_labels.extend(pickle.load(fh, encoding="latin1")["labels"])
    return {
        "class_names": class_names,
        "ordered_labels": train_labels,  # crossval/order matches EDA4ImageClassification (whole-CSV order)
        "train_count": len(train_labels),
        "test_count": len(test_labels),
        "validation_count": 0,
        "other_count": 0,
        "fixed_size": (32, 32),
    }


def _full_class_folders(path: Path):
    classes = sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))
    class_names = [c.name for c in classes]
    ordered_labels = []
    train_count = test_count = validation_count = other_count = 0
    for idx, class_dir in enumerate(classes):
        for entry in sorted(class_dir.iterdir()):
            if entry.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            ordered_labels.append(idx)
            lower = str(entry).lower()
            if "/train/" in lower:
                train_count += 1
            elif "/test/" in lower:
                test_count += 1
            elif "/validation/" in lower or "/val/" in lower:
                validation_count += 1
            else:
                other_count += 1
    if not ordered_labels:
        return None
    return {
        "class_names": class_names,
        "ordered_labels": ordered_labels,
        "train_count": train_count,
        "test_count": test_count,
        "validation_count": validation_count,
        "other_count": other_count,
        "fixed_size": None,
    }


# ---------------------------------------------------------------------------
# Pixel samples — for RGB histogram, sample grid, PCA/t-SNE.
# ---------------------------------------------------------------------------

def _sample_cifar10(path: Path, class_names):
    root = path / "cifar-10-batches-py"
    per_class = {}
    for batch_path in sorted(root.glob("data_batch_*")):
        with batch_path.open("rb") as fh:
            batch = pickle.load(fh, encoding="latin1")
        for row, label in zip(batch["data"], batch["labels"]):
            bucket = per_class.setdefault(label, [])
            if len(bucket) >= PER_CLASS_SAMPLE:
                continue
            bucket.append(row.reshape(3, 32, 32).transpose(1, 2, 0))
        if per_class and all(len(v) >= PER_CLASS_SAMPLE for v in per_class.values()) and len(per_class) >= len(class_names):
            break
    images, labels = [], []
    for label, imgs in sorted(per_class.items()):
        images.extend(imgs)
        labels.extend([label] * len(imgs))
    return images, labels, [None] * len(images)


def _sample_class_folders(path: Path, class_names):
    classes = sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))
    images, labels, paths = [], [], []
    for idx, class_dir in enumerate(classes):
        found = 0
        for entry in sorted(class_dir.iterdir()):
            if entry.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(entry) as im:
                    images.append(np.array(im.convert("RGB")))
                labels.append(idx)
                paths.append(entry)
                found += 1
            except (OSError, ValueError):
                continue
            if found >= PER_CLASS_SAMPLE:
                break
    return images, labels, paths


# ---------------------------------------------------------------------------
# Metrics (entropy / Gini / imbalance / CV), matching EDA4ImageClassification's
# calculate_metrics().
# ---------------------------------------------------------------------------

def _calculate_metrics(class_counts: Counter, class_names):
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    counts = np.array(sorted(class_counts.values()))
    probs = counts / total

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(num_classes) if num_classes > 1 else 1.0
    entropy_balance = int((entropy / max_entropy * 100) * 10000) / 10000 if max_entropy > 0 else 0.0

    ideal_count = total / num_classes
    sum_abs_dev = sum(abs(c - ideal_count) for c in counts)
    balancing_efficiency = int(((1 - (1 / num_classes) * (sum_abs_dev / ideal_count)) * 100) * 10000) / 10000

    max_count, min_count = int(counts.max()), int(counts.min())
    imbalance_ratio = round(max_count / min_count, 2) if min_count else float("inf")

    std_dev = round(float(counts.std()), 2)
    cv = round(std_dev / counts.mean() * 100, 2) if counts.mean() else 0.0

    n = len(counts)
    index = np.arange(1, n + 1)
    gini = ((2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n) * 100

    label_by_count = {v: k for k, v in class_counts.items()}

    def name(i):
        return class_names[i] if class_names and 0 <= i < len(class_names) else str(i)

    return {
        "num_classes": num_classes,
        "dataset_size": total,
        "avg_samples_per_class": round(total / num_classes, 2),
        "entropy_balance": entropy_balance,
        "balancing_efficiency": balancing_efficiency,
        "imbalance_ratio": imbalance_ratio,
        "std_dev": std_dev,
        "cv": cv,
        "gini": round(float(gini), 2),
        "max_samples": max_count,
        "min_samples": min_count,
        "max_class": name(label_by_count.get(max_count)),
        "min_class": name(label_by_count.get(min_count)),
    }


def _image_size_stats(images_or_paths, fixed_size):
    if fixed_size is not None:
        w, h = fixed_size
        return {
            "avg_width": w, "avg_height": h, "min_width": w, "min_height": h,
            "max_width": w, "max_height": h,
            "size_note": "Fixed-size pickled format; every image is identical resolution.",
            "approx_dataset_size_kb": None,
            "valid_images": None, "invalid_images": 0,
        }
    widths, heights, sizes_kb = [], [], []
    invalid = 0
    scanned = images_or_paths[:MAX_SIZE_SCAN]
    for p in scanned:
        try:
            with Image.open(p) as im:
                widths.append(im.width)
                heights.append(im.height)
            sizes_kb.append(p.stat().st_size / 1024)
        except (OSError, ValueError):
            invalid += 1
    truncated_note = f" (based on first {len(scanned)} of {len(images_or_paths)} images)" if len(images_or_paths) > MAX_SIZE_SCAN else ""
    if not widths:
        return {
            "avg_width": None, "avg_height": None, "min_width": None, "min_height": None,
            "max_width": None, "max_height": None,
            "size_note": "Could not read image dimensions.", "approx_dataset_size_kb": None,
            "valid_images": 0, "invalid_images": invalid,
        }
    return {
        "avg_width": round(sum(widths) / len(widths)), "avg_height": round(sum(heights) / len(heights)),
        "min_width": min(widths), "min_height": min(heights),
        "max_width": max(widths), "max_height": max(heights),
        "size_note": f"Measured directly from files{truncated_note}.",
        "approx_dataset_size_kb": round(sum(sizes_kb), 2),
        "valid_images": len(widths), "invalid_images": invalid,
    }


# ---------------------------------------------------------------------------
# Plots (matplotlib, base64 PNG) — mirror EDA4ImageClassification's generate_plots().
# ---------------------------------------------------------------------------

def _fig_to_formats():
    """PNG (raster, for on-screen display) + PDF (vector-backed text/lines,
    print/paper quality) of the current figure, both at PLOT_DPI."""
    png_buf = io.BytesIO()
    plt.savefig(png_buf, format="png", dpi=PLOT_DPI, bbox_inches="tight")
    png_buf.seek(0)
    png_b64 = base64.b64encode(png_buf.getvalue()).decode()
    pdf_buf = io.BytesIO()
    plt.savefig(pdf_buf, format="pdf", dpi=PLOT_DPI, bbox_inches="tight")
    pdf_buf.seek(0)
    pdf_b64 = base64.b64encode(pdf_buf.getvalue()).decode()
    plt.close()
    return {"png": png_b64, "pdf": pdf_b64}


def _plot_distribution(class_counts, class_names, entropy_balance):
    labels = [class_names[i] if class_names else str(i) for i in sorted(class_counts)]
    values = [class_counts[i] for i in sorted(class_counts)]
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette("husl", n_colors=len(values))
    ax = plt.gca()
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, padding=3, fontsize=9)
    plt.title("Class Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    ideal = sum(values) / len(values)
    plt.axhline(y=ideal, color="blue", linestyle=":", linewidth=2, label=f"Entropy Balance: {entropy_balance:.2f}%")
    plt.legend(loc="upper right")
    plt.tight_layout()
    return _fig_to_formats()


def _plot_insights(class_counts, class_names, entropy_balance, balancing_efficiency):
    labels = [class_names[i] if class_names else str(i) for i in sorted(class_counts)]
    values = [class_counts[i] for i in sorted(class_counts)]
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", n_colors=len(values))
    ax1 = plt.gca()
    bars = ax1.bar(range(len(values)), values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + height * 0.01, f"{int(height)}",
                  ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(range(len(values)))
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_xlabel("Classes")
    ax1.set_ylabel("Number of Samples")
    plt.title("Class Distribution with Balance Efficiency", fontsize=13, fontweight="bold")
    ax2 = ax1.twinx()
    ax2.set_ylabel("BE & EB (%)", color="purple")
    ax2.set_ylim(0, 100)
    ax2.axhline(y=balancing_efficiency, color="red", linestyle="--", linewidth=2,
                label=f"Balance Efficiency: {balancing_efficiency:.2f}%")
    ax2.axhline(y=entropy_balance, color="blue", linestyle=":", linewidth=2,
                label=f"Entropy Balance: {entropy_balance:.2f}%")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _fig_to_formats()


def _plot_rgb_histogram(images):
    rgb_means = np.array([img.reshape(-1, 3).mean(axis=0) for img in images]) if images else np.zeros((0, 3))
    plt.figure(figsize=(10, 5))
    if len(rgb_means):
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        plt.hist(rgb_means[:, 0], bins=40, alpha=0.6, color=colors[0], label="Red")
        plt.hist(rgb_means[:, 1], bins=40, alpha=0.6, color=colors[1], label="Green")
        plt.hist(rgb_means[:, 2], bins=40, alpha=0.6, color=colors[2], label="Blue")
        plt.legend(frameon=True)
        plt.grid(True, alpha=0.3)
    plt.title("RGB Channel Distribution (sample mean per image)", fontsize=13, fontweight="bold")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    formats = _fig_to_formats()
    if not len(rgb_means):
        return formats, None
    insights = {
        "channel_stats": {
            name: {
                "mean": round(float(rgb_means[:, i].mean()), 2),
                "median": round(float(np.median(rgb_means[:, i])), 2),
                "std": round(float(rgb_means[:, i].std()), 2),
            }
            for i, name in enumerate(["red", "green", "blue"])
        },
        "color_dominance": ["red", "green", "blue"][int(np.argmax(rgb_means.mean(axis=0)))],
        "brightness_level": "High" if rgb_means.mean() > 170 else "Medium" if rgb_means.mean() > 85 else "Low",
        "contrast_level": "High" if rgb_means.std() > 50 else "Medium" if rgb_means.std() > 25 else "Low",
    }
    return formats, insights


def _plot_sample_grid(images, labels, class_names):
    seen = {}
    for img, label in zip(images, labels):
        if label not in seen:
            seen[label] = img
    ordered = sorted(seen.items())
    n = len(ordered)
    if n == 0:
        return None
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(3 * cols, 3 * rows))
    plt.suptitle("Sample Image per Class", fontsize=16, y=1.0 if rows == 1 else 0.98)
    for i, (label, img) in enumerate(ordered):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img.astype("uint8"))
        plt.title(class_names[label] if class_names else str(label), fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    return _fig_to_formats()


def _plot_crossval(ordered_labels, class_names, k, strategy):
    n = len(ordered_labels)
    k = max(2, min(k, n))
    folds = [Counter() for _ in range(k)]
    if strategy == "stratified":
        for i, label in enumerate(ordered_labels):
            folds[i % k][label] += 1
    else:
        fold_size = n // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else n
            for label in ordered_labels[start:end]:
                folds[i][label] += 1
    classes = sorted(set(ordered_labels))
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(k)
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    for ci, cls in enumerate(classes):
        values = [folds[f].get(cls, 0) for f in range(k)]
        label = class_names[cls] if class_names else str(cls)
        plt.bar(range(k), values, bottom=bottom, label=label, color=colors[ci])
        bottom += values
    plt.xlabel("Fold Number")
    plt.ylabel("Number of Samples")
    plt.title(f"Cross-Validation Split ({strategy.capitalize()}, k={k})", fontsize=13, fontweight="bold")
    plt.xticks(range(k), [f"Fold {i + 1}" for i in range(k)])
    plt.legend(title="Classes", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    return _fig_to_formats(), k


# ---------------------------------------------------------------------------
# PCA (2D) / t-SNE (3D) on a per-image mean-RGB feature vector, matching
# EDA4ImageClassification's /dimred endpoint exactly (same feature, same
# n_components).
# ---------------------------------------------------------------------------

def _mean_rgb_features(images):
    return np.array([img.reshape(-1, 3).mean(axis=0) for img in images], dtype=np.float64) if images else np.zeros((0, 3))


def _pca_2d(X):
    if X.shape[0] < 2:
        return None, None
    centered = X - X.mean(axis=0)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    total_var = float((s ** 2).sum())
    explained = float((s[:2] ** 2).sum() / total_var) if total_var > 0 else 0.0
    return coords, explained


def _tsne_3d(X):
    if X.shape[0] < 8:
        return None
    from sklearn.manifold import TSNE
    perplexity = max(5, min(30, X.shape[0] // 3))
    reducer = TSNE(n_components=3, perplexity=perplexity, init="pca", random_state=42)
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def dataset_insights(root: Path, name: str, cv_k: int = 5, cv_strategy: str = "kfold"):
    path = root / name
    if not path.is_dir() or not any(path.iterdir()):
        return {"id": name, "available": False, "reason": "Dataset is not on disk."}

    if (path / "cifar-10-batches-py").exists():
        full = _full_cifar10(path)
    elif any(child.is_dir() and not child.name.startswith(".") for child in path.iterdir()):
        full = _full_class_folders(path)
    else:
        full = None

    if full is None:
        return {
            "id": name, "available": False,
            "reason": "This dataset's on-disk format isn't supported for EDA yet "
                      "(only CIFAR-10 pickled batches and class-folder layouts are).",
        }

    class_names = full["class_names"]
    ordered_labels = full["ordered_labels"]
    class_counts = Counter(ordered_labels)
    metrics = _calculate_metrics(class_counts, class_names)
    metrics["train_count"] = full["train_count"]
    metrics["test_count"] = full["test_count"]
    metrics["validation_count"] = full["validation_count"]
    metrics["other_count"] = full["other_count"]

    if (path / "cifar-10-batches-py").exists():
        images, labels, paths_or_none = _sample_cifar10(path, class_names)
        size_stats = _image_size_stats(paths_or_none, full["fixed_size"])
    else:
        images, labels, paths = _sample_class_folders(path, class_names)
        size_stats = _image_size_stats(paths, None)
    metrics.update(size_stats)

    if len(images) > MAX_TOTAL_SAMPLE:
        images, labels = images[:MAX_TOTAL_SAMPLE], labels[:MAX_TOTAL_SAMPLE]

    def cname(i):
        return class_names[i] if class_names and 0 <= i < len(class_names) else str(i)

    plots = {}
    if images:
        plots["distribution_plot"] = _plot_distribution(class_counts, class_names, metrics["entropy_balance"])
        plots["insights_plot"] = _plot_insights(class_counts, class_names, metrics["entropy_balance"], metrics["balancing_efficiency"])
        plots["rgb_histogram"], rgb_insights = _plot_rgb_histogram(images)
        sample_grid = _plot_sample_grid(images, labels, class_names)
        if sample_grid:
            plots["sample_per_class"] = sample_grid
    else:
        rgb_insights = None

    crossval_plot, applied_k = _plot_crossval(ordered_labels, class_names, cv_k, cv_strategy)

    X = _mean_rgb_features(images)
    pca_coords, pca_explained = _pca_2d(X)
    tsne_coords = _tsne_3d(X)

    pca_points = None
    if pca_coords is not None:
        pca_points = [{"x": float(x), "y": float(y), "label": cname(l)} for (x, y), l in zip(pca_coords, labels)]
    tsne_points = None
    if tsne_coords is not None:
        tsne_points = [{"x": float(x), "y": float(y), "z": float(z), "label": cname(l)}
                        for (x, y, z), l in zip(tsne_coords, labels)]

    return {
        "id": name,
        "available": True,
        "sample_size": len(images),
        "metrics": metrics,
        "class_counts": [{"label": cname(i), "count": class_counts[i]} for i in sorted(class_counts)],
        "plots": plots,
        "rgb_insights": rgb_insights,
        "crossval": {"k": applied_k, "strategy": cv_strategy, "plot": crossval_plot},
        "pca": pca_points,
        "pca_explained_variance": pca_explained,
        "tsne3d": tsne_points,
    }
