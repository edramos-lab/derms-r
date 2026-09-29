"""Confusion matrices and ROC curves. Conventional metrics; not part of DERMS-R."""
from __future__ import annotations
import json
import math
import os
import tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'derms-mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _class_names(names, nclasses):
    if names and len(names) == nclasses:
        return [str(name) for name in names]
    return [str(i) for i in range(nclasses)]


def classification_scores(y, pred, probs, nclasses):
    """ROC-AUC (macro OVR) and the integer confusion matrix. Missing-class folds yield null AUC."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    probs = np.asarray(probs, dtype=np.float64)
    labels = list(range(nclasses))
    matrix = confusion_matrix(y, pred, labels=labels)
    per_class = {}
    for i in labels:
        true = (y == i).astype(int)
        if true.min() == true.max():
            per_class[str(i)] = None
            continue
        fpr, tpr, _ = roc_curve(true, probs[:, i])
        per_class[str(i)] = float(auc(fpr, tpr))
    try:
        if len(np.unique(y)) < 2:
            macro = None
        elif nclasses == 2:
            macro = float(roc_auc_score(y, probs[:, 1]))
        else:
            macro = float(roc_auc_score(y, probs, multi_class='ovr', average='macro', labels=labels))
        if macro is not None and not math.isfinite(macro):
            macro = None
    except ValueError:
        macro = None
    return {
        'roc_auc_macro_ovr': macro,
        'roc_auc_per_class': per_class,
        'confusion_matrix': matrix.astype(int).tolist(),
    }


def confusion_matrix_figure(matrix, class_names, title='Confusion Matrix'):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(image, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    thresh = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > thresh else 'black')
    fig.tight_layout()
    return fig


def roc_figure(y, probs, class_names, title='ROC Curves'):
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, name in enumerate(class_names):
        true = (y == i).astype(int)
        if true.min() == true.max():
            continue
        fpr, tpr, _ = roc_curve(true, probs[:, i])
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def save_classification_reports(directory, y, pred, probs, class_names, stem='val'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = _class_names(class_names, np.asarray(probs).shape[1])
    scores = classification_scores(y, pred, probs, len(names))
    payload = {'classes': names, **scores}
    (directory / f'{stem}_classification_report.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n')
    fig_cm = confusion_matrix_figure(scores['confusion_matrix'], names,
                                     title=f'Confusion Matrix — {stem}')
    fig_cm.savefig(directory / f'{stem}_confusion_matrix.png', dpi=150)
    plt.close(fig_cm)
    fig_roc = roc_figure(y, probs, names, title=f'ROC Curves — {stem}')
    fig_roc.savefig(directory / f'{stem}_roc_curve.png', dpi=150)
    plt.close(fig_roc)
    return payload
