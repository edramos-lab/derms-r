#!/usr/bin/env python3
"""Background canary: can the class be predicted from cell-free glass pixels alone?

Cells carry the morphology the benchmark is meant to measure. The empty background carries only
acquisition information (stain tint, illumination, camera, slide). If a classifier that sees only
background pixels predicts the class well above chance on leakage-safe folds, the images carry a
slide/patient/acquisition fingerprint that is perfectly confounded with the (patient-level) label,
and image-level scores cannot be read as evidence about morphology alone.

Background = bright, low-saturation pixels; the bottom strip (scale bar) is excluded; per-image
features are percentiles of each channel over those pixels. Folds are the frozen splits.json folds,
so near-duplicates and filename groups never straddle train/validation.

    python audit_background_canary.py --root datasets/all --split results/all-run/all/splits.json --out canary
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PERCENTILES = (5, 25, 50, 75, 95)
BOTTOM_EXCLUDED = 0.15  # scale bar / caption strip
MIN_BACKGROUND = 100    # pixels needed for stable percentiles (at 256x192 working size)


def find(root, record_id):
    for split in ('development', 'test'):
        p = root / split / record_id
        if p.exists():
            return p
    raise FileNotFoundError(record_id)


def background_features(path, bright=70, pale=40):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB').resize((256, 192), Image.BILINEAR), dtype=np.float32)
    rgb = rgb[: int(rgb.shape[0] * (1 - BOTTOM_EXCLUDED))]
    peak, low = rgb.max(-1), rgb.min(-1)
    saturation = (peak - low) / np.maximum(peak, 1)
    mask = (low > np.percentile(low, bright)) & (saturation < np.percentile(saturation, pale))
    if mask.sum() < MIN_BACKGROUND:
        return None
    pixels = rgb[mask]
    return np.concatenate([np.percentile(pixels[:, c], PERCENTILES) for c in range(3)]
                          + [pixels.mean(0) / pixels.mean(), pixels.std(0)]), float(mask.mean())


def score(model, X, y, folds, rng, shuffle=False):
    pred = np.empty_like(y)
    for tr, va in folds:
        target = rng.permutation(y[tr]) if shuffle else y[tr]
        pred[va] = model().fit(X[tr], target).predict(X[va])
    return balanced_accuracy_score(y, pred)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True, help='frozen splits.json (StratifiedGroupKFold recommended)')
    p.add_argument('--out', type=Path, default=Path('audit_background_canary'))
    p.add_argument('--strict', action='store_true', help='tighter mask: brightest 10%% and palest 20%% only')
    args = p.parse_args()
    mask_args = (90, 20) if args.strict else (70, 40)

    split = json.loads(args.split.read_text())
    records = split['records']
    classes = split['classes']
    feats, keep = [], []
    for i, r in enumerate(records):
        f = background_features(find(args.root, r['id']), *mask_args)
        if f is not None:
            feats.append(f[0])
            keep.append(i)
    X = np.array(feats)
    y4 = np.array([records[i]['label'] for i in keep])
    position = {orig: new for new, orig in enumerate(keep)}
    folds = [([position[i] for i in f['train'] if i in position], [position[i] for i in f['val'] if i in position])
             for f in split['folds']]
    print(f'{len(keep)}/{len(records)} images usable, {len(classes)} classes, {len(folds)} folds, '
          f'cv={split["cv"]}, near-dup threshold {split.get("near_dup_threshold", 0)}')

    models = {
        'logistic regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        'gradient boosting': lambda: HistGradientBoostingClassifier(max_iter=150, random_state=0),
    }
    rng = np.random.default_rng(0)
    report = {'images': len(keep), 'classes': classes, 'cv': split['cv'], 'tasks': {}}
    # a binary task: Benign vs any malignant subtype
    benign = [i for i, c in enumerate(classes) if c.lower().startswith('benign')]
    tasks = {'multi-class': (y4, 1 / len(classes)),
             'benign vs malignant': (np.isin(y4, benign).astype(int), 0.5)}
    for task, (y, chance) in tasks.items():
        report['tasks'][task] = {'chance_balanced_accuracy': chance}
        for name, model in models.items():
            real = score(model, X, y, folds, rng)
            null = float(np.mean([score(model, X, y, folds, rng, shuffle=True) for _ in range(5)]))
            report['tasks'][task][name] = {'background_only': float(real), 'shuffled_labels': null}
            print(f'{task:>20} | {name:<20} | background-only balanced acc {real:.3f} | '
                  f'shuffled-label control {null:.3f} | chance {chance:.2f}')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'background_canary_strict.json' if args.strict else 'background_canary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
