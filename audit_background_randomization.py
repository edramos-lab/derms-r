#!/usr/bin/env python3
"""Does background randomization remove the acquisition shortcut, and what is left?

On the frozen leakage-safe folds, ResNet-50 (fixed epochs, no validation-based selection) is trained
three ways and evaluated three ways.

Training inputs
  raw        the image, caption strip blanked
  composite  every epoch, the image's cells over a freshly generated background (derms/background.py)
  mask-only  the binary cell mask alone: the floor of what layout and density can explain

Evaluation inputs (validation fold)
  raw        the image as is
  composite  cells over seeded generated backgrounds
  swap       cells of an image over the native, cell-inpainted glass of an image of a *different* class;
             reports how often the prediction follows the cells and how often it follows the glass

Generator checks on the composites: a classifier that sees only glass pixels, and one that sees only a
thin ring just outside each cell boundary, should sit at chance (the ring finds halo leakage). The same
ring on raw images is the reference.

    python audit_background_randomization.py --root datasets/all --split splits.json --out out
"""
import argparse
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

from audit_background_ablation import MEAN, STD, augment, find

spec = importlib.util.spec_from_file_location('background', Path(__file__).parent / 'derms' / 'background.py')
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)

DEVICE = 'cuda'
IMAGES, MASKS = None, None  # filled before the worker pool forks


def load_images(root, records):
    out = np.empty((len(records), bg.SIDE, bg.SIDE, 3), dtype=np.uint8)
    for i, r in enumerate(records):
        with Image.open(find(root, r['id'])) as image:
            out[i] = bg.fit(np.asarray(image.convert('RGB')))
    return out


def composite(args):
    index, seed = args
    return bg.randomize(IMAGES[index], np.random.default_rng(seed), MASKS[index])


def composite_constant(args):
    index, seed = args
    flat = np.full((bg.SIDE, bg.SIDE, 3), 150, dtype=np.uint8)
    return bg.randomize(flat, np.random.default_rng(seed), MASKS[index])


def to_tensor(arrays):
    x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255
    return ((x - MEAN) / STD).half()


def mask_tensor(indices):
    m = np.stack([(MASKS[i] > 0.5).astype(np.uint8) * 255 for i in indices])
    return to_tensor([np.repeat(a[..., None], 3, axis=2) for a in m])


def raw_tensor(indices):
    return to_tensor([bg.blank_strip(IMAGES[i].copy()) for i in indices])


def composite_tensor(pool, indices, seeds):
    return to_tensor(list(pool.map(composite, zip(indices, seeds), chunksize=32)))


def train(make_epoch_data, y, train_idx, classes, epochs, batch, seed):
    torch.manual_seed(seed)
    model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    steps = epochs * ((len(train_idx) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-4, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda')
    target = y[torch.as_tensor(train_idx)]
    for epoch in range(epochs):
        x = make_epoch_data(epoch)
        model.train()
        order = torch.randperm(len(train_idx))
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(augment(x[idx]).to(DEVICE)), target[idx].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, x):
    out = []
    with torch.autocast('cuda', dtype=torch.float16):
        for k in range(0, len(x), 128):
            out.append(model(x[k:k + 128].to(DEVICE)).argmax(1).cpu())
    return torch.cat(out).numpy()


def region_features(arr, region):
    pixels = arr[region].astype(np.float32)
    if len(pixels) < 100:
        return None
    return np.concatenate([np.percentile(pixels[:, c], (5, 25, 50, 75, 95)) for c in range(3)]
                           + [pixels.mean(0) / pixels.mean(), pixels.std(0)])


def canary(arrays, regions, y, folds):
    rows = [(i, region_features(a, r)) for i, (a, r) in enumerate(zip(arrays, regions))]
    keep = [i for i, f in rows if f is not None]
    X, yy = np.array([f for _, f in rows if f is not None]), y[keep]
    where = {orig: new for new, orig in enumerate(keep)}
    pred = np.empty_like(yy)
    for tr, va in folds:
        tr = [where[i] for i in tr if i in where]
        va = [where[i] for i in va if i in where]
        pred[va] = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[tr], yy[tr]).predict(X[va])
    return float(balanced_accuracy_score(yy, pred))


def contact_sheet(path, images, labels, names, rng):
    rows = []
    for i in rng.choice(len(images), 8, replace=False):
        tiles = [bg.blank_strip(images[i].copy()),
                 (np.repeat((MASKS[i][..., None] > 0.5) * 255, 3, 2)).astype(np.uint8)]
        tiles += [bg.randomize(images[i], rng, MASKS[i], protocol=p) for p in rng.choice(sorted(bg.PROTOCOLS), 5, False)]
        rows.append(np.concatenate(tiles, 1))
        print(f'  sheet row: {names[labels[i]]}', flush=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


def main():
    global IMAGES, MASKS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('audit_background_randomization'))
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--sheet-only', action='store_true')
    p.add_argument('--canary-only', action='store_true', help='generator checks only, no model training')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split.read_text())
    records, class_names = split['records'], split['classes']
    y = torch.tensor([r['label'] for r in records])
    IMAGES = load_images(args.root, records)
    MASKS = [bg.cell_mask(a) for a in IMAGES]
    print(f'{len(records)} images, mean cell fraction {np.mean([(m > 0.5).mean() for m in MASKS]):.2f}', flush=True)
    contact_sheet(args.out / 'background_randomization_sheet.png', IMAGES, y.numpy(), class_names, np.random.default_rng(0))
    if args.sheet_only:
        return

    folds = [(f['train'], f['val']) for f in split['folds'][:args.folds]]
    classes = len(class_names)
    report = {'epochs': args.epochs, 'cv': split['cv'], 'folds': len(folds), 'models': {}, 'generator_checks': {}}
    pool = ProcessPoolExecutor(args.workers, mp_context=get_context('fork'))

    class PoolMap:  # same .map interface the helpers use
        def map(self, fn, it, chunksize=1):
            return pool.map(fn, it, chunksize=chunksize)

    pmap = PoolMap()
    rng = np.random.default_rng(0)

    # --- generator checks on the composites (all images, one seeded composite each)
    seeds = [10_000 + i for i in range(len(records))]
    comps = list(pool.map(composite, zip(range(len(records)), seeds), chunksize=32))
    # control with no native cell pixels at all: a constant image composited through the same masks
    grey = list(pool.map(composite_constant, zip(range(len(records)), seeds), chunksize=32))
    labels = y.numpy()
    strip = int(bg.SIDE * (1 - bg.STRIP))

    def far_glass(m, distance=8):
        region = cv2.distanceTransform((m < 0.05).astype(np.uint8), cv2.DIST_L2, 3) > distance
        region[strip:] = False
        return region
    glass = [m < 0.05 for m in MASKS]
    far = [far_glass(m) for m in MASKS]
    rings = [bg.ring(m) for m in MASKS]
    raw = [bg.blank_strip(a.copy()) for a in IMAGES]
    for name, arrays, regions in (('composite_glass_only', comps, glass), ('composite_glass_far_from_cells', comps, far),
                                  ('composite_ring_only', comps, rings),
                                  ('grey_cells_glass_far_from_cells', grey, far), ('grey_cells_ring_only', grey, rings),
                                  ('raw_glass_only_reference', raw, glass), ('raw_ring_only_reference', raw, rings)):
        report['generator_checks'][name] = canary(arrays, regions, labels, folds)
        print(f'canary {name}: balanced accuracy {report["generator_checks"][name]:.3f} (chance {1 / classes:.2f})',
              flush=True)
    del comps, grey
    (args.out / 'background_randomization.json').write_text(json.dumps(report, indent=2))
    if args.canary_only:
        return

    # --- models
    recipes = {
        'raw': lambda tr, e, f: raw_tensor(tr),
        'composite': lambda tr, e, f: composite_tensor(pmap, tr, [f * 1_000_000 + e * 100_000 + int(i) for i in tr]),
        'mask_only': lambda tr, e, f: mask_tensor(tr),
    }
    for model_name, make in recipes.items():
        stats = {'raw_val': [], 'composite_val': [], 'swap_follow_cells': [], 'swap_follow_glass': []}
        for f, (tr, va) in enumerate(folds):
            t = time.time()
            tr_a, va_a = np.array(tr), np.array(va)
            model = train(lambda e: make(tr_a, e, f), y, tr, classes, args.epochs, args.batch, f)
            truth = labels[va_a]
            if model_name == 'mask_only':
                stats['raw_val'].append(float(balanced_accuracy_score(truth, predict(model, mask_tensor(va_a)))))
                print(f'{model_name} fold {f}: mask-only balanced accuracy {stats["raw_val"][-1]:.3f} ({time.time() - t:.0f}s)',
                      flush=True)
                del model
                continue
            stats['raw_val'].append(float(balanced_accuracy_score(truth, predict(model, raw_tensor(va_a)))))
            comp_val = composite_tensor(pmap, va_a, [20_000_000 + int(i) for i in va_a])
            stats['composite_val'].append(float(balanced_accuracy_score(truth, predict(model, comp_val))))
            # swap: cells of a val image on the glass of a val image of another class
            hosts = np.array([rng.choice(va_a[labels[va_a] != labels[a]]) for a in va_a])
            swapped = to_tensor([bg.native_swap(IMAGES[a], MASKS[a], IMAGES[h], MASKS[h]) for a, h in zip(va_a, hosts)])
            pred = predict(model, swapped)
            stats['swap_follow_cells'].append(float((pred == truth).mean()))
            stats['swap_follow_glass'].append(float((pred == labels[hosts]).mean()))
            print(f'{model_name} fold {f}: raw {stats["raw_val"][-1]:.3f} composite {stats["composite_val"][-1]:.3f} '
                  f'swap follows cells {stats["swap_follow_cells"][-1]:.3f} / glass {stats["swap_follow_glass"][-1]:.3f} '
                  f'({time.time() - t:.0f}s)', flush=True)
            del model
        report['models'][model_name] = {k: {'folds': v, 'mean': float(np.mean(v)), 'std': float(np.std(v))}
                                        for k, v in stats.items() if v}
        print(f'{model_name}: ' + ', '.join(f'{k} {np.mean(v):.3f}' for k, v in stats.items() if v), flush=True)
        (args.out / 'background_randomization.json').write_text(json.dumps(report, indent=2))
    (args.out / 'background_randomization.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
