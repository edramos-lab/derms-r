#!/usr/bin/env python3
"""How much of a fine-tuned model's score comes from the background?

Trains the same ImageNet-pretrained network on the frozen leakage-safe folds under three inputs:
  full        the image (bottom strip with the scale bar blanked in every condition)
  cells       background replaced by neutral gray; only saturated (stained) cells remain
  background  stained cells replaced by neutral gray; only the glass remains
Fixed epoch budget, no best-epoch selection (nothing is chosen on the validation fold), flips and
90-degree rotations only. A small drop from `full` to `cells` means morphology carries the score;
`background` well above chance means the shortcut is there; `full` ~ `background` would mean the
network could be using it.

    python audit_background_ablation.py --root datasets/all --split results/all-run/all/splits.json --out ablation
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score

SIDE = 224
BOTTOM = 0.15
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find(root, record_id):
    for split in ('development', 'test'):
        p = root / split / record_id
        if p.exists():
            return p
    raise FileNotFoundError(record_id)


def load(root, records):
    out = np.empty((len(records), SIDE, SIDE, 3), dtype=np.uint8)
    for i, r in enumerate(records):
        with Image.open(find(root, r['id'])) as image:
            out[i] = np.asarray(image.convert('RGB').resize((SIDE, SIDE), Image.BILINEAR))
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


def otsu(values, bins=256):
    hist, edges = np.histogram(values, bins=bins, range=(0, 1))
    p = hist / hist.sum()
    omega, mu = np.cumsum(p), np.cumsum(p * edges[:-1])
    between = (mu[-1] * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    return float(edges[int(np.argmax(between))])


def foreground_masks(images):
    """Stained-cell mask per image: saturation above that image's Otsu threshold, slightly dilated."""
    x = images.float() / 255
    peak, low = x.max(1).values, x.min(1).values
    sat = (peak - low) / peak.clamp(min=1e-6)
    masks = torch.stack([(sat[i] > otsu(sat[i].flatten().numpy())).float() for i in range(len(x))])[:, None]
    return F.max_pool2d(masks, 5, 1, 2)


def condition(images, masks, mode):
    x = (images.float() / 255 - MEAN) / STD
    if mode == 'cells':
        x = x * masks
    elif mode == 'background':
        x = x * (1 - masks)
    x[:, :, int(SIDE * (1 - BOTTOM)):] = 0  # scale bar / caption strip, in every condition
    return x


def augment(x):
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))


def run_fold(model_name, x, y, tr, va, classes, epochs, batch, seed, device):
    torch.manual_seed(seed)
    model = timm.create_model(model_name, pretrained=True, num_classes=classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    steps = epochs * ((len(tr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-4, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(tr))
        for k in range(0, len(tr), batch):
            idx = torch.as_tensor(tr)[order[k:k + batch]]
            xb, yb = augment(x[idx]).to(device), y[idx].to(device)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    model.eval()
    pred = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for k in range(0, len(va), 64):
            pred.append(model(x[torch.as_tensor(va)[k:k + 64]].to(device)).argmax(1).cpu())
    pred = torch.cat(pred).numpy()
    truth = y[torch.as_tensor(va)].numpy()
    return float((pred == truth).mean()), float(balanced_accuracy_score(truth, pred))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('audit_background_ablation'))
    p.add_argument('--model', default='resnet50.a1_in1k')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    split = json.loads(args.split.read_text())
    records, classes = split['records'], len(split['classes'])
    device = 'cuda'
    images = load(args.root, records)
    y = torch.tensor([r['label'] for r in records])
    masks = foreground_masks(images)
    print(f'{len(records)} images, stained-cell fraction {masks.mean():.2f}, folds {split["cv"]}, '
          f'near-dup {split.get("near_dup_threshold", 0)}', flush=True)

    report = {'model': args.model, 'epochs': args.epochs, 'cv': split['cv'], 'cell_fraction': float(masks.mean()),
              'conditions': {}}
    for mode in ('full', 'cells', 'background'):
        x = condition(images, masks, mode)
        accs, bals = [], []
        for f, fold in enumerate(split['folds'][:args.folds]):
            t = time.time()
            acc, bal = run_fold(args.model, x, y, fold['train'], fold['val'], classes, args.epochs, args.batch,
                                args.seed + f, device)
            accs.append(acc)
            bals.append(bal)
            print(f'{mode:>10} fold {f}: accuracy {acc:.3f} balanced {bal:.3f} ({time.time() - t:.0f}s)', flush=True)
        report['conditions'][mode] = {'accuracy': accs, 'balanced_accuracy': bals,
                                      'mean_balanced_accuracy': float(np.mean(bals)),
                                      'std_balanced_accuracy': float(np.std(bals))}
        print(f'{mode:>10} mean balanced accuracy {np.mean(bals):.3f} +/- {np.std(bals):.3f}', flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'background_ablation.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
