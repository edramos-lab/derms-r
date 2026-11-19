#!/usr/bin/env python3
"""External check against C-NMC 2019, a public ALL dataset that records the patient of every image.

Two questions, both binary (normal vs ALL) because C-NMC has no subtype labels:

1. transfer    A network trained on `all` (every development image, Benign vs any malignant subtype)
               is applied unchanged to every C-NMC training image. This is a domain shift (field
               images versus single-cell crops, different hospital, camera and stain), so a low score
               is a statement about transfer, not about the network.
2. grouping    Inside C-NMC, the same network and recipe is cross-validated twice: image-level
               (StratifiedKFold) and patient-level (StratifiedGroupKFold on subject_id). The gap
               between the two is a direct measurement of how much image-level evaluation overstates
               patient-level generalization, with real patient identifiers.

Fixed epoch budget, no best-epoch selection.

    python audit_external_cnmc.py --all-root datasets/all --all-split splits.json --cnmc external/cnmc --out out
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from audit_background_ablation import MEAN, SIDE, STD, augment, find

DEVICE = 'cuda'


def to_tensor(arrays):
    x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255
    return ((x - MEAN) / STD).half()


def load_all(root, records):
    arrays = []
    for r in records:
        with Image.open(find(root, r['id'])) as image:
            arrays.append(np.asarray(image.convert('RGB').resize((SIDE, SIDE), Image.BILINEAR)))
    return to_tensor(arrays)


def load_cnmc(folder):
    frames = [pd.read_parquet(p, columns=['subject_id', 'label', 'image']) for p in sorted(folder.glob('*.parquet'))]
    frame = pd.concat(frames, ignore_index=True)
    arrays, keep = [], []
    for i, cell in enumerate(frame['image']):
        try:
            with Image.open(io.BytesIO(cell['bytes'])) as image:
                arrays.append(np.asarray(image.convert('RGB').resize((SIDE, SIDE), Image.BILINEAR)))
            keep.append(i)
        except Exception:  # the Hugging Face mirror stores some files zero-filled
            pass
    skipped = frame.drop(index=keep)
    print(f'skipped {len(skipped)} undecodable images from subjects {sorted(set(skipped.subject_id))}', flush=True)
    frame = frame.iloc[keep].reset_index(drop=True)
    y = (frame['label'].str.lower().str.startswith('cancer')).astype(int).to_numpy()
    return to_tensor(arrays), torch.from_numpy(y), frame['subject_id'].astype(str).to_numpy()


def fit(x, y, train, epochs, batch, seed):
    torch.manual_seed(seed)
    model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    steps = epochs * ((len(train) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-4, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda')
    train = torch.as_tensor(train)
    for _ in range(epochs):
        model.train()
        order = train[torch.randperm(len(train))]
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(augment(x[idx]).to(DEVICE)), y[idx].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return model.eval()


@torch.no_grad()
def malignant_probability(model, x, idx):
    idx = torch.as_tensor(idx)
    out = []
    with torch.autocast('cuda', dtype=torch.float16):
        for k in range(0, len(idx), 128):
            out.append(model(x[idx[k:k + 128]].to(DEVICE)).float().softmax(1)[:, 1].cpu())
    return torch.cat(out).numpy()


def metrics(truth, prob):
    return {'balanced_accuracy': float(balanced_accuracy_score(truth, prob > 0.5)),
            'auc': float(roc_auc_score(truth, prob)), 'predicted_cancer_rate': float((prob > 0.5).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--all-root', type=Path, required=True)
    p.add_argument('--all-split', type=Path, required=True)
    p.add_argument('--cnmc', type=Path, required=True, help='folder with the C-NMC parquet files')
    p.add_argument('--out', type=Path, default=Path('audit_external_cnmc'))
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--cnmc-epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=32)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}

    split = json.loads(args.all_split.read_text())
    benign = {i for i, c in enumerate(split['classes']) if c.lower().startswith('benign')}
    xa = load_all(args.all_root, split['records'])
    ya = torch.tensor([0 if r['label'] in benign else 1 for r in split['records']])
    xc, yc, subjects = load_cnmc(args.cnmc)
    print(f'all: {len(ya)} images ({int(ya.sum())} malignant); C-NMC: {len(yc)} images ({int(yc.sum())} ALL), '
          f'{len(set(subjects))} subjects', flush=True)
    report['data'] = {'undecodable_skipped': 10661 - len(yc), 'all_images': len(ya), 'cnmc_images': len(yc), 'cnmc_subjects': len(set(subjects))}

    model = fit(xa, ya, np.arange(len(ya)), args.epochs, args.batch, 0)
    prob = malignant_probability(model, xc, np.arange(len(yc)))
    truth = yc.numpy()
    report['transfer_all_to_cnmc'] = metrics(truth, prob)
    # per-subject: mean predicted probability, then subject-level accuracy
    per = pd.DataFrame({'s': subjects, 't': truth, 'p': prob}).groupby('s').agg(t=('t', 'first'), p=('p', 'mean'))
    report['transfer_all_to_cnmc']['subject_level_auc'] = float(roc_auc_score(per['t'], per['p']))
    print('transfer all -> C-NMC:', report['transfer_all_to_cnmc'], flush=True)
    del model

    for name, splitter, groups in (('image_level', StratifiedKFold(5, shuffle=True, random_state=0), None),
                                   ('patient_level', StratifiedGroupKFold(5, shuffle=True, random_state=0), subjects)):
        bals, aucs = [], []
        for f, (tr, va) in enumerate(splitter.split(np.zeros(len(truth)), truth, groups)):
            if groups is not None:
                assert not set(subjects[tr]) & set(subjects[va])
            t = time.time()
            m = fit(xc, yc, tr, args.cnmc_epochs, args.batch, f)
            r = metrics(truth[va], malignant_probability(m, xc, va))
            bals.append(r['balanced_accuracy'])
            aucs.append(r['auc'])
            print(f'{name} fold {f}: balanced {r["balanced_accuracy"]:.3f} auc {r["auc"]:.3f} ({time.time() - t:.0f}s)',
                  flush=True)
            del m
        report[f'cnmc_cv_{name}'] = {'balanced_accuracy': bals, 'auc': aucs,
                                     'mean_balanced_accuracy': float(np.mean(bals)),
                                     'std_balanced_accuracy': float(np.std(bals)), 'mean_auc': float(np.mean(aucs))}
        print(f'{name} mean balanced accuracy {np.mean(bals):.3f} +/- {np.std(bals):.3f}, auc {np.mean(aucs):.3f}',
              flush=True)
        (args.out / 'external_cnmc.json').write_text(json.dumps(report, indent=2))
    (args.out / 'external_cnmc.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
