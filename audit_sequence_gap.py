#!/usr/bin/env python3
"""How does image similarity change with the distance in capture number?

For ImageFolder datasets whose filenames carry a capture number (Snap_086.jpg), images of one
patient/slide are expected to be similar and to sit at nearby numbers. This measures, within each
class, the colour-histogram and perceptual-hash distance between images whose capture numbers differ
by g, relative to a baseline of images from *different classes* (different patients by construction).
A curve that rises and then plateaus at the baseline indicates the typical run length of one
patient/slide; consecutive-number jumps estimate how many separate runs the numbering contains.

Read-only. Uses every split (development + test) because the question is about the data, not the split.

    python audit_sequence_gap.py --root datasets/all --out audit_sequence_gap
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import imagehash
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
BINS = 16
GAPS = (1, 2, 3, 5, 8, 12, 18, 25, 36, 50, 75, 100)


def features(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        small = np.asarray(rgb.resize((64, 64), Image.BILINEAR))
        hist = np.concatenate([np.histogram(small[..., c], bins=BINS, range=(0, 256))[0] for c in range(3)])
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(str(imagehash.phash(rgb, hash_size=16))), dtype=np.uint8))
    return hist / hist.sum() * 3, bits


def hist_dist(a, b):
    return 0.5 * np.abs(a - b).sum() / 3  # 0..1, mean over channels of total-variation distance


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('audit_sequence_gap'))
    p.add_argument('--max-gap', type=int, default=120)
    p.add_argument('--baseline-pairs', type=int, default=20000)
    args = p.parse_args()

    items = []  # (class, number, hist, bits)
    for split in sorted(d for d in args.root.iterdir() if d.is_dir()):
        for cls in sorted(d for d in split.iterdir() if d.is_dir()):
            for f in sorted(cls.iterdir()):
                if f.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                m = re.search(r'(\d+)', f.stem)
                if not m:
                    continue
                h, b = features(f)
                items.append((cls.name, int(m.group(1)), h, b))
    classes = sorted({i[0] for i in items})
    H = np.array([i[2] for i in items])
    B = np.array([i[3] for i in items], dtype=np.float32)
    cls_of = np.array([i[0] for i in items])
    num_of = np.array([i[1] for i in items])
    print(f'{len(items)} images, {len(classes)} classes')

    rng = np.random.default_rng(0)

    def hist_between(i, j):
        return 0.5 * np.abs(H[i] - H[j]).sum(-1) / 3

    def bits_between(i, j):
        return (B[i] != B[j]).sum(-1)

    # baseline: images of different classes are different patients
    i = rng.integers(0, len(items), args.baseline_pairs)
    j = rng.integers(0, len(items), args.baseline_pairs)
    keep = cls_of[i] != cls_of[j]
    base_h, base_b = np.median(hist_between(i[keep], j[keep])), np.median(bits_between(i[keep], j[keep]))
    same = cls_of[i] == cls_of[j]
    within_h = np.median(hist_between(i[same], j[same]))
    print(f'baseline (different classes): median hist distance {base_h:.4f}, median phash bits {base_b:.0f}')
    print(f'baseline (random pair, same class): median hist distance {within_h:.4f} '
          f'({within_h / base_h:.2f} of the different-class baseline)')

    report = {'images': len(items), 'baseline': {'hist': float(base_h), 'phash_bits': float(base_b),
                                               'hist_same_class_random': float(within_h)}, 'gaps': {}, 'classes': {}}
    by_class_num = {c: defaultdict(list) for c in classes}
    for idx, (c, n, _, _) in enumerate(items):
        by_class_num[c][n].append(idx)

    curve = {}
    for g in range(1, args.max_gap + 1):
        hs, bs = [], []
        for c in classes:
            nums = by_class_num[c]
            for n, idxs in nums.items():
                other = nums.get(n + g)
                if other:
                    a = np.repeat(idxs, len(other))
                    b = np.tile(other, len(idxs))
                    hs.append(hist_between(a, b))
                    bs.append(bits_between(a, b))
        if hs:
            hs, bs = np.concatenate(hs), np.concatenate(bs)
            curve[g] = (float(np.median(hs)) / base_h, float(np.median(bs)) / base_b, int(len(hs)))
    print('\n gap | hist / diff-class | hist / same-class random | phash / baseline | pairs')
    for g in GAPS:
        if g in curve:
            print(f'{g:>4} | {curve[g][0]:>17.2f} | {curve[g][0] * base_h / within_h:>24.2f} | {curve[g][1]:>16.2f} | {curve[g][2]:>5}')
    report['gaps'] = {str(g): {'hist_ratio': v[0], 'hist_ratio_same_class': v[0] * base_h / within_h,
                               'phash_ratio': v[1], 'pairs': v[2]} for g, v in curve.items()}

    # first gap at which the smoothed histogram ratio reaches 90% / 100% of the different-class baseline
    gs = sorted(curve)
    smooth = {g: float(np.mean([curve[k][0] for k in gs if abs(k - g) <= 2])) for g in gs}
    for level in (0.5, 0.8, 0.9, 1.0):
        reach = next((g for g in gs if smooth[g] >= level), None)
        report[f'gap_reaching_{int(level * 100)}pct'] = reach
        print(f'smoothed histogram distance first reaches {level:.0%} of baseline at gap {reach}')

    # run boundaries: jumps between consecutive capture numbers of one class
    print('\nrun-boundary estimate (consecutive capture numbers, gap <= 2):')
    jumps = {}
    for c in classes:
        nums = sorted(by_class_num[c])
        d = []
        for a, b in zip(nums, nums[1:]):
            if b - a <= 2:
                d.append(float(np.median(hist_between(np.repeat(by_class_num[c][a], len(by_class_num[c][b])),
                                                      np.tile(by_class_num[c][b], len(by_class_num[c][a]))))))
        jumps[c] = (np.array(d), len(nums))
    all_d = np.concatenate([v[0] for v in jumps.values()])
    median, mad = np.median(all_d), np.median(np.abs(all_d - np.median(all_d)))
    for k in (3, 5, 8):
        threshold = median + k * 1.4826 * mad
        per = {c: int((v[0] > threshold).sum()) for c, v in jumps.items()}
        segments = sum(per.values()) + len(classes)
        report['classes'][f'threshold_median_plus_{k}_mad'] = {'threshold': float(threshold), 'jumps': per, 'segments': segments}
        print(f'  threshold median+{k}*MAD = {threshold:.4f}: boundaries {sum(per.values())} -> about {segments} runs (89 patients expected)')
    print(f'  consecutive-number distance: median {median:.4f}, baseline {base_h:.4f}')

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'sequence_gap.json').write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gs, [curve[g][0] for g in gs], label='colour histogram', color='#1f77b4')
    ax.plot(gs, [curve[g][1] for g in gs], label='perceptual hash', color='#d62728')
    ax.axhline(1.0, color='grey', ls='--', lw=1, label='different classes (= different patients)')
    ax.axhline(within_h / base_h, color='#2ca02c', ls=':', lw=1.2, label='random pair, same class')
    ax.set_xlabel('difference in capture number')
    ax.set_ylabel('median distance / different-class baseline')
    ax.set_title('Image similarity versus capture-number gap')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / 'sequence_gap.png', dpi=160)


if __name__ == '__main__':
    main()
