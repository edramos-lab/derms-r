"""
Export MedMNIST's BloodMNIST (8-class peripheral blood cell classification) into
this repository's ImageFolder layout for the 'bloodmnist' dataset
(derms/config.py's DATASETS / IMAGEFOLDER_DATASETS).

MedMNIST ships train/val/test splits, but a handful of images are
byte-identical across the official train/val and test splits (observed: 6 out
of ~17k at size=224) -- a data-quality artifact of the source dataset, not
something introduced here. derms/engine.py's untouched_test_data() hashes
decoded image content and refuses to evaluate if development and test share
any image, by design (see test.py's module docstring: leakage checks have
"always" applied before first opening the test set). Rather than discover
that after a full training run, this script hashes test images first and
drops any development (train+val) image with the same content hash, logging
how many were dropped. Test itself is left untouched.

Usage:
    python prepare_bloodmnist.py --out_dir ./datasets/bloodmnist --size 224
"""

import argparse
import hashlib
import re
from pathlib import Path

import medmnist
import numpy as np
from torch.utils.data import ConcatDataset


def parse_args():
    p = argparse.ArgumentParser(description='Export BloodMNIST to an ImageFolder layout')
    p.add_argument('--out_dir', type=str, required=True,
                    help='Output root; will contain out_dir/development (train+val) and out_dir/test')
    p.add_argument('--size', type=int, default=224, choices=[28, 64, 128, 224],
                    help='MedMNIST image resolution to download')
    p.add_argument('--download_root', type=str, default='./medmnist_raw',
                    help='Where medmnist caches its downloaded .npz file')
    return p.parse_args()


def safe_name(label):
    return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')


def content_hash(img):
    arr = np.asarray(img)
    return hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()


def export_split(dataset, class_names, dst_root, exclude_hashes=None):
    counts, hashes, skipped = {}, set(), 0
    for i in range(len(dataset)):
        img, label = dataset[i]
        img = img.convert('RGB')
        h = content_hash(img)
        if exclude_hashes is not None and h in exclude_hashes:
            skipped += 1
            continue
        cls = class_names[int(label[0])]
        dst_dir = dst_root / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        idx = counts.get(cls, 0)
        img.save(dst_dir / f'{idx:05d}.png')
        counts[cls] = idx + 1
        hashes.add(h)
    return counts, hashes, skipped


def main():
    args = parse_args()
    info = medmnist.INFO['bloodmnist']
    data_class = getattr(medmnist, info['python_class'])
    class_names = {int(k): safe_name(v) for k, v in info['label'].items()}
    print('Classes:', class_names)

    train_ds = data_class(split='train', download=True, size=args.size, root=args.download_root)
    val_ds = data_class(split='val', download=True, size=args.size, root=args.download_root)
    test_ds = data_class(split='test', download=True, size=args.size, root=args.download_root)

    out_dir = Path(args.out_dir)
    dev_root = out_dir / 'development'
    test_root = out_dir / 'test'

    test_counts, test_hashes, _ = export_split(test_ds, class_names, test_root)
    dev_counts, _, dev_skipped = export_split(ConcatDataset([train_ds, val_ds]), class_names, dev_root,
                                               exclude_hashes=test_hashes)

    print('development (train+val):', dev_counts)
    print('test:', test_counts)
    if dev_skipped:
        print(f'Dropped {dev_skipped} development image(s) that exactly duplicate a test image '
              f'(cross-split content collision; test left untouched).')
    print('Written to:', dev_root, '/', test_root)


if __name__ == '__main__':
    main()
