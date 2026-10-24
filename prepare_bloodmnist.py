"""
Export MedMNIST's BloodMNIST (8-class peripheral blood cell classification) into
this repository's ImageFolder layout for the 'bloodmnist' dataset
(derms/config.py's DATASETS / IMAGEFOLDER_DATASETS).

MedMNIST already ships deduplicated, patient-independent train/val/test splits,
so unlike prepare_dataset.py (built for the raw, leaky Blood Cell Cancer [ALL]
dataset, which is the separate 'all' dataset) this script does no dedup/grouping
-- it only decodes the official .npz arrays into PNGs on disk.

Usage:
    python prepare_bloodmnist.py --out_dir ./datasets/bloodmnist --size 224
"""

import argparse
import re
from pathlib import Path

import medmnist
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


def export_split(dataset, class_names, dst_root):
    counts = {}
    for i in range(len(dataset)):
        img, label = dataset[i]
        cls = class_names[int(label[0])]
        dst_dir = dst_root / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        idx = counts.get(cls, 0)
        img.convert('RGB').save(dst_dir / f'{idx:05d}.png')
        counts[cls] = idx + 1
    return counts


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

    dev_counts = export_split(ConcatDataset([train_ds, val_ds]), class_names, dev_root)
    test_counts = export_split(test_ds, class_names, test_root)

    print('development (train+val):', dev_counts)
    print('test:', test_counts)
    print('Written to:', dev_root, '/', test_root)


if __name__ == '__main__':
    main()
