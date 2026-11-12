"""
Build the 9-class 'Footprint-Type' dataset from the raw 11-class Roboflow export.

The raw export has 11 classes, but after grouping Roboflow's augmented copies
(3 per source image, see derms/datasets.py inferred_groups) two classes have too
few independent images for the canonical 5-fold protocol (derms/config.py FOLDS):
Cavus-Foot-III has 3 and Flat-Foot-IV has 4. Rather than lower FOLDS, each is
merged into its adjacent grade:

    Cavus-Foot-II + Cavus-Foot-III -> Cavus-Foot-II_III
    Flat-Foot-III + Flat-Foot-IV   -> Flat-Foot-III_IV

All other classes are copied unchanged. The input is never modified.

Usage:
    python prepare_footprint.py --src ./raw-datasets/Footprint-Type --out_dir ./datasets/Footprint-Type
"""

import argparse
import shutil
from pathlib import Path

MERGES = {
    'Cavus-Foot-II': 'Cavus-Foot-II_III',
    'Cavus-Foot-III': 'Cavus-Foot-II_III',
    'Flat-Foot-III': 'Flat-Foot-III_IV',
    'Flat-Foot-IV': 'Flat-Foot-III_IV',
}


def main():
    p = argparse.ArgumentParser(description='Merge rare Footprint-Type classes into an ImageFolder layout')
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f'{args.out_dir} already exists; refusing to overwrite.')
    classes = sorted(d for d in args.src.iterdir() if d.is_dir())
    missing = set(MERGES) - {d.name for d in classes}
    if missing:
        raise SystemExit(f'Source is missing expected classes: {sorted(missing)}')
    for cls in classes:
        target = args.out_dir / MERGES.get(cls.name, cls.name)
        target.mkdir(parents=True, exist_ok=True)
        for f in sorted(cls.iterdir()):
            if (target / f.name).exists():
                raise SystemExit(f'File name collision while merging: {target / f.name}')
            shutil.copy2(f, target / f.name)
    for d in sorted(args.out_dir.iterdir()):
        print(f'{d.name}: {sum(1 for _ in d.iterdir())}')


if __name__ == '__main__':
    main()
