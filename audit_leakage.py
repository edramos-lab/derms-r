#!/usr/bin/env python3
"""Audit a development/test ImageFolder dataset for data leakage.

Reports exact duplicates, filename groups (the training rule), and perceptual-hash
near-duplicates, split into pairs inside one group, across groups in the same split, and across
development/test. Read-only: it never modifies the dataset.

    python audit_leakage.py --root datasets/all --thresholds 6 12 16 20 --json audit.json
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from derms.config import NEAR_DUP_HASH_SIZE
from prepare_dataset import base_id

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
SPLITS = (('development', 'test'), ('train', 'test'), ('Training', 'Testing'))
ROBOFLOW = re.compile(r'\.rf\.[0-9a-f]+(?=\.[^.]+$)')


def scan(root):
    for dev, test in SPLITS:
        if (root / dev).is_dir() and (root / test).is_dir():
            break
    else:
        raise SystemExit(f'{root}: expected a development/test, train/test or Training/Testing pair.')
    rows = []
    for split in (dev, test):
        for cls in sorted(p for p in (root / split).iterdir() if p.is_dir()):
            rows += [(split == test, cls.name, f) for f in sorted(cls.iterdir()) if f.suffix.lower() in IMAGE_SUFFIXES]
    return rows


def fingerprint(rows):
    sha, bits = [], []
    for _, _, path in rows:
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            arr = np.asarray(rgb, dtype=np.uint8)
            sha.append(hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest())
            digest = bytes.fromhex(str(imagehash.phash(rgb, hash_size=NEAR_DUP_HASH_SIZE)))
            bits.append(np.unpackbits(np.frombuffer(digest, dtype=np.uint8)))
    return sha, np.array(bits, dtype=np.float32)


def units(groups, near_pairs):
    parent = list(range(len(groups)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    for i, g in enumerate(groups):
        if g in first:
            parent[find(i)] = find(first[g])
        else:
            first[g] = i
    for i, j in near_pairs:
        parent[find(i)] = find(j)
    return [find(i) for i in range(len(groups))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--thresholds', type=int, nargs='+', default=[6, 12, 16, 20])
    p.add_argument('--json', type=Path)
    args = p.parse_args()

    rows = scan(args.root)
    is_test = np.array([r[0] for r in rows])
    label = np.array([r[1] for r in rows])
    names = [f'{"test" if r[0] else "development"}/{r[1]}/{r[2].name}' for r in rows]
    groups = [f'{r[1]}/{base_id(ROBOFLOW.sub("", r[2].name))}' for r in rows]
    sha, bits = fingerprint(rows)
    dist = bits @ (1 - bits).T + (1 - bits) @ bits.T
    np.fill_diagonal(dist, 10 ** 6)

    report = {'images': {'development': int((~is_test).sum()), 'test': int(is_test.sum())},
              'classes': sorted(set(label))}
    print(f'{args.root}: {report["images"]["development"]} development / {report["images"]["test"]} test images, '
          f'{len(report["classes"])} classes')

    by_sha = defaultdict(list)
    for i, h in enumerate(sha):
        by_sha[h].append(i)
    exact = [v for v in by_sha.values() if len(v) > 1]
    report['exact_duplicate_clusters'] = len(exact)
    report['exact_cross_split'] = sum(1 for v in exact if len({bool(is_test[i]) for i in v}) > 1)
    print(f'exact duplicate clusters: {len(exact)} (cross-split: {report["exact_cross_split"]})')

    dev_groups, test_groups = Counter(g for g, t in zip(groups, is_test) if not t), Counter(g for g, t in zip(groups, is_test) if t)
    report['filename_groups'] = {'development': len(dev_groups), 'test': len(test_groups),
                                 'largest_development_group': max(dev_groups.values()),
                                 'shared_development_test': len(set(dev_groups) & set(test_groups))}
    print(f'filename groups: development {len(dev_groups)} (largest {max(dev_groups.values())}), '
          f'test {len(test_groups)}, shared across development/test {report["filename_groups"]["shared_development_test"]}')

    report['thresholds'] = {}
    for threshold in sorted(args.thresholds):
        pairs = [(int(i), int(j)) for i, j in np.argwhere(np.triu(dist <= threshold))]
        same_group = sum(1 for i, j in pairs if groups[i] == groups[j])
        cross = [(i, j) for i, j in pairs if is_test[i] != is_test[j]]
        other = len(pairs) - same_group - len(cross)
        roots = np.array(units(groups, pairs))
        members = defaultdict(list)
        for i, r in enumerate(roots):
            members[r].append(i)
        entry = {'near_pairs': len(pairs), 'same_group': same_group, 'different_group_same_split': other,
                 'cross_split': len(cross), 'units': len(members),
                 'largest_unit': max(len(v) for v in members.values()),
                 'cross_split_units': sum(1 for v in members.values() if len({bool(is_test[i]) for i in v}) > 1),
                 'label_conflict_units': sum(1 for v in members.values() if len({label[i] for i in v}) > 1),
                 'cross_split_pairs': [{'a': names[i], 'b': names[j], 'bits': int(dist[i, j])} for i, j in cross]}
        report['thresholds'][threshold] = entry
        print(f'threshold {threshold:>3}: near pairs {entry["near_pairs"]} (same group {same_group}, different group '
              f'{other}, cross-split {entry["cross_split"]}) -> {entry["units"]} units, largest {entry["largest_unit"]}, '
              f'cross-split units {entry["cross_split_units"]}, label-conflict units {entry["label_conflict_units"]}')
        for pair in entry['cross_split_pairs']:
            print(f'    cross-split ({pair["bits"]} bits): {pair["a"]}  <->  {pair["b"]}')
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
