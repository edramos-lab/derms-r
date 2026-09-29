"""
Build a leaked-free train/test split for the Blood Cell Cancer [ALL] 4-class dataset.

The original split (see ER2_Blood_Cancer_ALL.ipynb) picked 10% of files per class
at random with `random.sample`, then moved them to dataset/test. That leaks in
three ways:

  1. Exact duplicate files ("Snap_018.jpg" / "Snap_018 - Copy.jpg") can land on
     opposite sides of the split -> the same image is in both train and test.
  2. Near-duplicate burst-capture frames (consecutive "Snap_NNN" files, phash
     distance ~2) can also land on opposite sides.
  3. Multiple crops from the same patient/sample ("Sap_XXX (N).jpg") are treated
     as independent images, so a patient can appear in both train and test.

This script fixes all three: it groups images by patient/sample identity AND by
perceptual-hash near-duplicate clusters, deduplicates exact byte-identical
copies, and then splits whole groups (never individual images) between train
and test so no group crosses the boundary.

Usage:
    python prepare_dataset.py --raw_dir "/path/to/Blood cell Cancer [ALL]" \
                               --out_dir /path/to/output --test_ratio 0.10

    # Layout consumed by this repository's ALL dataset loader:
    python prepare_dataset.py --raw_dir "/path/to/Blood cell Cancer [ALL]" \
                               --out_dir ./datasets/all --train_name development
"""

import argparse
import hashlib
import os
import re
import random
import shutil
from collections import defaultdict

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Leak-free grouped train/test split")
    p.add_argument("--raw_dir", type=str, required=True,
                    help="Path to the raw 'Blood cell Cancer [ALL]' folder (one subfolder per class)")
    p.add_argument("--out_dir", type=str, required=True,
                    help="Output root; will contain out_dir/train and out_dir/test")
    p.add_argument("--test_ratio", type=float, default=0.10,
                    help="Target fraction of images per class held out for test")
    p.add_argument("--phash_size", type=int, default=16,
                    help="Perceptual hash size (bits^2); higher = more sensitive")
    p.add_argument("--phash_threshold", type=int, default=6,
                    help="Max Hamming distance to consider two images near-duplicates")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_name", type=str, default="train",
                    help="Development folder name (use 'development' for this repo's ALL layout)")
    p.add_argument("--test_name", type=str, default="test")
    return p.parse_args()


def base_id(fname):
    """Strip ' (N)' and ' - Copy' suffixes to recover the patient/sample identity."""
    name, _ext = os.path.splitext(fname)
    prev = None
    while prev != name:
        prev = name
        name = re.sub(r"\s\(\d+\)$", "", name)
        name = re.sub(r"\s-\s*copy(\s\(\d+\))?$", "", name, flags=re.IGNORECASE)
    return name


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    args = parse_args()
    random.seed(args.seed)

    classes = sorted(
        d for d in os.listdir(args.raw_dir)
        if os.path.isdir(os.path.join(args.raw_dir, d))
    )
    print("Classes:", classes)

    train_out = os.path.join(args.out_dir, args.train_name)
    test_out = os.path.join(args.out_dir, args.test_name)

    grand_total = {"train": 0, "test": 0}

    for cls in classes:
        cdir = os.path.join(args.raw_dir, cls)
        files = sorted(
            f for f in os.listdir(cdir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        n = len(files)
        print(f"\n=== {cls}: {n} raw images ===")

        # --- hash every image (md5 for exact dupes, phash for near-dupes) ---
        import imagehash
        md5s, phashes = [], []
        for f in files:
            path = os.path.join(cdir, f)
            with open(path, "rb") as fh:
                md5s.append(hashlib.md5(fh.read()).hexdigest())
            img = Image.open(path).convert("RGB")
            phashes.append(imagehash.phash(img, hash_size=args.phash_size))

        # --- union-find groups: same base_id, same md5, or near-dup phash ---
        uf = UnionFind(n)
        bucket_by_base = defaultdict(list)
        for i, f in enumerate(files):
            bucket_by_base[base_id(f)].append(i)
        for idxs in bucket_by_base.values():
            for i in idxs[1:]:
                uf.union(idxs[0], i)

        bucket_by_md5 = defaultdict(list)
        for i, h in enumerate(md5s):
            bucket_by_md5[h].append(i)
        for idxs in bucket_by_md5.values():
            for i in idxs[1:]:
                uf.union(idxs[0], i)

        # near-duplicate phash clustering (O(n^2) but n is small per class)
        for i in range(n):
            for j in range(i + 1, n):
                if uf.find(i) == uf.find(j):
                    continue
                if phashes[i] - phashes[j] <= args.phash_threshold:
                    uf.union(i, j)

        groups = defaultdict(list)
        for i in range(n):
            groups[uf.find(i)].append(i)
        print(f"  {n} images -> {len(groups)} identity groups "
              f"(patient/sample + exact + near-duplicate clusters)")

        # --- dedup exact copies within a group: keep one file per md5 ---
        keep_idx = []
        for idxs in groups.values():
            seen_md5 = {}
            for i in idxs:
                seen_md5.setdefault(md5s[i], i)  # first occurrence wins
            keep_idx.extend(seen_md5.values())
        keep_idx = set(keep_idx)
        n_dropped = n - len(keep_idx)
        if n_dropped:
            print(f"  dropped {n_dropped} exact-duplicate files")

        # --- group-level stratified split (never split a group) ---
        group_ids = list(groups.keys())
        random.shuffle(group_ids)
        target = max(1, int(round(n * args.test_ratio)))

        test_group_ids = []
        test_count = 0
        for gid in group_ids:
            if test_count >= target:
                break
            members = [i for i in groups[gid] if i in keep_idx]
            if not members:
                continue
            test_group_ids.append(gid)
            test_count += len(members)
        test_group_ids = set(test_group_ids)

        train_files, test_files = [], []
        for gid, idxs in groups.items():
            members = [files[i] for i in idxs if i in keep_idx]
            if gid in test_group_ids:
                test_files.extend(members)
            else:
                train_files.extend(members)

        print(f"  final split: train={len(train_files)}  test={len(test_files)} "
              f"({len(test_files) / (len(train_files) + len(test_files)):.1%} test)")

        for subset_name, subset_files, out_root in [
            ("train", train_files, train_out),
            ("test", test_files, test_out),
        ]:
            dst_dir = os.path.join(out_root, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for f in subset_files:
                shutil.copy2(os.path.join(cdir, f), os.path.join(dst_dir, f))
            grand_total[subset_name] += len(subset_files)

    print(f"\nTOTAL: train={grand_total['train']}  test={grand_total['test']}")
    print(f"Written to: {train_out}  /  {test_out}")


if __name__ == "__main__":
    main()
