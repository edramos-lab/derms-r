#!/usr/bin/env python3
"""Offline real timm MobileNetV4 train/evaluate check on synthetic RGB images."""
import json
from pathlib import Path
import tempfile
import torch
from derms import engine as b
from verify_benchmark import make_data


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        make_data(root)
        ds = b.tv.ImageFolder(root / 'all/development')
        cfg = dict(device='cpu', pretrained=False, lr=.0001, weight_decay=.01, batch_size=4,
            workers=0, augmentation=True, flip_probability=.5, rotation=10., rotation_probability=.3,
            seed=42, warmup=1, latency_repeats=2)
        model, metrics, epoch, peak = b.fit(ds, [0,1,25,26], [2,27], cfg, 'mobilenetv4',
                                           2, root / 'fit', 'synthetic-smoke', 42, 1)
        print(json.dumps({'synthetic_only': True, 'model': b.REGISTRY['mobilenetv4'],
                          'metrics': metrics, 'best_epoch': epoch, 'environment': b.environment()}, indent=2))


if __name__ == '__main__':
    main()
