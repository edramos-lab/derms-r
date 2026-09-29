"""E2E-ImageClassification training augmentations (albumentations).

Source: https://github.com/edramos-lab/E2E-ImageClassification/blob/main/train.py
Geometric and stain/illumination transforms are applied on uint8 RGB, then this
benchmark applies ImageNet mean/std so pretrained timm checkpoints stay valid.
"""
from __future__ import annotations
import inspect
import cv2
import numpy as np
import albumentations as A


class LabABShift(A.ImageOnlyTransform):
    """Shift OpenCV LAB a* (±5) and b* (±7) for stain color-balance variation."""

    def __init__(self, p=0.5):
        super().__init__(p=p)

    def apply(self, img, a_shift=0.0, b_shift=0.0, **params):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_shift, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift, 0, 255)
        return cv2.cvtColor(np.round(lab).astype(np.uint8), cv2.COLOR_LAB2RGB)

    def get_params(self):
        return {
            'a_shift': float(np.random.uniform(-5, 5)),
            'b_shift': float(np.random.uniform(-7, 7)),
        }

    def get_transform_init_args_names(self):
        return ()


def _supports_kwarg(transform_cls, name):
    return name in inspect.signature(transform_cls.__init__).parameters


def _hsv_stain_shift(p):
    """Hue ±10°, saturation ±15, value ±8 (stain / illumination)."""
    if p <= 0:
        return None
    if _supports_kwarg(A.HueSaturationValue, 'hue_shift_range'):
        return A.HueSaturationValue(
            hue_shift_range=(-10, 10),
            sat_shift_range=(-15, 15),
            val_shift_range=(-10, 10),
            p=p,
        )
    return A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=p,
    )


def _brightness_contrast_shift(p):
    """Brightness ±8%, contrast ±12% (microscope / slide variation)."""
    if p <= 0:
        return None
    if _supports_kwarg(A.RandomBrightnessContrast, 'brightness_range'):
        return A.RandomBrightnessContrast(
            brightness_range=(-0.08, 0.08),
            contrast_range=(-0.12, 0.12),
            p=p,
        )
    return A.RandomBrightnessContrast(
        brightness_limit=0.08,
        contrast_limit=0.12,
        p=p,
    )


def _gauss_noise(p):
    """Low-probability sensor / compression noise."""
    if p <= 0:
        return None
    if _supports_kwarg(A.GaussNoise, 'std_range'):
        return A.GaussNoise(std_range=(0.01, 0.04), p=p)
    return A.GaussNoise(var_limit=(10.0, 40.0), p=p)


def training_augmentations(cfg):
    """Ordered E2E train-time transforms, excluding resize/normalize/tensor."""
    ops = [
        A.HorizontalFlip(p=cfg.get('flip_probability', 0.5)),
        A.Rotate(limit=cfg.get('rotation', 20.0), p=cfg.get('rotation_probability', 1.0),
                 border_mode=0, fill=0),
    ]
    for item in (
        _hsv_stain_shift(cfg.get('hsv_probability', 0.8)),
        LabABShift(p=cfg.get('lab_probability', 0.8)),
        _brightness_contrast_shift(cfg.get('brightness_contrast_probability', 0.8)),
        _gauss_noise(cfg.get('gauss_noise_probability', 0.15)),
    ):
        if item is not None:
            ops.append(item)
    blur_p = cfg.get('blur_probability', 0.15)
    if blur_p > 0:
        ops.append(A.Blur(blur_limit=(3, 3), p=blur_p))
    return ops
