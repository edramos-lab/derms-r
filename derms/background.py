"""Class-independent background randomization for blood-smear images.

Training images are split into cells and glass; the cells are composited over a freshly generated
background drawn from a table of staining protocols and optical conditions. Every random quantity is
drawn independently of the class, so the background cannot carry class information. The parameter
ranges below are plausible, hand-set values for common Romanowsky stains and microscope set-ups, not
calibrated measurements.

Used as a training-only augmentation (`--background-randomization`): evaluation sees raw images, with
the bottom caption/scale-bar strip blanked in both, so the strip is never a cue.
"""
import cv2
import numpy as np

SIDE = 224
STRIP = 0.15  # bottom fraction holding the scale bar / caption
FILL = np.array([124, 116, 104], dtype=np.uint8)  # ImageNet mean colour, written over the strip
MAX_HOLE = 60       # enclosed background smaller than this is filled (cell pallor, not a glass pocket)
MIN_COMPONENT = 40  # pixels (at 224x224) below which a detected blob is treated as debris, not a cell

# Protocol table: glass colour (unstained slide under the lamp), plasma/protein film colour, and the
# strength of the haematoxylin-like (basophilic) and eosin-like stain channels applied to the cells.
# 'ph' shifts both colours toward pink (acidic buffer) or blue (alkaline buffer).
PROTOCOLS = {
    'wright_giemsa':      dict(glass=(248, 246, 250), film=(240, 224, 236), hema=(0.9, 1.25), eosin=(0.9, 1.3)),
    'may_grunwald_giemsa': dict(glass=(244, 240, 248), film=(230, 222, 240), hema=(1.05, 1.45), eosin=(0.8, 1.15)),
    'leishman':           dict(glass=(246, 238, 240), film=(240, 218, 228), hema=(0.85, 1.2), eosin=(1.0, 1.4)),
    'jenner_giemsa':      dict(glass=(238, 241, 249), film=(224, 230, 246), hema=(1.0, 1.4), eosin=(0.7, 1.0)),
    'wright':             dict(glass=(240, 244, 248), film=(218, 232, 246), hema=(0.8, 1.15), eosin=(0.8, 1.1)),
    'diff_quik':          dict(glass=(250, 252, 252), film=(236, 243, 250), hema=(1.1, 1.6), eosin=(0.9, 1.3)),
    'giemsa_acidic':      dict(glass=(250, 238, 238), film=(246, 212, 222), hema=(0.7, 1.0), eosin=(1.2, 1.6)),
    'high_key':           dict(glass=(252, 252, 250), film=(250, 246, 248), hema=(0.8, 1.1), eosin=(0.8, 1.1)),
    'thick_grey':         dict(glass=(228, 228, 232), film=(214, 214, 224), hema=(1.0, 1.3), eosin=(0.7, 1.0)),
}

# Ruifrok & Johnston stain vectors (rows: haematoxylin, eosin, residual).
_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32)
_HED_INV = np.linalg.inv(_HED)


def fit(arr):
    """Resize to the working size (a copy, so the source array is never modified)."""
    return cv2.resize(arr, (SIDE, SIDE), interpolation=cv2.INTER_LINEAR)


def blank_strip(arr):
    """Overwrite the caption/scale-bar strip with a constant colour (in place) and return the array."""
    arr[int(arr.shape[0] * (1 - STRIP)):] = FILL
    return arr


def _otsu(values):
    scaled = np.clip(values, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold)


def glass_colour(arr):
    """Median colour of the brightest, least-tinted 10% of pixels."""
    low = arr.min(-1)
    return np.median(arr[low >= np.percentile(low, 90)], axis=0)


def cell_mask(arr, mode='all'):
    """Soft cell mask in [0, 1] (float32, HxW). 'all' keeps every cell, red cells included, by colour
    distance from the estimated glass colour; 'stained' keeps only strongly saturated (nucleated) cells."""
    h, w = arr.shape[:2]
    if mode == 'stained':
        score = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
        threshold = _otsu(score)
    else:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        glass = cv2.cvtColor(glass_colour(arr).reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
        score = np.linalg.norm(lab - glass.astype(np.float32), axis=-1) * 2.5
        # a low threshold: faint red-cell bodies count as cells, thin plasma film does not
        threshold = max(0.65 * _otsu(score), 24.0)
    hard = (score > threshold).astype(np.uint8)
    hard[int(h * (1 - STRIP)):] = 0
    hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labelled, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_COMPONENT
    hard = keep[labelled].astype(np.uint8)
    # fill only small enclosed holes (pallor inside a cell); larger enclosed glass pockets stay background
    count, labelled, stats, _ = cv2.connectedComponentsWithStats(1 - hard, connectivity=4)
    small = np.zeros(count, dtype=bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < MAX_HOLE
    border = np.unique(np.concatenate([labelled[0], labelled[-1], labelled[:, 0], labelled[:, -1]]))
    small[border] = False
    hard = np.where(small[labelled], 1, hard).astype(np.uint8)
    hard = cv2.erode(hard, np.ones((3, 3), np.uint8))  # cut the tinted halo at the cell edge
    return cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.0)


def _range(rng, low, high):
    return float(rng.uniform(low, high))


def _smooth_field(rng, h, w, cells, amplitude):
    small = rng.standard_normal((cells, cells)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amplitude


def make_background(h, w, rng, protocol=None):
    """A synthetic empty field: glass and plasma film in a protocol's colours, uneven illumination,
    optional ghost red cells and debris. Returns (float32 HxWx3 in 0..255, protocol name)."""
    name = protocol or str(rng.choice(sorted(PROTOCOLS)))
    spec = PROTOCOLS[name]
    ph = _range(rng, -1.0, 1.0)  # >0 acidic (pink), <0 alkaline (blue)
    shift = np.array([8.0, -6.0, -2.0]) * max(ph, 0) + np.array([-8.0, 2.0, 8.0]) * max(-ph, 0)
    glass = np.array(spec['glass'], dtype=np.float32) + rng.normal(0, 2.0, 3) + shift * 0.4
    film = np.array(spec['film'], dtype=np.float32) + rng.normal(0, 3.0, 3) + shift
    # plasma film is patchy: blend glass and film through a low-frequency field
    patch = np.clip(0.5 + _smooth_field(rng, h, w, int(rng.integers(3, 8)), _range(rng, 0.2, 0.9)), 0, 1)[..., None]
    bg = glass * (1 - patch) + film * patch
    # ghost red cells: pale discs with central pallor in the film colour
    if rng.random() < 0.7:
        ghost = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(0, int(_range(rng, 4, 45))))):
            centre = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            axes = (int(rng.integers(11, 19)), int(rng.integers(11, 19)))
            cv2.ellipse(ghost, centre, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, 2)
            cv2.ellipse(ghost, centre, (axes[0] * 6 // 10, axes[1] * 6 // 10), 0, 0, 360, -0.4, -1)
        ghost = cv2.GaussianBlur(ghost, (0, 0), _range(rng, 1.0, 2.5))
        bg = bg - ghost[..., None] * (np.array([18.0, 22.0, 14.0]) * _range(rng, 0.2, 1.0))
    # illumination: white balance, vignette, gradient, blotches
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = _range(rng, 0.3, 0.7) * w, _range(rng, 0.3, 0.7) * h
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.hypot(w, h)
    illumination = 1.0 - _range(rng, 0, 0.22) * radius ** 2 * 4
    angle = _range(rng, 0, 2 * np.pi)
    illumination += _range(rng, 0, 0.07) * ((xx / w - 0.5) * np.cos(angle) + (yy / h - 0.5) * np.sin(angle))
    illumination += _smooth_field(rng, h, w, int(rng.integers(3, 7)), _range(rng, 0, 0.035))
    gains = np.array([_range(rng, 0.94, 1.07), 1.0, _range(rng, 0.93, 1.08)], dtype=np.float32)  # lamp colour
    bg = bg * illumination[..., None] * gains * _range(rng, 0.88, 1.03)
    # debris and platelets
    for _ in range(int(rng.integers(0, 22))):
        cv2.circle(bg, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 4)),
                   tuple(float(c) for c in (150 + rng.normal(0, 12, 3) * [1, 1, 1] + [0, -35, 25])), -1)
    sigma = _range(rng, 0, 1.2) if rng.random() < 0.5 else 0.0  # defocused background
    if sigma:
        bg = cv2.GaussianBlur(bg, (0, 0), sigma)
    return np.clip(bg, 0, 255), name


def stain_jitter(arr, rng, protocol):
    """Re-stain the cells: scale the haematoxylin/eosin channels within the protocol's range, then a
    small gamma and brightness change."""
    spec = PROTOCOLS[protocol]
    od = -np.log((arr.astype(np.float32) + 1.0) / 256.0)
    hed = od @ _HED_INV
    hed[..., 0] *= _range(rng, *spec['hema'])
    hed[..., 1] *= _range(rng, *spec['eosin'])
    hed[..., 2] *= _range(rng, 0.85, 1.15)
    out = 256.0 * np.exp(-(hed @ _HED)) - 1.0
    out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, _range(rng, 0.85, 1.15)) * _range(rng, 0.92, 1.06)
    return np.clip(out, 0, 255)


def randomize(arr, rng, mask=None, protocol=None):
    """Composite the cells of `arr` (uint8 RGB, SIDExSIDE) over a generated background. Returns uint8."""
    h, w = arr.shape[:2]
    mask = cell_mask(arr) if mask is None else mask
    bg, name = make_background(h, w, rng, protocol)
    out = mask[..., None] * stain_jitter(arr, rng, name) + (1 - mask[..., None]) * bg
    sigma = _range(rng, 0, 0.7)
    if sigma > 0.05:  # blends the cell edge into the new background
        out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), sigma)
    out = out + rng.normal(0, _range(rng, 0, 4.0), out.shape).astype(np.float32)  # sensor noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:  # compression artifacts, applied to cells and glass alike
        ok, enc = cv2.imencode('.jpg', out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 96))])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy()
    return blank_strip(out)


def native_swap(cells, cells_mask, host, host_mask):
    """Counterfactual image: the cells of `cells` on the glass of `host`, whose own cells are inpainted."""
    hole = cv2.dilate((host_mask > 0.05).astype(np.uint8), np.ones((5, 5), np.uint8))
    clean = cv2.inpaint(np.ascontiguousarray(host), hole, 3, cv2.INPAINT_TELEA)
    out = cells_mask[..., None] * cells.astype(np.float32) + (1 - cells_mask[..., None]) * clean.astype(np.float32)
    return blank_strip(np.clip(out, 0, 255).astype(np.uint8))


def ring(mask, width=4, gap=2):
    """Thin annulus just outside the cell boundary (bool HxW), for halo-leakage checks."""
    hard = (mask > 0.5).astype(np.uint8)
    outer = cv2.dilate(hard, np.ones((2 * (gap + width) + 1,) * 2, np.uint8))
    inner = cv2.dilate(hard, np.ones((2 * gap + 1,) * 2, np.uint8))
    return (outer > 0) & (inner == 0)
