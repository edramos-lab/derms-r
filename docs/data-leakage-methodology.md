# Data-leakage methodology

This document records how DERMS-R prevents train/validation/test leakage, what an audit of the
pipeline and of the `all` (blood-cell ALL) dataset found, what was changed, and what remains a
limitation. It is the reference for the methods section of any result produced with this code.

## 1. Principle

A held-out score is only meaningful if no information about the held-out images reached training,
model selection, or hyperparameter choice. Leakage is a property of the *data*, not of what is
known about it: images from the same patient, slide or acquisition session are correlated whether
or not an identifier was recorded. When identifiers are missing, the splitter must therefore infer
dependence from what the data does reveal (filenames, pixel content, perceptual similarity) and the
write-up must state that patient-level independence cannot be guaranteed.

## 2. Controls in the pipeline

| Layer | Control | Where |
|---|---|---|
| Unit of splitting | Exact duplicates (SHA-256 of decoded RGB), filename groups and, optionally, near-duplicates are merged into one unit; units are never split across folds | `derms/datasets.py` `make_units` |
| Filename groups | `' (N)'` / `' - Copy'` suffixes and Roboflow `.rf.<hash>` suffixes stripped, prefixed with the class; optional `--all-groups` CSV overrides | `inferred_groups`, `prepare_dataset.base_id` |
| Near-duplicates | 256-bit perceptual hash (`imagehash.phash`, `hash_size=16`); images within `--near-dup-threshold` bits are one unit | `inventory`, `make_units` |
| CV strategy | `StratifiedGroupKFold` (recommended), `GroupKFold`, or `StratifiedKFold`, applied to units weighted by image count | `cv_splits` |
| Frozen splits | Folds written once to `splits.json` and hash-checked on resume | `prepare` |
| Ratio subsets | Nested 0.10-1.00 subsets are drawn only from each fold's training units | `nested` |
| Preprocessing | Augmentation only on training; normalization uses fixed ImageNet constants (nothing fitted on validation/test) | `transform`, `Images` |
| Development/test | Exact-hash, group and (when enabled) near-duplicate overlap abort the run; for folder-split datasets the near-duplicate check runs in `prepare`, before any training | `prepare`, `engine.untouched_test_data`, `test.py` |
| Final refit | Fixed epoch budget, no validation or test selection | `fit(..., val_ids=None)` |

`--near-dup-threshold` defaults to `0` (off) and is omitted from `config.json` when off, so
existing outputs and their hashes are unchanged. **Use `--near-dup-threshold 16` for `all`.**

## 3. Audit findings

The audit tool is `audit_leakage.py` (read-only); its outputs for `all` are in
`docs/leakage-audit/`. Counts are minimums: the audit detects near-identical frames, not every
image that shares a patient.

| ID | Finding | Evidence | Status |
|---|---|---|---|
| F1 | `all` carries no patient identifier. Filenames reveal only sample sequences | 2,877 development images fall into 1,829 filename groups; the largest holds 153 images (`Sap_013 (1)...`) | Mitigated by grouping; residual limit, see section 5 |
| F2 | Sequential snapshots (`Snap_086`/`Snap_087`) are near-identical but sit in different filename groups, so they could straddle folds | 27 near-duplicate pairs (<= 12 bits) across different groups inside development; 22 at <= 6 bits | Fixed by near-duplicate units |
| F3 | Three test images were near-duplicates of development images; the exact-hash and filename checks both passed | `Snap_048 - Copy` / `Snap_049 - Copy` (10 bits), `Snap_089` / `Snap_090 - Copy` (12 bits), `Snap_167 - Copy` / `Snap_168 - Copy` (10 bits) | Fixed: images quarantined, section 4 |
| F4 | `prepare_dataset.py` clusters near-duplicates only at `phash` distance <= 6, which misses F3 (10-12 bits) | The three F3 pairs measured 10, 12 and 10 bits | Use `--phash_threshold 16` when preparing a split |
| F5 | Training did not cluster near-duplicates at all | `make_units` unioned only SHA-256 and filename groups | Fixed: `--near-dup-threshold` |
| F6 | Orchestrator EDA grouped by file instead of patient/sample: the image lacked `prepare_dataset`, and the import failed silently | EDA reported one group per file (2,877) while training saw 1,829 | Fixed: `base_id` inlined in `orchestrator/insights.py`, covered by a test |
| F7 | Filename typos defeat filename grouping (`Snap_0s10`, `Snasp_009`, `Snap_#`, `Sap_#`, `Snp_#`) | The two examples are 1 bit apart | Mitigated by near-duplicate units |
| F8 | CV scores are optimistic: each fold keeps the epoch with the lowest validation loss and reports that fold's metrics at that epoch | `engine.fit`, selection on `val_ids` | **Fixed by option**: `--epoch-selection last` scores every fold at its final epoch, so nothing is chosen on validation data. The default stays `best` so existing outputs keep their config hash and resume; use `last` for new runs (the orchestrator defaults to it). The held-out test was never affected because the locked refit uses a fixed budget |
| F9 | A sweep tests every configuration on the test set; choosing a final configuration by test score is test-set selection | `benchmark()` locks and tests every architecture automatically | Process rule: select by CV, report test only |
| F10 | Exact decoded-image duplicates: none | 0 clusters in `all`, development and test | No action |
| F11 | Capture number is not a usable proxy for patient identity in `all`. The collection has 89 patients (about 36 images each), but no patient mapping is on disk | Colour-histogram distance between same-class images at capture-number gap 1-8 is 0.95-1.05 of the distance between random same-class pairs, with no rise-and-plateau at any gap up to 120 (`audit_sequence_gap.py`, `docs/leakage-audit/all_sequence_gap.*`); the run-boundary count ranges from 10 to 304 depending on the threshold, against 89 expected | Not usable for grouping; the real mapping must come from the dataset's authors |
| F12 | Acquisition fingerprint: the class is predictable from cell-free background pixels alone, so image-level scores cannot be attributed to cell morphology only | Background-only classifier on the frozen StratifiedGroupKFold folds (threshold 16): 4-class balanced accuracy 0.92 (gradient boosting; chance 0.25, shuffled-label control 0.25), benign vs malignant 0.92; with a tighter background mask 0.82 and 0.82 (`audit_background_canary.py`, `docs/leakage-audit/background_canary*.json`) | Open; limits any universality claim. Mitigate with stain/colour normalization or augmentation and report the canary |
| F13 | A fine-tuned ResNet-50 reaches nearly the same score with the stained cells removed as with the full image, so a high score does not show it uses morphology | Same folds, fixed 8 epochs, no best-epoch selection, 5 folds, balanced accuracy: full 0.986 +/- 0.007; cells only (background grayed) 0.974 +/- 0.010; background only (stained cells grayed) 0.971 +/- 0.009 (`audit_background_ablation.py`, `docs/leakage-audit/background_ablation.json`) | Open; the two inputs are redundant, so the contribution of each is not isolated |
| F14 | Removing global tint and contrast does not remove the shortcut: glass-only input scores 0.82 with or without per-image channel standardization, and the full image scores 0.99 after it | Same protocol as F13, balanced accuracy over 5 folds: glass only 0.820 +/- 0.031; glass, per-image standardized 0.827 +/- 0.022; full, standardized 0.990 +/- 0.006; background, standardized 0.963 +/- 0.011 (`docs/leakage-audit/background_ablation_norm.json`) | Open; tint is not the only cue |
| F15 | With real patient IDs (C-NMC 2019), image-level CV overstates patient-level generalization, and a model trained on `all` does not transfer to C-NMC | ResNet-50, fixed 5 epochs, 5 folds, normal vs ALL, 10,311 images from 72 subjects: image-level CV balanced accuracy 0.902 +/- 0.004 (AUC 0.970); patient-level 0.820 +/- 0.053 (AUC 0.911). `all` -> C-NMC: balanced accuracy 0.50, every image predicted ALL, AUC 0.19 (`audit_external_cnmc.py`, `docs/leakage-audit/external_cnmc.json`) | Documented; `all` is reported as image-level only |

## 4. Remediation applied to `all`

1. The three F3 test images were moved out of `datasets/all/test` into a quarantine folder with a
   manifest (`docs/leakage-audit/all_quarantine_manifest.txt`). They were not deleted; the originals
   also remain in `raw-datasets/Blood`. The test set went from 325 to **322** images.
2. Re-audit (`all_after.json`): zero development/test pairs within 16 bits, zero exact duplicates,
   zero shared filename groups.
3. The CV split was rebuilt with `--cv-strategy StratifiedGroupKFold --near-dup-threshold 16`:
   **1,801 units** from 2,877 development images (1,829 filename groups before near-duplicate
   merging). For all five folds, validation/training share **0** near-duplicate pairs and **0**
   filename groups.

| Threshold (bits) | Units before / after curation (development + test) | Cross-split units before | Cross-split units after |
|---|---|---|---|
| 0 (filename groups only) | 2,068 / 2,067 | 0 | 0 |
| 6 | 2,047 / 2,046 (dev+test) | 0 | 0 |
| 12 | 2,039 / 2,041 | 3 | 0 |
| 16 | 2,034 / 2,036 | 3 | 0 |
| 20 | 2,030 / 2,032 | 4 | 1 |
| 24 | 2,027 / 2,029 | 4 | 1 |

The count is stable from 12 to 24 bits, no unit mixes class
labels, and the largest unit never exceeds the 153-image filename group, so the threshold does not
chain unrelated images together. 16 was chosen because it sits on that plateau and above the
10-12 bit F3 pairs.

## 5. Residual risks and limits

- **No patient identifiers, and no reliable proxy for them (F11).** The image-similarity-versus-capture-number analysis found no patient-sized blocks, so the filename-and-similarity units are the best available control and are much finer (1,801 units) than 89 patients. Grouping by filename and similarity controls near-identical images,
  not patients. Images of one patient that look different (other fields, other focus) can still sit
  in different folds or in development and test. Report cross-validation and test results for `all`
  as *image-level* generalization; patient- or slide-level generalization is not established. If
  identifiers become available, pass them with `--all-groups` and re-run.
- **Acquisition confound (F12).** Background pixels alone predict the class far above chance, so the images carry stain, illumination or slide information that tracks the patient-level label. A residual cell-derived signal in the mask is possible (the tighter mask lowers but does not remove the effect). Do not claim morphology-only or patient-universal performance; the canary is a diagnostic, not a patient-level test.
- **Redundant cues (F13).** The background-only input is the non-stained region, which still contains pale red cells and cell outlines, so it is not pure glass; the 0.97 therefore shows that non-cell content suffices, not that stain tint alone does. The full model may use either cue. Report CV scores as image-level and do not attribute them to morphology.
- **Not only tint (F14).** Standardizing each channel per image leaves every score unchanged, so global colour is not what the network needs. The glass mask itself encodes where cells are not, so glass-only input still reveals cell layout and density; with the remaining cues (illumination gradients, noise and compression texture) this design cannot separate acquisition from layout. Colour normalization alone is not a mitigation.
- **External check (F15).** On C-NMC the image-to-patient gap is about 8 points of balanced accuracy and 6 of AUC, with large patient-to-patient spread (0.72-0.86). It was measured with no near-duplicate grouping inside C-NMC and a single seed. The transfer test from `all` is a strong domain shift (field images versus single-cell crops, different centre, stain and camera; `all` Benign cells are not C-NMC normal lymphocytes), so its failure says the `all` models do not carry over, not that either dataset is wrong. 350 C-NMC files from subjects 49, 67 and 68 are zero-filled in the Hugging Face mirror and were skipped; use the TCIA release for a final result.
- **One test pair at 18 bits.** `Snap_489` (test) and `Snap_490` (development), both Pre-B, are 18
  bits apart. They are outside the chosen threshold of 16 and were left in place. A threshold of 20
  would flag and require quarantining this pair.
- **Uneven folds.** Because a 153-image group cannot be split, fold sizes range from 463 to 702
  validation images and the per-class counts are uneven (for example early Pre-B: 302, 298, 87, 87,
  107). Per-fold metrics should be read with that in mind.
- **Selection bias (F8)** inflates CV scores in any run made with `--epoch-selection best` (the CLI default, for compatibility); runs with `last` are free of it. Do not compare the two policies' CV scores directly; test results are not affected.
- **Other datasets.** Footprint-Type relies on Roboflow `.rf.` filename grouping and an in-code
  group-aware holdout; it was not audited for near-duplicates. BloodMNIST was not available to
  audit. CIFAR and Flowers/Food101 inherit ImageNet-pretraining overlap that this code cannot
  control.

## 6. Reproduce

```bash
# can the class be predicted from empty background alone? (F12)
python audit_background_canary.py --root datasets/all --split results/all-run/all/splits.json --out canary [--strict]

# audit a development/test dataset (read-only)
python audit_leakage.py --root datasets/all --thresholds 6 12 16 20 24 --json audit.json

# does image similarity change with the distance in capture number? (F11)
python audit_sequence_gap.py --root datasets/all --out audit_sequence_gap

# train the same network on full / cells-only / background-only images (F13)
python audit_background_ablation.py --root datasets/all --split results/all-run/all/splits.json --out ablation
python audit_background_ablation.py --root datasets/all --split results/all-run/all/splits.json --out ablation \
    --conditions glass glass_norm full_norm background_norm --tag background_ablation_norm   # F14

# external check on C-NMC 2019 (patient IDs known): transfer from `all` and image- versus patient-level CV (F15)
python audit_external_cnmc.py --all-root datasets/all --all-split results/all-run/all/splits.json --cnmc external/cnmc

# rebuild the split with the leakage controls on
python train.py --data datasets --output results/all-run --datasets all --models resnet50 \
    --prepare-only --cv-strategy StratifiedGroupKFold --near-dup-threshold 16
```

In the orchestrator, set CV Strategy to Stratified Group K-Fold and Near-Duplicate Threshold to 16.

## 7. Reporting guidance

State the grouping rule (filename sequence, exact hash, 256-bit perceptual hash within 16 bits),
the number of units, the CV strategy, that patient identifiers were unavailable, the three
quarantined test images, and the epoch-selection policy (`--epoch-selection last` is selection-free; CV scores from `best` are optimistic).
