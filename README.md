# DERMS-R benchmark

Development-only model selection, followed by a separately locked final test evaluation.
The defaults run **12 architectures × 5 ratios × 5 folds = 300 fits per dataset**
(1,500 fits for all five datasets), plus one final refit per selected dataset.
This is an implementation package, not a completed scientific benchmark.

## Install and verify

Python 3.10–3.12 on Linux/macOS. CUDA is recommended; CPU is supported. File locking uses
POSIX `fcntl`; on Windows, use WSL. Create a fresh environment and install a matching
PyTorch/torchvision build for your hardware, then:

```bash
python -m pip install -r requirements.txt
python train.py --verify-models
python verify_benchmark.py
python smoke_test.py  # real timm MobileNetV4, synthetic images, no downloads
```

The registry pins explicit ImageNet-1k checkpoint tags in `timm==1.0.29`, including
ConvNeXt-V2's ImageNet-1k FCMAE pretraining/fine-tuning. Aliases are:
`resnet50`, `efficientnet_b0`, `efficientnetv2_s`, `convnextv2_tiny`, `xception41`,
`vit_base`, `deit3_base`, `swin_tiny`, `maxvit_tiny`, `nextvit_base`,
`efficientvit_b0`, `mobilenetv4`. See `model_registry_verification.json` for the
exact identifiers and pretrained configurations. Missing IDs, unavailable weights,
deterministic-operation failures, and training errors stop the run; models are never substituted.
Initial pretrained weight downloads require internet access. `--no-pretrained` is an explicit
alternative experimental condition and must use a separate output directory.

## Run, resume, lock, evaluate

For one dataset, keep the same training arguments on every invocation:

```bash
python train.py --data ./datasets --output ./results/cifar10 --datasets cifar10 --download
# Restart after interruption, with identical configuration:
python train.py --data ./datasets --output ./results/cifar10 --datasets cifar10 --resume
# After every candidate/fold is complete: select, refit, and lock the winner:
python train.py --data ./datasets --output ./results/cifar10 --datasets cifar10 --resume --lock cifar10
# Only this program opens official test data:
python test.py --output ./results/cifar10 --dataset cifar10
```

Use `--prepare-only` to build/check splits without fitting. It creates a run configuration;
subsequent training therefore needs `--resume`. For all datasets:

```bash
python train.py --data ./datasets --output ./results/all_datasets \
  --datasets cifar10 cifar100 flowers102 food101 all --all-groups ./patients.csv --download
```

Use the same arguments plus `--resume --lock DATASET` to lock each dataset separately.
For a smaller **separate** experiment, use e.g. `--models resnet50 mobilenetv4 --epochs 10`.
Ratios and fold count are fixed; there is no option that accidentally changes them.
Data roots are `datasets/cifar10`, `datasets/cifar100`, `datasets/flowers102`,
`datasets/food101`, and `datasets/all` (torchvision may create its own subdirectories inside).
Use `--no-augmentation` to disable flip/rotation, or set `--flip-probability`,
`--rotation`, and `--rotation-probability`. Predeclare these choices before benchmarking.

## ALL data and leakage protection

Recommended structure:

```text
datasets/all/development/benign/*.png
datasets/all/development/early_pre_b/*.png
datasets/all/development/pre_b/*.png
datasets/all/development/pro_b/*.png
datasets/all/test/<same class directories>/*.png
```

Alternatively, `datasets/all/<class>/*` creates a deterministic stratified 20% test holdout
(`--holdout 0.20`, `--seed 42`). Both development/test directories must exist, or neither.
Class folder names are arbitrary but must match exactly between development and test.

For correlated images, supply `--all-groups patients.csv` with `path,group` columns;
paths are relative to `datasets/all`, such as `development/benign/001.png,patient_17`.
Every image must have a group ID. Each patient/group must have one class label. If images
really are independent patients/specimens, explicitly use `--all-independent-images`
instead. The program cannot infer patient identity from image pixels or filenames.

Exact decoded-RGB duplicates are grouped together, transitively with patient groups,
before splitting. StratifiedKFold is applied to these independent units: ordinary image
stratification when images are unique/independent, otherwise group-label stratification.
Groups never cross CV folds or the generated holdout. Conflicting group/duplicate labels
and fewer than five development units per class are rejected. With unequal group sizes,
image class proportions and realized image ratios need not be exact.

Training with explicit official/ALL partitions does not inspect test image content.
At final evaluation, duplicate images or patient overlap across development/test cause
an explicit failure; samples are never silently removed. This preserves official splits,
but a dataset containing official cross-partition duplicates may fail this strict audit.
Near-duplicates and unknown patient relationships still require dataset curation. For a
flat ALL dataset, content/labels must be read once to define the grouped holdout; thereafter
test samples are excluded from all development loaders, metrics, and selection decisions.

## Methodology

- CIFAR-10/100: official train is development; official test is held out. Flowers-102:
  official train + val is development; official test is held out. Food-101: official
  train/test. No official partitions are recombined with test data.
- Five fixed shuffled stratified folds (`seed=42`). Each fold's validation partition is
  identical across every architecture and ratio. Within each fold's training partition,
  seeded per-class unit orderings produce nested subsets at 0.10/0.25/0.50/0.75/1.00.
  Counts use `max(1, floor(r * class_unit_count))`; realized image ratios are recorded.
  Nesting is non-strict for very small classes (two adjacent subsets can coincide).
  Splits/subsets are generated once, persisted, and reused across all architectures.
- All images: RGB, direct bilinear resize to 224×224, common ImageNet mean/std.
  Default training augmentation: horizontal flip p=0.5 and rotation ±10° p=0.3,
  constant black fill. Validation/test have no stochastic augmentation. This common
  preprocessing intentionally overrides model-specific pretrained preprocessing;
  report this limitation, especially for Xception and models trained at other resolutions.
- Full fine-tuning in FP32, AdamW (`lr=1e-4`, weight decay 0.01), cosine decay to zero,
  50 epochs, no early stopping, no model-specific tuning. Minimum validation loss
  selects each fold checkpoint; earliest epoch breaks equal losses. A singleton final
  training batch uses frozen BatchNorm running statistics, without dropping samples.
- Fold metrics at the selected checkpoint: accuracy, macro precision/recall/F1 over
  all classes (`zero_division=0`), MCC, and sample-weighted cross-entropy. Fold summary
  means are unweighted; standard deviations use `ddof=1`. Validation is also used for
  epoch selection, so CV performance has selection optimism; final test is the external
  assessment, not an independent claim that CV is unbiased or nested CV.

For each **architecture–ratio** candidate, with accuracy and standard deviation in [0,1]
and trainable parameters P in **millions**:

```text
DERMS-R = A_mean^alpha * P^(-beta) * r^(-gamma) * (1 + A_std/epsilon)^(-delta)
alpha=2; beta=gamma=delta=0.5; epsilon=0.001
```

`r` in the score is the **arithmetic mean of the five realized fractions**
`train_n / full_fold_training_n`. The requested ratio remains the candidate label.
Both realized mean/SD and nominal ratio are reported, so rounding and unequal patient-group
sizes remain visible. This is an explicit aggregate-score convention, not the mean of
five fold-level scores. Final refit uses the same requested ratio and reports its own
realized fraction; this can differ from CV. Ratios refer to available training pools,
not to the test set or the whole dataset.
Components, raw scores, max-normalized scores, average-tie ranks, accuracy ranks, and
within-ratio ranks are saved. Normalization/ranking is per dataset over the configured
architecture–ratio candidate set; never pool raw scores across datasets. All-zero raw
scores normalize to zero. Scores encode chosen preferences; they are not inferential tests.

`--lock` requires the complete configured candidate set, selects highest raw DERMS-R
(ties: alias alphabetically, then ascending ratio), and saves the selection **before** refitting.
It trains a fresh instance on the winning ratio's nested subset of all development data
for the median of that candidate's five best epochs. No validation/test chooses the final
checkpoint: the fixed-budget final epoch is locked. This preserves the selected data-budget
condition rather than automatically training at r=1. `test.py` verifies the lock and reports
test metrics once. Repeated calls return the saved result. An interrupted evaluation can
retry only with the same lock and test fingerprint. Test metrics never enter DERMS-R tables.

## Outputs and reproducibility

`config.json` records scientific arguments, seeds, code hashes, dependency versions, hardware, and
pretrained configurations. Each dataset has `splits.json`, `fold_results.csv`, `summary.csv`,
`summary.json`; each fit has `last.pt`, `best.pt`, `history.json`, `result.json`.
`locked/` contains the final selection, checkpoint, lock, and eventual test results/predictions.
Store the entire output directory; optimizer checkpoints can require substantial disk space.
Only load checkpoints you trust (PyTorch pickle serialization).

Each epoch saves model, optimizer, scheduler, best weights/metrics, and history atomically.
Epoch-specific RNG seeds also seed Albumentations and each loader worker. Interrupted epochs
restart from the previous committed epoch. An OS lock prevents concurrent writers. Resume
requires identical configuration, code, dependency/hardware metadata, and content fingerprints;
changed images, group metadata, splits, checkpoints, or candidate sets are rejected. Keep the
same data path and device for resume; use a new output directory for changed experiments.
Reproducibility is scoped to the same software/hardware; CUDA determinism can fail explicitly
for unsupported operations. Test artifacts are an auditable protocol, not a tamper-proof
security system; do not use test results to revise selection and rerun the experiment.

Hardware fields include total/trainable parameters; batch-1 FP32 forward latency
(mean/median/p95 after warmup, synchronized on CUDA; excludes preprocessing/transfers);
and CUDA peak **allocated** bytes for training and inference (null on CPU). Training peak
includes model/optimizer allocations. CPU threads, warmups, repetitions, device and shape
are recorded. Use an otherwise idle device; CUDA measurements were not validated locally.
FLOPs use fvcore's convention of **one fused multiply-add = one operation**. Unsupported
operations are listed; `flops_supported_ops` is a partial count and `flops` is null whenever
coverage is incomplete or tracing fails. Even a complete trace follows fvcore's ignored-op
conventions, not a universal operation definition. Hardware metrics do not enter DERMS-R.

## Verification included

`model_registry_verification.json`: installed pretrained ID/config checks.
`training_smoke_verification.txt`: real MobileNetV4 optimization/evaluation on
synthetic images. These are execution checks, not evidence of predictive performance.
`verify_benchmark.py`: offline regression tests, including the full 25-fit workflow with a
small substitute network, lock/test isolation, nesting/grouping, formula correctness,
stale configuration/data/split rejection, duplicate leakage rejection, and identical
interrupted/resumed training. Pretrained downloads, real dataset training, and CUDA execution, all-architecture forward passes, and live W&B uploads
were not exercised in this delivery.

API references: [W&B run API](https://docs.wandb.ai/ref/python/experiments/run/),
[W&B Tables](https://docs.wandb.ai/guides/track/log/log-tables), [timm models](https://huggingface.co/docs/timm/main/reference/models),
[torchvision datasets](https://docs.pytorch.org/vision/stable/datasets.html),
[fvcore FLOP conventions](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis).


## Modular project and explicit analysis

```text
train.py                  development CV and final refit/lock CLI
analyze.py                complete-fold aggregation, ablations, ranks, W&B Tables
test.py                   locked untouched-test CLI
smoke_test.py              real MobileNetV4 synthetic optimization check
verify_benchmark.py       synthetic protocol regression suite
derms/config.py           fixed ratios and 12-model registry
derms/datasets.py         partitions, fingerprints, grouping, nesting, augmentation
derms/models.py           installed timm validation and construction
derms/metric.py           raw formula and component decomposition
derms/tracking.py         optional W&B mirror
derms/engine.py           training, profiling, aggregation, locking
derms/utils.py            atomic writes, RNG, metadata, file locks
```

The benchmark also refreshes basic local summaries when CV finishes. Run `analyze.py`
for publication tables and ablations. It requires all five folds for every configured
candidate and validates split/config identities and checkpoint hashes; it never reads
test results. `analysis/derms_summary.csv` includes components, raw/max-normalized scores,
average-tie ranks and no-data/no-complexity/no-stability/accuracy-only ablations.
The dataset×ratio rank matrices compare architectures at each ratio. The per-dataset
architecture rank matrix ranks the arithmetic mean of within-ratio ranks, giving the
five ratios equal weight. This differs from selecting the best architecture–ratio pair.
The candidate set and metric exponents are frozen in `config.json`; changing them
requires a separate experiment directory. No inferential significance is claimed.

```bash
# Download official archives and prepare deterministic splits, without training.
# torchvision archives can include test files; test samples are not evaluated here.
python train.py --data ./datasets --output ./results/cifar10 --datasets cifar10 --download --prepare-only
# Continue that exact experiment:
python train.py --data ./datasets --output ./results/cifar10 --datasets cifar10 --resume

# One model / one dataset, all five ratios and folds, separate output:
python train.py --data ./datasets --output ./results/one --datasets cifar10 --models mobilenetv4 --download --no-wandb
python analyze.py --output ./results/one --no-wandb
python train.py --data ./datasets --output ./results/one --datasets cifar10 --models mobilenetv4 --resume --lock cifar10
python test.py --output ./results/one --dataset cifar10

# Full primary benchmark, 1,500 CV fits (ALL requires your own images and group CSV):
python train.py --data ./datasets --output ./results/full --datasets cifar10 cifar100 flowers102 food101 all --all-groups ./patients.csv --download --wandb --wandb-project derms-r-benchmark
python analyze.py --output ./results/full --wandb --wandb-project derms-r-benchmark
# Use the identical full training arguments plus --resume --lock DATASET to refit/lock
# each selected winner, then test.py --output ./results/full --dataset DATASET.
```

## W&B tracking and local authority

Authenticate with `wandb login` for online operation. No credentials belong in config files.
`--wandb` enables tracking, `--no-wandb` (default) runs without importing W&B.
`--wandb-project`, `--wandb-entity`, `--wandb-mode online|offline|disabled`,
`--wandb-config tracking.json`, and `--wandb-artifacts` are supported by all three CLIs.
The optional JSON object becomes `tracking_metadata`; it cannot override scientific settings.
Tracking options are operational and excluded from scientific run IDs, so tracking can be
changed on resume. Keep the same W&B project/entity when resuming an online run.

Each CV fit has one deterministic run ID from dataset/split/config/model/ratio/fold.
Epoch logs contain training loss/LR and validation classification metrics; **DERMS-R is
never an epoch metric**. Fold summaries include selected metrics and hardware profiling.
`--wandb-artifacts` uploads each completed best checkpoint. Analysis uses a separate
`derms-analysis` job with a W&B Table and a results Artifact; final testing uses a
`locked-test` job with isolated test results. No W&B Sweeps are used.

```bash
# Tracking metadata example:
printf '{"study":"DERMS-R primary benchmark"}\n' > tracking.json
python train.py --datasets cifar10 --models mobilenetv4 --output ./results/tracked --data ./datasets --wandb --wandb-project derms-r --wandb-entity YOUR_ENTITY --wandb-config tracking.json --wandb-artifacts

# Offline W&B, no server connection (data and pretrained weights must be cached):
python train.py --datasets cifar10 --models mobilenetv4 --output ./results/offline --data ./datasets --wandb --wandb-mode offline
python analyze.py --output ./results/offline --wandb --wandb-mode offline
# Later upload each generated offline session:
wandb sync ./results/offline/wandb/offline-run-*

# Completely offline from-scratch condition, kept separate from pretrained benchmark:
python train.py --datasets cifar10 --models mobilenetv4 --output ./results/scratch --data ./datasets --no-pretrained --no-wandb
```

Tracking failures emit warnings and do not invalidate committed local results. Online
runs use `resume=allow`; offline restarts create local W&B sessions, since W&B offline
mode does not support server resume. Completed local fits are skipped, not replayed to W&B.
An interruption between a local commit and remote logging can leave gaps in remote history;
`history.json` and `last.pt` retain the authoritative history. Analysis can always regenerate
and upload final result tables. W&B network operations may wait for SDK timeouts.

`predictions.npz` stores stable IDs, targets, predicted classes and float32 probabilities
at each best CV checkpoint; `locked/test_probabilities.npz` is separate. Class ordering
is `splits.json:classes`. Disable these potentially large files with
`--no-save-probabilities` at experiment creation. Predictions are compressed and evaluated
in deterministic order. Probability files are optional outputs, not inputs to model selection.

For a metric paper, predeclare candidates, ratios, preprocessing, score exponents and
normalization scope. Keep ALL patient groups disjoint; image-level stratification cannot
establish patient independence. Fold SD describes this fixed CV partition and is not a
confidence interval. The same CV results choose epochs and candidates; use the locked test
only once for external assessment. Pretrained benchmark datasets may have unknown overlap
with ImageNet pretraining, which these local fingerprints cannot audit.
