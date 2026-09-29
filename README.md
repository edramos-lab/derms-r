# DERMS-R: Data-Efficient Robust Model Score

## Canonical Specification and Implementation Guide

This document defines the **canonical DERMS-R formulation** to be used throughout this repository.

When implementing, refactoring, testing, analyzing, or documenting DERMS-R, treat this README as the primary mathematical and methodological specification.

Do not modify the canonical equation, default constants, operating-point selection procedure, or interpretation unless explicitly instructed.

---

# 1. Purpose

**DERMS-R (Data-Efficient Robust Model Score)** is a multi-objective metric for evaluating deep-learning architectures by jointly considering:

1. predictive performance;
2. training-data efficiency;
3. model complexity; and
4. cross-validation stability.

DERMS-R is intended for **model selection and benchmarking**.

It does not replace conventional predictive metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- MCC
- ROC-AUC

Instead, DERMS-R addresses the question:

> Which architecture and training-data operating point provides the most favorable predefined balance between predictive performance, data requirements, model complexity, and cross-validation stability?

Higher DERMS-R values indicate a more favorable balance according to the defined weighting parameters.

---

# 2. Canonical DERMS-R Equation

For architecture \(i\), evaluated at training-data ratio \(r\):

$$
\mathrm{DERMS\text{-}R}_{i,r}
=
\frac{
\bar{A}_{i,r}^{\alpha}
}{
\left(P_i/P_0\right)^{\beta}
r^{\gamma}
\left(1+s_{i,r}/\sigma_0\right)^{\delta}
}
$$

The equivalent multiplicative representation is:

$$
\mathrm{DERMS\text{-}R}_{i,r}
=
\bar{A}_{i,r}^{\alpha}
\cdot
r^{-\gamma}
\cdot
\left(P_i/P_0\right)^{-\beta}
\cdot
\left(1+s_{i,r}/\sigma_0\right)^{-\delta}.
$$

This second representation exposes the four components of DERMS-R:

$$
\mathrm{DERMS\text{-}R}
=
\mathrm{Performance}
\times
\mathrm{DataEfficiency}
\times
\mathrm{ComplexityEfficiency}
\times
\mathrm{Stability}.
$$

---

# 3. Canonical Baseline Configuration

Unless explicitly overridden for sensitivity analysis, use:

```text
alpha  = 2.0
beta   = 0.5
gamma  = 0.5
delta  = 0.5

P0     = 1,000,000 parameters
sigma0 = 0.001

dataset_ratios = [0.10, 0.25, 0.50, 0.75, 1.00]
```

Therefore, the canonical baseline equation becomes:

$$
\mathrm{DERMS\text{-}R}_{i,r}
=
\frac{
\bar{A}_{i,r}^{2}
}{
\sqrt{P_i/P_0}
\sqrt{r}
\sqrt{1+s_{i,r}/\sigma_0}
}
$$

with

$$
P_0=10^6
$$

and

$$
\sigma_0=0.001.
$$

---

# 4. Variable Definitions

## 4.1 Mean Validation Accuracy

Let

$$
A_{i,r,k}
$$

represent the validation accuracy of architecture \(i\), at data ratio \(r\), for cross-validation fold \(k\).

For \(K\) folds:

$$
\bar A_{i,r}
=
\frac{1}{K}
\sum_{k=1}^{K}
A_{i,r,k}
$$

Accuracy MUST be represented on the interval:

$$
0\leq A\leq1.
$$

For example:

```text
Correct:   0.987
Incorrect: 98.7
```

Do not mix percentage-scale and fractional-scale accuracies.

---

## 4.2 Cross-Validation Standard Deviation

Cross-validation stability is represented using the **sample standard deviation**:

$$
s_{i,r}
=
\sqrt{
\frac{1}{K-1}
\sum_{k=1}^{K}
\left(
A_{i,r,k}-\bar A_{i,r}
\right)^2
}
$$

The implementation MUST therefore use sample SD.

For PyTorch:

```python
sd_accuracy = accuracies.std(unbiased=True)
```

For NumPy:

```python
sd_accuracy = np.std(accuracies, ddof=1)
```

Do not use population SD.

---

# 5. Fixed vs. Estimated Quantities

DERMS-R intentionally combines empirical estimates with fixed experimental quantities.

## Estimated from cross-validation

The following quantities are estimated from validation folds:

```text
mean validation accuracy = Abar
validation accuracy SD   = s
```

## Fixed for an operating point

The following are fixed:

```text
P_i = number of trainable parameters
r   = training-data ratio
```

Neither \(P_i\) nor \(r\) should be averaged across folds.

## Fixed DERMS-R parameters

Unless performing sensitivity analysis:

```text
alpha
beta
gamma
delta
P0
sigma0
```

remain fixed.

---

# 6. DERMS-R Components

## 6.1 Performance Component

$$
Q_{i,r}
=
\bar A_{i,r}^{\alpha}
$$

Canonical baseline:

$$
Q_{i,r}=\bar A_{i,r}^{2}.
$$

This rewards predictive performance.

---

## 6.2 Data-Efficiency Component

$$
D(r)=r^{-\gamma}
$$

For the canonical

$$
\gamma=0.5,
$$

the experimentally evaluated ratios produce:

| Training ratio | Data-efficiency factor |
|---:|---:|
| 0.10 | 3.1623 |
| 0.25 | 2.0000 |
| 0.50 | 1.4142 |
| 0.75 | 1.1547 |
| 1.00 | 1.0000 |

Lower training-data requirements therefore increase the data-efficiency component.

The canonical experiment uses:

$$
\mathcal R=\{0.10,0.25,0.50,0.75,1.00\}
$$

---

# 7. Complexity-Efficiency Component

Model complexity is represented by:

$$
C(P_i)
=
\left(
\frac{P_i}{P_0}
\right)^{-\beta}
$$

where

$$
P_0=1,000,000.
$$

Using \(P_i/P_0\) makes the complexity term dimensionless.

The canonical baseline is:

$$
\beta=0.5.
$$

Therefore:

$$
C(P_i)
=
\frac{1}{
\sqrt{P_i/P_0}
}.
$$

Use the number of **trainable parameters** unless the experiment explicitly defines another convention.

For a PyTorch model:

```python
num_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
```

---

# 8. Stability Component

Cross-validation stability is represented by:

$$
R(s_{i,r})
=
\left(
1+\frac{s_{i,r}}{\sigma_0}
\right)^{-\delta}
$$

Canonical values:

```text
delta  = 0.5
sigma0 = 0.001
```

Therefore:

$$
R(s_{i,r})
=
\frac{1}{
\sqrt{
1+s_{i,r}/0.001
}
}.
$$

A model with lower cross-validation variability receives a larger stability component.

---

# 9. Critical Operating-Point Rule

This rule is mandatory.

## DO NOT

Do NOT first identify the training-data ratio with the highest validation accuracy and then calculate DERMS-R only for that ratio.

That procedure defeats the purpose of explicitly including data efficiency in DERMS-R.

## DO

Calculate DERMS-R independently for every architecture × ratio operating point.

For architecture \(i\):

$$
\mathrm{DERMS\text{-}R}_{i,0.10}
\quad
\mathrm{DERMS\text{-}R}_{i,0.25}
\quad
\mathrm{DERMS\text{-}R}_{i,0.50}
\quad
\mathrm{DERMS\text{-}R}_{i,0.75}
\quad
\mathrm{DERMS\text{-}R}_{i,1.00}.
$$

Only after all five scores have been calculated should the optimal operating point be selected.

---

# 10. DERMS-R Optimal Operating Point

For architecture \(i\), define:

$$
r_i^*
=
\underset{
r\in\{0.10,0.25,0.50,0.75,1.00\}
}{
\arg\max
}
\;
\mathrm{DERMS\text{-}R}_{i,r}
$$

The architecture-level optimal DERMS-R score is:

$$
\mathrm{DERMS\text{-}R}_{i}^{*}
=
\max_{
r\in\{0.10,0.25,0.50,0.75,1.00\}
}
\mathrm{DERMS\text{-}R}_{i,r}
$$

This distinction MUST be preserved throughout the software.

### Operating-point score

```text
DERMS-R(i, r)
```

evaluates a particular architecture at a particular training-data ratio.

### Architecture-level optimal score

```text
DERMS-R*(i)
```

represents the best DERMS-R operating point observed for architecture \(i\).

Do not confuse these two quantities.

---

# 11. Cross-Validation Requirements

DERMS-R stability must measure variation across cross-validation folds under a fixed experimental configuration.

For a particular architecture \(i\) and ratio \(r\):

```text
fold_1_accuracy
fold_2_accuracy
...
fold_K_accuracy
```

must correspond to the same:

```text
architecture
dataset
dataset ratio
optimizer configuration
learning rate
batch size
training protocol
preprocessing protocol
```

Do NOT pool accuracies obtained from different learning rates, batch sizes, or unrelated hyperparameter configurations to calculate \(s_{i,r}\).

Otherwise, the SD would mix:

```text
cross-validation variability
+
hyperparameter variability
```

and would no longer represent the intended DERMS-R stability term.

---

# 12. Recommended Five-Fold Configuration

The primary benchmark should use:

```text
K = 5
```

For example:

```python
fold_accuracies = [
    0.982,
    0.991,
    0.987,
    0.979,
    0.986,
]
```

Then:

```python
mean_accuracy = mean(fold_accuracies)
sd_accuracy   = sample_std(fold_accuracies)
```

These values become:

$$
\bar A_{i,r}
$$

and

$$
s_{i,r}.
$$

---

# 13. Normalization

Raw DERMS-R and normalized DERMS-R must remain distinct.

When comparing architectures at the **same dataset ratio**, normalized DERMS-R may be calculated as:

$$
\mathrm{DERMS\text{-}R}^{Norm}_{i,r}
=
\frac{
\mathrm{DERMS\text{-}R}_{i,r}
}{
\max_j
\mathrm{DERMS\text{-}R}_{j,r}
}
$$

Therefore:

$$
0<
\mathrm{DERMS\text{-}R}^{Norm}_{i,r}
\leq1.
$$

A normalized value of:

```text
1.000
```

means:

> Highest DERMS-R within the explicitly defined comparison set.

It does NOT mean:

```text
100% accuracy
perfect model
perfect generalization
```

Always preserve the raw DERMS-R value.

Never overwrite raw scores with normalized scores.

---

# 14. Canonical PyTorch Implementation

The repository implementation is `derms.metric.derms_r`. It follows this specification:

```python
import torch
from typing import Sequence, Dict


def derms_r(
    fold_accuracies: Sequence[float],
    num_params: int,
    data_ratio: float,
    alpha: float = 2.0,
    beta: float = 0.5,
    gamma: float = 0.5,
    delta: float = 0.5,
    p0: float = 1_000_000.0,
    sigma0: float = 0.001,
) -> Dict[str, float]:

    acc = torch.as_tensor(
        fold_accuracies,
        dtype=torch.float64,
    )

    if acc.numel() < 2:
        raise ValueError(
            "DERMS-R requires at least two CV folds."
        )

    if torch.any((acc < 0.0) | (acc > 1.0)):
        raise ValueError(
            "Accuracy must be represented on the 0-1 scale."
        )

    if num_params <= 0:
        raise ValueError(
            "num_params must be greater than zero."
        )

    if not 0.0 < data_ratio <= 1.0:
        raise ValueError(
            "data_ratio must satisfy 0 < r <= 1."
        )

    if p0 <= 0:
        raise ValueError(
            "p0 must be greater than zero."
        )

    if sigma0 <= 0:
        raise ValueError(
            "sigma0 must be greater than zero."
        )

    mean_acc = acc.mean()

    # Sample standard deviation
    sd_acc = acc.std(unbiased=True)

    performance = mean_acc.pow(alpha)

    data_efficiency = torch.tensor(
        data_ratio,
        dtype=torch.float64,
    ).pow(-gamma)

    complexity_efficiency = torch.tensor(
        num_params / p0,
        dtype=torch.float64,
    ).pow(-beta)

    stability = (
        1.0 + sd_acc / sigma0
    ).pow(-delta)

    score = (
        performance
        * data_efficiency
        * complexity_efficiency
        * stability
    )

    return {
        "mean_accuracy": mean_acc.item(),
        "sd_accuracy": sd_acc.item(),
        "performance": performance.item(),
        "data_efficiency": data_efficiency.item(),
        "complexity_efficiency":
            complexity_efficiency.item(),
        "stability": stability.item(),
        "derms_r": score.item(),
    }
```

---

# 15. Evaluating All Operating Points

The software must evaluate all five canonical data ratios.

Example:

```python
DATA_RATIOS = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

scores = {}

for ratio in DATA_RATIOS:

    scores[ratio] = derms_r(
        fold_accuracies=cv_results[ratio],
        num_params=num_params,
        data_ratio=ratio,
    )
```

Then determine:

```python
best_ratio = max(
    scores,
    key=lambda r: scores[r]["derms_r"],
)

best_score = scores[best_ratio]["derms_r"]
```

This corresponds mathematically to:

$$
r_i^*=\arg\max_r \mathrm{DERMS\text{-}R}_{i,r}
$$

and

$$
\mathrm{DERMS\text{-}R}_i^*=\max_r \mathrm{DERMS\text{-}R}_{i,r}.
$$

---

# 16. Required Output per Operating Point

For every architecture × dataset × ratio combination, store at least:

```text
dataset
architecture
num_parameters
data_ratio
number_of_folds

fold_1_accuracy
fold_2_accuracy
...
fold_K_accuracy

mean_validation_accuracy
sd_validation_accuracy

performance_component
data_efficiency_component
complexity_efficiency_component
stability_component

derms_r_raw
derms_r_normalized
```

At the architecture level additionally store:

```text
optimal_ratio
optimal_derms_r
```

---

# 17. Recommended Result Schema

A result table should resemble:

| Dataset | Architecture | Params | Ratio | Mean Acc | SD | Performance | Data Eff. | Complexity | Stability | DERMS-R |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dataset_A | model_A | ... | 0.10 | ... | ... | ... | 3.1623 | ... | ... | ... |
| dataset_A | model_A | ... | 0.25 | ... | ... | ... | 2.0000 | ... | ... | ... |
| dataset_A | model_A | ... | 0.50 | ... | ... | ... | 1.4142 | ... | ... | ... |
| dataset_A | model_A | ... | 0.75 | ... | ... | ... | 1.1547 | ... | ... | ... |
| dataset_A | model_A | ... | 1.00 | ... | ... | ... | 1.0000 | ... | ... | ... |

Do not discard non-optimal ratios.

All five operating points are scientifically relevant because they describe the architecture's performance/data-efficiency trajectory.

---

# 18. Recommended Software Architecture

Prefer separating metric computation from model training.

This repository:

```text
derms-r/
|
|-- derms/
|   |-- __init__.py
|   |-- metric.py
|   |-- components.py
|   |-- normalization.py
|   |-- analysis.py
|   |-- engine.py
|   |-- datasets.py
|   |-- augmentation.py
|   |-- models.py
|   |-- plots.py
|   |-- tracking.py
|   |-- config.py
|   |-- utils.py
|
|-- train.py
|-- analyze.py
|-- test.py
|-- prepare_dataset.py
|-- export_models.py
|-- requirements_export.txt
|
|-- tests/
|   |-- test_metric.py
|   |-- test_components.py
|   |-- test_normalization.py
|   |-- test_operating_points.py
|   |-- test_prepare_and_augmentation.py
|   |-- test_plots.py
|
|-- results/
|
|-- README.md
```

`metric.py` contains the canonical DERMS-R calculation.

Training code does not redefine the equation.

---

# 19. Unit Tests

Implement unit tests for at least the following properties.

## Perfect stability

If:

$$
s=0,
$$

then:

$$
R(s)=1.
$$

## Increasing variability

Holding everything else constant:

```text
higher SD -> lower DERMS-R
```

## Increasing parameter count

Holding everything else constant:

```text
higher parameter count -> lower DERMS-R
```

## Increasing data requirement

Holding everything else constant:

```text
higher r -> lower DERMS-R
```

## Increasing accuracy

Holding everything else constant:

```text
higher mean accuracy -> higher DERMS-R
```

## Reference complexity

If:

$$
P_i=P_0,
$$

then:

$$
(P_i/P_0)^{-\beta}=1.
$$

## Full dataset

If:

$$
r=1,
$$

then:

$$
r^{-\gamma}=1.
$$

These properties are tested in `tests/`.

---

# 20. Sensitivity Analysis

The baseline parameters are:

$$
(\alpha,\beta,\gamma,\delta)
=
(2,0.5,0.5,0.5).
$$

These should be treated as the **canonical baseline configuration**, not automatically as universal constants.

The research implementation should support alternative values to study ranking sensitivity.

Examples include:

### Performance-heavy

```text
alpha = 4.0
beta  = 0.25
gamma = 0.25
delta = 0.25
```

### Complexity-heavy

```text
alpha = 2.0
beta  = 1.0
gamma = 0.5
delta = 0.5
```

### Data-efficiency-heavy

```text
alpha = 2.0
beta  = 0.5
gamma = 1.0
delta = 0.5
```

### Stability-heavy

```text
alpha = 2.0
beta  = 0.5
gamma = 0.5
delta = 1.0
```

The implementation therefore exposes:

```python
alpha
beta
gamma
delta
p0
sigma0
```

as configurable parameters.

Do not hard-code them internally except as documented defaults.

---

# 21. Ablation Analysis

The research implementation should permit DERMS-R component ablation.

Useful variants include:

```text
Performance + Complexity

Performance + Complexity + Data Efficiency

Performance + Complexity + Stability

Full DERMS-R
```

The objective is to quantify how the data-efficiency and stability terms affect architecture rankings.

Do not describe an ablation variant as canonical DERMS-R.

Only the complete four-component equation is canonical DERMS-R.

---

# 22. Interpretation Rules

DERMS-R should be interpreted as a **multi-objective scalar model-selection criterion**.

Higher DERMS-R means a more favorable balance among the explicitly modeled objectives.

It does NOT establish that one model is universally superior to another.

A model may have:

```text
higher accuracy
but lower DERMS-R
```

if the accuracy improvement requires substantially:

```text
more parameters,
more training data,
or produces less stable CV performance.
```

Similarly, a very small model is not automatically optimal.

Poor predictive performance can reduce its DERMS-R through:

$$
\bar A^\alpha.
$$

DERMS-R therefore represents a trade-off rather than a single-objective efficiency measure.

---

# 23. Conventional Metrics Must Still Be Reported

DERMS-R should be reported alongside conventional predictive metrics.

At minimum consider reporting:

```text
Accuracy
Precision
Recall
F1-score
MCC
```

and ROC-AUC where methodologically appropriate.

DERMS-R should complement these metrics rather than replace them.

---

# 24. Important Methodological Distinction

Do not confuse:

```text
fold
```

with:

```text
hyperparameter experiment
```

For example, suppose an experiment contains:

```text
5 folds
3 learning rates
2 batch sizes
```

This produces:

```text
5 x 3 x 2 = 30 training runs.
```

Do NOT calculate one SD from all 30 validation accuracies.

The DERMS-R stability term should normally be calculated from the five folds belonging to one fixed training configuration.

Hyperparameter optimization should be handled separately from the CV stability estimate.

---

# 25. Canonical Experimental Ratios

The canonical experimental ratios for the current DERMS-R research project are:

$$
\mathcal R=
\{0.10,\;0.25,\;0.50,\;0.75,\;1.00\}
$$

Do not automatically add other ratios.

Do not interpolate missing ratios.

Only experimentally evaluated operating points may be used to determine:

$$
r_i^*.
$$

---

# 26. Cursor Implementation Instructions

When using this README as context in Cursor:

1. Treat the mathematical definitions in this document as authoritative.
2. Do not silently alter the DERMS-R equation.
3. Do not change the canonical baseline constants.
4. Use fractional accuracy values between 0 and 1.
5. Use sample SD across CV folds.
6. Keep parameter count and data ratio fixed for each operating point.
7. Calculate DERMS-R for every architecture × ratio combination.
8. Never select the highest-accuracy ratio before DERMS-R calculation.
9. Determine the optimal ratio using DERMS-R itself.
10. Preserve raw DERMS-R values before normalization.
11. Clearly distinguish raw, normalized, and optimal DERMS-R.
12. Keep hyperparameter variability separate from cross-validation variability.
13. Implement input validation.
14. Implement unit tests for the mathematical properties described above.
15. Make alpha, beta, gamma, delta, P0, and sigma0 configurable.
16. Do not introduce undocumented modifications to the metric.
17. If an implementation decision is ambiguous, preserve the mathematical definition rather than inventing new DERMS-R behavior.

---

# 27. Canonical Definition Summary

The authoritative DERMS-R definition for this project is:

$$
\mathrm{DERMS\text{-}R}_{i,r}
=
\frac{
\bar A_{i,r}^{2}
}{
\sqrt{P_i/10^6}
\sqrt r
\sqrt{1+s_{i,r}/0.001}
}
$$

for:

$$
r\in\{0.10,0.25,0.50,0.75,1.00\}
$$

where:

- \(\bar A_{i,r}\) = mean validation accuracy across CV folds;
- \(s_{i,r}\) = sample SD of validation accuracy across those folds;
- \(P_i\) = trainable parameter count;
- \(P_0=10^6\) parameters;
- \(r\) = experimentally evaluated training-data fraction;
- accuracy and SD use the 0–1 scale.

For each architecture:

$$
r_i^*
=
\arg\max_r
\mathrm{DERMS\text{-}R}_{i,r}
$$

and

$$
\mathrm{DERMS\text{-}R}_i^*
=
\max_r
\mathrm{DERMS\text{-}R}_{i,r}
$$

This equation, together with the operating-point selection procedure, constitutes the **canonical DERMS-R specification** for the project.

---

# Repository implementation

This package trains a development-only candidate set, scores every architecture × ratio with canonical DERMS-R, then optionally locks one winner for a separately evaluated test set.

Defaults run **12 architectures × 5 ratios × 5 folds = 300 fits per dataset** (1,500 fits for all five datasets), plus one final refit per selected dataset. This is an implementation package, not a completed scientific benchmark.

The score uses the **nominal** operating-point ratio \(r\in\{0.10,0.25,0.50,0.75,1.00\}\) and the trainable parameter count \(P_i\). Realized training fractions are recorded for audit and are not averaged into \(r\).

## Install and verify

Python 3.10–3.12 on Linux/macOS. CUDA is recommended; CPU is supported. File locking uses POSIX `fcntl`; on Windows, use WSL.

```bash
python -m pip install -r requirements.txt
python train.py --verify-models
python -m unittest discover -s tests -v
python verify_benchmark.py
python smoke_test.py
```

The registry pins explicit ImageNet-1k checkpoint tags in `timm==1.0.29`. Aliases are: `resnet50`, `efficientnet_b0`, `efficientnetv2_s`, `convnextv2_tiny`, `xception41`, `vit_base`, `deit3_base`, `swin_tiny`, `maxvit_tiny`, `nextvit_base`, `efficientvit_b0`, `mobilenetv4`. Missing IDs, unavailable weights, deterministic-operation failures, and training errors stop the run; models are never substituted.

## Run, resume, lock, evaluate

```bash
python train.py --data ./datasets --output ./results/cifar10 --datasets cifar10 --download
python train.py --data ./datasets --output ./results/cifar10 --datasets cifar10 --resume
python train.py --data ./datasets --output ./results/cifar10 --datasets cifar10 --resume --lock cifar10
python test.py --output ./results/cifar10 --dataset cifar10
python analyze.py --output ./results/cifar10
```

`--lock` requires the complete configured candidate set. It selects the highest raw DERMS-R over all architecture × ratio operating points (ties: model alias alphabetically, then ascending ratio) and saves the selection **before** refitting. It does not first keep the highest-accuracy ratio.

Exponents and scale constants are configurable (`--alpha`, `--beta`, `--gamma`, `--delta`, `--p0`, `--sigma0`) and default to the canonical baseline. Changing them requires a new output directory.

## ALL data, splits, and training protocol

- CIFAR-10/100: official train is development; official test is held out. Flowers-102: official train + val is development. Food-101: official train/test. No official partitions are recombined with test data.
- Five fixed shuffled **StratifiedKFold** folds (`n_splits=5`, `shuffle=True`, `random_state=42`), matching the E2E-ImageClassification protocol. Stratification is applied to independent units after grouping exact duplicates and patient/sample identities, so groups never cross folds. Nested subsets at 0.10/0.25/0.50/0.75/1.00. Counts use `max(1, floor(r * class_unit_count))`.
- Images: RGB, bilinear resize to 224×224, then the E2E training augmentations (horizontal flip p=0.5, rotation ±20° p=1.0, HSV stain shift p=0.8, LAB a*/b* shift p=0.8, brightness/contrast p=0.8, Gauss noise p=0.15, blur p=0.15). Validation/test omit those transforms. ImageNet mean/std is applied after augmentation so pretrained timm weights remain valid.
- Full fine-tuning in FP32, AdamW (`lr=1e-4`, weight decay 0.01), cosine decay, 50 epochs. Minimum validation loss selects each fold checkpoint.
- Fold metrics: accuracy, macro precision/recall/F1, MCC, ROC-AUC (macro one-vs-rest), and sample-weighted cross-entropy. Each fold writes `val_confusion_matrix.png` and `val_roc_curve.png`. Locked test evaluation writes the same reports as `test_*.png` / `test_classification_report.json`. These conventional reports complement DERMS-R; they do not replace it.

For the ALL / blood-cell dataset, first build a leak-free train/test split with the E2E script:

```bash
python prepare_dataset.py --raw_dir "/path/to/Blood cell Cancer [ALL]" \
    --out_dir ./datasets/all --train_name development --test_ratio 0.10
```

`prepare_dataset.py` groups images by patient/sample filename identity, exact MD5 duplicates, and perceptual-hash near-duplicates, then splits whole groups. Training infers the same filename groups unless you pass `--all-groups patients.csv` or `--all-independent-images`. Folder layouts `development/test`, `train/test`, and `Training/Testing` are accepted. Exact decoded-RGB duplicates and groups never cross CV folds. Test overlap fails explicitly.

`analyze.py` writes ablations (Performance+Complexity, +Data, +Stability) and sensitivity variants. Only the four-component equation is canonical DERMS-R.

## Export ONNX (GPU) and TensorRT

Install export extras, then convert a locked checkpoint. TensorRT engines are GPU-specific and should be built on the target device.

```bash
python -m pip install -r requirements_export.txt
# ONNX with CUDAExecutionProvider; skip TensorRT if it is not installed:
python export_models.py --lock-dir ./results/cifar10/cifar10/locked --device cuda --onnx-only
# ONNX + TensorRT FP16 (requires TensorRT on this machine):
python export_models.py --lock-dir ./results/cifar10/cifar10/locked --device cuda --fp16
```

`--lock-dir` reads `best.pt`, the winner alias, and class names from `lock.json`. You can also pass `--model_path` / `--model_name` as in E2E-ImageClassification. Place `test_image.jpg` in the working directory to run a smoke inference on the exported graph.

