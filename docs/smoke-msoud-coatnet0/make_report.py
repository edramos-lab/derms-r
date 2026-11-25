"""Build figures, tables and main.tex for the CoAtNet-0 / Msoud_v2 smoke-test report from the downloaded results."""
import ast, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def roc_curve(t, sc):
    o = np.argsort(-sc, kind='stable'); t = np.asarray(t)[o]; sc = sc[o]
    d = np.where(np.diff(sc))[0]; idx = np.r_[d, len(t) - 1]
    tps = np.cumsum(t)[idx]; fps = 1 + idx - tps
    return np.r_[0, fps / fps[-1]], np.r_[0, tps / tps[-1]], None


def auc(x, y_):
    return float(np.trapz(y_, x))


def matthews_corrcoef(yt, yp):
    c = np.bincount(yt * 4 + yp, minlength=16).reshape(4, 4).astype(float)
    t, pk_, cc, s_ = c.sum(1), c.sum(0), np.trace(c), c.sum()
    return (cc * s_ - t @ pk_) / np.sqrt((s_ ** 2 - pk_ @ pk_) * (s_ ** 2 - t @ t))

ROOT = Path(sys.argv[1])
R = ROOT / 'Msoud_v2'
OUT = ROOT / 'report'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
NICE = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
OI = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'verm': '#D55E00', 'sky': '#56B4E9', 'purple': '#CC79A7', 'grey': '#666666'}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 8, 'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

cfg = json.load(open(ROOT / 'config.json'))
fr = pd.read_csv(R / 'fold_results.csv')
sm = pd.read_csv(R / 'summary.csv').sort_values('ratio').reset_index(drop=True)
lock = R / 'locked' / 'coatnet_0'
rep = json.load(open(lock / 'test_classification_report.json'))
prob = np.load(lock / 'test_probabilities.npz')
dr = json.load(open(lock / 'derms_r.json'))
sel = json.load(open(lock / 'selection.json'))
history = json.load(open(lock / 'history.json'))
import shutil
gc = json.load(open(ROOT / 'gradcam' / 'gradcam_selection.json'))
shutil.copy(ROOT / 'gradcam' / 'gradcam_overview.pdf', FIG / 'fig6_gradcam.pdf')
MLRUN = '9bad2a85b27e4d60baa4b6ded10d7fe7'

cm = np.array(rep['confusion_matrix'])
n = cm.sum()
support = cm.sum(1)
tp = np.diag(cm)
rec = tp / support
prc = tp / cm.sum(0)
f1 = 2 * prc * rec / (prc + rec)
acc = tp.sum() / n
bal = rec.mean()
y, p = prob['targets'], prob['predictions']
assert (np.bincount(y) == support).all()
assert (np.bincount(y * 4 + p, minlength=16).reshape(4, 4) == cm).all()
mcc = matthews_corrcoef(y, p)
macro_f1 = f1.mean()
z = 1.96
wil_c = (acc + z * z / (2 * n)) / (1 + z * z / n)
wil_h = z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
conf = prob['probabilities'].max(1)
ok = p == y
chance = support.max() / n


def pooled_cm(ratio):
    s = fr[np.isclose(fr.ratio, ratio)]
    return sum(np.array(ast.literal_eval(c)) for c in s['metrics.confusion_matrix'])


# ---------- Figure 1: CV accuracy and DERMS-R vs. ratio
ratios = sm.ratio.values
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
for r in ratios:
    a = fr[np.isclose(fr.ratio, r)]['metrics.accuracy'].values
    ax[0].scatter(np.full(len(a), r) + np.linspace(-0.012, 0.012, len(a)), a, s=9, color=OI['sky'], zorder=2, linewidths=0)
ax[0].errorbar(ratios, sm.accuracy_mean, yerr=sm.accuracy_std, fmt='o-', color=OI['blue'], capsize=3, lw=1.2, ms=4, zorder=3, label='mean $\\pm$ s.d.')
ax[0].axhline(acc, color=OI['verm'], ls='--', lw=1, label=f'BRISC test accuracy ({acc:.3f})')
ax[0].set_xlabel('Training ratio $r$'); ax[0].set_ylabel('Validation accuracy')
ax[0].set_xticks(ratios); ax[0].set_ylim(0.5, 1.0); ax[0].set_title('(a) Five-fold CV accuracy', loc='left')
ax[0].legend(frameon=False, loc='lower right')
ax[1].plot(ratios, sm.derms_r_raw, 'o-', color=OI['blue'], lw=1.2, ms=4)
i = int(sm.derms_r_raw.idxmax())
ax[1].scatter([ratios[i]], [sm.derms_r_raw[i]], s=60, facecolors='none', edgecolors=OI['verm'], linewidths=1.3, zorder=4)
ax[1].annotate('locked', (ratios[i], sm.derms_r_raw[i]), xytext=(8, 6), textcoords='offset points', color=OI['verm'], fontsize=8)
ax[1].set_xlabel('Training ratio $r$'); ax[1].set_ylabel('DERMS-R (raw)'); ax[1].set_xticks(ratios)
ax[1].set_title('(b) DERMS-R score', loc='left')
fig.savefig(FIG / 'fig1_cv_ratio.pdf'); plt.close(fig)

# ---------- Figure 2: test confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
for k, (mat, title, fmt) in enumerate([(cm, '(a) Counts', 'd'), (cm / support[:, None], '(b) Row-normalised (recall)', '.2f')]):
    im = ax[k].imshow(mat if k == 0 else mat, cmap='Blues', vmin=0, vmax=(cm.max() if k == 0 else 1))
    for a_ in range(4):
        for b_ in range(4):
            v = mat[a_, b_]
            ax[k].text(b_, a_, format(v, fmt), ha='center', va='center', fontsize=8.5,
                       color='white' if v > (0.55 * (cm.max() if k == 0 else 1)) else 'black')
    ax[k].set_xticks(range(4)); ax[k].set_yticks(range(4))
    ax[k].set_xticklabels(NICE, rotation=30, ha='right'); ax[k].set_yticklabels(NICE)
    ax[k].set_xlabel('Predicted'); ax[k].set_ylabel('True'); ax[k].set_title(title, loc='left')
    for s_ in ax[k].spines.values(): s_.set_visible(False)
    ax[k].tick_params(length=0)
fig.savefig(FIG / 'fig2_test_confusion.pdf'); plt.close(fig)

# ---------- Figure 3: per-class recall, CV vs test
cv_rec = {}
for r in (0.10, 1.00):
    c = pooled_cm(r); cv_rec[r] = np.diag(c) / c.sum(1)
fig, ax = plt.subplots(figsize=(6.2, 2.9), constrained_layout=True)
w = 0.26; xs = np.arange(4)
for j, (lab, vals, col) in enumerate([('CV, $r{=}0.10$ (pooled folds)', cv_rec[0.10], OI['sky']),
                                      ('CV, $r{=}1.00$ (pooled folds)', cv_rec[1.00], OI['blue']),
                                      ('BRISC test (locked $r{=}0.10$ model)', rec, OI['verm'])]):
    b = ax.bar(xs + (j - 1) * w, vals, w, label=lab, color=col)
    for xx, vv in zip(xs + (j - 1) * w, vals):
        ax.text(xx, vv + 0.012, f'{vv:.2f}', ha='center', fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(NICE); ax.set_ylabel('Per-class recall'); ax.set_ylim(0, 1.12)
ax.legend(frameon=False, loc='lower center', ncol=1, bbox_to_anchor=(0.5, 1.0), ncols=3, columnspacing=1.0, handlelength=1.2)
fig.savefig(FIG / 'fig3_recall_cv_vs_test.pdf'); plt.close(fig)

# ---------- Figure 4: ROC (one-vs-rest) on test
fig, ax = plt.subplots(figsize=(3.6, 3.4), constrained_layout=True)
cols = [OI['blue'], OI['orange'], OI['green'], OI['verm']]
aucs = []
for k in range(4):
    fpr, tpr, _ = roc_curve(y == k, prob['probabilities'][:, k]); aucs.append(auc(fpr, tpr))
    ax.plot(fpr, tpr, color=cols[k], lw=1.3, label=f'{NICE[k]} (AUC {aucs[-1]:.3f})')
ax.plot([0, 1], [0, 1], color=OI['grey'], lw=0.7, ls=':')
ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate'); ax.set_aspect('equal')
ax.legend(frameon=False, loc='lower right', fontsize=7.5)
fig.savefig(FIG / 'fig4_test_roc.pdf'); plt.close(fig)
assert np.allclose(aucs, [rep['roc_auc_per_class'][str(k)] for k in range(4)], atol=1e-4), aucs

# ---------- Figure 5: confidence on correct vs incorrect test predictions
fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
bins = np.linspace(0.25, 1, 16)
ax.hist(conf[ok], bins=bins, alpha=0.8, color=OI['blue'], label=f'correct (n={ok.sum()})')
ax.hist(conf[~ok], bins=bins, alpha=0.8, color=OI['verm'], label=f'incorrect (n={(~ok).sum()})')
ax.set_xlabel('Maximum softmax probability'); ax.set_ylabel('Test images'); ax.legend(frameon=False)
fig.savefig(FIG / 'fig5_confidence.pdf'); plt.close(fig)

# ---------- tables
def f(x, d=3): return f'{x:.{d}f}'
DAG = ' $^{\\dagger}$'
rows = []
for _, r in sm.iterrows():
    sub = fr[np.isclose(fr.ratio, r.ratio)]
    rows.append(f"{r.ratio:.2f} & {int(round(sub.train_n.mean()))} & {f(r.accuracy_mean)} $\\pm$ {f(r.accuracy_std)} & {f(r.f1_macro_mean)} & {f(r.mcc_mean)} & {f(r.roc_auc_macro_ovr_mean)} & {f(r.derms_r_raw, 4)}{DAG if np.isclose(r.ratio, dr['locked_ratio']) else ''} \\\\")
t_cv = '\n'.join(rows)

rows = []
for _, r in sm.iterrows():
    rows.append(f"{r.ratio:.2f} & {f(r.performance_component)} & {f(r.data_efficiency_component)} & {f(r.complexity_efficiency_component)} & {f(r.stability_component)} & {f(r.derms_r_raw, 4)} \\\\")
t_comp = '\n'.join(rows)

rows = []
for k in range(4):
    rows.append(f"{NICE[k]} & {support[k]} & {f(prc[k])} & {f(rec[k])} & {f(f1[k])} & {f(aucs[k])} \\\\")
rows.append('\\midrule')
rows.append(f"Macro average & {n} & {f(prc.mean())} & {f(rec.mean())} & {f(macro_f1)} & {f(np.mean(aucs))} \\\\")
t_cls = '\n'.join(rows)

rows = []
for a_ in range(4):
    rows.append(f"{NICE[a_]} & " + ' & '.join((f'\\textbf{{{cm[a_, b_]}}}' if a_ == b_ else str(cm[a_, b_])) for b_ in range(4)) + f" & {support[a_]} \\\\")
rows.append('\\midrule')
rows.append('Predicted total & ' + ' & '.join(str(v) for v in cm.sum(0)) + f' & {n} \\\\')
t_cm = '\n'.join(rows)

rows = []
for r_ in gc['samples']:
    nm = r_['file'].split('/')[1].replace('_', '\\_')
    rows.append(f"{r_['class'].replace('_', ' ')} & \\texttt{{{nm}}} & {r_['predicted'].replace('_', ' ')} & {r_['prob_predicted']:.3f} & {'yes' if r_['correct'] else 'no'} \\\\")
t_gc = '\n'.join(rows)
s1 = fr[np.isclose(fr.ratio, 1.0)]
prof = fr.iloc[0]
lat = fr['profile.latency_ms_mean'].mean(); lat_sd = fr['profile.latency_ms_mean'].std()
infmem = fr['profile.cuda_inference_peak_bytes'].mean() / 2**20
trmem = fr['cuda_train_peak_bytes'].max() / 2**30
pa = ' & '.join

top_conf = {k: (cm[a, b], cm[a].sum()) for k, (a, b) in {'gl_me': (0, 1), 'gl_pi': (0, 3), 'nt_pi': (2, 3), 'nt_me': (2, 1), 'me_nt': (1, 2), 'me_pi': (1, 3)}.items()}
pct = lambda k: f'{100 * top_conf[k][0] / top_conf[k][1]:.1f}\\%'
gap_vs_r01 = sm.accuracy_mean[0] - acc
gap_vs_r1 = sm.accuracy_mean.iloc[-1] - acc
env = json.load(open(ROOT / 'config.json'))['environment']
pk = env['packages']
dev_n = {'glioma': 1313, 'meningioma': 1106, 'no_tumor': 1266, 'pituitary': 1146}
m10 = sm.iloc[0]

tex = r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage[margin=2.4cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,caption,subcaption,float,microtype,siunitx,array}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\emergencystretch=3em\setlength{\parindent}{1.2em}\setlength{\parskip}{0pt}
\renewcommand{\arraystretch}{1.12}
\sisetup{round-mode=none}
\title{\vspace{-1.2cm}\textbf{Smoke-Test Evaluation of CoAtNet-0 on a Curated Brain-MRI Benchmark:\\ Stratified Five-Fold Development with a Reserved External Test Set}}
\author{DERMS-R benchmark pipeline --- run \texttt{smoke-msoud-coatnet0}}
\date{26 November 2026 (run completed 25 November 2026, UTC)}
\begin{document}
\maketitle
\begin{abstract}
\noindent We report a deliberately short ("smoke") training run that exercises the full DERMS-R benchmark pipeline for the \texttt{coatnet\_0} architecture (\texttt{coatnet\_0\_rw\_224.sw\_in1k}, 26.7\,M parameters) on the four-class brain-MRI benchmark \texttt{Msoud\_v2} (4,831 development images). Models were fine-tuned for two epochs under stratified five-fold cross-validation (CV) at five training ratios, $r\in\{0.10,0.25,0.50,0.75,1.00\}$. The DERMS-R criterion selected $r=0.10$, after which the architecture was refitted once and evaluated a single time on a reserved, independently sourced test set (BRISC2025, $n=952$). Cross-validated accuracy rose monotonically from ''' + f'{sm.accuracy_mean.iloc[0]:.3f}' + r''' ($r=0.10$) to ''' + f'{sm.accuracy_mean.iloc[-1]:.3f}' + r''' ($r=1.00$), but the locked model reached only ''' + f'{acc:.3f}' + r''' accuracy (95\% Wilson CI ''' + f'{wil_c - wil_h:.3f}--{wil_c + wil_h:.3f}' + r''') on the external test set, with macro-F1 of ''' + f'{macro_f1:.3f}' + r''' and one-vs-rest macro AUC of ''' + f'{np.mean(aucs):.3f}' + r'''. Errors concentrated in glioma$\rightarrow$meningioma and no-tumor$\rightarrow$pituitary confusions. Grad-CAM maps for one randomly drawn test image per class were generated and logged to MLflow; in three of four examples the attribution fell largely on the skull, the image margin or non-brain tissue, and in the glioma example the visible lesion received little weight. Because the run is intentionally under-trained and the CV folds were not grouped by near-duplicate or patient, these figures verify the pipeline and characterise the cross-source gap; they should not be read as the attainable performance of the architecture.
\end{abstract}

\section{Introduction}
The purpose of this run was operational: to confirm that the newly registered architecture \texttt{coatnet\_0} and the newly registered datasets train, cross-validate, lock and test correctly end-to-end on the RTX~5000 (Ada) host, before committing GPU time to a full benchmark. The run follows the DERMS-R protocol, in which an architecture is trained at several fractions of the development data, scored by a composite criterion that rewards accuracy, data efficiency, parameter efficiency and fold-to-fold stability, refitted at the selected ratio, and evaluated exactly once on a held-out test partition that is never used for model or ratio selection.

\section{Materials and Methods}
\subsection{Data}
The development partition is a curated subset of the Msoud brain-tumor MRI collection (Kaggle, version~2; \texttt{Msoud\_v2\_development\_curated\_provisional\_v1}) and contains ''' + f'{sum(dev_n.values()):,}' + r''' images in four classes (glioma ''' + f"{dev_n['glioma']:,}" + r''', meningioma ''' + f"{dev_n['meningioma']:,}" + r''', no tumor ''' + f"{dev_n['no_tumor']:,}" + r''', pituitary ''' + f"{dev_n['pituitary']:,}" + r'''). The test partition is a curated subset of the BRISC2025 collection (Kaggle, version~6; \texttt{BRISC2025\_v6\_test\_curated\_provisional\_v1}) with 952 images (glioma 244, meningioma 294, no tumor 139, pituitary 275). The two packages share no image by SHA-256 and use an identical class-to-index mapping. Both packages are marked by their authors as \emph{provisional automated selections}: manual curation is incomplete, near-duplicate grouping within Msoud has not been finalised, and patient- or source-level independence has not been established. The test class distribution differs from development (no-tumor 14.6\% vs.\ 26.2\%), and the two sources originate from different repositories, so the test measures cross-source transfer in addition to generalisation.

\subsection{Model and training}
We used \texttt{coatnet\_0\_rw\_224.sw\_in1k} from \texttt{timm} ''' + pk['timm'] + r''' with ImageNet-1k pretrained weights and a freshly initialised four-way classification head. Images were resized to $224\times224$ and normalised with ImageNet statistics. Training used AdamW (weight decay ''' + f"{cfg['weight_decay']}" + r'''), a cosine-annealed learning rate starting at $10^{-4}$ (annealed over the two epochs; $5\times10^{-5}$ in the second epoch), batch size ''' + f"{cfg['batch_size']}" + r''', full-precision (float32) arithmetic, and the repository's default augmentation policy (flip, rotation up to $20^{\circ}$, HSV/LAB colour jitter, brightness/contrast, Gaussian noise and blur). Each fit ran for ''' + f"{cfg['epochs']}" + r''' epochs; with \texttt{--epoch-selection last} the final epoch is always evaluated, so no checkpoint is chosen using validation data.

\begin{table}[H]\centering\small
\caption{Run configuration (from \texttt{config.json}).}\label{tab:setup}
\begin{tabular}{ll@{\hspace{2.2em}}ll}
\toprule
Architecture & \texttt{coatnet\_0\_rw\_224.sw\_in1k} & Epochs per fit & ''' + f"{cfg['epochs']}" + r''' \\
Parameters & ''' + f"{int(sm.num_parameters[0]):,}" + r''' & Batch size & ''' + f"{cfg['batch_size']}" + r''' \\
CV strategy & \texttt{StratifiedKFold}, 5 folds & Learning rate / weight decay & $10^{-4}$ / ''' + f"{cfg['weight_decay']}" + r''' \\
Training ratios & 0.10, 0.25, 0.50, 0.75, 1.00 & Epoch selection & last epoch \\
Seed & ''' + f"{cfg['seed']}" + r''' & Pretrained & yes (ImageNet-1k) \\
GPU & ''' + env['gpu'][0] + r''' & Software & PyTorch ''' + pk['torch'].split('+')[0] + r''', timm ''' + pk['timm'] + r''' \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Protocol and scoring}
For each ratio $r$, a stratified subset of the development partition was drawn and divided into five folds; each fold served once as validation data. For every $r$ the five validation accuracies yield a mean $\bar A$ and unbiased standard deviation $s$. The DERMS-R score is the product of four components,
\begin{equation}
\text{DERMS-R}(r)=\underbrace{\bar A^{\alpha}}_{\text{performance}}\cdot\underbrace{r^{-\gamma}}_{\text{data eff.}}\cdot\underbrace{(P/P_0)^{-\beta}}_{\text{complexity eff.}}\cdot\underbrace{(1+s/\sigma_0)^{-\delta}}_{\text{stability}},
\end{equation}
with $\alpha=2$, $\gamma=\beta=\delta=0.5$, $P_0=10^{6}$ parameters and $\sigma_0=10^{-3}$ (canonical values), where $P$ is the number of trainable parameters. The ratio with the highest DERMS-R was locked; the model was then refitted once on the corresponding stratified subset (481 images, realised ratio 0.0996) for the median best epoch of its folds (2) and evaluated once on the BRISC2025 test partition. Test metrics were computed from the stored predictions and probabilities: accuracy with a 95\% Wilson interval, per-class precision, recall and F1, macro-F1, Matthews correlation coefficient (MCC), and one-vs-rest ROC AUC. The test set was not used for any selection.

\subsection{Grad-CAM}
To inspect which image regions drive the locked model's decisions we computed Grad-CAM maps\footnote{Selvaraju et al., ``Grad-CAM: Visual explanations from deep networks via gradient-based localization'', ICCV 2017.} on the test set for one image per class. Images were drawn uniformly at random within each class with a fixed seed (''' + f"{gc['seed']}" + r''') so that the examples are not hand-picked. The target layer is the output of the last CoAtNet stage ($7\times7$ spatial resolution); channel weights are the spatially averaged gradients of the class logit, the weighted sum is rectified, bilinearly upsampled to $224\times224$ and min--max normalised per image. Maps are shown for the predicted class and, for misclassified images, also for the true class. Predictions recomputed in this analysis were asserted to equal those stored by the test run, and the checkpoint was verified against the SHA-256 recorded in the lock file. Grad-CAM is a qualitative aid: its localisation is coarse, it was not validated here against lesion annotations, and it does not establish that a region is causally used by the network.

\section{Results}
\subsection{Cross-validation}
Cross-validated accuracy increased monotonically with the training ratio, from ''' + f'{sm.accuracy_mean.iloc[0]:.3f}' + r''' at $r=0.10$ to ''' + f'{sm.accuracy_mean.iloc[-1]:.3f}' + r''' at $r=1.00$, and the fold-to-fold standard deviation shrank from ''' + f'{sm.accuracy_std.iloc[0]:.3f}' + r''' to ''' + f'{sm.accuracy_std.iloc[-1]:.3f}' + r''' (Table~\ref{tab:cv}, Fig.~\ref{fig:cv}a). The parameter-efficiency component is constant across ratios (''' + f'{sm.complexity_efficiency_component[0]:.3f}' + r'''), so DERMS-R is decided by the trade-off between the performance, data-efficiency and stability terms (Table~\ref{tab:comp}). The data-efficiency term $r^{-1/2}$ penalises large ratios more strongly than the squared accuracy rewards them, so the score decreases monotonically from ''' + f'{sm.derms_r_raw.iloc[0]:.4f}' + r''' at $r=0.10$ to ''' + f'{sm.derms_r_raw.iloc[-1]:.4f}' + r''' at $r=1.00$ (Fig.~\ref{fig:cv}b) and the smallest ratio is locked. This is a property of the metric under these hyper-parameters, not evidence that 10\% of the data suffices.

\begin{table}[H]\centering\small
\caption{Five-fold cross-validation results for \texttt{coatnet\_0} on \texttt{Msoud\_v2}. $n_\text{train}$ is the mean training-fold size; metrics are fold means (macro-averaged where applicable). $^{\dagger}$Locked ratio.}\label{tab:cv}
\begin{tabular}{ccccccc}
\toprule
$r$ & $n_\text{train}$ & Accuracy (mean $\pm$ s.d.) & Macro-F1 & MCC & Macro AUC & DERMS-R \\
\midrule
''' + t_cv + r'''
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering\small
\caption{DERMS-R components by training ratio (parameter count $P=''' + f"{int(sm.num_parameters[0]):,}" + r'''$).}\label{tab:comp}
\begin{tabular}{cccccc}
\toprule
$r$ & Performance $\bar A^{2}$ & Data eff. $r^{-1/2}$ & Complexity eff. & Stability & DERMS-R \\
\midrule
''' + t_comp + r'''
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/fig1_cv_ratio.pdf}
\caption{(a) Validation accuracy of the five folds (light points) with mean $\pm$ s.d. (dark) against the training ratio; the dashed line marks the external test accuracy of the locked model. (b) DERMS-R score against the training ratio; the circle marks the locked ratio.}\label{fig:cv}
\end{figure}

\subsection{External test}
The locked model (trained on 481 development images for two epochs) classified ''' + f'{tp.sum()}' + r''' of ''' + f'{n}' + r''' test images correctly: accuracy ''' + f'{acc:.3f}' + r''' (95\% CI ''' + f'{wil_c - wil_h:.3f}--{wil_c + wil_h:.3f}' + r'''), balanced accuracy ''' + f'{bal:.3f}' + r''', macro-F1 ''' + f'{macro_f1:.3f}' + r''', MCC ''' + f'{mcc:.3f}' + r''' and macro AUC ''' + f'{np.mean(aucs):.3f}' + r'''. The majority-class rate is ''' + f'{chance:.3f}' + r''', so the model is clearly above chance but ''' + f'{gap_vs_r01:.3f}' + r''' below its own CV accuracy at the locked ratio and ''' + f'{gap_vs_r1:.3f}' + r''' below the CV accuracy at $r=1.00$. Per-class results are in Table~\ref{tab:cls}, the confusion matrix in Table~\ref{tab:cm} and Fig.~\ref{fig:cm}, and ROC curves in Fig.~\ref{fig:roc}a.

\begin{table}[H]\centering\small
\caption{Per-class results of the locked model on the BRISC2025 test set.}\label{tab:cls}
\begin{tabular}{lccccc}
\toprule
Class & Support & Precision & Recall & F1 & ROC AUC \\
\midrule
''' + t_cls + r'''
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering\small
\caption{Test confusion matrix (rows: true class; columns: predicted class).}\label{tab:cm}
\begin{tabular}{lccccr}
\toprule
 & Glioma & Meningioma & No tumor & Pituitary & True total \\
\midrule
''' + t_cm + r'''
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/fig2_test_confusion.pdf}
\caption{Confusion matrix of the locked model on the external test set, as counts (a) and row-normalised recall (b).}\label{fig:cm}
\end{figure}

The errors are systematic rather than diffuse. ''' + pct('gl_me') + r''' of glioma images were labelled meningioma and a further ''' + pct('gl_pi') + r''' pituitary, giving a glioma recall of ''' + f'{rec[0]:.3f}' + r'''. ''' + pct('nt_pi') + r''' of no-tumor images were predicted as pituitary and ''' + pct('nt_me') + r''' as meningioma (recall ''' + f'{rec[2]:.3f}' + r'''), while ''' + pct('me_nt') + r''' of meningiomas were predicted as no tumor. Pituitary recall is high (''' + f'{rec[3]:.3f}' + r''') but precision is only ''' + f'{prc[3]:.3f}' + r''' because the model predicts pituitary for ''' + f'{cm.sum(0)[3]}' + r''' images against ''' + f'{support[3]}' + r''' true cases: the classifier is biased toward this class on the new source. Fig.~\ref{fig:rec} contrasts per-class recall in CV with that on the test set; the drop is largest for glioma and no tumor, and it is already visible against the equally under-trained CV models at $r=0.10$. The one-vs-rest AUCs (Fig.~\ref{fig:roc}a) show that the class scores retain ranking information (glioma ''' + f'{aucs[0]:.3f}' + r''', pituitary ''' + f'{aucs[3]:.3f}' + r''') except for meningioma (''' + f'{aucs[1]:.3f}' + r'''), whose probabilities are poorly separated from those of the other classes. Confidence is only modestly lower when the model errs: the mean maximum softmax probability is ''' + f'{conf[ok].mean():.3f}' + r''' for correct and ''' + f'{conf[~ok].mean():.3f}' + r''' for incorrect predictions (Fig.~\ref{fig:roc}b).

\begin{figure}[H]\centering
\includegraphics[width=0.86\linewidth]{figures/fig3_recall_cv_vs_test.pdf}
\caption{Per-class recall of pooled cross-validation folds (at the locked ratio $r=0.10$ and at $r=1.00$) versus the external test set.}\label{fig:rec}
\end{figure}

\begin{figure}[H]\centering
\begin{subfigure}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{figures/fig4_test_roc.pdf}\caption{One-vs-rest ROC curves.}\end{subfigure}
\begin{subfigure}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{figures/fig5_confidence.pdf}\caption{Prediction confidence.}\end{subfigure}
\caption{Test-set ROC curves per class and distribution of the maximum softmax probability for correct and incorrect predictions.}\label{fig:roc}
\end{figure}

\subsection{Grad-CAM examples}
Figure~\ref{fig:gc} shows the maps for the four randomly selected test images (Table~\ref{tab:gc}); three were classified correctly and the no-tumor example was predicted as pituitary with probability ''' + f"{gc['samples'][2]['prob_predicted']:.2f}" + r'''. This split is consistent with the overall accuracy but, with $n=4$, carries no statistical weight.

\begin{table}[H]\centering\small
\caption{Randomly selected test images used for Grad-CAM (one per class).}\label{tab:gc}
\begin{tabular}{llccc}
\toprule
True class & Image file & Predicted & Prob. & Correct \\
\midrule
''' + t_gc + r'''
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/fig6_gradcam.pdf}
\caption{Grad-CAM for the locked model on one random test image per class: input (top), map for the predicted class (middle) and, where the prediction was wrong, for the true class (bottom). Warmer colours indicate higher attribution; maps are normalised per image.}\label{fig:gc}
\end{figure}

Visual inspection suggests the following, which we state as observations on four images and not as findings. For the correctly classified glioma, attribution is concentrated on the skull margin and the corners of the frame, while the central region containing the visible lesion receives little weight. For the meningioma, the attribution is diffuse and spreads over the skull and facial structures; the bright mass at the left of the image is only weakly highlighted. For the no-tumor image predicted as pituitary, the map of the predicted class is dominated by the upper border of the image (the neck region) and the posterior brain, whereas the map of the true class covers the central cerebellar region. Only for the pituitary example is the attribution concentrated near the midline and suprasellar region, anatomically closer to the expected site. Taken together with the cross-source drop in accuracy, attention to the periphery, to non-brain tissue and to the image frame is compatible with the model using acquisition or framing cues that do not transfer between the two sources; this is a hypothesis for the longer run, to be tested with more images, annotated lesion masks and a faithfulness check such as occlusion or perturbation tests. The map resolution ($7\times7$) and the short two-epoch training may also contribute to the diffuse patterns.

\subsection{Computational profile}
Parameter count and a single-image GPU inference profile were recorded for every fit (Table~\ref{tab:prof}). Latency is the mean over 50 repeats at batch size 1 in float32 and was stable across ratios. The FLOP count is reported as partial by the profiler (several operators, including attention and GELU, are unsupported) and is therefore omitted from the table rather than reported as a lower bound.

\begin{table}[H]\centering\small
\caption{Computational profile of \texttt{coatnet\_0} (RTX~5000 Ada, float32, batch size 1).}\label{tab:prof}
\begin{tabular}{lr}
\toprule
Trainable parameters & ''' + f"{int(prof['profile.trainable_parameters']):,}" + r''' \\
Inference latency, mean over fits (ms) & ''' + f'{lat:.2f} (s.d. {lat_sd:.2f} across fits)' + r''' \\
Peak CUDA memory during inference (MiB) & ''' + f'{infmem:.0f}' + r''' \\
Peak CUDA memory during training (GiB) & ''' + f'{trmem:.2f}' + r''' \\
Epoch time, locked refit (s) & ''' + f"{np.mean([h['epoch_seconds'] for h in history]):.1f}" + r''' \\
\bottomrule
\end{tabular}
\end{table}

\section{Discussion and Limitations}
The run met its operational aim: the architecture and the datasets load, train, cross-validate, lock and evaluate end to end, and the reserved test set was opened exactly once. The scientific content is limited by design and by data, and the following caveats apply.
\begin{enumerate}
\item \textbf{Under-training.} Each model saw only two epochs. The locked refit used 481 images. The test accuracy therefore characterises a barely fine-tuned model.
\item \textbf{Ratio selection by DERMS-R.} With the canonical exponents the data-efficiency term dominates and the smallest ratio is always favoured at this accuracy level. The locked model is thus the weakest configuration in CV (Table~\ref{tab:cv}); a production run should inspect the score components rather than rely on the arg-max alone.
\item \textbf{Ungrouped folds.} Folds were stratified by class only. The data package states that near-duplicate grouping within Msoud is not complete and that patient-level independence is unestablished. Near-duplicates or same-patient slices straddling folds would inflate the CV accuracies in Table~\ref{tab:cv} (up to 0.976), and part of the CV--test gap may reflect this leakage rather than domain shift alone. The two effects cannot be separated with the present run.
\item \textbf{Cross-source shift.} The test images come from a different repository, with a different class balance. A fine-tuned network is known to be sensitive to such shifts, and the strongly asymmetric confusions (Fig.~\ref{fig:cm}) are consistent with a source-dependent decision rule. We did not examine image acquisition characteristics, so we make no causal claim.
\item \textbf{Single seed and single test look.} One seed (''' + f"{cfg['seed']}" + r''') and one evaluation were run; the reported interval reflects test-sample size only, not training variability.
\item \textbf{Qualitative explainability.} Grad-CAM maps cover one image per class, use a coarse $7\times7$ target layer and are unvalidated against lesion annotations; they can generate hypotheses but do not support claims about what the model has learned.
\end{enumerate}
A follow-up run should use grouped cross-validation once near-duplicate groups are available, train for several tens of epochs, report multiple seeds, and consider the per-class recalls and calibration, not accuracy alone, as selection and acceptance criteria.

\section*{Reproducibility}
All numbers and figures in this report were generated programmatically from the artefacts of the run (directory \texttt{smoke-msoud-coatnet0}: \texttt{config.json}, \texttt{Msoud\_v2/summary.csv}, \texttt{fold\_results.csv}, and the \texttt{locked/coatnet\_0/} test files). Configuration hash \texttt{''' + json.load(open(lock / 'lock.json'))['config_hash'][:12] + r'''}, split hash \texttt{''' + json.load(open(lock / 'lock.json'))['split_hash'][:12] + r'''}, Grad-CAM images and metadata are also logged in MLflow (experiment "derms-r-benchmark - smoke - Msoud\_v2", run \texttt{''' + MLRUN[:12] + r'''}). Locked checkpoint SHA-256 \texttt{''' + json.load(open(lock / 'lock.json'))['checkpoint_sha256'][:12] + r'''}. Software: Python ''' + env['python'].split()[0] + r''', PyTorch ''' + pk['torch'] + r''', timm ''' + pk['timm'] + r''', CUDA ''' + env['cuda'] + r'''.
\end{document}
'''
(OUT / 'report.tex').write_text(tex)
print('acc', acc, 'macroF1', macro_f1, 'mcc', mcc, 'bal', bal, 'ci', wil_c - wil_h, wil_c + wil_h, 'auc', aucs)
print('conf correct/incorrect', conf[ok].mean(), conf[~ok].mean(), 'chance', chance)
