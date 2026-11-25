"""Grad-CAM for the locked DERMS-R model on one randomly drawn test image per class; optionally mirrored to MLflow.

Runs inside the derms-r-train image (needs derms/, timm, torch, albumentations). The target layer is the output of the
last CoAtNet stage (7x7 feature map), so heatmaps are coarse. Images are drawn with a fixed seed, not hand-picked.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets as tv

from derms.config import REGISTRY, MEAN, STD

p = argparse.ArgumentParser()
p.add_argument('--lock-dir', type=Path, required=True)
p.add_argument('--test-dir', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--model', default='coatnet_0')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--mlflow-uri')
p.add_argument('--mlflow-experiment', default='derms-r-benchmark - smoke - Msoud_v2')
p.add_argument('--run-name', default='gradcam-coatnet_0-test')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

lock = json.load(open(a.lock_dir / 'lock.json'))
sha = hashlib.sha256((a.lock_dir / 'best.pt').read_bytes()).hexdigest()
assert sha == lock['checkpoint_sha256'], 'checkpoint does not match lock.json'
classes = lock['classes']
stored = json.load(open(a.lock_dir / 'test_predictions.json'))
stored_pred = dict(zip(stored['ids'], stored['predictions']))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = timm.create_model(REGISTRY[a.model], pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(a.lock_dir / 'best.pt', map_location='cpu', weights_only=False)['model'])
model.to(device).eval()

ds = tv.ImageFolder(a.test_dir)
assert ds.classes == classes
tf = A.Compose([A.Resize(224, 224), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

acts, grads = {}, {}
layer = model.stages[-1]
def _save(module, inputs, output):
    acts['a'] = output
    output.register_hook(lambda g: grads.__setitem__('g', g))


layer.register_forward_hook(_save)


def cam_for(x, target):
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[0, target].backward(retain_graph=True)
    w = grads['g'].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts['a']).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)[0, 0].detach().cpu().numpy()
    cam = cam - cam.min()
    return cam / cam.max() if cam.max() > 0 else cam, logits.softmax(1)[0].detach().cpu().numpy()


rng = np.random.default_rng(a.seed)
picked = []
for k, name in enumerate(classes):
    idx = [i for i, (_, t) in enumerate(ds.samples) if t == k]
    picked.append(int(rng.choice(idx)))

records, panels = [], []
for k, i in enumerate(picked):
    path, target = ds.samples[i]
    rel = f'{classes[target]}/{Path(path).name}'
    img = Image.open(path).convert('RGB')
    show = np.asarray(img.resize((224, 224), Image.BILINEAR))
    x = tf(image=np.asarray(img))['image'].unsqueeze(0).to(device)
    cam_pred_target = None
    with torch.enable_grad():
        x.requires_grad_(True)
        logits = model(x)
        pred = int(logits.argmax(1))
        assert pred == stored_pred[rel], f'{rel}: recomputed prediction {pred} != stored {stored_pred[rel]}'
        cam_p, prob = cam_for(x, pred)
        cam_t = cam_p if pred == target else cam_for(x, target)[0]
    panels.append((show, cam_p, cam_t, target, pred, prob))
    rec = {'class': classes[target], 'file': rel, 'predicted': classes[pred], 'correct': pred == target,
           'prob_predicted': float(prob[pred]), 'prob_true_class': float(prob[target])}
    records.append(rec)
    for tag, cam in (('predicted', cam_p), ('trueclass', cam_t)):
        if tag == 'trueclass' and pred == target:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(4.4, 2.3), constrained_layout=True)
        ax[0].imshow(show); ax[1].imshow(show); ax[1].imshow(cam, cmap='jet', alpha=0.45, vmin=0, vmax=1)
        ax[0].set_title(f'{classes[target]} (true)', fontsize=8); ax[1].set_title(f'Grad-CAM: {classes[pred if tag == "predicted" else target]}', fontsize=8)
        for x_ in ax: x_.axis('off')
        fig.savefig(a.out / f'gradcam_{classes[target]}_{tag}.png', dpi=200); plt.close(fig)

# combined figure: rows = original, Grad-CAM (predicted), Grad-CAM (true class, only when it differs)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'font.size': 8})
fig, ax = plt.subplots(3, 4, figsize=(7.0, 5.5), constrained_layout=True)
for c, (show, cam_p, cam_t, target, pred, prob) in enumerate(panels):
    ax[0, c].imshow(show)
    ax[0, c].set_title(f'True: {classes[target]}\nPred: {classes[pred]} ({prob[pred]:.2f})', fontsize=8,
                       color='#009E73' if pred == target else '#D55E00')
    ax[1, c].imshow(show); ax[1, c].imshow(cam_p, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    ax[2, c].imshow(show)
    if pred != target:
        ax[2, c].imshow(cam_t, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    else:
        ax[2, c].text(0.5, 0.5, '(same as\npredicted)', transform=ax[2, c].transAxes, ha='center', va='center', color='#222222', fontsize=8)
        ax[2, c].images[0].set_alpha(0.25)
    for r_ in range(3): ax[r_, c].axis('off')
for r_, lab in enumerate(['Test image', 'Grad-CAM,\npredicted class', 'Grad-CAM,\ntrue class']):
    ax[r_, 0].text(-0.06, 0.5, lab, transform=ax[r_, 0].transAxes, ha='right', va='center', fontsize=8)
fig.savefig(a.out / 'gradcam_overview.pdf'); fig.savefig(a.out / 'gradcam_overview.png', dpi=200); plt.close(fig)

meta = {'model': a.model, 'timm_id': REGISTRY[a.model], 'target_layer': 'stages[-1] output (7x7)', 'seed': a.seed,
        'checkpoint_sha256': sha, 'config_hash': lock['config_hash'], 'dataset': lock['dataset'], 'samples': records}
json.dump(meta, open(a.out / 'gradcam_selection.json', 'w'), indent=2)
print(json.dumps(records, indent=1))

if a.mlflow_uri:
    import mlflow
    mlflow.set_tracking_uri(a.mlflow_uri)
    mlflow.set_experiment(a.mlflow_experiment)
    with mlflow.start_run(run_name=a.run_name, tags={'dataset': lock['dataset'], 'job_type': 'gradcam', 'model': a.model}) as run:
        mlflow.log_params({'model': a.model, 'timm_id': REGISTRY[a.model], 'dataset': lock['dataset'],
                           'test_set': 'Msoud_v2/test (BRISC2025)', 'gradcam_target_layer': 'stages[-1]',
                           'selection_seed': a.seed, 'checkpoint_sha256': sha[:32], 'config_hash': lock['config_hash'][:32]})
        for r_ in records:
            mlflow.log_metric(f'prob_predicted_{r_["class"]}', r_['prob_predicted'])
            mlflow.log_param(f'sample_{r_["class"]}', f'{r_["file"]} -> {r_["predicted"]}')
        mlflow.log_artifacts(str(a.out), artifact_path='gradcam')
        print('MLflow run', run.info.run_id, run.info.experiment_id)
