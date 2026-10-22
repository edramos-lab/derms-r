#!/usr/bin/env python3
"""DERMS-R development-only benchmark. See README.md for protocol and limitations."""
from __future__ import annotations
import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
import timm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import datasets as tv

from .config import *
from .utils import *
from .datasets import *
from .models import *
from .metric import derms_r, metric_kwargs, trainable_parameter_count
from .normalization import add_normalized_column
from .analysis import attach_architecture_optima
from .plots import classification_scores, save_classification_reports
from .tracking import Tracker, add_tracking_args, tracking_options

def metrics(y, pred, loss, nclasses, probs=None):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=list(range(nclasses)), average='macro', zero_division=0)
    result = {'accuracy': float(accuracy_score(y, pred)), 'precision_macro': float(precision),
              'recall_macro': float(recall), 'f1_macro': float(f1),
              'mcc': float(matthews_corrcoef(y, pred)), 'loss': float(loss)}
    if probs is not None:
        scores = classification_scores(y, pred, probs, nclasses)
        result['roc_auc_macro_ovr'] = scores['roc_auc_macro_ovr']
        result['roc_auc_per_class'] = scores['roc_auc_per_class']
        result['confusion_matrix'] = scores['confusion_matrix']
    return result


def evaluate(model, dl, device, nclasses):
    model.eval()
    y, pred, batches, loss, n = [], [], [], 0.0, 0
    with torch.inference_mode():
        for x, target in dl:
            x, target = x.to(device), target.to(device)
            logits = model(x)
            batch_loss = nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(batch_loss):
                raise FloatingPointError('Non-finite evaluation loss.')
            loss += batch_loss.item() * len(target)
            n += len(target)
            y.extend(target.cpu().tolist())
            pred.extend(logits.argmax(1).cpu().tolist())
            batches.append(logits.softmax(1).cpu().numpy())
    if not n:
        raise ValueError('Empty evaluation set.')
    probs = np.concatenate(batches)
    return metrics(y, pred, loss / n, nclasses, probs), y, pred, probs


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(ds, train_ids, val_ids, cfg, name, nclasses, directory, identity, seed, epochs, tracker=None, class_names=None):
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    last = directory / 'last.pt'
    state = load_checkpoint(last) if last.exists() else None
    if state and state['identity'] != identity:
        raise RuntimeError('Checkpoint identity/config mismatch.')
    model = model_for(name, nclasses, cfg['pretrained'] and state is None).to(cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start, best_loss, best_epoch, best_state, best_metrics = 0, math.inf, None, None, None
    history, peak = [], None
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best_loss, best_epoch = state['epoch'], state['best_loss'], state['best_epoch']
        best_state, best_metrics, history = state['best_model'], state['best_metrics'], state['history']
        peak = state['cuda_train_peak_bytes']
    for epoch in range(start, epochs):
        # Epoch-local RNG makes an interrupted epoch replay identically on the same stack/hardware.
        epoch_seed = seed + 10000 + epoch
        seed_all(epoch_seed)
        dl = loader(ds, train_ids, cfg, True, epoch_seed)
        model.train()
        if cfg['device'].startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(cfg['device'])
        total, seen = 0.0, 0
        epoch_started = time.perf_counter()
        progress_every = max(1, len(dl) // 10)
        for batch_index, (x, target) in enumerate(dl, start=1):
            x, target = x.to(cfg['device']), target.to(cfg['device'])
            optimizer.zero_grad(set_to_none=True)
            # Avoid singleton-batch BatchNorm failure without discarding scarce samples.
            frozen = []
            if len(target) == 1:
                for module in model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
                        module.eval()
                        frozen.append(module)
            loss = nn.functional.cross_entropy(model(x), target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss.')
            loss.backward()
            optimizer.step()
            for module in frozen:
                module.train()
            total += loss.item() * len(target)
            seen += len(target)
            if batch_index % progress_every == 0 or batch_index == len(dl):
                elapsed = time.perf_counter() - epoch_started
                print(
                    f'{directory.name} epoch {epoch+1}/{epochs}: '
                    f'train batch {batch_index}/{len(dl)} '
                    f'({100 * batch_index / len(dl):.0f}%, {elapsed:.1f}s)',
                    flush=True,
                )
        if cfg['device'].startswith('cuda'):
            peak = max(peak or 0, torch.cuda.max_memory_allocated(cfg['device']))
        row = {
            'epoch': epoch + 1,
            'train_loss': total / seen,
            'lr': optimizer.param_groups[0]['lr'],
        }
        if val_ids is not None:
            result, _, _, _ = evaluate(model, loader(ds, val_ids, cfg, False, seed), cfg['device'], nclasses)
            row.update({f'val_{k}': v for k, v in result.items() if not isinstance(v, (list, dict))})
            if result['loss'] < best_loss:  # earliest epoch wins exact ties
                best_loss, best_epoch = result['loss'], epoch + 1
                best_state, best_metrics = cpu_state(model), result
        else:
            # Fixed-budget final refit: no validation or test-based checkpoint selection.
            best_loss, best_epoch = total / seen, epoch + 1
            best_state, best_metrics = cpu_state(model), None
        row['epoch_seconds'] = time.perf_counter() - epoch_started
        scheduler.step()
        history.append(row)
        atomic_torch(last, {'identity': identity, 'epoch': epoch + 1, 'model': cpu_state(model),
                           'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                           'best_model': best_state, 'best_loss': best_loss, 'best_epoch': best_epoch,
                           'best_metrics': best_metrics, 'history': history, 'cuda_train_peak_bytes': peak})
        atomic_json(directory / 'history.json', history)
        if tracker:
            tracker.epoch(row)
        print(f'{directory.name} epoch {epoch+1}/{epochs}: {row}', flush=True)
    model.load_state_dict(best_state)
    if val_ids is not None:
        best_metrics, y, pred, probs = evaluate(
            model, loader(ds, val_ids, cfg, False, seed), cfg['device'], nclasses)
        save_classification_reports(directory, y, pred, probs,
                                    class_names or [str(i) for i in range(nclasses)], stem='val')
    atomic_torch(directory / 'best.pt', {'identity': identity, 'model': best_state,
                                       'best_epoch': best_epoch, 'metrics': best_metrics})
    atomic_json(directory / 'history.json', history)
    return model, best_metrics, best_epoch, peak


def profile(model, cfg):
    from fvcore.nn import FlopCountAnalysis
    model.eval()
    device = cfg['device']
    x = torch.zeros(1, 3, 224, 224, device=device)
    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(cfg['warmup']):
            model(x)
        sync()
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(cfg['latency_repeats']):
            sync()
            t = time.perf_counter()
            model(x)
            sync()
            timings.append((time.perf_counter() - t) * 1000)
        peak = torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None
    result = {'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': trainable_parameter_count(model),
              'latency_ms_mean': float(np.mean(timings)), 'latency_ms_median': float(np.median(timings)),
              'latency_ms_p95': float(np.percentile(timings, 95)), 'cuda_inference_peak_bytes': peak,
              'latency_batch_size': 1, 'precision': 'float32', 'input_shape': [1, 3, 224, 224]}
    try:
        with torch.no_grad():
            analysis = FlopCountAnalysis(model, x).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            counted = int(analysis.total())
            unsupported = dict(analysis.unsupported_ops())
        result.update(flops_supported_ops=counted, flops=counted if not unsupported else None,
                      flops_status='partial' if unsupported else 'complete_under_fvcore_convention',
                      flops_unsupported_ops=unsupported)
    except Exception as e:
        result.update(flops=None, flops_supported_ops=None, flops_status='failed',
                      flops_error=f'{type(e).__name__}: {e}')
    if result['flops_status'] != 'complete_under_fvcore_convention':
        warnings.warn(f'FLOPs {result["flops_status"]}; see explicit profiling diagnostics.')
    return result


def run_identity(cfg, plan, name, ratio, fold):
    return digest({'config': cfg, 'split_hash': plan['hash'], 'model': name, 'ratio': ratio, 'fold': fold})


def aggregate(name, cfg, plan, output, log_tracking=True):
    rows, summary = [], []
    params = metric_kwargs(cfg)
    n_folds = int(cfg.get('folds', FOLDS))
    for model in cfg['models']:
        for ratio in RATIOS:
            cell = []
            for fold in range(n_folds):
                directory = output / name / model / f'r{ratio:.2f}' / f'fold{fold}'
                path = directory / 'result.json'
                if not path.exists():
                    raise RuntimeError(f'Incomplete candidate set: {path}')
                row = read_json(path)
                if row['identity'] != run_identity(cfg, plan, model, ratio, fold):
                    raise RuntimeError(f'Stale result: {path}')
                if file_hash(directory / 'best.pt') != row['checkpoint_sha256']:
                    raise RuntimeError(f'Modified checkpoint: {directory}')
                cell.append(row)
            rows.extend(cell)
            fold_accuracies = [r['metrics']['accuracy'] for r in cell]
            num_params = cell[0]['profile']['trainable_parameters']
            # Nominal operating-point ratio r is fixed; realized fractions are recorded, not scored.
            scored = derms_r(fold_accuracies, num_params, ratio, **params)
            s = {'dataset': name, 'architecture': model, 'model': model, 'timm_id': REGISTRY[model],
                 'num_parameters': num_params, 'data_ratio': ratio, 'ratio': ratio,
                 'number_of_folds': n_folds,
                 'realized_ratio_mean': float(np.mean([r['realized_ratio'] for r in cell])),
                 'best_epoch_median': int(np.median([r['best_epoch'] for r in cell]))}
            for k, acc in enumerate(fold_accuracies, 1):
                s[f'fold_{k}_accuracy'] = acc
            s['mean_validation_accuracy'] = scored['mean_accuracy']
            s['sd_validation_accuracy'] = scored['sd_accuracy']
            s['performance_component'] = scored['performance']
            s['data_efficiency_component'] = scored['data_efficiency']
            s['complexity_efficiency_component'] = scored['complexity_efficiency']
            s['stability_component'] = scored['stability']
            s['derms_r_raw'] = scored['derms_r']
            for key in ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'mcc', 'loss', 'roc_auc_macro_ovr'):
                values = [r['metrics'].get(key) for r in cell]
                if any(v is None for v in values):
                    s[f'{key}_mean'], s[f'{key}_std'] = None, None
                else:
                    s[f'{key}_mean'], s[f'{key}_std'] = float(np.mean(values)), float(np.std(values, ddof=1))
            for key in ('latency_ms_mean', 'flops', 'cuda_inference_peak_bytes'):
                values = [r['profile'].get(key) for r in cell]
                s[key + '_mean'] = float(np.mean(values)) if all(v is not None for v in values) else None
            s['realized_ratio_std'] = float(np.std([r['realized_ratio'] for r in cell], ddof=1))
            summary.append(s)
            if log_tracking:
                derms_identity = digest({'config': cfg, 'split_hash': plan['hash'], 'model': model,
                                          'ratio': ratio, 'kind': 'derms-r-summary'})
                with Tracker(derms_identity, f'{name}-{model}-r{ratio:.2f}-derms-r', name,
                             {**cfg, 'dataset': name, 'model': model, 'ratio': ratio},
                             job_type='derms-r-summary') as derms_tracker:
                    # The DERMS-R score and the four terms it's built from (README §score):
                    # performance (mean/sd validation accuracy), data efficiency (ratio),
                    # complexity efficiency (trainable params), and stability (sd accuracy).
                    derms_tracker.summary({
                        'derms_r': s['derms_r_raw'],
                        'mean_validation_accuracy': s['mean_validation_accuracy'],
                        'sd_validation_accuracy': s['sd_validation_accuracy'],
                        'num_parameters': s['num_parameters'],
                        'data_ratio': s['data_ratio'],
                        'performance_component': s['performance_component'],
                        'data_efficiency_component': s['data_efficiency_component'],
                        'complexity_efficiency_component': s['complexity_efficiency_component'],
                        'stability_component': s['stability_component'],
                    })
    df = pd.DataFrame(summary)
    df = add_normalized_column(df)
    df = attach_architecture_optima(df)
    df['derms_rank'] = df.derms_r_raw.rank(method='average', ascending=False)
    df['accuracy_rank'] = df.accuracy_mean.rank(method='average', ascending=False)
    df['derms_rank_within_ratio'] = df.groupby('data_ratio').derms_r_raw.rank(method='average', ascending=False)
    df = df.sort_values(['derms_r_raw', 'model', 'ratio'], ascending=[False, True, True])
    atomic_json(output / name / 'summary.json', df.astype(object).where(pd.notna(df), None).to_dict(orient='records'))
    df.to_csv(output / name / 'summary.csv', index=False)
    pd.json_normalize(rows).to_csv(output / name / 'fold_results.csv', index=False)
    optima = df.drop_duplicates(['dataset', 'architecture'])[
        ['dataset', 'architecture', 'num_parameters', 'optimal_ratio', 'optimal_derms_r']]
    optima.to_csv(output / name / 'architecture_optima.csv', index=False)
    return df


def benchmark(name, ds, plan, cfg, output, download=False):
    nclasses = len(plan['classes'])
    for model in cfg['models']:
        for ratio in RATIOS:
            for fold, split in enumerate(plan['folds']):
                directory = output / name / model / f'r{ratio:.2f}' / f'fold{fold}'
                identity = run_identity(cfg, plan, model, ratio, fold)
                if (directory / 'result.json').exists():
                    result = read_json(directory / 'result.json')
                    if result['identity'] != identity or file_hash(directory / 'best.pt') != result['checkpoint_sha256']:
                        raise RuntimeError(f'Stale or corrupted completed run: {directory}')
                    continue
                indices = split['subsets'][f'{ratio:.2f}']
                with Tracker(identity, f'{name}-{model}-r{ratio:.2f}-f{fold}', name,
                             {**cfg, 'dataset': name, 'model': model, 'ratio': ratio, 'fold': fold,
                              'run_id': identity, 'split_hash': plan['hash'],
                              'realized_ratio': len(indices) / len(split['train'])}) as tracker:
                    trained, result, epoch, peak = fit(ds, indices, split['val'], cfg, model, nclasses,
                        directory, identity, cfg['seed'] + fold * 100000, cfg['epochs'], tracker,
                        class_names=plan['classes'])
                    prof = profile(trained, cfg)
                    if cfg.get('save_probabilities', True):
                        save_predictions(trained, ds, split['val'], cfg, directory / 'predictions.npz')
                    reports = [
                        directory / 'val_confusion_matrix.png',
                        directory / 'val_roc_curve.png',
                        directory / 'val_classification_report.json',
                    ]
                    tracker.artifact([p for p in reports if p.exists()], kind='classification')
                    tracker.summary({**{k: v for k, v in result.items() if not isinstance(v, (list, dict))},
                                    **prof, 'best_epoch': epoch, 'cuda_train_peak_bytes': peak})
                    tracker.checkpoint(directory / 'best.pt')
                atomic_json(directory / 'result.json', {'identity': identity, 'dataset': name, 'model': model,
                    'ratio': ratio, 'fold': fold, 'train_n': len(indices), 'val_n': len(split['val']),
                    'realized_ratio': len(indices) / len(split['train']), 'best_epoch': epoch,
                    'metrics': result, 'profile': prof, 'cuda_train_peak_bytes': peak,
                    'checkpoint_sha256': file_hash(directory / 'best.pt')})
                del trained
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # This architecture's full ratio x fold sweep just finished. Lock and
        # test it immediately, at its own best ratio, rather than waiting for
        # every other architecture to finish so a single overall winner can be
        # picked: with only the winner ever locked and tested, a crash partway
        # through a multi-architecture run (or simply choosing to stop early)
        # leaves no held-out result for any architecture at all, and the full
        # picture across architectures -- not just the eventual top-ranked one
        # -- is exactly what a held-out comparison is for.
        try:
            lock = lock_architecture(name, ds, plan, cfg, output, model)
            test_architecture(name, cfg, plan, output, model, lock, download)
        except Exception as error:
            print(f'{model}: lock/test failed, continuing to the next architecture: '
                  f'{type(error).__name__}: {error}', flush=True)
    return aggregate(name, cfg, plan, output)


def architecture_scores(name, cfg, plan, output, model):
    """Raw DERMS-R at each of the five ratios for one already-fully-trained
    architecture, used to pick that architecture's own best ratio to lock.
    Needs only this architecture's own results, not the other candidates'."""
    params = metric_kwargs(cfg)
    n_folds = int(cfg.get('folds', FOLDS))
    rows = []
    for ratio in RATIOS:
        cell = []
        for fold in range(n_folds):
            directory = output / name / model / f'r{ratio:.2f}' / f'fold{fold}'
            path = directory / 'result.json'
            row = read_json(path)
            if row['identity'] != run_identity(cfg, plan, model, ratio, fold):
                raise RuntimeError(f'Stale result: {path}')
            if file_hash(directory / 'best.pt') != row['checkpoint_sha256']:
                raise RuntimeError(f'Modified checkpoint: {directory}')
            cell.append(row)
        fold_accuracies = [r['metrics']['accuracy'] for r in cell]
        num_params = cell[0]['profile']['trainable_parameters']
        scored = derms_r(fold_accuracies, num_params, ratio, **params)
        rows.append({'ratio': ratio, 'derms_r_raw': scored['derms_r'],
                      'best_epoch_median': int(np.median([r['best_epoch'] for r in cell]))})
    return rows


def lock_architecture(name, ds, plan, cfg, output, model):
    """Refit one architecture at its own best ratio (never filtering by max
    accuracy first) and save the refit checkpoint under its own subdirectory,
    independently of every other architecture's results or ranking."""
    rows = architecture_scores(name, cfg, plan, output, model)
    winner = max(rows, key=lambda r: r['derms_r_raw'])
    directory = output / name / 'locked' / model
    directory.mkdir(parents=True, exist_ok=True)
    selection = {'config_hash': digest(cfg), 'split_hash': plan['hash'], 'model': model,
                 'ratio': winner['ratio'], 'derms_r_raw': winner['derms_r_raw'],
                 'final_realized_ratio': len(plan['final_subsets'][f"{winner['ratio']:.2f}"]) / len(ds),
                 'epochs': winner['best_epoch_median'],
                 'policy': 'max_derms_r_own_architecture',
                 'training_indices': plan['final_subsets'][f"{winner['ratio']:.2f}"]}
    selection['identity'] = digest(selection)
    selection_path = directory / 'selection.json'
    if selection_path.exists() and read_json(selection_path) != selection:
        raise RuntimeError('Locked selection mismatch.')
    atomic_json(selection_path, selection)
    lock_path = directory / 'lock.json'
    if lock_path.exists():
        lock = read_json(lock_path)
        if lock['selection'] != selection or file_hash(directory / 'best.pt') != lock['checkpoint_sha256']:
            raise RuntimeError('Existing locked artifact changed.')
        print(f'{model}: already locked at r={winner["ratio"]:.2f}.', flush=True)
        return lock
    trained, _, _, _ = fit(ds, selection['training_indices'], None, cfg, model,
                          len(plan['classes']), directory, selection['identity'], cfg['seed'] + 700000,
                          selection['epochs'])
    del trained
    lock = {'schema': SCHEMA, 'dataset': name, 'selection': selection,
            'classes': plan['classes'], 'checkpoint_sha256': file_hash(directory / 'best.pt'),
            'config_hash': digest(cfg), 'split_hash': plan['hash'],
            'locked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    atomic_json(lock_path, lock)
    print(f'{model}: locked at r={winner["ratio"]:.2f} (own best; raw DERMS-R={winner["derms_r_raw"]:.4f}).',
          flush=True)
    return lock


def untouched_test_data(name, cfg, plan, download):
    """The official, never-trained-on test partition for one dataset, with
    the same leakage checks test.py has always applied before first opening it."""
    root = Path(cfg['data']) / name
    kind = plan['partition']['kind']
    if name != 'all':
        ds = official(name, root, test=True, download=download)
        records = inventory(ds)
        classes = list(getattr(ds, 'classes', [str(i) for i in sorted(set(labels(ds)))]))
        groups = None
    elif kind == 'all_folders':
        test_name = plan['partition'].get('test_name', 'test')
        ds = tv.ImageFolder(root / test_name)
        records = inventory(ds)
        classes = list(ds.classes)
        groups = resolve_groups(cfg, records, prefix=f'{test_name}/')
    elif kind == 'all_holdout':
        full = tv.ImageFolder(root)
        records_all = inventory(full)
        if digest(records_all) != plan['partition']['inventory_hash']:
            raise RuntimeError('ALL holdout source changed since development split.')
        ids = plan['partition']['test_indices']
        ds = Subset(full, ids)
        records = [records_all[i] for i in ids]
        classes = list(full.classes)
        all_groups = resolve_groups(cfg, records_all)
        groups = [all_groups[i] for i in ids] if all_groups else None
    else:
        raise RuntimeError('Invalid partition mode.')
    if classes != plan['classes']:
        raise RuntimeError('Development/test class mapping mismatch.')
    if set(labels(ds)) != set(range(len(classes))):
        raise RuntimeError('Test set must contain every class in the locked mapping.')
    overlap = {r['sha256'] for r in records} & {r['sha256'] for r in plan['records']}
    if overlap:
        raise RuntimeError(f'Data leakage: {len(overlap)} identical images shared by development/test. '
                           'Evaluation aborted; curate a new benchmark, never silently drop test samples.')
    if groups is not None and set(groups) & set(plan['groups']):
        raise RuntimeError('Patient/group leakage between development and test; evaluation aborted.')
    return ds, records


def test_architecture(name, cfg, plan, output, model, lock, download=False):
    """One-time untouched-test evaluation of one architecture's locked
    checkpoint. Safe to call repeatedly: returns the saved result instead of
    re-reading the test set if this exact lock was already evaluated."""
    directory = output / name / 'locked' / model
    if (directory / 'test_result.json').exists():
        result = read_json(directory / 'test_result.json')
        if result['lock_hash'] != digest(lock):
            raise RuntimeError('Previous test result belongs to another lock.')
        print(f'{model}: test already evaluated (accuracy={result["metrics"]["accuracy"]:.4f}).', flush=True)
        return result
    selection = lock['selection']
    ds, records = untouched_test_data(name, cfg, plan, download)
    test_hash = digest(records)
    started_path = directory / 'test_started.json'
    ticket = {'lock_hash': digest(lock), 'test_fingerprint': test_hash}
    if started_path.exists() and read_json(started_path) != ticket:
        raise RuntimeError('Interrupted evaluation cannot resume with a different lock/test set.')
    atomic_json(started_path, ticket)
    seed_all(cfg['seed'] + 800000)
    net = model_for(model, len(plan['classes']), False)
    checkpoint = load_checkpoint(directory / 'best.pt')
    if checkpoint['identity'] != selection['identity']:
        raise RuntimeError('Checkpoint selection identity mismatch.')
    net.load_state_dict(checkpoint['model'])
    net.to(cfg['device'])
    result, y, pred, probs = evaluate(net, loader(ds, list(range(len(ds))), cfg, False, cfg['seed']),
                                      cfg['device'], len(plan['classes']))
    save_classification_reports(directory, y, pred, probs, plan['classes'], stem='test')
    if cfg.get('save_probabilities', True):
        save_predictions(net, ds, list(range(len(ds))), cfg, directory / 'test_probabilities.npz')
    atomic_json(directory / 'test_predictions.json', {'ids': [r['id'] for r in records],
                                                       'targets': y, 'predictions': pred})
    test_result = {**ticket, 'dataset': name, 'model': model, 'n_test': len(ds), 'metrics': result,
                   'selection': selection, 'evaluated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    atomic_json(directory / 'test_result.json', test_result)
    artifacts = [directory / 'test_result.json', directory / 'test_confusion_matrix.png',
                 directory / 'test_roc_curve.png', directory / 'test_classification_report.json']
    with Tracker(digest(ticket), f'{name}-{model}-test', name,
                 {'lock': lock, 'test_fingerprint': test_hash}, 'locked-test') as tracker:
        tracker.summary({k: v for k, v in result.items() if not isinstance(v, (list, dict))})
        tracker.artifact([p for p in artifacts if p.exists()])
    print(f'{model}: test accuracy {result["accuracy"]:.4f} (n={len(ds)}).', flush=True)
    del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_result


def lock_selected(name, ds, plan, cfg, output):
    """Legacy global-winner lock: refits only the single best (architecture,
    ratio) pair across the whole candidate set, at the old single-directory
    path `locked/`. Superseded by benchmark()'s automatic per-architecture
    lock_architecture()/test_architecture() calls, which now lock and test
    every architecture at its own best ratio as soon as that architecture's
    training finishes, not only the eventual overall winner. Kept only for
    backward compatibility with `--lock DATASET` / orchestrator jobs that
    still invoke it explicitly; by the time it runs, every architecture
    already has its own locked+tested checkpoint under `locked/<model>/`."""
    # log_tracking=False: this re-aggregate is only to pick the winner -- the
    # DERMS-R summary for every (model, ratio) was already logged once when
    # benchmark() produced these results; re-logging here would duplicate it.
    df = aggregate(name, cfg, plan, output, log_tracking=False)
    # Select after scoring every architecture × ratio. Never filter by max accuracy first.
    winner = df.astype(object).where(pd.notna(df), None).iloc[0].to_dict()
    directory = output / name / 'locked'
    directory.mkdir(exist_ok=True)
    selection = {'config_hash': digest(cfg), 'split_hash': plan['hash'], 'winner': winner,
                 'final_realized_ratio': len(plan['final_subsets'][f'{winner["ratio"]:.2f}']) / len(ds),
                 'epochs': int(winner['best_epoch_median']),
                 'policy': 'max_derms_r_then_model_then_ratio',
                 'training_indices': plan['final_subsets'][f'{winner["ratio"]:.2f}']}
    selection['identity'] = digest(selection)
    selection_path = directory / 'selection.json'
    if selection_path.exists() and read_json(selection_path) != selection:
        raise RuntimeError('Locked selection mismatch.')
    atomic_json(selection_path, selection)
    lock_path = directory / 'lock.json'
    if lock_path.exists():
        lock = read_json(lock_path)
        if lock['selection'] != selection or file_hash(directory / 'best.pt') != lock['checkpoint_sha256']:
            raise RuntimeError('Existing locked artifact changed.')
        print(f'Already locked: {lock_path}')
        return
    trained, _, _, _ = fit(ds, selection['training_indices'], None, cfg, winner['model'],
                          len(plan['classes']), directory, selection['identity'], cfg['seed'] + 700000,
                          selection['epochs'])
    del trained
    atomic_json(lock_path, {'schema': SCHEMA, 'dataset': name, 'selection': selection,
                'classes': plan['classes'], 'checkpoint_sha256': file_hash(directory / 'best.pt'),
                'config_hash': digest(cfg), 'split_hash': plan['hash'],
                'locked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print(f'Final refit locked. Evaluate only with test.py --output {output} --dataset {name}', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('datasets'))
    p.add_argument('--output', type=Path, default=Path('results'))
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--models', nargs='+', choices=REGISTRY, default=list(REGISTRY))
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--holdout', type=float, default=0.20)
    p.add_argument('--all-groups', type=Path)
    p.add_argument('--all-independent-images', action='store_true')
    p.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--augmentation', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--flip-probability', type=float, default=0.5)
    p.add_argument('--rotation', type=float, default=20.0)
    p.add_argument('--rotation-probability', type=float, default=1.0)
    p.add_argument('--hsv-probability', type=float, default=0.8)
    p.add_argument('--lab-probability', type=float, default=0.8)
    p.add_argument('--brightness-contrast-probability', type=float, default=0.8)
    p.add_argument('--gauss-noise-probability', type=float, default=0.15)
    p.add_argument('--blur-probability', type=float, default=0.15)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--latency-repeats', type=int, default=50)
    p.add_argument('--alpha', type=float, default=ALPHA)
    p.add_argument('--beta', type=float, default=BETA)
    p.add_argument('--gamma', type=float, default=GAMMA)
    p.add_argument('--delta', type=float, default=DELTA)
    p.add_argument('--p0', type=float, default=P0)
    p.add_argument('--sigma0', type=float, default=SIGMA0)
    p.add_argument('--download', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--verify-models', action='store_true', help='verify installed IDs and print report; no data access')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--lock', choices=DATASETS, help='select from completed CV, refit and lock; requires --resume')
    add_tracking_args(p)
    p.add_argument('--save-probabilities', action=argparse.BooleanOptionalAction, default=True)
    return p


def main():
    args = parser().parse_args()
    tracking = tracking_options(args)
    if args.verify_models:
        print(json.dumps({'timm': timm.__version__, 'models': verify_registry(args.models)}, indent=2))
        return
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0 or args.threads < 1:
        raise ValueError('epochs/threads >=1, batch-size >=2 and workers >=0 required.')
    if not 0 < args.holdout < 1 or args.warmup < 1 or args.latency_repeats < 2:
        raise ValueError('Invalid holdout/profiling configuration.')
    if args.device != 'cpu' and not args.device.startswith('cuda'):
        raise ValueError('Use cpu or cuda[:index]; MPS is excluded from this reproducibility protocol.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in
           ('lr', 'alpha', 'beta', 'gamma', 'delta', 'p0', 'sigma0')):
        raise ValueError('lr and score parameters must be finite and strictly positive.')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.rotation <= 180:
        raise ValueError('Invalid weight decay or rotation.')
    if not all(0 <= x <= 1 for x in (args.flip_probability, args.rotation_probability,
                                     args.hsv_probability, args.lab_probability,
                                     args.brightness_contrast_probability,
                                     args.gauss_noise_probability, args.blur_probability)):
        raise ValueError('Augmentation probabilities must be in [0,1].')
    if len(set(args.models)) != len(args.models) or len(set(args.datasets)) != len(args.datasets):
        raise ValueError('Duplicate dataset/model selections.')
    if args.lock and (not args.resume or args.lock not in args.datasets):
        raise ValueError('--lock requires --resume and inclusion in --datasets.')
    if args.seed < 0 or args.seed > 2**32 - 1000000 - args.epochs:
        raise ValueError('Seed outside supported range.')
    torch.set_num_threads(args.threads)
    cfg = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
           if not k.startswith(('wandb', 'mlflow'))
           and k not in ('output', 'download', 'resume', 'verify_models', 'prepare_only', 'lock')}
    cfg.update(schema=SCHEMA, ratios=list(RATIOS), folds=FOLDS, image_size=224,
               normalization={'mean': MEAN, 'std': STD}, precision='float32',
               source_hashes={n: file_hash(project_root() / n) for n in source_files()},
               groups_sha256=file_hash(args.all_groups) if args.all_groups else None,
               environment=environment(), registry=verify_registry(args.models))
    cfg = json.loads(json.dumps(cfg))
    output = args.output.resolve()
    Tracker.configure(tracking, output)
    with exclusive(output):
        path = output / 'config.json'
        if path.exists():
            if not args.resume:
                raise RuntimeError('Output already exists; use --resume with identical configuration.')
            old = read_json(path)
            if old != cfg:
                changed = [k for k in cfg if old.get(k) != cfg[k]]
                raise RuntimeError(f'Configuration/environment/code mismatch: {changed}. Use a new output directory.')
        else:
            if args.resume:
                raise RuntimeError('Cannot resume: config.json does not exist.')
            if any(x.name != '.writer.lock' for x in output.iterdir()):
                raise RuntimeError('New output directory must be empty.')
            atomic_json(path, cfg)
        for name in ([args.lock] if args.lock else args.datasets):
            ds, plan = prepare(name, cfg, output, args.download)
            if args.lock:
                lock_selected(name, ds, plan, cfg, output)
            elif not args.prepare_only:
                benchmark(name, ds, plan, cfg, output, download=args.download)



def project_root():
    return Path(__file__).resolve().parent.parent


def source_files():
    root = project_root()
    return ['train.py', 'analyze.py', 'test.py', 'prepare_dataset.py'] + sorted(p.relative_to(root).as_posix() for p in (root / 'derms').glob('*.py'))


def save_predictions(model, ds, indices, cfg, path):
    """Compressed float32 probabilities, aligned with stable dataset sample IDs."""
    model.eval()
    probabilities, targets = [], []
    with torch.inference_mode():
        for x, y in loader(ds, indices, cfg, False, cfg['seed']):
            probabilities.append(model(x.to(cfg['device'])).softmax(1).cpu().numpy())
            targets.extend(y.tolist())
    p = np.concatenate(probabilities)
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, ids=np.asarray([sample_id(ds, i) for i in indices]),
                            targets=np.asarray(targets), predictions=p.argmax(1), probabilities=p)
    os.replace(temporary, path)
