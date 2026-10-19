#!/usr/bin/env python3
"""One-time untouched-test evaluation of a previously selected/refitted locked model."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torchvision import datasets as tv
from torch.utils.data import Subset
from derms import engine as b
from derms.plots import save_classification_reports


def untouched_test(name, cfg, plan, download):
    root = Path(cfg['data']) / name
    kind = plan['partition']['kind']
    if name != 'all':
        ds = b.official(name, root, test=True, download=download)
        records = b.inventory(ds)
        classes = list(getattr(ds, 'classes', [str(i) for i in sorted(set(b.labels(ds)))]))
        groups = None
    elif kind == 'all_folders':
        test_name = plan['partition'].get('test_name', 'test')
        ds = tv.ImageFolder(root / test_name)
        records = b.inventory(ds)
        classes = list(ds.classes)
        groups = b.resolve_groups(cfg, records, prefix=f'{test_name}/')
    elif kind == 'all_holdout':
        full = tv.ImageFolder(root)
        records_all = b.inventory(full)
        if b.digest(records_all) != plan['partition']['inventory_hash']:
            raise RuntimeError('ALL holdout source changed since development split.')
        ids = plan['partition']['test_indices']
        ds = Subset(full, ids)
        records = [records_all[i] for i in ids]
        classes = list(full.classes)
        all_groups = b.resolve_groups(cfg, records_all)
        groups = [all_groups[i] for i in ids] if all_groups else None
    else:
        raise RuntimeError('Invalid partition mode.')
    if classes != plan['classes']:
        raise RuntimeError('Development/test class mapping mismatch.')
    if set(b.labels(ds)) != set(range(len(classes))):
        raise RuntimeError('Test set must contain every class in the locked mapping.')
    overlap = {r['sha256'] for r in records} & {r['sha256'] for r in plan['records']}
    if overlap:
        raise RuntimeError(f'Data leakage: {len(overlap)} identical images shared by development/test. '
                           'Evaluation aborted; curate a new benchmark, never silently drop test samples.')
    if groups is not None and set(groups) & set(plan['groups']):
        raise RuntimeError('Patient/group leakage between development and test; evaluation aborted.')
    return ds, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', choices=b.DATASETS, required=True)
    p.add_argument('--download', action='store_true')
    b.add_tracking_args(p)
    args = p.parse_args()
    b.Tracker.configure(b.tracking_options(args), args.output)
    output = args.output.resolve()
    with b.exclusive(output):
        cfg = b.read_json(output / 'config.json')
        plan = b.read_json(output / args.dataset / 'splits.json')
        folder = output / args.dataset / 'locked'
        if not (folder / 'lock.json').exists():
            raise RuntimeError('No locked final checkpoint. Run train.py --resume --lock DATASET first.')
        lock = b.read_json(folder / 'lock.json')
        torch.set_num_threads(cfg['threads'])
        for filename, sha in cfg['source_hashes'].items():
            if b.file_hash(Path(__file__).parent / filename) != sha:
                raise RuntimeError(f'Changed benchmark source: {filename}')
        if b.environment() != cfg['environment']:
            raise RuntimeError('Evaluation environment differs from locked benchmark environment.')
        if cfg['all_groups'] and b.file_hash(cfg['all_groups']) != cfg['groups_sha256']:
            raise RuntimeError('Patient/group metadata changed.')
        payload = {k: v for k, v in plan.items() if k != 'hash'}
        if b.digest(payload) != plan['hash'] or plan['hash'] != lock['split_hash']:
            raise RuntimeError('Split manifest integrity mismatch.')
        if b.digest(cfg) != lock['config_hash'] or plan['config_hash'] != lock['config_hash']:
            raise RuntimeError('Locked configuration mismatch.')
        if b.file_hash(folder / 'best.pt') != lock['checkpoint_sha256']:
            raise RuntimeError('Locked checkpoint was modified.')
        selection = lock['selection']
        if b.digest({k: v for k, v in selection.items() if k != 'identity'}) != selection['identity']:
            raise RuntimeError('Selection manifest integrity mismatch.')
        if b.read_json(folder / 'selection.json') != selection or lock['classes'] != plan['classes']:
            raise RuntimeError('Locked selection/class mapping mismatch.')
        if (folder / 'test_result.json').exists():
            result = b.read_json(folder / 'test_result.json')
            if result['lock_hash'] != b.digest(lock):
                raise RuntimeError('Previous test result belongs to another lock.')
            print('Test already evaluated; returning the saved result without reading test data.')
            print(result['metrics'])
            return
        # Verify development fingerprint before first opening the untouched test dataset.
        _, current_plan = b.prepare(args.dataset, cfg, output, False)
        if current_plan != plan:
            raise RuntimeError('Development data changed.')
        ds, records = untouched_test(args.dataset, cfg, plan, args.download)
        test_hash = b.digest(records)
        started_path = folder / 'test_started.json'
        ticket = {'lock_hash': b.digest(lock), 'test_fingerprint': test_hash}
        if started_path.exists() and b.read_json(started_path) != ticket:
            raise RuntimeError('Interrupted evaluation cannot resume with a different lock/test set.')
        b.atomic_json(started_path, ticket)
        b.seed_all(cfg['seed'] + 800000)
        model = b.model_for(selection['winner']['model'], len(plan['classes']), False)
        checkpoint = b.load_checkpoint(folder / 'best.pt')
        if checkpoint['identity'] != selection['identity']:
            raise RuntimeError('Checkpoint selection identity mismatch.')
        model.load_state_dict(checkpoint['model'])
        model.to(cfg['device'])
        result, y, pred, probs = b.evaluate(model, b.loader(ds, list(range(len(ds))), cfg, False,
                                      cfg['seed']), cfg['device'], len(plan['classes']))
        save_classification_reports(folder, y, pred, probs, plan['classes'], stem='test')
        if cfg.get('save_probabilities', True):
            b.save_predictions(model, ds, list(range(len(ds))), cfg, folder / 'test_probabilities.npz')
        b.atomic_json(folder / 'test_predictions.json', {'ids': [r['id'] for r in records],
                                                       'targets': y, 'predictions': pred})
        b.atomic_json(folder / 'test_result.json', {**ticket, 'dataset': args.dataset,
                      'n_test': len(ds), 'metrics': result, 'selection': selection['winner'],
                      'evaluated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        artifacts = [folder / 'test_result.json', folder / 'test_confusion_matrix.png',
                     folder / 'test_roc_curve.png', folder / 'test_classification_report.json']
        run_name = f"{args.dataset}-{selection['winner']['model']}-test"
        with b.Tracker(b.digest(ticket), run_name, args.dataset,
                       {'lock': lock, 'test_fingerprint': test_hash}, 'locked-test') as tracker:
            tracker.summary({k: v for k, v in result.items() if not isinstance(v, (list, dict))})
            tracker.artifact([path for path in artifacts if path.exists()])
        print(result)


if __name__ == '__main__':
    main()
