#!/usr/bin/env python3
"""Aggregate complete development CV, score ablations, and mirror results to W&B."""
import argparse
from pathlib import Path
import pandas as pd
from derms import engine as b
from derms.tracking import Tracker, add_tracking_args, tracking_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    add_tracking_args(parser)
    args = parser.parse_args()
    output = args.output.resolve()
    Tracker.configure(tracking_options(args), output)
    with b.exclusive(output):
        cfg = b.read_json(output / 'config.json')
        for name, sha in cfg['source_hashes'].items():
            if b.file_hash(b.project_root() / name) != sha:
                raise RuntimeError(f'Changed benchmark source: {name}')
        tables = []
        for dataset in cfg['datasets']:
            plan = b.read_json(output / dataset / 'splits.json')
            if b.digest({k: v for k, v in plan.items() if k != 'hash'}) != plan['hash'] or plan['config_hash'] != b.digest(cfg):
                raise RuntimeError('Split/config integrity mismatch')
            df = b.aggregate(dataset, cfg, plan, output)
            for label, overrides in {'no_data': {'gamma': 0}, 'no_complexity': {'beta': 0},
                                     'no_stability': {'delta': 0}, 'accuracy_only': {'beta': 0, 'gamma': 0, 'delta': 0}}.items():
                df[label + '_raw'] = [b.components(r.accuracy_mean, r.params_m, r.metric_ratio,
                    r.accuracy_std, {**cfg, **overrides})['derms_raw'] for r in df.itertuples()]
                df[label + '_rank'] = df[label + '_raw'].rank(method='average', ascending=False)
            df['accuracy_rank_within_ratio'] = df.groupby('ratio').accuracy_mean.rank(method='average', ascending=False)
            tables.append(df)
        summary = pd.concat(tables, ignore_index=True)
        folder = output / 'analysis'
        folder.mkdir(exist_ok=True)
        summary.to_csv(folder / 'derms_summary.csv', index=False)
        for column, filename in [('derms_rank_within_ratio', 'derms_rank_matrix_dataset_ratio.csv'),
                                 ('accuracy_rank_within_ratio', 'accuracy_rank_matrix_dataset_ratio.csv')]:
            summary.pivot(index=['dataset', 'ratio'], columns='model', values=column).to_csv(folder / filename)
        # Per-dataset architecture ranks use a predeclared mean of within-ratio ranks.
        ranks = summary.groupby(['dataset', 'model']).derms_rank_within_ratio.mean().unstack()
        ranks.rank(axis=1, method='average').to_csv(folder / 'derms_rank_matrix_dataset.csv')
        b.atomic_json(folder / 'analysis_metadata.json', {'config_hash': b.digest(cfg),
            'ratio_policy': 'arithmetic mean of fold realized training fractions', 'sd_ddof': 1,
            'normalization': 'per dataset across configured model-ratio candidates',
            'fold_result_hashes': {str(p.relative_to(output)): b.file_hash(p) for p in sorted(output.glob('*/**/fold*/result.json'))},
            'config': cfg})
        with Tracker(b.digest({'analysis': b.digest(cfg)}), 'derms-analysis-' + b.digest(cfg)[:12],
                     'benchmark-analysis', cfg, 'derms-analysis') as tracker:
            tracker.table('DERMS-R Benchmark', summary)
            tracker.artifact(sorted(folder.iterdir()))
        print(f'Analysis saved to {folder}')


if __name__ == '__main__':
    main()
