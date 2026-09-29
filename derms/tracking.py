"""Optional best-effort W&B / MLflow mirrors. Local files are authoritative."""
import json
import os
import warnings
from pathlib import Path


def add_tracking_args(parser):
    import argparse
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb-project', default='derms-r-benchmark')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--wandb-config', type=Path,
        help='JSON object of additional tracking metadata for enabled trackers; cannot override protocol')
    parser.add_argument('--wandb-artifacts', action='store_true', help='Upload best checkpoint artifacts')
    parser.add_argument('--mlflow', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--mlflow-tracking-uri',
        help='Defaults to MLFLOW_TRACKING_URI env var, else a local ./mlruns store under --output')
    parser.add_argument('--mlflow-experiment', default='derms-r-benchmark')
    parser.add_argument('--mlflow-artifacts', action='store_true', help='Upload best checkpoint artifacts')


def tracking_options(args):
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
               if k.startswith('wandb') or k.startswith('mlflow')}
    extra = json.loads(Path(args.wandb_config).read_text()) if args.wandb_config else {}
    if not isinstance(extra, dict):
        raise ValueError('--wandb-config must contain a JSON object')
    options['extra_metadata'] = extra
    return options


def _flatten_for_mlflow(value, prefix=''):
    flat = {}
    if isinstance(value, dict):
        for k, v in value.items():
            flat.update(_flatten_for_mlflow(v, f'{prefix}.{k}' if prefix else str(k)))
    else:
        text = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        flat[prefix[:250]] = text[:250]
    return flat


class Tracker:
    options = {}
    directory = Path('.')

    @classmethod
    def configure(cls, options, directory):
        cls.options, cls.directory = options, Path(directory)
        cls.directory.mkdir(parents=True, exist_ok=True)

    def __init__(self, identity, name, dataset, config, job_type='cv-fold'):
        self.identity, self.name = identity, name
        self.run = None
        self.mlflow = None
        self.mlflow_run = None
        config = {**config, 'tracking_metadata': self.options.get('extra_metadata', {})}
        if self.options.get('wandb'):
            self._init_wandb(identity, name, dataset, config, job_type)
        if self.options.get('mlflow'):
            self._init_mlflow(identity, name, dataset, config, job_type)

    def _init_wandb(self, identity, name, dataset, config, job_type):
        try:
            import wandb
            self.wandb = wandb
            mode = self.options['wandb_mode']
            self.run = wandb.init(project=self.options['wandb_project'], entity=self.options.get('wandb_entity'),
                id=identity[:32], name=name, group=dataset, job_type=job_type, mode=mode,
                resume='allow' if mode == 'online' else None, dir=str(self.directory), config=config)
            self.run.define_metric('epoch')
            self.run.define_metric('train/*', step_metric='epoch')
            self.run.define_metric('val/*', step_metric='epoch')
        except Exception as error:
            warnings.warn(f'W&B unavailable; local results retained: {error}')

    def _init_mlflow(self, identity, name, dataset, config, job_type):
        try:
            import mlflow
            self.mlflow = mlflow
            uri = (self.options.get('mlflow_tracking_uri') or os.environ.get('MLFLOW_TRACKING_URI')
                   or f'file:{(self.directory / "mlruns").resolve()}')
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(self.options.get('mlflow_experiment') or 'derms-r-benchmark')
            self.mlflow_run = mlflow.start_run(run_name=name,
                tags={'dataset': dataset, 'job_type': job_type, 'identity': identity[:32]})
            mlflow.log_params(_flatten_for_mlflow(config))
        except Exception as error:
            warnings.warn(f'MLflow unavailable; local results retained: {error}')
            self.mlflow_run = None

    def _safe(self, backend, operation):
        try:
            operation()
        except Exception as error:
            warnings.warn(f'{backend} mirror failed; local files remain authoritative: {error}')

    def epoch(self, row):
        values = {'epoch': row['epoch'], 'train/loss': row['train_loss'], 'train/learning_rate': row['lr']}
        values.update({k.replace('val_', 'val/', 1): v for k, v in row.items() if k.startswith('val_')})
        if self.run is not None:
            self._safe('W&B', lambda: self.run.log(values))
        if self.mlflow_run is not None:
            metrics = {k: v for k, v in values.items() if k != 'epoch'}
            self._safe('MLflow', lambda: self.mlflow.log_metrics(metrics, step=int(row['epoch'])))

    def summary(self, values):
        if self.run is not None:
            self._safe('W&B', lambda: self.run.summary.update(values))
        if self.mlflow_run is not None:
            numeric = {k: v for k, v in values.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            other = {k: str(v)[:250] for k, v in values.items() if k not in numeric}

            def log_summary():
                if numeric:
                    self.mlflow.log_metrics(numeric)
                if other:
                    self.mlflow.log_params(other)
            self._safe('MLflow', log_summary)

    def _upload_wandb_artifact(self, paths, kind):
        def upload():
            artifact = self.wandb.Artifact(self.name, type=kind)
            for path in paths:
                artifact.add_file(str(path), name=Path(path).name)
            self.run.log_artifact(artifact)
        self._safe('W&B', upload)

    def _upload_mlflow_artifact(self, paths, kind):
        self._safe('MLflow', lambda: [self.mlflow.log_artifact(str(path), artifact_path=kind) for path in paths])

    def artifact(self, paths, kind='benchmark-results'):
        if self.run is not None:
            self._upload_wandb_artifact(paths, kind)
        if self.mlflow_run is not None:
            self._upload_mlflow_artifact(paths, kind)

    def checkpoint(self, path):
        if self.options.get('wandb_artifacts') and self.run is not None:
            self._upload_wandb_artifact([path], 'model')
        if self.options.get('mlflow_artifacts') and self.mlflow_run is not None:
            self._upload_mlflow_artifact([path], 'model')

    def table(self, name, dataframe):
        if self.run is not None:
            self._safe('W&B', lambda: self.run.log({name: self.wandb.Table(dataframe=dataframe)}))
        if self.mlflow_run is not None:
            self._safe('MLflow', lambda: self.mlflow.log_table(data=dataframe, artifact_file=f'{name}.json'))

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.run is not None:
            self._safe('W&B', lambda: self.run.finish(exit_code=1 if kind else 0))
        if self.mlflow_run is not None:
            self._safe('MLflow', lambda: self.mlflow.end_run(status='FAILED' if kind else 'FINISHED'))
        return False
