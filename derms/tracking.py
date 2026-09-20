"""Optional best-effort W&B mirror. Local files are authoritative."""
import json
import warnings
from pathlib import Path


def add_tracking_args(parser):
    import argparse
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb-project', default='derms-r-benchmark')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--wandb-config', type=Path, help='JSON object of additional tracking metadata; cannot override protocol')
    parser.add_argument('--wandb-artifacts', action='store_true', help='Upload best checkpoint artifacts')


def tracking_options(args):
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k.startswith('wandb')}
    extra = json.loads(Path(args.wandb_config).read_text()) if args.wandb_config else {}
    if not isinstance(extra, dict):
        raise ValueError('--wandb-config must contain a JSON object')
    options['extra_metadata'] = extra
    return options


class Tracker:
    options = {}
    directory = Path('.')

    @classmethod
    def configure(cls, options, directory):
        cls.options, cls.directory = options, Path(directory)
        cls.directory.mkdir(parents=True, exist_ok=True)

    def __init__(self, identity, name, dataset, config, job_type='cv-fold'):
        self.run = None
        self.identity, self.name = identity, name
        if not self.options.get('wandb'):
            return
        try:
            import wandb
            self.wandb = wandb
            mode = self.options['wandb_mode']
            self.run = wandb.init(project=self.options['wandb_project'], entity=self.options.get('wandb_entity'),
                id=identity[:32], name=name, group=dataset, job_type=job_type, mode=mode,
                resume='allow' if mode == 'online' else None, dir=str(self.directory),
                config={**config, 'tracking_metadata': self.options.get('extra_metadata', {})})
            self.run.define_metric('epoch')
            self.run.define_metric('train/*', step_metric='epoch')
            self.run.define_metric('val/*', step_metric='epoch')
        except Exception as error:
            warnings.warn(f'W&B unavailable; local results retained: {error}')

    def safe(self, operation):
        if self.run is not None:
            try:
                operation()
            except Exception as error:
                warnings.warn(f'W&B mirror failed; local files remain authoritative: {error}')

    def epoch(self, row):
        values = {'epoch': row['epoch'], 'train/loss': row['train_loss'], 'train/learning_rate': row['lr']}
        values.update({k.replace('val_', 'val/', 1): v for k, v in row.items() if k.startswith('val_')})
        self.safe(lambda: self.run.log(values))

    def summary(self, values):
        self.safe(lambda: self.run.summary.update(values))

    def artifact(self, paths, kind='benchmark-results'):
        def upload():
            artifact = self.wandb.Artifact(self.name, type=kind)
            for path in paths:
                artifact.add_file(str(path), name=Path(path).name)
            self.run.log_artifact(artifact)
        self.safe(upload)

    def checkpoint(self, path):
        if self.options.get('wandb_artifacts'):
            self.artifact([path], 'model')

    def table(self, name, dataframe):
        self.safe(lambda: self.run.log({name: self.wandb.Table(dataframe=dataframe)}))

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.safe(lambda: self.run.finish(exit_code=1 if kind else 0))
        return False
