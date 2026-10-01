"""Build and launch train.py containers that can reach a separate MLflow container."""
import importlib.util
import json
import math
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _load_protocol():
    """Load canonical constants without importing derms/__init__.py (that pulls in torch)."""
    path = Path(__file__).resolve().parents[1] / "derms" / "config.py"
    spec = importlib.util.spec_from_file_location("derms_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROTOCOL = _load_protocol()
DATASETS = _PROTOCOL.DATASETS
FOLDS = _PROTOCOL.FOLDS
RATIOS = _PROTOCOL.RATIOS
REGISTRY = _PROTOCOL.REGISTRY

MAX_COMBINATIONS = 12
EXCLUDED_NETWORKS = {"bridge", "host", "none"}

MODEL_LABELS = {
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnetv2_s": "EfficientNetV2-S",
    "convnextv2_tiny": "ConvNeXtV2-Tiny",
    "xception41": "Xception41",
    "vit_base": "ViT-Base",
    "deit3_base": "DeiT3-Base",
    "swin_tiny": "Swin-Tiny",
    "maxvit_tiny": "MaxViT-Tiny",
    "nextvit_base": "NextViT-Base",
    "efficientvit_b0": "EfficientViT-B0",
    "mobilenetv4": "MobileNetV4",
}
DEFAULT_MODELS = (
    "resnet50",
    "efficientnet_b0",
    "efficientnetv2_s",
    "convnextv2_tiny",
    "vit_base",
    "swin_tiny",
)
DATASET_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "flowers102": 102,
    "food101": 101,
    "all": None,
}


class ConfigError(ValueError):
    """The training form cannot be turned into a train.py invocation."""


class DockerError(RuntimeError):
    """The Docker daemon or a docker command failed."""


@dataclass
class Settings:
    image: str
    in_docker: bool
    visible_datasets: Path
    visible_results: Path
    jobs_dir: Path
    host_datasets: Path | None = None
    host_results: Path | None = None
    mlflow_container: str | None = None
    network_name: str = "derms-r"
    paths_resolved: bool = False
    hf_token: str | None = None
    host_hf_cache: Path | None = None

    @classmethod
    def from_env(cls):
        root = Path(__file__).resolve().parents[1]
        in_docker = os.environ.get("ORCHESTRATOR_IN_DOCKER") == "1"
        visible_datasets = Path(os.environ.get("SCAN_DATASETS", root / "datasets"))
        visible_results = Path(os.environ.get("SCAN_RESULTS", root / "results"))
        return cls(
            image=os.environ.get("DERMS_IMAGE", "derms-r-train:latest"),
            in_docker=in_docker,
            visible_datasets=visible_datasets,
            visible_results=visible_results,
            jobs_dir=Path(os.environ.get("JOBS_DIR", visible_results / "orchestrator")),
            host_datasets=Path(os.environ["HOST_DATASETS"]) if os.environ.get("HOST_DATASETS") else None,
            host_results=Path(os.environ["HOST_RESULTS"]) if os.environ.get("HOST_RESULTS") else None,
            mlflow_container=os.environ.get("MLFLOW_CONTAINER") or None,
            network_name=os.environ.get("DERMS_NETWORK", "derms-r"),
            paths_resolved=not in_docker,
            hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
            host_hf_cache=Path(os.environ["HOST_HF_CACHE"]) if os.environ.get("HOST_HF_CACHE") else None,
        )


@dataclass
class MLflowPlan:
    container: str
    tracking_uri: str
    network: str
    connect: bool
    host_url: str | None
    internal_port: int

    def public(self):
        return {
            "ok": True,
            "container": self.container,
            "tracking_uri": self.tracking_uri,
            "network": self.network,
            "connect": self.connect,
            "host_url": self.host_url,
            "error": None,
        }


@dataclass
class RunSpec:
    action: str
    datasets: list
    models: list
    device: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    holdout: float
    augmentation: bool
    pretrained: bool
    experiment: str
    artifacts: bool
    notify_email: str
    download: bool = False
    label: str = ""


def fmt_float(value):
    return format(float(value), ".12g")


def canonical_ratios():
    return [float(r) for r in RATIOS]


def model_catalog():
    return [
        {"id": name, "label": MODEL_LABELS.get(name, name), "default": name in DEFAULT_MODELS}
        for name in REGISTRY
    ]


def dataset_catalog(root: Path):
    rows = []
    for name in DATASETS:
        path = root / name
        present = path.is_dir() and any(path.iterdir())
        rows.append({
            "id": name,
            "classes": DATASET_CLASSES.get(name),
            "present": present,
            "container_path": f"/workspace/datasets/{name}",
            "default": name in {"cifar10", "cifar100"},
        })
    return rows


def _as_number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a number.") from None
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite.")
    return number


def linear_series(spec, name, *, integer):
    lo = _as_number(spec.get("min"), f"{name} min")
    hi = _as_number(spec.get("max"), f"{name} max")
    if lo > hi:
        raise ConfigError(f"{name} min cannot exceed max.")
    if math.isclose(lo, hi):
        return [int(round(lo)) if integer else lo]
    step = _as_number(spec.get("step"), f"{name} step")
    if step <= 0:
        raise ConfigError(f"{name} step must be positive.")
    if integer:
        lo_i, hi_i, step_i = int(round(lo)), int(round(hi)), int(round(step))
        if step_i < 1:
            raise ConfigError(f"{name} step must be a positive integer.")
        values = list(range(lo_i, hi_i + 1, step_i))
        if not values:
            raise ConfigError(f"{name} range is empty.")
        return values
    values = []
    cursor = lo
    for _ in range(64):
        if cursor > hi + step * 1e-9:
            break
        values.append(cursor)
        cursor += step
    else:
        raise ConfigError(f"{name} range has too many steps.")
    return values


def log_series(spec, name):
    lo = _as_number(spec.get("min"), f"{name} min")
    hi = _as_number(spec.get("max"), f"{name} max")
    if lo <= 0 or hi <= 0:
        raise ConfigError(f"{name} log range must be strictly positive.")
    if lo > hi:
        raise ConfigError(f"{name} min cannot exceed max.")
    values = []
    cursor = lo
    for _ in range(12):
        values.append(cursor)
        nxt = cursor * 10
        if nxt > hi * (1 + 1e-9):
            break
        cursor = nxt
    else:
        raise ConfigError(f"{name} log range is too wide.")
    if not math.isclose(values[-1], hi, rel_tol=1e-6, abs_tol=0):
        values.append(hi)
    return values


def series(spec, name, *, integer=False):
    if not isinstance(spec, dict):
        raise ConfigError(f"{name} range must be an object.")
    scale = spec.get("scale", "linear")
    if scale == "log":
        if integer:
            raise ConfigError(f"{name} cannot use a log scale.")
        return log_series(spec, name)
    if scale != "linear":
        raise ConfigError(f"{name} scale must be linear or log.")
    return linear_series(spec, name, integer=integer)


def _check_ratios(payload):
    ratios = payload.get("ratios", canonical_ratios())
    if not isinstance(ratios, list) or len(ratios) != len(RATIOS):
        raise ConfigError("DERMS-R training uses the canonical ratios 0.10, 0.25, 0.50, 0.75, and 1.00.")
    for got, expected in zip(ratios, RATIOS):
        if not math.isclose(_as_number(got, "ratio"), expected, rel_tol=0, abs_tol=1e-9):
            raise ConfigError("DERMS-R training uses the canonical ratios 0.10, 0.25, 0.50, 0.75, and 1.00.")


def _check_names(payload):
    datasets = payload.get("datasets")
    models = payload.get("models")
    if not isinstance(datasets, list) or not datasets:
        raise ConfigError("Select at least one dataset.")
    if not isinstance(models, list) or not models:
        raise ConfigError("Select at least one architecture.")
    unknown_data = [name for name in datasets if name not in DATASETS]
    unknown_models = [name for name in models if name not in REGISTRY]
    if unknown_data or unknown_models:
        raise ConfigError(f"Unknown dataset or architecture: {unknown_data + unknown_models}.")
    if len(set(datasets)) != len(datasets) or len(set(models)) != len(models):
        raise ConfigError("Duplicate dataset or architecture selections.")
    return list(datasets), list(models)


def _shared(payload, datasets, models):
    if payload.get("deployment", "single") != "single":
        raise ConfigError("Cluster mode is not available. Use Single Machine.")
    folds = payload.get("folds", FOLDS)
    if int(folds) != FOLDS:
        raise ConfigError(f"K-folds is fixed at {FOLDS} by the DERMS-R protocol.")
    _check_ratios(payload)
    device = str(payload.get("device") or "cpu")
    if device != "cpu" and not device.startswith("cuda"):
        raise ConfigError("Device must be cpu or cuda[:index].")
    if device.startswith("cuda"):
        index = device.split(":", 1)[1] if ":" in device else "0"
        if index == "":
            device = "cuda:0"
        elif not index.isdigit():
            raise ConfigError("CUDA device index must be a number.")
        else:
            device = f"cuda:{index}"
    seed = int(_as_number(payload.get("seed", 42), "seed"))
    holdout = _as_number(payload.get("holdout", 0.20), "holdout")
    if seed < 0 or not 0 < holdout < 1:
        raise ConfigError("Seed must be non-negative and holdout must be between 0 and 1.")
    experiment = str(payload.get("mlflow_experiment") or "derms-r-benchmark").strip()
    if not experiment:
        raise ConfigError("MLflow experiment name cannot be empty.")
    if any(ord(c) < 0x20 for c in experiment):
        raise ConfigError("MLflow experiment name cannot contain control characters.")
    email = str(payload.get("notify_email") or "").strip()
    return {
        "datasets": datasets,
        "models": models,
        "device": device,
        "seed": seed,
        "holdout": holdout,
        "augmentation": bool(payload.get("augmentation", True)),
        "pretrained": bool(payload.get("pretrained", True)),
        "experiment": experiment,
        "artifacts": bool(payload.get("mlflow_artifacts", False)),
        "notify_email": email,
    }


def _fixed_values(payload):
    epochs = int(_as_number(payload.get("epochs", 50), "epochs"))
    batch = int(_as_number(payload.get("batch_size", 32), "batch size"))
    lr = _as_number(payload.get("lr", 1e-4), "learning rate")
    decay = _as_number(payload.get("weight_decay", 0.01), "weight decay")
    if epochs < 1 or batch < 2 or lr <= 0 or decay < 0:
        raise ConfigError("Epochs must be >= 1, batch size >= 2, learning rate > 0, and weight decay >= 0.")
    return epochs, batch, lr, decay


def parse_runs(payload):
    if not isinstance(payload, dict):
        raise ConfigError("Request body must be a JSON object.")
    action = payload.get("action", "train")
    if action not in {"train", "prepare"}:
        raise ConfigError("Action must be train or prepare.")
    datasets, models = _check_names(payload)
    shared = _shared(payload, datasets, models)
    mode = payload.get("mode", "fixed")
    if action == "prepare" or mode == "fixed":
        epochs, batch, lr, decay = _fixed_values(payload)
        run = RunSpec(action=action, epochs=epochs, batch_size=batch, lr=lr, weight_decay=decay,
                      download=action == "prepare", **shared)
        run.label = "prepare" if action == "prepare" else "train"
        return [run]
    if mode != "range":
        raise ConfigError("Mode must be fixed or range.")
    ranges = payload.get("ranges")
    if not isinstance(ranges, dict):
        raise ConfigError("Range mode requires a ranges object.")
    batches = series(ranges.get("batch_size"), "batch size", integer=True)
    epochs_list = series(ranges.get("epochs"), "epochs", integer=True)
    lrs = series(ranges.get("lr"), "learning rate")
    decays = series(ranges.get("weight_decay"), "weight decay")
    runs = []
    for batch in batches:
        for epochs in epochs_list:
            for lr in lrs:
                for decay in decays:
                    if int(epochs) < 1 or int(batch) < 2 or float(lr) <= 0 or float(decay) < 0:
                        raise ConfigError("A range includes an invalid epoch, batch size, learning rate, or weight decay.")
                    run = RunSpec(
                        action="train", epochs=int(epochs), batch_size=int(batch), lr=float(lr),
                        weight_decay=float(decay), download=False, **shared)
                    run.label = f"bs{batch}-ep{epochs}-lr{fmt_float(lr)}-wd{fmt_float(decay)}"
                    runs.append(run)
    if len(runs) > MAX_COMBINATIONS:
        raise ConfigError(
            f"Range mode would start {len(runs)} containers (limit {MAX_COMBINATIONS}). "
            "Narrow the ranges or use Fixed mode.")
    if not runs:
        raise ConfigError("Range mode produced no runs.")
    return runs


def gpu_args(device):
    if device == "cpu":
        return []
    index = device.split(":", 1)[1]
    return ["--gpus", f"device={index}"]


def train_args(run: RunSpec, tracking_uri: str | None):
    args = [
        "train.py",
        "--data", "/workspace/datasets",
        "--output", "/workspace/results",
        "--datasets", *run.datasets,
        "--models", *run.models,
        "--epochs", str(run.epochs),
        "--batch-size", str(run.batch_size),
        "--lr", fmt_float(run.lr),
        "--weight-decay", fmt_float(run.weight_decay),
        "--seed", str(run.seed),
        "--holdout", fmt_float(run.holdout),
        "--device", run.device,
    ]
    if not run.augmentation:
        args.append("--no-augmentation")
    if not run.pretrained:
        args.append("--no-pretrained")
    if run.action == "prepare":
        args.extend(["--prepare-only", "--download"])
    elif tracking_uri:
        args.extend([
            "--mlflow",
            "--mlflow-tracking-uri", tracking_uri,
            "--mlflow-experiment", run.experiment,
        ])
        if run.artifacts:
            args.append("--mlflow-artifacts")
    return args


def container_argv(run: RunSpec, settings: Settings, plan: MLflowPlan | None, job_id: str):
    if settings.host_datasets is None or settings.host_results is None:
        raise DockerError("Host dataset and results paths are not resolved.")
    output = settings.host_results / "runs" / job_id
    tracking = None if run.action == "prepare" or plan is None else plan.tracking_uri
    command = [
        "run", "-d",
        "--name", f"derms-train-{job_id}",
        "--label", f"derms.job={job_id}",
        "--label", "derms.role=train",
        "--shm-size", "2g",
        "-v", f"{settings.host_datasets}:/workspace/datasets",
        "-v", f"{output}:/workspace/results",
    ]
    if plan is not None and run.action == "train":
        command.extend(["--network", plan.network, "-e", f"MLFLOW_TRACKING_URI={plan.tracking_uri}"])
    if settings.host_hf_cache is not None:
        settings.host_hf_cache.mkdir(parents=True, exist_ok=True)
        command.extend([
            "-v", f"{settings.host_hf_cache}:/root/.cache/huggingface",
            "-e", "HF_HOME=/root/.cache/huggingface",
        ])
    if settings.hf_token:
        # huggingface_hub reads HUGGING_FACE_HUB_TOKEN; timm downloads weights through that API.
        command.extend([
            "-e", f"HF_TOKEN={settings.hf_token}",
            "-e", f"HUGGING_FACE_HUB_TOKEN={settings.hf_token}",
        ])
    command.extend(gpu_args(run.device))
    command.append(settings.image)
    command.extend(train_args(run, tracking))
    return command


def allocate_output(settings: Settings, job_id: str):
    visible = settings.visible_results / "runs" / job_id
    visible.mkdir(parents=True, exist_ok=True)
    if any(visible.iterdir()):
        raise DockerError(f"Output directory {visible} is not empty.")
    return visible


SECRET_ENV_PREFIXES = ("HF_TOKEN=", "HUGGING_FACE_HUB_TOKEN=")


def _redact(part):
    for prefix in SECRET_ENV_PREFIXES:
        if part.startswith(prefix):
            return f"{prefix}***"
    return part


def command_preview(argv):
    # Stored in jobs.json and shown in the UI (Configuration/Dashboard/Audit
    # trail) verbatim, so any injected secret env vars must be masked here —
    # this is a display/audit string, not what's actually passed to `docker run`.
    return "docker " + " ".join(shlex.quote(_redact(part)) for part in argv)


def _published_port(bindings, internal):
    if not bindings:
        return None
    for binding in bindings:
        if binding and binding.get("HostPort"):
            return int(binding["HostPort"]), internal
    return None


def endpoint_from_inspect(name, inspect):
    ports = (inspect.get("NetworkSettings") or {}).get("Ports") or {}
    chosen_internal = None
    host_port = None
    for key, bindings in ports.items():
        internal = int(str(key).split("/")[0])
        published = _published_port(bindings, internal)
        if internal == 5000 or (published and published[0] == 5000):
            chosen_internal = internal
            host_port = published[0] if published else None
            break
        if chosen_internal is None:
            chosen_internal = internal
            host_port = published[0] if published else None
    if chosen_internal is None:
        exposed = (inspect.get("Config") or {}).get("ExposedPorts") or {}
        if "5000/tcp" in exposed:
            chosen_internal = 5000
        else:
            chosen_internal = 5000
    networks = list(((inspect.get("NetworkSettings") or {}).get("Networks") or {}).keys())
    user = [net for net in networks if net not in EXCLUDED_NETWORKS]
    preferred = next((net for net in user if "mlflow" in net.lower()), None)
    network = preferred or (user[0] if user else None)
    container = name[1:] if name.startswith("/") else name
    return {
        "container": container,
        "internal_port": chosen_internal,
        "host_port": host_port,
        "user_network": network,
    }


def plan_mlflow(endpoint, network_name):
    container = endpoint["container"]
    port = int(endpoint["internal_port"])
    if endpoint.get("user_network"):
        network = endpoint["user_network"]
        connect = False
    else:
        network = network_name
        connect = True
    host_url = f"http://localhost:{endpoint['host_port']}" if endpoint.get("host_port") else None
    return MLflowPlan(
        container=container,
        tracking_uri=f"http://{container}:{port}",
        network=network,
        connect=connect,
        host_url=host_url,
        internal_port=port,
    )


def parse_ps(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name, image = line.split("\t", 1)
        rows.append({"name": name, "image": image})
    return rows


def choose_mlflow(rows, preferred):
    if preferred:
        match = [row for row in rows if row["name"] == preferred]
        if not match:
            raise DockerError(f"MLFLOW_CONTAINER={preferred} is not a running container.")
        return match[0]
    matches = [row for row in rows if "mlflow" in row["name"].lower() or "mlflow" in row["image"].lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise DockerError(
            "No running MLflow container. Start MLflow, or set MLFLOW_CONTAINER to its name.")
    names = ", ".join(row["name"] for row in matches)
    raise DockerError(f"Multiple MLflow containers ({names}). Set MLFLOW_CONTAINER to the one training should use.")


@dataclass
class Completed:
    returncode: int
    stdout: str
    stderr: str


class Docker:
    def call(self, args, timeout=20, check=True):
        try:
            proc = subprocess.run(
                ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False)
        except FileNotFoundError:
            raise DockerError("docker is not installed on PATH.") from None
        except subprocess.TimeoutExpired:
            raise DockerError("Docker did not respond. Is Docker Desktop running?") from None
        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "docker command failed").strip()
            raise DockerError(detail)
        return Completed(proc.returncode, proc.stdout, proc.stderr)

    def ps_rows(self):
        proc = self.call(["ps", "--filter", "status=running", "--format", "{{.Names}}\t{{.Image}}"], timeout=8)
        return parse_ps(proc.stdout)

    def inspect(self, name):
        proc = self.call(["inspect", name], timeout=8)
        data = json.loads(proc.stdout)
        if not data:
            raise DockerError(f"docker inspect returned nothing for {name}.")
        return data[0]

    def _raise_if_daemon_unreachable(self, proc):
        detail = f"{proc.stderr or ''}\n{proc.stdout or ''}".lower()
        if proc.returncode != 0 and any(token in detail for token in (
                "permission denied", "cannot connect", "is the docker daemon running", "docker.sock")):
            raise DockerError((proc.stderr or proc.stdout or "Docker is not reachable.").strip())

    def image_present(self, image):
        proc = self.call(["image", "inspect", image], check=False, timeout=8)
        self._raise_if_daemon_unreachable(proc)
        return proc.returncode == 0

    def has_gpu(self):
        proc = self.call(["info", "--format", "{{json .Runtimes}}"], check=False, timeout=8)
        self._raise_if_daemon_unreachable(proc)
        if proc.returncode != 0:
            return False
        try:
            runtimes = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return "nvidia" in (proc.stdout or "").lower()
        if isinstance(runtimes, dict):
            return any("nvidia" in key.lower() for key in runtimes)
        return False

    def ensure_network(self, name):
        proc = self.call(["network", "inspect", name], check=False)
        if proc.returncode != 0:
            self.call(["network", "create", name])

    def connect(self, network, container):
        proc = self.call(["network", "connect", network, container], check=False)
        if proc.returncode == 0:
            return
        detail = (proc.stderr or proc.stdout or "").lower()
        if "already exists" in detail:
            return
        raise DockerError((proc.stderr or proc.stdout or "network connect failed").strip())

    def resolve_host_paths(self, settings: Settings):
        if settings.paths_resolved and settings.host_datasets and settings.host_results:
            return settings
        if not settings.in_docker:
            root = Path(__file__).resolve().parents[1]
            settings.host_datasets = settings.host_datasets or (root / "datasets")
            settings.host_results = settings.host_results or (root / "results")
            settings.host_hf_cache = settings.host_hf_cache or (root / "hf-cache")
            settings.paths_resolved = True
            return settings
        info = self.inspect(os.environ.get("HOSTNAME") or "localhost")
        mounts = {mount.get("Destination"): mount.get("Source") for mount in info.get("Mounts") or []}
        datasets = settings.host_datasets or (Path(mounts["/data/datasets"]) if mounts.get("/data/datasets") else None)
        results = settings.host_results or (Path(mounts["/data/results"]) if mounts.get("/data/results") else None)
        hf_cache = settings.host_hf_cache or (Path(mounts["/data/hf-cache"]) if mounts.get("/data/hf-cache") else None)
        if datasets is None or results is None:
            raise DockerError(
                "The orchestrator container must mount datasets at /data/datasets and results at /data/results.")
        settings.host_datasets = Path(datasets)
        settings.host_results = Path(results)
        settings.host_hf_cache = Path(hf_cache) if hf_cache else None
        settings.paths_resolved = True
        return settings

    def mlflow_plan(self, settings: Settings):
        chosen = choose_mlflow(self.ps_rows(), settings.mlflow_container)
        endpoint = endpoint_from_inspect(chosen["name"], self.inspect(chosen["name"]))
        return plan_mlflow(endpoint, settings.network_name)

    def prepare_network(self, plan: MLflowPlan):
        if not plan.connect:
            return
        self.ensure_network(plan.network)
        self.connect(plan.network, plan.container)

    def run_detached(self, args):
        proc = self.call(args, timeout=120)
        container_id = proc.stdout.strip()
        if not container_id:
            raise DockerError("docker run did not return a container id.")
        return container_id

    def logs(self, name, tail=200):
        proc = self.call(["logs", "--tail", str(tail), name], check=False, timeout=30)
        return (proc.stdout or "") + (proc.stderr or "")

    def state(self, name):
        proc = self.call(["inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", name], check=False)
        if proc.returncode != 0:
            return {"status": "missing", "exit_code": None}
        parts = proc.stdout.split()
        exit_code = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
        return {"status": parts[0] if parts else "unknown", "exit_code": exit_code}

    def stop(self, name):
        self.call(["stop", "-t", "20", name], timeout=40)


def gpu_required(runs):
    return any(run.device.startswith("cuda") for run in runs)
