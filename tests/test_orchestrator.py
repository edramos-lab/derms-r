"""Training-page launcher: Docker argv, MLflow networking, and the HTTP API."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from orchestrator.launcher import (
    command_preview,
    ConfigError,
    Settings,
    choose_mlflow,
    container_argv,
    endpoint_from_inspect,
    log_series,
    parse_runs,
    plan_mlflow,
)
from orchestrator.server import App
from orchestrator.survey import collect_scores, profile_dataset


def settings(root: Path):
    data = root / "datasets"
    results = root / "results"
    data.mkdir()
    (data / "cifar10").mkdir()
    (data / "cifar10" / "ready").write_text("ok")
    return Settings(
        image="derms-r-train:latest",
        in_docker=False,
        visible_datasets=data,
        visible_results=results,
        jobs_dir=results / "orchestrator",
        host_datasets=data,
        host_results=results,
        paths_resolved=True,
    )


def fixed_payload(**overrides):
    payload = {
        "action": "train",
        "datasets": ["cifar10"],
        "models": ["resnet50"],
        "deployment": "single",
        "device": "cpu",
        "mode": "fixed",
        "epochs": 50,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "seed": 42,
        "holdout": 0.2,
        "folds": 5,
        "ratios": [0.10, 0.25, 0.50, 0.75, 1.00],
        "augmentation": True,
        "pretrained": True,
        "mlflow_experiment": "derms-r-benchmark",
    }
    payload.update(overrides)
    return payload


class FakeDocker:
    def __init__(self, inspect=None, gpu=False, image=True, rows=None):
        self.inspect_doc = inspect or {
            "NetworkSettings": {
                "Ports": {"5000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "5000"}]},
                "Networks": {"mlops_default": {}},
            }
        }
        self.gpu = gpu
        self.image = image
        self.rows = rows or [{"name": "mlflow", "image": "ghcr.io/mlflow/mlflow:v2.16.0"}]
        self.ran = []
        self.connected = []

    def resolve_host_paths(self, current):
        current.paths_resolved = True
        return current

    def image_present(self, image):
        return self.image

    def has_gpu(self):
        return self.gpu

    def mlflow_plan(self, current):
        chosen = choose_mlflow(self.rows, current.mlflow_container)
        endpoint = endpoint_from_inspect(chosen["name"], self.inspect_doc)
        return plan_mlflow(endpoint, current.network_name)

    def prepare_network(self, plan):
        if plan.connect:
            self.connected.append((plan.network, plan.container))

    def run_detached(self, args):
        self.ran.append(args)
        return "container-id"

    def logs(self, name):
        return "training log"

    def state(self, name):
        return {"status": "running", "exit_code": 0}

    def stop(self, name):
        self.stopped = name


class PlanTests(unittest.TestCase):
    def test_user_network_uses_container_dns(self):
        endpoint = endpoint_from_inspect("mlflow", {
            "NetworkSettings": {
                "Ports": {"5000/tcp": [{"HostPort": "5001"}]},
                "Networks": {"bridge": {}, "tracking_net": {}},
            }
        })
        plan = plan_mlflow(endpoint, "derms-r")
        self.assertEqual(plan.tracking_uri, "http://mlflow:5000")
        self.assertEqual(plan.network, "tracking_net")
        self.assertFalse(plan.connect)
        self.assertEqual(plan.host_url, "http://localhost:5001")

    def test_bridge_only_mlflow_is_attached_to_derms_network(self):
        endpoint = endpoint_from_inspect("/mlflow-server", {
            "NetworkSettings": {
                "Ports": {"5000/tcp": None},
                "Networks": {"bridge": {}},
            }
        })
        plan = plan_mlflow(endpoint, "derms-r")
        self.assertEqual(plan.container, "mlflow-server")
        self.assertTrue(plan.connect)
        self.assertEqual(plan.network, "derms-r")
        self.assertEqual(plan.tracking_uri, "http://mlflow-server:5000")
        self.assertIsNone(plan.host_url)

    def test_multiple_mlflow_containers_need_an_explicit_name(self):
        rows = [
            {"name": "mlflow", "image": "mlflow"},
            {"name": "mlflow-backup", "image": "mlflow"},
        ]
        with self.assertRaises(Exception):
            choose_mlflow(rows, None)
        chosen = choose_mlflow(rows, "mlflow-backup")
        self.assertEqual(chosen["name"], "mlflow-backup")

    def test_log_range_includes_the_upper_bound(self):
        learning = log_series({"min": 1e-5, "max": 1e-3}, "learning rate")
        decay = log_series({"min": 0.001, "max": 0.05}, "weight decay")
        self.assertEqual(len(learning), 3)
        self.assertEqual(len(decay), 3)
        for got, expected in zip(learning, (1e-5, 1e-4, 1e-3)):
            self.assertAlmostEqual(got, expected)
        for got, expected in zip(decay, (0.001, 0.01, 0.05)):
            self.assertAlmostEqual(got, expected)


class ConfigTests(unittest.TestCase):
    def test_fixed_run_keeps_canonical_protocol(self):
        runs = parse_runs(fixed_payload())
        self.assertEqual(len(runs), 1)
        self.assertEqual((runs[0].epochs, runs[0].batch_size), (50, 32))

    def test_partial_ratios_are_rejected(self):
        with self.assertRaises(ConfigError):
            parse_runs(fixed_payload(ratios=[1.0]))

    def test_cluster_mode_is_rejected(self):
        with self.assertRaises(ConfigError):
            parse_runs(fixed_payload(deployment="cluster"))

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaises(ConfigError):
            parse_runs(fixed_payload(models=["not-a-model"]))

    def test_range_over_the_cap_is_rejected(self):
        payload = fixed_payload(mode="range", ranges={
            "batch_size": {"min": 16, "max": 64, "step": 16, "scale": "linear"},
            "epochs": {"min": 25, "max": 100, "step": 25, "scale": "linear"},
            "lr": {"min": 1e-5, "max": 1e-3, "scale": "log"},
            "weight_decay": {"min": 0.001, "max": 0.05, "scale": "log"},
        })
        with self.assertRaises(ConfigError) as caught:
            parse_runs(payload)
        self.assertIn("12", str(caught.exception))

    def test_narrow_range_is_one_container_per_combination(self):
        payload = fixed_payload(mode="range", ranges={
            "batch_size": {"min": 16, "max": 32, "step": 16, "scale": "linear"},
            "epochs": {"min": 50, "max": 50, "step": 1, "scale": "linear"},
            "lr": {"min": 1e-4, "max": 1e-4, "scale": "log"},
            "weight_decay": {"min": 0.01, "max": 0.01, "scale": "log"},
        })
        runs = parse_runs(payload)
        self.assertEqual([run.batch_size for run in runs], [16, 32])


class CommandTests(unittest.TestCase):
    def test_training_container_targets_mlflow_dns(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = settings(Path(tmp))
            run = parse_runs(fixed_payload(device="cuda:0", augmentation=False))[0]
            endpoint = endpoint_from_inspect("mlflow", {
                "NetworkSettings": {
                    "Ports": {"5000/tcp": [{"HostPort": "5000"}]},
                    "Networks": {"mlops_default": {}},
                }
            })
            plan = plan_mlflow(endpoint, "derms-r")
            argv = container_argv(run, current, plan, "abcd1234")
        self.assertIn("--network", argv)
        self.assertIn("mlops_default", argv)
        self.assertIn("http://mlflow:5000", argv)
        self.assertNotIn("localhost", argv)
        self.assertNotIn("127.0.0.1", argv)
        self.assertIn("--gpus", argv)
        self.assertIn("device=0", argv)
        self.assertIn("--no-augmentation", argv)
        self.assertIn("--mlflow-experiment", argv)
        self.assertIn("derms-r-benchmark", argv)

    def test_prepare_container_does_not_contact_mlflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = settings(Path(tmp))
            run = parse_runs(fixed_payload(action="prepare"))[0]
            argv = container_argv(run, current, None, "abcd1234")
        self.assertIn("--prepare-only", argv)
        self.assertIn("--download", argv)
        self.assertNotIn("--mlflow", argv)
        self.assertNotIn("--network", argv)

    def test_huggingface_token_is_passed_and_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = settings(Path(tmp))
            current.hf_token = "secret-token"
            current.host_hf_cache = Path(tmp) / "hf"
            run = parse_runs(fixed_payload())[0]
            argv = container_argv(run, current, None, "abcd1234")
            preview = command_preview(argv)
        self.assertIn("HF_TOKEN=secret-token", argv)
        self.assertIn("HUGGING_FACE_HUB_TOKEN=secret-token", argv)
        self.assertIn("HF_HOME=/root/.cache/huggingface", argv)
        self.assertNotIn("secret-token", preview)
        self.assertIn("HF_TOKEN=***", preview)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.docker = FakeDocker()
        self.app = App(docker=self.docker, settings=settings(Path(self.tmp.name)))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self.app.handler_class())
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def request(self, method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req) as response:
            body = response.read().decode()
            return response.status, json.loads(body) if path.startswith("/api/") else body

    def test_page_lists_start_training(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("Start Training", body)
        self.assertIn("Prepare Dataset", body)
        self.assertIn("Dashboard", body)
        self.assertIn("EDA", body)
        self.assertIn("Administration", body)

    def test_status_reports_mlflow_on_its_network(self):
        status, body = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(body["mlflow"]["tracking_uri"], "http://mlflow:5000")
        self.assertEqual(body["mlflow"]["network"], "mlops_default")
        self.assertFalse(body["mlflow"]["connect"])
        cifar = next(row for row in body["datasets"] if row["id"] == "cifar10")
        self.assertTrue(cifar["present"])

    def test_start_launches_a_container_on_the_mlflow_network(self):
        status, body = self.request("POST", "/api/jobs", fixed_payload())
        self.assertEqual(status, 201)
        self.assertEqual(len(self.docker.ran), 1)
        self.assertIn("http://mlflow:5000", self.docker.ran[0])
        self.assertEqual(body["jobs"][0]["state"], "running")
        self.assertEqual(self.docker.connected, [])
        _, jobs = self.request("GET", "/api/jobs")
        self.assertEqual(jobs["jobs"][0]["id"], body["jobs"][0]["id"])

    def test_cuda_without_a_gpu_runtime_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("POST", "/api/jobs", fixed_payload(device="cuda:0"))
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.docker.ran, [])

    def test_bridge_mlflow_is_connected_before_launch(self):
        self.docker.inspect_doc = {
            "NetworkSettings": {
                "Ports": {"5000/tcp": [{"HostPort": "5000"}]},
                "Networks": {"bridge": {}},
            }
        }
        status, body = self.request("POST", "/api/jobs", fixed_payload())
        self.assertEqual(status, 201)
        self.assertEqual(self.docker.connected, [("derms-r", "mlflow")])
        self.assertIn("--network", self.docker.ran[0])
        self.assertIn("derms-r", self.docker.ran[0])
        self.assertIn("http://mlflow:5000", body["jobs"][0]["command"])
        self.assertIn("created_at", body["jobs"][0])

    def test_eda_reports_a_missing_dataset(self):
        status, body = self.request("GET", "/api/eda")
        self.assertEqual(status, 200)
        cifar = next(row for row in body["datasets"] if row["id"] == "cifar10")
        self.assertTrue(cifar["present"])
        self.assertEqual(cifar["layout"], "files")


class SurveyTests(unittest.TestCase):
    def test_class_folders_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "all" / "blast" / "a.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"jpg")
            profile = profile_dataset(root, "all", None)
        self.assertEqual(profile["layout"], "class-folders")
        self.assertEqual(profile["classes"][0]["name"], "blast")
        self.assertEqual(profile["classes"][0]["images"], 1)

    def test_torchvision_archive_is_recognized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cifar10" / "cifar-10-batches-py").mkdir(parents=True)
            profile = profile_dataset(root, "cifar10", 10)
        self.assertEqual(profile["layout"], "torchvision")
        self.assertEqual(profile["class_count"], 10)

    def test_scores_use_the_best_raw_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            folder = results / "runs" / "abcd" / "cifar10"
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps([
                {"dataset": "cifar10", "model": "resnet50", "ratio": 0.1, "derms_r_raw": 1.2},
                {"dataset": "cifar10", "model": "vit_base", "ratio": 0.5, "derms_r_raw": 2.4},
            ]))
            scores = collect_scores(results)
        self.assertEqual(scores[0]["model"], "vit_base")
        self.assertAlmostEqual(scores[0]["derms_r"], 2.4)


if __name__ == "__main__":
    unittest.main()
