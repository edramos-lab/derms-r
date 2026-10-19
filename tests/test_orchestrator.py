"""Training-page launcher: Docker argv, MLflow networking, and the HTTP API."""
import json
import smtplib
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from datetime import datetime

from orchestrator.launcher import (
    command_preview,
    ConfigError,
    DockerError,
    Settings,
    choose_mlflow,
    container_argv,
    endpoint_from_inspect,
    in_window,
    log_series,
    parse_runs,
    plan_mlflow,
)
from orchestrator.eta import estimate_job_hours
from orchestrator.mailer import MailError, Mailer
from orchestrator.settings_store import SettingsStore
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
        self.state_result = {"status": "running", "exit_code": 0}

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
        return self.state_result

    def stop(self, name):
        self.stopped = name

    def remove(self, name):
        self.removed = name


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


class WindowHoursTests(unittest.TestCase):
    def test_overnight_window_wraps_midnight(self):
        self.assertTrue(in_window(datetime(2026, 1, 1, 23, 0), start_hour=20, end_hour=8))
        self.assertTrue(in_window(datetime(2026, 1, 1, 3, 0), start_hour=20, end_hour=8))
        self.assertFalse(in_window(datetime(2026, 1, 1, 12, 0), start_hour=20, end_hour=8))

    def test_same_day_window_does_not_wrap(self):
        self.assertTrue(in_window(datetime(2026, 1, 1, 10, 0), start_hour=9, end_hour=17))
        self.assertFalse(in_window(datetime(2026, 1, 1, 20, 0), start_hour=9, end_hour=17))
        self.assertFalse(in_window(datetime(2026, 1, 1, 8, 0), start_hour=9, end_hour=17))

    def test_equal_start_and_end_means_always_open(self):
        self.assertTrue(in_window(datetime(2026, 1, 1, 3, 0), start_hour=5, end_hour=5))


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


class LockTestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.docker = FakeDocker()
        self.app = App(docker=self.docker, settings=settings(Path(self.tmp.name)))

    def tearDown(self):
        self.tmp.cleanup()

    def _completed_train_job(self):
        result = self.app.start(fixed_payload())
        job = self.app._job(result["jobs"][0]["id"])
        job["state"] = "exited"
        job["exit_code"] = 0
        return job

    def test_lock_launches_a_container_against_the_parents_output_dir(self):
        job = self._completed_train_job()
        entry = self.app.lock(job["id"], "cifar10")
        self.assertEqual(entry["action"], "lock")
        self.assertEqual(entry["state"], "running")
        self.assertEqual(entry["output"], job["output"])
        self.assertEqual(entry["parent_job"], job["id"])
        argv = self.docker.ran[-1]
        self.assertIn("--resume", argv)
        self.assertIn("--lock", argv)
        self.assertIn("cifar10", argv)
        self.assertIn(job["output"], " ".join(argv))

    def test_test_is_rejected_before_locking(self):
        job = self._completed_train_job()
        with self.assertRaises(DockerError):
            self.app.test(job["id"], "cifar10")

    def test_test_launches_after_locking(self):
        job = self._completed_train_job()
        locked_dir = Path(job["output"]) / "cifar10" / "locked"
        locked_dir.mkdir(parents=True)
        (locked_dir / "lock.json").write_text("{}")
        entry = self.app.test(job["id"], "cifar10")
        self.assertEqual(entry["action"], "test")
        self.assertEqual(entry["state"], "running")
        argv = self.docker.ran[-1]
        self.assertIn("test.py", argv)
        self.assertIn("cifar10", argv)

    def test_lock_rejected_for_a_dataset_not_in_the_job(self):
        job = self._completed_train_job()
        with self.assertRaises(DockerError):
            self.app.lock(job["id"], "flowers102")

    def test_lock_rejected_before_the_job_finishes(self):
        result = self.app.start(fixed_payload())
        job_id = result["jobs"][0]["id"]
        with self.assertRaises(DockerError):
            self.app.lock(job_id, "cifar10")

    def test_lock_status_reflects_filesystem_state(self):
        job = self._completed_train_job()
        jobs = self.app.list_jobs()
        row = next(j for j in jobs if j["id"] == job["id"])
        self.assertEqual(row["lock_status"], [{"dataset": "cifar10", "locked": False, "tested": False}])
        locked_dir = Path(job["output"]) / "cifar10" / "locked"
        locked_dir.mkdir(parents=True)
        (locked_dir / "lock.json").write_text("{}")
        jobs = self.app.list_jobs()
        row = next(j for j in jobs if j["id"] == job["id"])
        self.assertEqual(row["lock_status"], [{"dataset": "cifar10", "locked": True, "tested": False}])

    def test_train_exiting_successfully_auto_triggers_lock(self):
        result = self.app.start(fixed_payload())
        job_id = result["jobs"][0]["id"]
        self.docker.state_result = {"status": "exited", "exit_code": 0}
        with self.app._lock:
            self.app._refresh_unlocked()
        actions = {j["action"] for j in self.app.jobs}
        self.assertIn("lock", actions)
        lock_job = next(j for j in self.app.jobs if j["action"] == "lock")
        self.assertEqual(lock_job["parent_job"], job_id)
        self.assertEqual(lock_job["datasets"], ["cifar10"])
        self.assertEqual(lock_job["state"], "running")

    def test_lock_exiting_successfully_auto_triggers_test(self):
        result = self.app.start(fixed_payload())
        job_id = result["jobs"][0]["id"]
        self.docker.state_result = {"status": "exited", "exit_code": 0}
        with self.app._lock:
            self.app._refresh_unlocked()  # train exits -> lock auto-created
        lock_job = next(j for j in self.app.jobs if j["action"] == "lock")
        # Simulate what the real lock container would have written to disk.
        locked_dir = Path(self.app._job(job_id)["output"]) / "cifar10" / "locked"
        locked_dir.mkdir(parents=True)
        (locked_dir / "lock.json").write_text("{}")
        with self.app._lock:
            self.app._refresh_unlocked()  # lock job observed exited -> test auto-created
        test_job = next(j for j in self.app.jobs if j["action"] == "test")
        self.assertEqual(test_job["parent_job"], job_id)
        self.assertEqual(test_job["datasets"], ["cifar10"])
        self.assertEqual(test_job["state"], "running")
        argv = self.docker.ran[-1]
        self.assertIn("test.py", argv)

    def test_auto_chain_does_not_fire_on_a_failed_train_job(self):
        result = self.app.start(fixed_payload())
        self.docker.state_result = {"status": "exited", "exit_code": 1}
        with self.app._lock:
            self.app._refresh_unlocked()
        self.assertFalse(any(j["action"] == "lock" for j in self.app.jobs))

    def test_refresh_does_not_skip_jobs_when_auto_chain_inserts_new_ones(self):
        # Two train jobs exiting in the same refresh pass: inserting the first
        # job's auto-lock into self.jobs must not cause the second job to be
        # skipped by the same iteration (regression test for mutating a list
        # while iterating over it).
        first = self.app.start(fixed_payload())["jobs"][0]["id"]
        second = self.app.start(fixed_payload(datasets=["cifar10"]))["jobs"][0]["id"]
        self.docker.state_result = {"status": "exited", "exit_code": 0}
        with self.app._lock:
            self.app._refresh_unlocked()
        self.assertEqual(self.app._job(first)["state"], "exited")
        self.assertEqual(self.app._job(second)["state"], "exited")
        lock_parents = {j["parent_job"] for j in self.app.jobs if j["action"] == "lock"}
        self.assertEqual(lock_parents, {first, second})


class LockTestWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.docker = FakeDocker()
        self.app = App(docker=self.docker, settings=settings(Path(self.tmp.name)))
        self.app.mailer = RecordingMailer()

    def tearDown(self):
        self.tmp.cleanup()

    def _completed_window_train_job(self):
        with patch("orchestrator.server.in_window", return_value=True):
            result = self.app.start(fixed_payload(window=True, notify_emails="owner@example.com"))
        job = self.app._job(result["jobs"][0]["id"])
        job["state"] = "exited"
        job["exit_code"] = 0
        return job

    def test_lock_inherits_window_and_queues_when_window_is_closed(self):
        job = self._completed_window_train_job()
        with patch("orchestrator.server.in_window", return_value=False):
            entry = self.app.lock(job["id"], "cifar10")
        self.assertTrue(entry["window"])
        self.assertEqual(entry["state"], "queued")
        self.assertIsNone(entry["container_name"])

    def test_lock_launches_immediately_when_window_is_open(self):
        job = self._completed_window_train_job()
        with patch("orchestrator.server.in_window", return_value=True):
            entry = self.app.lock(job["id"], "cifar10")
        self.assertEqual(entry["state"], "running")
        self.assertTrue(entry["window"])

    def test_non_window_job_locks_are_unaffected_by_window_state(self):
        with patch("orchestrator.server.in_window", return_value=True):
            result = self.app.start(fixed_payload())  # window not requested
        job = self.app._job(result["jobs"][0]["id"])
        job["state"] = "exited"
        job["exit_code"] = 0
        with patch("orchestrator.server.in_window", return_value=False):
            entry = self.app.lock(job["id"], "cifar10")
        self.assertFalse(entry["window"])
        self.assertEqual(entry["state"], "running")

    def test_lock_job_is_paused_when_window_closes_mid_run_and_resumes_later(self):
        job = self._completed_window_train_job()
        with patch("orchestrator.server.in_window", return_value=True):
            entry = self.app.lock(job["id"], "cifar10")
        lock_id = entry["id"]

        with patch("orchestrator.server.in_window", return_value=False):
            with self.app._lock:
                self.app._apply_window_unlocked()
        paused = self.app._job(lock_id)
        self.assertEqual(paused["state"], "queued")
        self.assertTrue(paused["resumed"])
        self.assertIsNone(paused["container_name"])
        self.assertEqual(self.docker.stopped, f"derms-lock-{lock_id}")
        self.assertEqual(len(self.app.mailer.sent), 1)
        self.assertIn("paused", self.app.mailer.sent[0][1].lower())

        with patch("orchestrator.server.in_window", return_value=True):
            with self.app._lock:
                self.app._apply_window_unlocked()
        resumed = self.app._job(lock_id)
        self.assertEqual(resumed["state"], "running")
        self.assertEqual(resumed["container_name"], f"derms-lock-{lock_id}")
        self.assertEqual(len(self.app.mailer.sent), 2)
        self.assertIn("resumed", self.app.mailer.sent[1][1].lower())
        # Relaunch used lock_container_argv again (--resume --lock), same as the first launch.
        argv = self.docker.ran[-1]
        self.assertIn("--resume", argv)
        self.assertIn("--lock", argv)

    def test_auto_chained_test_job_also_inherits_window(self):
        job = self._completed_window_train_job()
        with patch("orchestrator.server.in_window", return_value=True):
            self.app.lock(job["id"], "cifar10")
        # Simulate what the real lock container writes to disk, then its exit.
        locked_dir = Path(job["output"]) / "cifar10" / "locked"
        locked_dir.mkdir(parents=True)
        (locked_dir / "lock.json").write_text("{}")
        self.docker.state_result = {"status": "exited", "exit_code": 0}
        with patch("orchestrator.server.in_window", return_value=True):
            with self.app._lock:
                self.app._refresh_unlocked()
        test_job = next(j for j in self.app.jobs if j["action"] == "test")
        self.assertTrue(test_job["window"])
        self.assertEqual(test_job["state"], "running")


class WindowSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.docker = FakeDocker()
        self.app = App(docker=self.docker, settings=settings(Path(self.tmp.name)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_window_job_outside_the_window_is_queued_not_launched(self):
        with patch("orchestrator.server.in_window", return_value=False):
            result = self.app.start(fixed_payload(window=True))
        job = result["jobs"][0]
        self.assertEqual(job["state"], "queued")
        self.assertIsNone(job["container_name"])
        self.assertEqual(self.docker.ran, [])
        self.assertIn(job["id"], self.app._run_specs)

    def test_window_job_inside_the_window_launches_immediately(self):
        with patch("orchestrator.server.in_window", return_value=True):
            result = self.app.start(fixed_payload(window=True))
        job = result["jobs"][0]
        self.assertEqual(job["state"], "running")
        self.assertEqual(len(self.docker.ran), 1)

    def test_scheduler_launches_a_queued_job_once_the_window_opens(self):
        with patch("orchestrator.server.in_window", return_value=False):
            result = self.app.start(fixed_payload(window=True))
        job_id = result["jobs"][0]["id"]
        with patch("orchestrator.server.in_window", return_value=True):
            with self.app._lock:
                self.app._apply_window_unlocked()
        job = self.app._job(job_id)
        self.assertEqual(job["state"], "running")
        self.assertEqual(len(self.docker.ran), 1)
        self.assertNotIn("--resume", self.docker.ran[0])

    def test_scheduler_stops_a_running_window_job_when_the_window_closes_then_resumes_it(self):
        with patch("orchestrator.server.in_window", return_value=True):
            result = self.app.start(fixed_payload(window=True))
        job_id = result["jobs"][0]["id"]
        with patch("orchestrator.server.in_window", return_value=False):
            with self.app._lock:
                self.app._apply_window_unlocked()
        job = self.app._job(job_id)
        self.assertEqual(job["state"], "queued")
        self.assertTrue(job["resumed"])
        self.assertIsNone(job["container_name"])
        self.assertEqual(self.docker.stopped, "derms-train-" + job_id)
        self.assertEqual(self.docker.removed, "derms-train-" + job_id)
        with patch("orchestrator.server.in_window", return_value=True):
            with self.app._lock:
                self.app._apply_window_unlocked()
        job = self.app._job(job_id)
        self.assertEqual(job["state"], "running")
        self.assertEqual(len(self.docker.ran), 2)
        self.assertIn("--resume", self.docker.ran[-1])

    def test_a_queued_job_can_be_cancelled(self):
        with patch("orchestrator.server.in_window", return_value=False):
            result = self.app.start(fixed_payload(window=True))
        job_id = result["jobs"][0]["id"]
        job = self.app.stop(job_id)
        self.assertEqual(job["state"], "cancelled")
        self.assertNotIn(job_id, self.app._run_specs)


class RecordingMailer:
    def __init__(self):
        self.sent = []

    def send(self, to_addrs, subject, body):
        self.sent.append((tuple(to_addrs), subject, body))


class EmailParsingTests(unittest.TestCase):
    def test_multiple_addresses_are_parsed_and_deduplicated(self):
        runs = parse_runs(fixed_payload(notify_emails="a@x.com, a@x.com ; b@y.com  c@z.com"))
        self.assertEqual(runs[0].notify_emails, ["a@x.com", "b@y.com", "c@z.com"])

    def test_blank_field_means_no_recipients(self):
        runs = parse_runs(fixed_payload(notify_emails=""))
        self.assertEqual(runs[0].notify_emails, [])

    def test_invalid_address_is_rejected(self):
        with self.assertRaises(ConfigError):
            parse_runs(fixed_payload(notify_emails="not-an-email"))

    def test_one_invalid_address_among_valid_ones_is_still_rejected(self):
        with self.assertRaises(ConfigError):
            parse_runs(fixed_payload(notify_emails="good@example.com, bad"))


class MailerTests(unittest.TestCase):
    def test_send_starts_tls_and_logs_in_before_sending(self):
        with patch("smtplib.SMTP") as smtp_cls:
            client = smtp_cls.return_value.__enter__.return_value
            mailer = Mailer(host="smtp.example.com", port=587, username="u", password="p", sender="from@example.com")
            mailer.send(["to@example.com"], "subject", "body")
        client.starttls.assert_called_once()
        client.login.assert_called_once_with("u", "p")
        client.send_message.assert_called_once()

    def test_send_skips_login_without_credentials(self):
        with patch("smtplib.SMTP") as smtp_cls:
            client = smtp_cls.return_value.__enter__.return_value
            mailer = Mailer(host="smtp.example.com", port=25, username=None, password=None, sender="from@example.com")
            mailer.send(["to@example.com"], "subject", "body")
        client.login.assert_not_called()

    def test_smtp_failure_is_wrapped_in_mail_error(self):
        with patch("smtplib.SMTP") as smtp_cls:
            smtp_cls.return_value.__enter__.side_effect = smtplib.SMTPException("boom")
            mailer = Mailer(host="smtp.example.com", port=587, username=None, password=None, sender="from@example.com")
            with self.assertRaises(MailError):
                mailer.send(["to@example.com"], "subject", "body")

    def test_from_env_is_none_without_an_smtp_host(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(Mailer.from_env())

    def test_from_env_reads_configured_values(self):
        env = {
            "SMTP_HOST": "smtp.example.com", "SMTP_PORT": "2525",
            "SMTP_USERNAME": "u", "SMTP_PASSWORD": "p", "SMTP_USE_TLS": "0",
        }
        with patch.dict("os.environ", env, clear=True):
            mailer = Mailer.from_env()
        self.assertEqual((mailer.host, mailer.port, mailer.username, mailer.use_tls), ("smtp.example.com", 2525, "u", False))


class EmailNotificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.docker = FakeDocker()
        self.app = App(docker=self.docker, settings=settings(Path(self.tmp.name)))
        self.app.mailer = RecordingMailer()

    def tearDown(self):
        self.tmp.cleanup()

    def test_job_completion_sends_an_email(self):
        result = self.app.start(fixed_payload(notify_emails="owner@example.com"))
        self.docker.state_result = {"status": "exited", "exit_code": 0}
        with self.app._lock:
            self.app._refresh_unlocked()
        self.assertEqual(len(self.app.mailer.sent), 1)
        to_addrs, subject, _ = self.app.mailer.sent[0]
        self.assertEqual(to_addrs, ("owner@example.com",))
        self.assertIn("RIGAL-AI Notification", subject)
        self.assertIn("succeeded", subject)
        job = self.app._job(result["jobs"][0]["id"])
        self.assertEqual(job["mail_delivery"], "sent (completed)")

    def test_no_recipients_means_no_email_is_sent(self):
        self.app.start(fixed_payload())
        self.docker.state_result = {"status": "exited", "exit_code": 0}
        with self.app._lock:
            self.app._refresh_unlocked()
        self.assertEqual(self.app.mailer.sent, [])

    def test_completion_email_includes_derms_r_terms_and_system_utilization(self):
        result = self.app.start(fixed_payload(notify_emails="owner@example.com"))
        job = result["jobs"][0]
        summary_dir = Path(job["output"]) / "cifar10"
        summary_dir.mkdir(parents=True)
        (summary_dir / "summary.json").write_text(json.dumps([{
            "dataset": "cifar10", "model": "resnet50", "ratio": 0.75,
            "derms_r_raw": 0.2624, "mean_validation_accuracy": 0.893,
            "sd_validation_accuracy": 0.0047, "num_parameters": 2141818,
        }]))
        fake_gpu = {
            "gpu_percent": 87.0, "power_watts": 210.5, "memory_used_mb": 9000.0,
            "memory_total_mb": 32000.0, "temperature_c": 62.0,
        }
        self.docker.state_result = {"status": "exited", "exit_code": 0}
        # sysinfo reads /proc and nvidia-smi, neither of which exist in this test
        # sandbox -- stub both so the test isn't macOS/Linux/GPU-dependent.
        fake_system = {"cpu": {"percent": 12.3, "cores": 16}, "memory": {"percent": 45.2, "used_gb": 14.2, "total_gb": 32.0}}
        with patch("orchestrator.server.collect_system", return_value=fake_system), \
             patch("orchestrator.server.gpu_utilization", return_value=fake_gpu):
            with self.app._lock:
                self.app._refresh_unlocked()
        _, _, body = self.app.mailer.sent[0]
        self.assertIn("DERMS-R = 0.2624", body)
        self.assertIn("mean val accuracy = 0.8930", body)
        self.assertIn("params = 2,141,818", body)
        self.assertIn("CPU: 12.3% across 16 cores", body)
        self.assertIn("RAM: 45.2% (14.2/32.0 GB)", body)
        self.assertIn("87.0% util, 210.5 W", body)

    def test_completion_email_reports_gpu_telemetry_unavailable(self):
        self.app.start(fixed_payload(notify_emails="owner@example.com"))
        self.docker.state_result = {"status": "exited", "exit_code": 0}
        with self.app._lock:
            self.app._refresh_unlocked()
        _, _, body = self.app.mailer.sent[0]
        self.assertIn("GPU: telemetry unavailable", body)

    def test_missing_smtp_configuration_is_reported_on_the_job(self):
        self.app.mailer = None
        result = self.app.start(fixed_payload(notify_emails="owner@example.com"))
        self.assertEqual(result["jobs"][0]["mail_delivery"], "SMTP not configured")

    def test_window_lifecycle_sends_queued_paused_and_resumed_emails(self):
        with patch("orchestrator.server.in_window", return_value=False):
            result = self.app.start(fixed_payload(window=True, notify_emails="owner@example.com"))
        job_id = result["jobs"][0]["id"]
        self.assertEqual(len(self.app.mailer.sent), 1)
        self.assertIn("queued", self.app.mailer.sent[0][1].lower())

        with patch("orchestrator.server.in_window", return_value=True):
            with self.app._lock:
                self.app._apply_window_unlocked()
        self.assertEqual(len(self.app.mailer.sent), 1, "first launch of a queued job is not a resume")

        with patch("orchestrator.server.in_window", return_value=False):
            with self.app._lock:
                self.app._apply_window_unlocked()
        self.assertEqual(len(self.app.mailer.sent), 2)
        self.assertIn("paused", self.app.mailer.sent[1][1].lower())

        with patch("orchestrator.server.in_window", return_value=True):
            with self.app._lock:
                self.app._apply_window_unlocked()
        self.assertEqual(len(self.app.mailer.sent), 3)
        self.assertIn("resumed", self.app.mailer.sent[2][1].lower())
        job = self.app._job(job_id)
        self.assertEqual(job["mail_delivery"], "sent (resumed)")


class SettingsStoreTests(unittest.TestCase):
    def test_defaults_when_no_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SettingsStore(Path(tmp) / "system_settings.json")
        self.assertEqual(store.data["window_start_hour"], 20)
        self.assertEqual(store.data["window_end_hour"], 8)
        self.assertEqual(store.data["gpu_temp_threshold_c"], 85.0)
        self.assertEqual(store.data["gpu_alert_emails"], [])

    def test_update_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "system_settings.json"
            store = SettingsStore(path)
            store.update({"window_start_hour": 9, "training_defaults": {"epochs": 10}})
            reloaded = SettingsStore(path)
        self.assertEqual(reloaded.data["window_start_hour"], 9)
        self.assertEqual(reloaded.data["window_end_hour"], 8)
        self.assertEqual(reloaded.data["training_defaults"], {"epochs": 10})


class SettingsApiAndGpuMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.docker = FakeDocker()
        self.app = App(docker=self.docker, settings=settings(Path(self.tmp.name)))
        self.app.mailer = RecordingMailer()

    def tearDown(self):
        self.tmp.cleanup()

    def test_window_hours_are_validated_and_persisted(self):
        updated = self.app.update_settings({"window_start_hour": 7, "window_end_hour": 19})
        self.assertEqual((updated["window_start_hour"], updated["window_end_hour"]), (7, 19))
        self.assertEqual(self.app.get_settings()["window_start_hour"], 7)
        with self.assertRaises(ConfigError):
            self.app.update_settings({"window_start_hour": 25, "window_end_hour": 8})

    def test_gpu_alert_emails_are_parsed_and_validated(self):
        updated = self.app.update_settings({"gpu_alert_emails": "a@x.com, b@y.com"})
        self.assertEqual(updated["gpu_alert_emails"], ["a@x.com", "b@y.com"])
        with self.assertRaises(ConfigError):
            self.app.update_settings({"gpu_alert_emails": "not-an-email"})

    def test_training_defaults_round_trip(self):
        self.app.update_settings({"training_defaults": {"datasets": ["cifar10"], "epochs": 5}})
        self.assertEqual(self.app.get_settings()["training_defaults"], {"datasets": ["cifar10"], "epochs": 5})
        with self.assertRaises(ConfigError):
            self.app.update_settings({"training_defaults": "not-a-dict"})

    def test_gpu_status_reports_latest_and_history(self):
        # Exercise the single-sample poll step the background thread runs on
        # each tick, without actually running the thread (avoids sleep-based
        # flakiness) or needing a real GPU.
        with patch("orchestrator.server.gpu_utilization", return_value={
            "gpu_percent": 42.0, "power_watts": 150.0, "memory_used_mb": 1000.0,
            "memory_total_mb": 32000.0, "temperature_c": 55.0,
        }):
            self.app._poll_gpu_once()
        status = self.app.gpu_status()
        self.assertEqual(status["current"]["gpu_percent"], 42.0)
        self.assertEqual(len(status["history"]), 1)

    def test_gpu_status_is_empty_without_a_gpu(self):
        with patch("orchestrator.server.gpu_utilization", return_value=None):
            self.app._poll_gpu_once()
        status = self.app.gpu_status()
        self.assertIsNone(status["current"])
        self.assertEqual(status["history"], [])

    def test_high_temperature_triggers_an_alert_email(self):
        self.app.update_settings({"gpu_temp_threshold_c": 80.0, "gpu_alert_emails": "oncall@example.com"})
        hot_reading = {"gpu_percent": 99.0, "power_watts": 300.0, "memory_used_mb": 30000.0,
                       "memory_total_mb": 32000.0, "temperature_c": 90.0}
        self.app._check_gpu_alert(hot_reading)
        self.assertEqual(len(self.app.mailer.sent), 1)
        to_addrs, subject, body = self.app.mailer.sent[0]
        self.assertEqual(to_addrs, ("oncall@example.com",))
        self.assertIn("GPU temperature alert", subject)
        self.assertIn("90", body)
        # A second reading within the cooldown window should not re-alert.
        self.app._check_gpu_alert(hot_reading)
        self.assertEqual(len(self.app.mailer.sent), 1)

    def test_temperature_below_threshold_does_not_alert(self):
        self.app.update_settings({"gpu_temp_threshold_c": 80.0, "gpu_alert_emails": "oncall@example.com"})
        cool_reading = {"gpu_percent": 10.0, "power_watts": 50.0, "memory_used_mb": 1000.0,
                        "memory_total_mb": 32000.0, "temperature_c": 55.0}
        self.app._check_gpu_alert(cool_reading)
        self.assertEqual(self.app.mailer.sent, [])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.docker = FakeDocker()
        self.app = App(docker=self.docker, settings=settings(Path(self.tmp.name)))
        # Bypass password hashing/the bootstrap admin's forced password change:
        # tests only need an authenticated session, not real credentials.
        self.app.auth.users["tester"] = {
            "salt": "00", "hash": "", "role": "sysadmin",
            "created_at": "2026-01-01T00:00:00+00:00", "must_change_password": False,
        }
        token = self.app.auth.create_session("tester", "sysadmin")
        self.session_cookie = f"session={token}"
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
        req.add_header("Cookie", self.session_cookie)
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
        self.assertIn("Audit trail", body)

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


class TestEtaEstimate(unittest.TestCase):
    RUN_COMBOS = 25  # 5 canonical ratios x 5 folds

    def test_uses_direct_history_when_available(self):
        job = {"models": ["resnet50"], "datasets": ["cifar10"], "epochs": 10}
        hours = estimate_job_hours(job, history={"resnet50": 2.0}, params_by_model={"resnet50": 25_000_000})
        self.assertAlmostEqual(hours, 2.0 * 10 * self.RUN_COMBOS / 3600)

    def test_falls_back_to_model_size_scaling(self):
        job = {"models": ["new_model"], "datasets": ["cifar10"], "epochs": 10}
        # resnet50 (25M params) is the only run history; new_model (50M, 2x
        # the size) has no history of its own, so its rate should scale
        # linearly off resnet50's seconds-per-epoch-per-million-parameters.
        history = {"resnet50": 2.0}
        params = {"resnet50": 25_000_000, "new_model": 50_000_000}
        hours = estimate_job_hours(job, history, params)
        expected_seconds_per_epoch = (2.0 / 25) * 50
        self.assertAlmostEqual(hours, expected_seconds_per_epoch * 10 * self.RUN_COMBOS / 3600)

    def test_sums_across_models_and_datasets(self):
        job = {"models": ["resnet50", "vit_base"], "datasets": ["cifar10", "cifar100"], "epochs": 5}
        history = {"resnet50": 1.0, "vit_base": 3.0}
        hours = estimate_job_hours(job, history, params_by_model={})
        expected_seconds = (1.0 + 3.0) * 5 * 2 * self.RUN_COMBOS
        self.assertAlmostEqual(hours, expected_seconds / 3600)

    def test_no_history_or_params_returns_none(self):
        job = {"models": ["mystery_model"], "datasets": ["cifar10"], "epochs": 10}
        self.assertIsNone(estimate_job_hours(job, history={}, params_by_model={}))

    def test_missing_epochs_returns_none(self):
        job = {"models": ["resnet50"], "datasets": ["cifar10"], "epochs": 0}
        self.assertIsNone(estimate_job_hours(job, history={"resnet50": 2.0}, params_by_model={}))


if __name__ == "__main__":
    unittest.main()
