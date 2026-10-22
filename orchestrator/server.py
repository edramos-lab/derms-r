"""HTTP server for the DERMS-R training configuration page."""
import json
import secrets
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from orchestrator.auth import AuthError, AuthStore
from orchestrator.eta import estimate_job_hours, seconds_per_epoch_by_model
from orchestrator.inference_proxy import InferenceProxyError, fetch_json, proxy_multipart
from orchestrator.launcher import (
    ConfigError,
    Docker,
    DockerError,
    Settings,
    allocate_output,
    command_preview,
    container_argv,
    dataset_catalog,
    deploy_container_argv,
    gpu_required,
    in_window,
    lock_container_argv,
    model_catalog,
    parse_runs,
    test_container_argv,
)
from orchestrator.insights import dataset_insights
from orchestrator.mailer import MailError, Mailer, parse_emails
from orchestrator.settings_store import SettingsStore
from orchestrator.survey import collect_scores, profile_datasets
from orchestrator.sysinfo import collect as collect_system
from orchestrator.sysinfo import gpu_utilization
from orchestrator.xai import explain_section, section_context

# Paths reachable without a session. Everything else under /api/ requires auth.
PUBLIC_PATHS = {"/", "/index.html"}
PUBLIC_API_GET = {"/api/auth/me"}
PUBLIC_API_POST = {"/api/auth/login"}

STATIC = Path(__file__).resolve().parent / "static" / "index.html"
WINDOW_POLL_SECONDS = 60
GPU_POLL_SECONDS = 15
GPU_HISTORY_MAXLEN = 240


def _format_hour12(hour):
    period = "AM" if hour < 12 else "PM"
    h12 = hour % 12 or 12
    return f"{h12}:00 {period}"


def _format_scores(scores):
    if not scores:
        return "  (no summary.json found for this job yet)"
    lines = []
    for row in scores:
        acc = row.get("mean_val_accuracy")
        std = row.get("std_val_accuracy")
        params = row.get("num_parameters")
        derms_r = row.get("derms_r")
        acc_text = f"{acc:.4f}" if isinstance(acc, (int, float)) else "n/a"
        std_text = f"{std:.4f}" if isinstance(std, (int, float)) else "n/a"
        params_text = f"{params:,}" if isinstance(params, (int, float)) else "n/a"
        derms_text = f"{derms_r:.4f}" if isinstance(derms_r, (int, float)) else "n/a"
        lines.append(
            f"  {row.get('dataset')} / {row.get('model')} @ ratio {row.get('ratio')}: DERMS-R = {derms_text}\n"
            f"    mean val accuracy = {acc_text} (+/- {std_text}), params = {params_text}"
        )
    return "\n".join(lines)


def _format_system(system, gpu):
    cpu = system.get("cpu") or {}
    mem = system.get("memory") or {}
    lines = [f"  CPU: {cpu.get('percent')}% across {cpu.get('cores')} cores"]
    if mem:
        lines.append(f"  RAM: {mem.get('percent')}% ({mem.get('used_gb')}/{mem.get('total_gb')} GB)")
    if gpu:
        lines.append(
            f"  GPU: {gpu['gpu_percent']}% util, {gpu['power_watts']} W power draw, "
            f"{gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB"
        )
    else:
        lines.append("  GPU: telemetry unavailable (no NVIDIA runtime reachable right now)")
    return "\n".join(lines)


class App:
    def __init__(self, docker=None, settings=None):
        self.docker = docker or Docker()
        self.settings = settings or Settings.from_env()
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.settings.jobs_dir / "jobs.json"
        self._lock = threading.Lock()
        self.jobs = self._read()
        self._insights_cache = {}  # (dataset_id, cv_k, cv_strategy) -> result dict
        self._xai_cache_path = self.settings.jobs_dir / "xai_cache.json"
        self._xai_cache = self._read_xai_cache()  # {dataset_id: {section: result}}, persisted to disk
        self.auth = AuthStore(self.settings.jobs_dir / "users.json")
        self._run_specs = {}  # job_id -> RunSpec; in-memory only, used to relaunch queued window jobs
        self.mailer = Mailer.from_env()
        self.settings_store = SettingsStore(self.settings.jobs_dir / "system_settings.json")
        self._gpu_history = deque(maxlen=GPU_HISTORY_MAXLEN)
        self._last_gpu_alert_at = None
        self._eta_cache = {"at": 0.0, "history": {}, "params": {}}  # refreshed at most once/minute
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _read_xai_cache(self):
        if not self._xai_cache_path.exists():
            return {}
        try:
            data = json.loads(self._xai_cache_path.read_text())
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_xai_cache(self):
        temp = self._xai_cache_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self._xai_cache, indent=2))
        temp.replace(self._xai_cache_path)

    def _read(self):
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def _write(self):
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.jobs, indent=2))
        temporary.replace(self.path)

    def handler_class(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                print(f"[orchestrator] {self.address_string()} {fmt % args}", flush=True)

            def _session_token(self):
                cookie = SimpleCookie(self.headers.get("Cookie", ""))
                morsel = cookie.get("session")
                return morsel.value if morsel else None

            def _current_user(self):
                return app.auth.session_user(self._session_token())

            def _set_session_cookie(self, token):
                self.send_header("Set-Cookie", f"session={token}; HttpOnly; Path=/; SameSite=Strict")

            def _clear_session_cookie(self):
                self.send_header("Set-Cookie", "session=; HttpOnly; Path=/; SameSite=Strict; Max-Age=0")

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                if path in PUBLIC_PATHS:
                    body = STATIC.read_bytes()
                    self._send(200, body, "text/html; charset=utf-8")
                    return
                if path == "/api/auth/me":
                    user = self._current_user()
                    self._send_json({"authenticated": user is not None, "user": user})
                    return
                user = self._current_user()
                if user is None:
                    self._send_json({"error": "Not authenticated."}, 401)
                    return
                if user.get("must_change_password"):
                    self._send_json({"error": "Password change required before continuing."}, 403)
                    return
                try:
                    if path == "/api/status":
                        self._send_json(app.status())
                    elif path == "/api/system":
                        self._send_json(app.system())
                    elif path == "/api/jobs":
                        self._send_json({"jobs": app.list_jobs()})
                    elif path == "/api/eda":
                        self._send_json(app.eda())
                    elif path == "/api/eda/insights":
                        query = parse_qs(parsed.query)
                        dataset_id = (query.get("dataset") or [""])[0]
                        known = {row["id"] for row in dataset_catalog(app.settings.visible_datasets)}
                        if dataset_id not in known:
                            self._send_json({"error": f"Unknown dataset {dataset_id!r}."}, 404)
                        else:
                            cv_k = int((query.get("cv_k") or ["5"])[0])
                            cv_strategy = (query.get("cv_strategy") or ["kfold"])[0]
                            if cv_strategy not in {"kfold", "stratified"}:
                                cv_strategy = "kfold"
                            self._send_json(app.eda_insights(dataset_id, cv_k=cv_k, cv_strategy=cv_strategy))
                    elif path == "/api/eda/explain":
                        query = parse_qs(parsed.query)
                        dataset_id = (query.get("dataset") or [""])[0]
                        section = (query.get("section") or [""])[0]
                        known = {row["id"] for row in dataset_catalog(app.settings.visible_datasets)}
                        if dataset_id not in known:
                            self._send_json({"error": f"Unknown dataset {dataset_id!r}."}, 404)
                        elif section not in {"metrics", "distribution", "rgb", "crossval", "pca", "tsne"}:
                            self._send_json({"error": f"Unknown section {section!r}."}, 404)
                        else:
                            self._send_json(app.eda_explain(dataset_id, section))
                    elif path == "/api/scores":
                        self._send_json({"scores": app.scores()})
                    elif path == "/api/gpu":
                        self._send_json(app.gpu_status())
                    elif path == "/api/settings":
                        self._send_json(app.get_settings())
                    elif path == "/api/users":
                        if user["role"] != "sysadmin":
                            self._send_json({"error": "Sys Admin role required."}, 403)
                        else:
                            self._send_json({"users": app.auth.list_users()})
                    elif path.startswith("/api/jobs/") and path.endswith("/logs"):
                        job_id = path.split("/")[3]
                        self._send_json(app.logs(job_id))
                    elif path == "/api/inference/deployable":
                        self._send_json(fetch_json("/deployable"))
                    elif path == "/api/inference/stats":
                        query = parse_qs(parsed.query)
                        self._send_json(fetch_json("/stats", {
                            "backend": (query.get("backend") or [None])[0],
                            "dataset": (query.get("dataset") or [None])[0],
                        }))
                    elif path == "/api/inference/records":
                        query = parse_qs(parsed.query)
                        self._send_json(fetch_json("/records", {
                            "limit": (query.get("limit") or ["100"])[0],
                            "offset": (query.get("offset") or ["0"])[0],
                            "backend": (query.get("backend") or [None])[0],
                            "dataset": (query.get("dataset") or [None])[0],
                        }))
                    else:
                        self._send_json({"error": "Not found."}, 404)
                except (ConfigError, DockerError) as error:
                    self._send_json({"error": str(error)}, 400)
                except InferenceProxyError as error:
                    self._send_json({"error": str(error)}, 502)

            def do_POST(self):
                path = urlparse(self.path).path
                if path == "/api/auth/login":
                    try:
                        payload = self._json()
                    except json.JSONDecodeError:
                        self._send_json({"error": "Request body must be JSON."}, 400)
                        return
                    result = app.auth.authenticate(payload.get("username", ""), payload.get("password", ""))
                    if result is None:
                        self._send_json({"error": "Invalid username or password."}, 401)
                        return
                    token = app.auth.create_session(result["username"], result["role"])
                    self.send_response(200)
                    self._set_session_cookie(token)
                    body = json.dumps({"user": result}).encode()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/api/auth/logout":
                    app.auth.destroy_session(self._session_token())
                    self.send_response(200)
                    self._clear_session_cookie()
                    body = b'{"ok": true}'
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/api/auth/change-password":
                    user = self._current_user()
                    if user is None:
                        self._send_json({"error": "Not authenticated."}, 401)
                        return
                    try:
                        payload = self._json()
                        app.auth.change_password(
                            user["username"],
                            payload.get("current_password", ""),
                            payload.get("new_password", ""),
                        )
                        self._send_json({"ok": True})
                    except json.JSONDecodeError:
                        self._send_json({"error": "Request body must be JSON."}, 400)
                    except AuthError as error:
                        self._send_json({"error": str(error)}, 400)
                    return
                user = self._current_user()
                if user is None:
                    self._send_json({"error": "Not authenticated."}, 401)
                    return
                if user.get("must_change_password"):
                    self._send_json({"error": "Password change required before continuing."}, 403)
                    return
                if path in ("/api/inference/predict", "/api/inference/predict/triton"):
                    # Image upload: the body is raw multipart/form-data, not
                    # JSON, so this bypasses the generic self._json() dispatch
                    # below entirely and forwards the bytes as-is.
                    self._proxy_inference_predict(path)
                    return
                try:
                    payload = self._json()
                    if path == "/api/jobs":
                        status = 200 if payload.get("dry_run") else 201
                        self._send_json(app.start(payload), status)
                    elif path.startswith("/api/jobs/") and path.endswith("/stop"):
                        job_id = path.split("/")[3]
                        self._send_json(app.stop(job_id))
                    elif path.startswith("/api/jobs/") and path.endswith("/lock"):
                        job_id = path.split("/")[3]
                        self._send_json(app.lock(job_id, payload.get("dataset", "")), 201)
                    elif path.startswith("/api/jobs/") and path.endswith("/test"):
                        job_id = path.split("/")[3]
                        self._send_json(app.test(job_id, payload.get("dataset", "")), 201)
                    elif path == "/api/inference/deploy":
                        job_id = payload.get("job_id", "")
                        self._send_json(app.deploy(job_id, payload.get("dataset", "")), 201)
                    elif path == "/api/settings":
                        self._send_json(app.update_settings(payload))
                    elif path == "/api/users":
                        if user["role"] != "sysadmin":
                            self._send_json({"error": "Sys Admin role required."}, 403)
                        else:
                            app.auth.create_user(payload.get("username", ""), payload.get("password", ""), payload.get("role", ""))
                            self._send_json({"users": app.auth.list_users()}, 201)
                    else:
                        self._send_json({"error": "Not found."}, 404)
                except json.JSONDecodeError:
                    self._send_json({"error": "Request body must be JSON."}, 400)
                except (ConfigError, DockerError) as error:
                    self._send_json({"error": str(error)}, 400)
                except AuthError as error:
                    self._send_json({"error": str(error)}, 400)

            def do_DELETE(self):
                path = urlparse(self.path).path
                user = self._current_user()
                if user is None:
                    self._send_json({"error": "Not authenticated."}, 401)
                    return
                if user.get("must_change_password"):
                    self._send_json({"error": "Password change required before continuing."}, 403)
                    return
                if path.startswith("/api/users/"):
                    if user["role"] != "sysadmin":
                        self._send_json({"error": "Sys Admin role required."}, 403)
                        return
                    target = path.split("/")[3]
                    try:
                        app.auth.delete_user(target, user["username"])
                        self._send_json({"users": app.auth.list_users()})
                    except AuthError as error:
                        self._send_json({"error": str(error)}, 400)
                else:
                    self._send_json({"error": "Not found."}, 404)

            def _json(self):
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length > 100_000:
                    raise ConfigError("Request body is too large.")
                raw = self.rfile.read(length) if length else b"{}"
                data = json.loads(raw.decode())
                if not isinstance(data, dict):
                    raise ConfigError("Request body must be a JSON object.")
                return data

            def _send_json(self, payload, status=200):
                body = json.dumps(payload).encode()
                self._send(status, body, "application/json")

            def _send(self, status, body, content_type):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _proxy_inference_predict(self, path):
                """Forward an image-upload prediction request to inference-api
                unparsed: job_id/dataset travel as query params (already on
                self.path), the body is the raw multipart upload. Capped well
                above a single 224x224 RGB image's realistic upload size."""
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length > 20_000_000:
                    self._send_json({"error": "Uploaded file is too large (20MB limit)."}, 413)
                    return
                body = self.rfile.read(length) if length else b""
                content_type = self.headers.get("Content-Type", "multipart/form-data")
                query = parse_qs(urlparse(self.path).query)
                forward_query = {
                    "job_id": (query.get("job_id") or [""])[0],
                    "dataset": (query.get("dataset") or [""])[0],
                }
                target = "/predict/triton" if path.endswith("/triton") else "/predict"
                try:
                    status, resp_content_type, resp_body = proxy_multipart(target, forward_query, content_type, body)
                except InferenceProxyError as error:
                    self._send_json({"error": str(error)}, 502)
                    return
                self._send(status, resp_body, resp_content_type)

        return Handler

    def status(self):
        docker_ok = True
        docker_error = None
        image_present = False
        gpu = False
        mlflow = {"ok": False, "error": None, "container": None, "tracking_uri": None,
                  "network": None, "connect": False, "host_url": None}
        datasets_path = str(self.settings.visible_datasets)
        try:
            self.docker.resolve_host_paths(self.settings)
            image_present = self.docker.image_present(self.settings.image)
            gpu = self.docker.has_gpu()
            try:
                plan = self.docker.mlflow_plan(self.settings)
                mlflow = plan.public()
            except DockerError as error:
                mlflow["error"] = str(error)
        except DockerError as error:
            docker_ok = False
            docker_error = str(error)
            mlflow["error"] = str(error)
        return {
            "docker": docker_ok,
            "docker_error": docker_error,
            "image": self.settings.image,
            "image_present": image_present,
            "gpu": gpu,
            "datasets_path": datasets_path,
            "host_datasets": str(self.settings.host_datasets) if self.settings.host_datasets else None,
            "mlflow": mlflow,
            "datasets": dataset_catalog(self.settings.visible_datasets),
            "models": model_catalog(),
            "folds": 5,
            "ratios": [0.10, 0.25, 0.50, 0.75, 1.00],
            "max_combinations": 12,
            "window_open": in_window(start_hour=self.settings_store.data["window_start_hour"],
                                      end_hour=self.settings_store.data["window_end_hour"]),
            "window_hours": f"{_format_hour12(self.settings_store.data['window_start_hour'])}-"
                            f"{_format_hour12(self.settings_store.data['window_end_hour'])}",
        }

    def system(self):
        return collect_system(self.docker)

    def eda(self):
        catalog = dataset_catalog(self.settings.visible_datasets)
        return {"datasets": profile_datasets(self.settings.visible_datasets, catalog)}

    def eda_insights(self, dataset_id, cv_k=5, cv_strategy="kfold"):
        key = (dataset_id, cv_k, cv_strategy)
        with self._lock:
            cache = self._insights_cache.get(key)
        if cache is not None:
            return cache
        result = dataset_insights(self.settings.visible_datasets, dataset_id, cv_k=cv_k, cv_strategy=cv_strategy)
        with self._lock:
            self._insights_cache[key] = result
        return result

    def eda_explain(self, dataset_id, section):
        with self._lock:
            cached = self._xai_cache.get(dataset_id, {}).get(section)
        if cached is not None:
            return cached
        insights = self.eda_insights(dataset_id)
        if not insights.get("available"):
            return {"error": insights.get("reason", "Dataset insights are not available.")}
        context = section_context(section, insights)
        result = explain_section(section, dataset_id, context)
        if "text" in result:
            # Only persist genuine successes — a transient OpenRouter error (rate
            # limit, timeout, missing key at the time) should be retried on the
            # next request, not baked into the cache forever.
            with self._lock:
                self._xai_cache.setdefault(dataset_id, {})[section] = result
                self._write_xai_cache()
        return result

    def scores(self):
        rows = collect_scores(self.settings.visible_results)
        with self._lock:
            experiments = {job["id"]: job.get("mlflow_experiment") for job in self.jobs}
        for row in rows:
            job_id = row.pop("job_id", None)
            row["experiment"] = experiments.get(job_id) or "—"
        return rows

    def list_jobs(self):
        with self._lock:
            self._refresh_unlocked()
            jobs = [dict(job) for job in self.jobs]
        for job in jobs:
            job["lock_status"] = self._lock_status(job)
        self._apply_eta(jobs)
        return jobs

    def _eta_inputs(self):
        # MLflow REST + the scores scan are both too slow to run on every
        # 3s dashboard poll, so refresh at most once a minute.
        now = time.monotonic()
        if now - self._eta_cache["at"] > 60:
            history = self._eta_cache["history"]
            try:
                plan = self.docker.mlflow_plan(self.settings)
                history = seconds_per_epoch_by_model(plan.tracking_uri)
            except DockerError:
                pass
            params = self._eta_cache["params"]
            try:
                params = {row["model"]: row["num_parameters"] for row in self.scores() if row.get("num_parameters")}
            except Exception:
                pass
            self._eta_cache = {"at": now, "history": history, "params": params}
        return self._eta_cache["history"], self._eta_cache["params"]

    def _apply_eta(self, jobs):
        running_train_ids = {j["id"] for j in jobs if j.get("action") == "train" and j.get("state") == "running"}
        history, params = self._eta_inputs() if running_train_ids else ({}, {})
        for job in jobs:
            total_hours = None
            remaining_hours = None
            if job["id"] in running_train_ids:
                total_hours = estimate_job_hours(job, history, params)
                if total_hours is not None and job.get("started_at"):
                    started = datetime.fromisoformat(job["started_at"])
                    elapsed_hours = (datetime.now(timezone.utc) - started).total_seconds() / 3600
                    remaining_hours = max(total_hours - elapsed_hours, 0.0)
            job["eta_hours_total"] = total_hours
            job["eta_hours_remaining"] = remaining_hours

    def _lock_status(self, job):
        # Filesystem-derived, not persisted: whether each dataset this training
        # job covered has a locked final refit, and whether that lock has been
        # evaluated against the untouched official test set. Each architecture
        # now locks and tests itself automatically as soon as its own sweep
        # finishes (results/<dataset>/locked/<model>/), not only a single
        # overall winner picked at the end -- so "locked"/"tested" below are
        # true only once every one of this job's architectures has its own
        # lock/test result, with per-architecture counts alongside for the
        # partial case. The legacy single global-winner path
        # (results/<dataset>/locked/lock.json, no model subdirectory) is
        # checked as a fallback so a manually-triggered legacy Lock/Test still
        # shows up.
        if job.get("action") != "train" or job.get("state") != "exited" or job.get("exit_code") != 0:
            return []
        rows = []
        for dataset in job.get("datasets") or []:
            dataset_dir = Path(job["output"]) / dataset
            models = job.get("models") or []
            locked_count = sum((dataset_dir / "locked" / m / "lock.json").exists() for m in models)
            tested_count = sum((dataset_dir / "locked" / m / "test_result.json").exists() for m in models)
            legacy_folder = dataset_dir / "locked"
            rows.append({
                "dataset": dataset,
                "locked": bool(models) and locked_count == len(models) or (legacy_folder / "lock.json").exists(),
                "tested": bool(models) and tested_count == len(models) or (legacy_folder / "test_result.json").exists(),
                "locked_count": locked_count,
                "tested_count": tested_count,
                "total": len(models),
            })
        return rows

    def _refresh_unlocked(self):
        changed = False
        # Snapshot: _auto_chain_unlocked can insert new jobs into self.jobs
        # (via lock/test auto-chaining), which must not perturb this iteration.
        for job in list(self.jobs):
            if not job.get("container_name") or job.get("state") in {"exited", "failed", "missing", "preview"}:
                continue
            try:
                state = self.docker.state(job["container_name"])
            except DockerError:
                continue
            if state["status"] != job.get("state") or state["exit_code"] != job.get("exit_code"):
                job["state"] = state["status"]
                job["exit_code"] = state["exit_code"]
                changed = True
                if state["status"] in {"exited", "dead", "missing"}:
                    self._notify(job, "completed")
                    self._auto_chain_unlocked(job)
        if changed:
            self._write()

    def _auto_chain_unlocked(self, job):
        """A train job no longer auto-chains a legacy global-winner lock job:
        derms/engine.py's benchmark() now locks and tests each architecture
        automatically, at its own best ratio, as soon as that architecture's
        own sweep finishes -- inside the training container itself, not as a
        separate chained job. Launching the legacy lock here on top of that
        would just re-refit the eventual best-ranked architecture a second
        time for no benefit. A lock job that finishes successfully (only ever
        launched manually now, via the legacy Lock button) is still
        auto-chained to a test job, since that two-step legacy path still
        needs it."""
        if job.get("exit_code") != 0:
            return
        if job.get("action") == "lock":
            parent_id = job.get("parent_job")
            dataset = (job.get("datasets") or [None])[0]
            if parent_id and dataset:
                try:
                    self._satellite_unlocked(parent_id, dataset, "test", require_locked=True)
                except DockerError as error:
                    job["error"] = f"Auto-test failed for {dataset!r}: {error}"

    def _event_message(self, event, job):
        ref = f"Job {job['id']} ({job['action']} {job['label']})".strip()
        start = _format_hour12(self.settings_store.data["window_start_hour"])
        end = _format_hour12(self.settings_store.data["window_end_hour"])
        if event == "queued":
            return (
                f"RIGAL-AI Notification: job queued for the {start} window",
                f"{ref} is outside the {start}-{end} operating window and will start automatically at {start}.",
            )
        if event == "paused":
            return (
                f"RIGAL-AI Notification: job paused at {end}",
                f"{ref} was stopped at the {end} window boundary and will resume at {start} "
                "from its last checkpointed epoch.",
            )
        if event == "resumed":
            return ("RIGAL-AI Notification: job resumed", f"{ref} resumed training at the {start} window open.")
        if event == "completed":
            status = "succeeded" if job.get("exit_code") == 0 else f"exited with code {job.get('exit_code')}"
            scores = collect_scores(Path(job["output"])) if job.get("output") else []
            system = collect_system(self.docker)
            gpu = gpu_utilization()
            body = (
                f"{ref} finished: {status}.\n"
                f"Datasets: {', '.join(job.get('datasets') or [])} | "
                f"Models: {', '.join(job.get('models') or [])} | Device: {job.get('device')}\n"
                f"Output: {job.get('output')}\n\n"
                "DERMS-R results (score, and the terms it's computed from):\n"
                f"{_format_scores(scores)}\n\n"
                "System utilization at completion:\n"
                f"{_format_system(system, gpu)}"
            )
            return (f"RIGAL-AI Notification: training {status} ({job['id']})", body)
        raise ValueError(f"Unknown notification event: {event!r}")

    def _notify(self, job, event):
        emails = job.get("notify_emails") or []
        if not emails:
            return
        sent = job.setdefault("notified", [])
        if event in sent:
            return
        if self.mailer is None:
            job["mail_delivery"] = "SMTP not configured"
            return
        subject, body = self._event_message(event, job)
        try:
            self.mailer.send(emails, subject, body)
            sent.append(event)
            job["mail_delivery"] = f"sent ({event})"
        except MailError as error:
            job["mail_delivery"] = f"failed ({event}): {error}"

    def _scheduler_loop(self):
        while True:
            time.sleep(WINDOW_POLL_SECONDS)
            with self._lock:
                self._apply_window_unlocked()

    def _monitor_loop(self):
        while True:
            time.sleep(GPU_POLL_SECONDS)
            self._poll_gpu_once()

    def _poll_gpu_once(self):
        reading = gpu_utilization()
        if reading is None:
            return None
        reading = {**reading, "ts": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self._gpu_history.append(reading)
        self._check_gpu_alert(reading)
        return reading

    def _check_gpu_alert(self, reading):
        temp = reading.get("temperature_c")
        threshold = self.settings_store.data.get("gpu_temp_threshold_c")
        emails = self.settings_store.data.get("gpu_alert_emails") or []
        if temp is None or threshold is None or not emails or self.mailer is None:
            return
        if temp < threshold:
            return
        interval = timedelta(minutes=self.settings_store.data.get("gpu_alert_interval_minutes", 10))
        now = datetime.now(timezone.utc)
        if self._last_gpu_alert_at is not None and now - self._last_gpu_alert_at < interval:
            return
        subject = f"RIGAL-AI Notification: GPU temperature alert ({temp:.0f}C)"
        body = (
            f"GPU temperature is {temp:.0f}C, at or above the configured threshold of {threshold:.0f}C.\n"
            f"Utilization: {reading.get('gpu_percent')}% | Power: {reading.get('power_watts')} W | "
            f"Memory: {reading.get('memory_used_mb', 0):.0f}/{reading.get('memory_total_mb', 0):.0f} MB\n"
            f"Next alert (if still above threshold) in at least "
            f"{self.settings_store.data.get('gpu_alert_interval_minutes', 10)} minutes."
        )
        try:
            self.mailer.send(emails, subject, body)
            self._last_gpu_alert_at = now
        except MailError:
            pass

    def gpu_status(self):
        with self._lock:
            history = list(self._gpu_history)
        return {"current": history[-1] if history else None, "history": history}

    def get_settings(self):
        return dict(self.settings_store.data)

    def update_settings(self, payload):
        if not isinstance(payload, dict):
            raise ConfigError("Request body must be a JSON object.")
        patch = {}
        if "window_start_hour" in payload or "window_end_hour" in payload:
            start = int(payload.get("window_start_hour", self.settings_store.data["window_start_hour"]))
            end = int(payload.get("window_end_hour", self.settings_store.data["window_end_hour"]))
            if not (0 <= start <= 23) or not (0 <= end <= 23):
                raise ConfigError("Window hours must be between 0 and 23.")
            patch["window_start_hour"] = start
            patch["window_end_hour"] = end
        if "gpu_temp_threshold_c" in payload:
            try:
                threshold = float(payload["gpu_temp_threshold_c"])
            except (TypeError, ValueError):
                raise ConfigError("GPU temperature threshold must be a number.") from None
            if threshold <= 0:
                raise ConfigError("GPU temperature threshold must be positive.")
            patch["gpu_temp_threshold_c"] = threshold
        if "gpu_alert_interval_minutes" in payload:
            try:
                interval = float(payload["gpu_alert_interval_minutes"])
            except (TypeError, ValueError):
                raise ConfigError("GPU alert interval must be a number.") from None
            if interval <= 0:
                raise ConfigError("GPU alert interval must be positive.")
            patch["gpu_alert_interval_minutes"] = interval
        if "gpu_alert_emails" in payload:
            raw = payload["gpu_alert_emails"]
            try:
                patch["gpu_alert_emails"] = parse_emails(raw if isinstance(raw, str) else "")
            except ValueError as error:
                raise ConfigError(str(error)) from None
        if "training_defaults" in payload:
            if not isinstance(payload["training_defaults"], dict):
                raise ConfigError("training_defaults must be a JSON object.")
            patch["training_defaults"] = payload["training_defaults"]
        return self.settings_store.update(patch)

    def _launch_queued_unlocked(self, job):
        run = self._run_specs.get(job["id"])
        if run is None:
            job["state"] = "failed"
            job["error"] = "Lost the run configuration for this scheduled job (orchestrator restarted?)."
            return
        plan = None
        if run.action == "train":
            try:
                plan = self.docker.mlflow_plan(self.settings)
                self.docker.prepare_network(plan)
            except DockerError as error:
                job["error"] = str(error)
                return
        argv = container_argv(run, self.settings, plan, job["id"], resume=job.get("resumed", False))
        job["command"] = command_preview(argv)
        job["tracking_uri"] = plan.tracking_uri if plan and run.action == "train" else None
        job["container_name"] = f"derms-train-{job['id']}"
        try:
            job["container_id"] = self.docker.run_detached(argv)
            job["state"] = "running"
            job["started_at"] = datetime.now(timezone.utc).isoformat()
            job["error"] = None
            if job.get("resumed"):
                self._notify(job, "resumed")
        except DockerError as error:
            job["state"] = "failed"
            job["error"] = str(error)

    def _launch_queued_satellite_unlocked(self, job):
        """Relaunch a paused lock/test job in place (same job id/container
        name). Lock resumes via train.py's per-epoch checkpoint (the same
        --resume flag it already always passes); test has no partial-progress
        checkpointing, so an interrupted test simply restarts its (fast)
        evaluation from scratch against the same lock/test fingerprint."""
        try:
            parent = self._job(job["parent_job"])
        except DockerError as error:
            job["state"] = "failed"
            job["error"] = str(error)
            return
        dataset = job["datasets"][0]
        plan = None
        if job.get("tracking_uri"):
            try:
                plan = self.docker.mlflow_plan(self.settings)
                self.docker.prepare_network(plan)
            except DockerError as error:
                job["error"] = str(error)
                return
        argv = (lock_container_argv if job["action"] == "lock" else test_container_argv)(
            parent, dataset, self.settings, plan, job["id"])
        job["command"] = command_preview(argv)
        job["tracking_uri"] = plan.tracking_uri if plan else None
        job["container_name"] = f"derms-{job['action']}-{job['id']}"
        try:
            job["container_id"] = self.docker.run_detached(argv)
            job["state"] = "running"
            job["error"] = None
            if job.get("resumed"):
                self._notify(job, "resumed")
        except DockerError as error:
            job["state"] = "failed"
            job["error"] = str(error)

    def _apply_window_unlocked(self):
        self._refresh_unlocked()
        window_open = in_window(start_hour=self.settings_store.data["window_start_hour"],
                                 end_hour=self.settings_store.data["window_end_hour"])
        changed = False
        for job in self.jobs:
            if not job.get("window"):
                continue
            if job["state"] == "queued" and window_open:
                if job.get("action") == "train":
                    self._launch_queued_unlocked(job)
                else:
                    self._launch_queued_satellite_unlocked(job)
                changed = True
            elif job["state"] == "running" and not window_open and job.get("container_name"):
                try:
                    self.docker.stop(job["container_name"])
                    self.docker.remove(job["container_name"])
                except DockerError as error:
                    job["error"] = str(error)
                    continue
                job["state"] = "queued"
                job["resumed"] = True
                job["container_name"] = None
                job["container_id"] = None
                self._notify(job, "paused")
                changed = True
        if changed:
            self._write()

    def logs(self, job_id):
        job = self._job(job_id)
        if not job.get("container_name"):
            return {"id": job_id, "logs": job.get("error") or ""}
        try:
            text = self.docker.logs(job["container_name"])
        except DockerError as error:
            text = str(error)
        return {"id": job_id, "logs": text}

    def stop(self, job_id):
        with self._lock:
            job = self._job(job_id)
            if job["state"] == "queued":
                job["state"] = "cancelled"
                self._run_specs.pop(job_id, None)
                self._write()
                return dict(job)
            name = job.get("container_name")
            if not name:
                raise DockerError("That job has no container to stop.")
            self.docker.stop(name)
            state = self.docker.state(name)
            job["state"] = state["status"]
            job["exit_code"] = state["exit_code"]
            self._write()
            return dict(job)

    def _job(self, job_id):
        if not job_id or any(c not in "0123456789abcdef" for c in job_id):
            raise DockerError("Unknown job.")
        for job in self.jobs:
            if job["id"] == job_id:
                return job
        raise DockerError("Unknown job.")

    def _satellite_unlocked(self, job_id, dataset, kind, require_locked):
        """Shared setup for `lock` and `test`: validate the parent job/dataset,
        resolve an MLflow plan, and launch a new tracked container against the
        parent job's existing output directory (never a fresh one -- that's
        what lets `--resume --lock` see the same development split/config).
        Assumes self._lock is already held (called both from the public
        lock()/test() entry points and from the auto-chain in _refresh_unlocked)."""
        job = self._job(job_id)
        if job.get("action") != "train":
            raise DockerError("Only a completed training job can be locked or tested.")
        if dataset not in (job.get("datasets") or []):
            raise DockerError(f"{dataset!r} was not trained in this job.")
        if job.get("state") != "exited" or job.get("exit_code") != 0:
            raise DockerError("The training job must finish successfully first.")
        folder = Path(job["output"]) / dataset / "locked"
        if require_locked and not (folder / "lock.json").exists():
            raise DockerError(f"{dataset!r} is not locked yet. Run Lock before Test.")
        self.docker.resolve_host_paths(self.settings)
        plan = None
        if job.get("tracking_uri"):
            plan = self.docker.mlflow_plan(self.settings)
            self.docker.prepare_network(plan)
        satellite_id = secrets.token_hex(4)
        argv = (lock_container_argv if kind == "lock" else test_container_argv)(
            job, dataset, self.settings, plan, satellite_id)
        window = job.get("window", False)
        queued = window and not in_window(start_hour=self.settings_store.data["window_start_hour"],
                                           end_hour=self.settings_store.data["window_end_hour"])
        entry = {
            "id": satellite_id, "action": kind, "label": f"{kind}-{dataset}",
            "state": "queued" if queued else "created", "datasets": [dataset], "models": job.get("models"),
            "device": job.get("device"), "epochs": job.get("epochs"),
            "batch_size": job.get("batch_size"), "lr": job.get("lr"),
            "weight_decay": job.get("weight_decay"), "window": window, "resumed": False,
            "container_name": None if queued else f"derms-{kind}-{satellite_id}",
            "container_id": None, "exit_code": None,
            "command": command_preview(argv),
            "tracking_uri": plan.tracking_uri if plan else None,
            "notify_emails": job.get("notify_emails") or [], "mail_delivery": None, "notified": [],
            "output": job["output"], "parent_job": job_id, "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if queued:
            self.jobs.insert(0, entry)
            self._write()
            self._notify(entry, "queued")
            return dict(entry)
        try:
            entry["container_id"] = self.docker.run_detached(argv)
            entry["state"] = "running"
        except DockerError as error:
            entry["state"] = "failed"
            entry["error"] = str(error)
        self.jobs.insert(0, entry)
        self._write()
        return dict(entry)

    def lock(self, job_id, dataset):
        with self._lock:
            return self._satellite_unlocked(job_id, dataset, "lock", require_locked=False)

    def test(self, job_id, dataset):
        with self._lock:
            return self._satellite_unlocked(job_id, dataset, "test", require_locked=True)

    def _deploy_unlocked(self, job_id, dataset):
        """One-off export_models.py --deploy-triton run against an already
        locked model. Simpler than _satellite_unlocked: no MLflow plan (not a
        training/eval run) and no scheduling window (export is fast, never
        worth deferring to the training window)."""
        job = self._job(job_id)
        if job.get("action") != "train":
            raise DockerError("Only a completed training job can be deployed.")
        if dataset not in (job.get("datasets") or []):
            raise DockerError(f"{dataset!r} was not trained in this job.")
        folder = Path(job["output"]) / dataset / "locked"
        if not (folder / "lock.json").exists():
            raise DockerError(f"{dataset!r} is not locked yet. Run Lock (and Test) before deploying.")
        self.docker.resolve_host_paths(self.settings)
        deploy_id = secrets.token_hex(4)
        argv = deploy_container_argv(job, dataset, self.settings, deploy_id)
        entry = {
            "id": deploy_id, "action": "deploy", "label": f"deploy-{dataset}",
            "state": "created", "datasets": [dataset], "models": job.get("models"),
            "device": job.get("device"), "epochs": job.get("epochs"),
            "batch_size": job.get("batch_size"), "lr": job.get("lr"),
            "weight_decay": job.get("weight_decay"), "window": False, "resumed": False,
            "container_name": f"derms-deploy-{deploy_id}",
            "container_id": None, "exit_code": None,
            "command": command_preview(argv),
            "tracking_uri": None,
            "notify_emails": job.get("notify_emails") or [], "mail_delivery": None, "notified": [],
            "output": job["output"], "parent_job": job_id, "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            entry["container_id"] = self.docker.run_detached(argv)
            entry["state"] = "running"
        except DockerError as error:
            entry["state"] = "failed"
            entry["error"] = str(error)
        self.jobs.insert(0, entry)
        self._write()
        return dict(entry)

    def deploy(self, job_id, dataset):
        with self._lock:
            return self._deploy_unlocked(job_id, dataset)

    def start(self, payload):
        runs = parse_runs(payload)
        dry_run = bool(payload.get("dry_run"))
        self.docker.resolve_host_paths(self.settings)
        if not dry_run and not self.docker.image_present(self.settings.image):
            raise DockerError(
                f"Training image {self.settings.image} is not built. "
                "From the repo root run: docker compose --profile train build")
        if not dry_run and gpu_required(runs) and not self.docker.has_gpu():
            raise DockerError("This Docker daemon has no NVIDIA runtime. Set Device to cpu, or run on a GPU host.")
        plan = None
        if any(run.action == "train" for run in runs):
            plan = self.docker.mlflow_plan(self.settings)
            if not dry_run:
                self.docker.prepare_network(plan)
        created = []
        for run in runs:
            job_id = secrets.token_hex(4)
            queued = not dry_run and run.window and not in_window(
                start_hour=self.settings_store.data["window_start_hour"],
                end_hour=self.settings_store.data["window_end_hour"])
            if not dry_run:
                allocate_output(self.settings, job_id)
            argv = container_argv(run, self.settings, plan, job_id)
            job = {
                "id": job_id,
                "action": run.action,
                "label": run.label,
                "state": "preview" if dry_run else ("queued" if queued else "created"),
                "datasets": run.datasets,
                "models": run.models,
                "device": run.device,
                "epochs": run.epochs,
                "batch_size": run.batch_size,
                "lr": run.lr,
                "weight_decay": run.weight_decay,
                "seed": run.seed,
                "holdout": run.holdout,
                "augmentation": run.augmentation,
                "pretrained": run.pretrained,
                "mlflow_experiment": run.experiment,
                "mlflow_artifacts": run.artifacts,
                "window": run.window,
                "resumed": False,
                "container_name": None if (dry_run or queued) else f"derms-train-{job_id}",
                "container_id": None,
                "exit_code": None,
                "command": command_preview(argv),
                "tracking_uri": plan.tracking_uri if plan and run.action == "train" and not queued else None,
                "notify_emails": run.notify_emails,
                "mail_delivery": None,
                "notified": [],
                "output": str(self.settings.visible_results / "runs" / job_id),
                "error": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if run.notify_emails and self.mailer is None:
                job["mail_delivery"] = "SMTP not configured"
            if dry_run:
                created.append(job)
                continue
            if run.window:
                # Keep the spec around even for an immediate launch: a window
                # job can still be stopped and need relaunching later tonight.
                self._run_specs[job_id] = run
            if queued:
                with self._lock:
                    self.jobs.insert(0, job)
                    self._notify(job, "queued")
                    self._write()
                created.append(job)
                continue
            try:
                job["container_id"] = self.docker.run_detached(argv)
                job["state"] = "running"
                job["started_at"] = datetime.now(timezone.utc).isoformat()
            except DockerError as error:
                job["state"] = "failed"
                job["error"] = str(error)
            created.append(job)
            with self._lock:
                self.jobs.insert(0, job)
                self._write()
            if job["state"] == "failed":
                break
        return {
            "dry_run": dry_run,
            "mlflow": plan.public() if plan else None,
            "jobs": created,
        }


def main():
    import os
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    app = App()
    server = ThreadingHTTPServer((host, port), app.handler_class())
    print(f"RIGAL-AI training configuration at http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
