"""HTTP server for the DERMS-R training configuration page."""
import json
import secrets
import threading
import time
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from orchestrator.auth import AuthError, AuthStore
from orchestrator.eta import estimate_job_hours, seconds_per_epoch_by_model
from orchestrator.launcher import (
    ConfigError,
    Docker,
    DockerError,
    Settings,
    allocate_output,
    command_preview,
    container_argv,
    dataset_catalog,
    gpu_required,
    model_catalog,
    parse_runs,
)
from orchestrator.insights import dataset_insights
from orchestrator.survey import collect_scores, profile_datasets
from orchestrator.sysinfo import collect as collect_system
from orchestrator.xai import explain_section, section_context

# Paths reachable without a session. Everything else under /api/ requires auth.
PUBLIC_PATHS = {"/", "/index.html"}
PUBLIC_API_GET = {"/api/auth/me"}
PUBLIC_API_POST = {"/api/auth/login"}

STATIC = Path(__file__).resolve().parent / "static" / "index.html"


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
        self._eta_cache = {"at": 0.0, "history": {}, "params": {}}  # refreshed at most once/minute

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
                    elif path == "/api/users":
                        if user["role"] != "sysadmin":
                            self._send_json({"error": "Sys Admin role required."}, 403)
                        else:
                            self._send_json({"users": app.auth.list_users()})
                    elif path.startswith("/api/jobs/") and path.endswith("/logs"):
                        job_id = path.split("/")[3]
                        self._send_json(app.logs(job_id))
                    else:
                        self._send_json({"error": "Not found."}, 404)
                except (ConfigError, DockerError) as error:
                    self._send_json({"error": str(error)}, 400)

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
                try:
                    payload = self._json()
                    if path == "/api/jobs":
                        status = 200 if payload.get("dry_run") else 201
                        self._send_json(app.start(payload), status)
                    elif path.startswith("/api/jobs/") and path.endswith("/stop"):
                        job_id = path.split("/")[3]
                        self._send_json(app.stop(job_id))
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
            row["experiment"] = experiments.get(job_id) or "\u2014"
        return rows

    def list_jobs(self):
        with self._lock:
            self._refresh_unlocked()
            jobs = [dict(job) for job in self.jobs]
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

    def _refresh_unlocked(self):
        changed = False
        for job in self.jobs:
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
            if not dry_run:
                allocate_output(self.settings, job_id)
            argv = container_argv(run, self.settings, plan, job_id)
            job = {
                "id": job_id,
                "action": run.action,
                "label": run.label,
                "state": "preview" if dry_run else "created",
                "datasets": run.datasets,
                "models": run.models,
                "device": run.device,
                "epochs": run.epochs,
                "batch_size": run.batch_size,
                "lr": run.lr,
                "weight_decay": run.weight_decay,
                "container_name": None if dry_run else f"derms-train-{job_id}",
                "container_id": None,
                "exit_code": None,
                "command": command_preview(argv),
                "tracking_uri": plan.tracking_uri if plan and run.action == "train" else None,
                "notify_email": run.notify_email,
                "mail_delivery": "not configured" if run.notify_email else None,
                "output": str(self.settings.visible_results / "runs" / job_id),
                "error": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if dry_run:
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
