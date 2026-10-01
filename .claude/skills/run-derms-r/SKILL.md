---
name: run-derms-r
description: Build and launch the RIGAL-AI (formerly DERMS-R) Docker stack (training orchestrator web UI + MLflow tracking server + training image) on this machine. Use when asked to "run the container(s)", "start the orchestrator", "run mlflow", or "bring the stack up" for this repo.
version: 1.1.0
---

# Run RIGAL-AI (repo: derms-r)

Full local stack: `orchestrator` (web UI at :8080, drives Docker to launch
training runs), `mlflow` (tracking server at :5001), and the `train` image
(built on demand, launched by the orchestrator itself via the host Docker
socket — you normally don't run it directly). UI is branded "RIGAL-AI";
repo/image/container names still say `derms-r` — that's intentional, only
the display name changed.

## Login (auth added 2026-09-30)

The orchestrator now requires login — no more "no accounts" local console.
On first boot with no `users.json`, it creates a default Sys Admin and prints
the one-time password to container logs:

```bash
docker logs derms-r-orchestrator-1 2>&1 | grep -A4 "default Sys Admin"
```

Sessions are in-memory and do NOT survive an orchestrator container restart
(recreate, not just docker restart of the same container — actually any
process restart clears them) — expect to log in again after every rebuild.
`users.json` itself (and the accounts in it) persists on disk at
`./results/orchestrator/users.json`, bind-mounted, so accounts you create
survive restarts even though sessions don't.

Roles: `sysadmin` (full access, including User Management) and `ml_user`
(everything except User Management — enforced both in the UI nav and
server-side on `/api/users*`, verified with curl during implementation).

## OpenRouter API key (XAI panels, added 2026-09-30)

The Exploratory Dataset Analysis tab's "AI Insight" panels call OpenRouter
(`deepseek/deepseek-chat`) server-side. The key lives in the repo-root `.env`
(gitignored, never commit it):

```
OPENROUTER_API_KEY=sk-or-v1-...
```

`docker-compose.yml` passes it through as `OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:-}`.
docker-compose auto-loads `.env` from the same directory — no extra flag
needed. If `.env` is missing/empty, the EDA page still works fully; XAI
panels just show an error string instead of a stack trace.

XAI results are cached to disk at `./results/orchestrator/xai_cache.json`
(`{dataset_id: {section: {text, model}}}`), not just in-memory — survives
container restarts/rebuilds, so revisiting the EDA tab for a
dataset+section already explained does NOT re-call OpenRouter (verified:
7.3s cold call vs 0.01s after a real `docker restart`). Only successful
responses are persisted; a transient OpenRouter error is retried on the
next request rather than cached forever. Delete that file (or the relevant
dataset/section keys inside it) to force fresh explanations.

## One-time environment facts (already fixed in this repo/machine as of 2026-09-30)

- Docker context on this Mac is **colima**, not Docker Desktop. `~/.docker/config.json`
  must NOT set `"credsStore": "desktop"` — that binary doesn't exist under colima and
  breaks every `docker pull`/`build` with a credential-helper error. Already removed.
- `Dockerfile.orchestrator` detects arch via `uname -m` at build time (not the
  buildx-only `TARGETARCH` build arg), so it builds correctly with the classic
  builder (no buildx plugin installed here).
- Host port **5000 is taken by macOS AirPlay Receiver** — MLflow is published on
  **5001** instead (internal container port is still 5000; that's what the
  orchestrator uses over the `derms-r` Docker network).
- `mlflow` and the `derms-r` network are defined directly in `docker-compose.yml`
  now — no manual `docker run` / `docker network create` needed.
- **Colima's VM must have enough RAM, or training/prepare containers get SIGKILLed
  by the kernel OOM killer (`exit_code 137`, `State.OOMKilled=true`).** Default
  colima profile was 2GiB — too small for even a `cifar10` download running
  alongside the orchestrator build and mlflow. Bumped to 4 CPU / 8GiB (host has
  16GB/10 cores):
  ```bash
  colima stop
  colima start --cpu 4 --memory 8 --disk 100
  ```
  Check current allocation with `colima list`. If you see repeated jobs dying
  with exit code 137 in the Dashboard/Administration audit trail, this is why —
  don't just retry, check `colima list` and `docker inspect <container> --format
  '{{.State.OOMKilled}}'` first. `colima stop`/`start` resets the active Docker
  context to `default` and back to `colima`; that's expected, not a regression.
  Containers with `restart: unless-stopped` (orchestrator, mlflow) resume
  automatically after `colima start`; one-shot `derms-train-*` job containers do
  not — those need to be relaunched from the UI.

## Launch

From the repo root:

```bash
docker-compose up -d --build orchestrator mlflow
```

(Use `docker-compose`, the standalone binary — the `docker compose` plugin is
not installed on this machine.)

This builds the orchestrator image and starts both `derms-r-orchestrator-1`
and `mlflow`, on a shared `derms-r` network.

## Build the training image (only needed before first "Start Training"/"Prepare Dataset" click)

```bash
docker-compose --profile train build train
```

Without this, the Configuration page shows: "Training image derms-r-train:latest
is not built."

## Verify

```bash
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'
curl -sS -o /dev/null -w "orchestrator: %{http_code}\n" http://127.0.0.1:8080
curl -sS -o /dev/null -w "mlflow: %{http_code}\n" http://127.0.0.1:5001
```

Both should return `200`. Open http://localhost:8080 — you'll hit the login
screen first (see "Login" above). After logging in, sidebar should show
Dashboard, Configuration, Exploratory Dataset Analysis, Audit trail, and
(sysadmin only) User Management. Dashboard's MLflow tile should read
`mlflow` (auto-detected by container name), not an error.

If the UI looks stale (e.g. missing tabs, a button that should be hidden
still showing) after editing `orchestrator/static/index.html` or
`orchestrator/server.py`, the image needs rebuilding — those files are baked
in at build time, not mounted live:

```bash
docker-compose build orchestrator && docker-compose up -d orchestrator
```

**CSS gotcha (bit us twice, 2026-09-30):** an element toggled via the
`hidden` attribute will NOT actually hide if any CSS rule sets `display` on
it by class/tag/id — author styles always beat the browser's built-in
`[hidden]{display:none}` regardless of specificity or source order. Fixed
globally with `[hidden]{display:none!important}` near the top of the
stylesheet. If you add a new toggled element and it "doesn't hide," this is
almost certainly why — check you're not fighting an explicit `display:` rule,
and don't remove the global override.

## Stop

```bash
docker-compose down
```

This removes the orchestrator and mlflow containers but leaves the `derms-r`
network definition in compose (recreated on next `up`) and leaves data on disk
(`./mlflow-data`, `./results`, `./datasets` are all bind-mounted, not in
container-only volumes).
