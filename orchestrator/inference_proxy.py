"""Thin proxy helpers from the orchestrator to the internal inference-api
container (derms/inference_server.py), same urllib style as eta.py's MLflow
REST calls. inference-api isn't exposed on a host port -- the orchestrator,
already the one authenticated public entry point, is the only thing that
talks to it directly, over the shared `derms-r` Docker network.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://inference-api:8060")


class InferenceProxyError(Exception):
    """inference-api is unreachable, or returned something unexpected."""


def _url(path, query=None):
    url = f"{INFERENCE_API_URL.rstrip('/')}{path}"
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
    return url


def fetch_json(path, query=None, timeout=10):
    try:
        with urllib.request.urlopen(_url(path, query), timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise InferenceProxyError(f"inference-api {path} returned HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        raise InferenceProxyError(f"inference-api is unreachable ({path}): {error}") from error


def post_json(path, payload, query=None, timeout=15):
    request = urllib.request.Request(
        _url(path, query), data=json.dumps(payload).encode(),
        method="POST", headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise InferenceProxyError(f"inference-api {path} returned HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        raise InferenceProxyError(f"inference-api is unreachable ({path}): {error}") from error


def proxy_multipart(path, query, content_type, body, timeout=60):
    """Forward a raw multipart/form-data body byte-for-byte and relay the
    response. Used for image-upload prediction requests: the browser's
    request to the orchestrator already carries job_id/dataset as query
    params and the image as the entire body, so there's nothing to parse --
    just forward the bytes and the Content-Type header (which carries the
    multipart boundary) unchanged.

    Returns (status_code, content_type, body_bytes) so the caller can relay
    the response verbatim instead of re-encoding it.
    """
    request = urllib.request.Request(
        _url(path, query), data=body, method="POST", headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers.get("Content-Type", "application/json"), response.read()
    except urllib.error.HTTPError as error:
        return error.code, "application/json", error.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise InferenceProxyError(f"inference-api is unreachable ({path}): {error}") from error
