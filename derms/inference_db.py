"""SQLite inference log for the RIGAL-AI Inference tab.

Generalized from a sibling project's inference log for a single fixed
4-class dataset: this one serves CIFAR-10/100, Flowers-102, and Food-101
(10/100/102/101 classes respectively), so per-class probability columns
don't work -- `probabilities` is stored as a JSON blob instead, and a
`dataset` column distinguishes which of the four a given prediction came
from.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

DB_PATH = os.getenv("INFERENCE_DB_PATH", "/workspace/results/inference_log.db")

_lock = threading.Lock()
_initialized = False


def _get_conn() -> sqlite3.Connection:
    global _initialized
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if not _initialized:
        conn.execute("PRAGMA journal_mode=WAL")
        _init_tables(conn)
        _initialized = True
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inference_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            backend TEXT NOT NULL,
            dataset TEXT NOT NULL,
            model_name TEXT NOT NULL,
            filename TEXT,
            predicted_class TEXT NOT NULL,
            class_index INTEGER NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            inference_time_ms REAL,
            batch_size INTEGER DEFAULT 1,
            device TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON inference_log(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_backend ON inference_log(backend)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_model ON inference_log(dataset, model_name)")
    conn.commit()


def log_inference(backend: str, dataset: str, model_name: str, filename: str,
                  predicted_class: str, class_index: int, confidence: float,
                  probabilities: dict, inference_time_ms: float | None,
                  batch_size: int = 1, device: str = "cuda") -> None:
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("""
                INSERT INTO inference_log
                (timestamp, backend, dataset, model_name, filename, predicted_class,
                 class_index, confidence, probabilities, inference_time_ms, batch_size, device)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                backend, dataset, model_name, filename, predicted_class, class_index,
                confidence, json.dumps(probabilities), inference_time_ms, batch_size, device,
            ))
            conn.commit()
        finally:
            conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    record = dict(row)
    try:
        record["probabilities"] = json.loads(record["probabilities"])
    except (TypeError, ValueError):
        record["probabilities"] = {}
    return record


def get_records(limit: int = 100, offset: int = 0, backend: str | None = None,
                dataset: str | None = None, model_name: str | None = None) -> list[dict]:
    conn = _get_conn()
    try:
        clauses, params = [], []
        if backend:
            clauses.append("backend=?")
            params.append(backend)
        if dataset:
            clauses.append("dataset=?")
            params.append(dataset)
        if model_name:
            clauses.append("model_name=?")
            params.append(model_name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM inference_log {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_total_count(backend: str | None = None, dataset: str | None = None) -> int:
    conn = _get_conn()
    try:
        clauses, params = [], []
        if backend:
            clauses.append("backend=?")
            params.append(backend)
        if dataset:
            clauses.append("dataset=?")
            params.append(dataset)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return conn.execute(f"SELECT COUNT(*) FROM inference_log {where}", params).fetchone()[0]
    finally:
        conn.close()


def get_stats(backend: str | None = None, dataset: str | None = None) -> dict:
    conn = _get_conn()
    try:
        clauses, params = [], []
        if backend:
            clauses.append("backend=?")
            params.append(backend)
        if dataset:
            clauses.append("dataset=?")
            params.append(dataset)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params = tuple(params)

        row = conn.execute(f"""
            SELECT
                COUNT(*) as total_inferences,
                AVG(inference_time_ms) as avg_latency_ms,
                MIN(inference_time_ms) as min_latency_ms,
                MAX(inference_time_ms) as max_latency_ms,
                AVG(confidence) as avg_confidence,
                MIN(confidence) as min_confidence
            FROM inference_log {where}
        """, params).fetchone()

        class_dist = conn.execute(f"""
            SELECT predicted_class, COUNT(*) as count
            FROM inference_log {where}
            GROUP BY predicted_class ORDER BY count DESC LIMIT 20
        """, params).fetchall()

        latency_trend = conn.execute(f"""
            SELECT
                SUBSTR(timestamp, 1, 16) as minute,
                AVG(inference_time_ms) as avg_latency,
                COUNT(*) as count
            FROM inference_log {where}
            GROUP BY minute ORDER BY minute DESC LIMIT 60
        """, params).fetchall()

        confidence_dist = conn.execute(f"""
            SELECT
                CASE
                    WHEN confidence >= 0.99 THEN '99-100%'
                    WHEN confidence >= 0.95 THEN '95-99%'
                    WHEN confidence >= 0.90 THEN '90-95%'
                    WHEN confidence >= 0.80 THEN '80-90%'
                    ELSE '<80%'
                END as bucket,
                COUNT(*) as count
            FROM inference_log {where}
            GROUP BY bucket ORDER BY bucket
        """, params).fetchall()

        backend_where = f"WHERE {' AND '.join(c for c in clauses if not c.startswith('backend'))}" \
            if any(not c.startswith('backend') for c in clauses) else ""
        backend_params = tuple(p for c, p in zip(clauses, params) if not c.startswith('backend'))
        backend_stats = conn.execute(f"""
            SELECT backend, COUNT(*) as count, AVG(inference_time_ms) as avg_latency
            FROM inference_log {backend_where} GROUP BY backend
        """, backend_params).fetchall()

        return {
            "summary": dict(row),
            "class_distribution": [dict(r) for r in class_dist],
            "latency_trend": [dict(r) for r in reversed(latency_trend)],
            "confidence_distribution": [dict(r) for r in confidence_dist],
            "backend_comparison": [dict(r) for r in backend_stats],
        }
    finally:
        conn.close()
