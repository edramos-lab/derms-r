"""Persisted system settings: the scheduling window, GPU alert thresholds, and
the last-used training form -- all small enough to keep in one JSON file, all
needing to survive both a page reload and an orchestrator restart."""
import json
from pathlib import Path

DEFAULTS = {
    "window_start_hour": 20,
    "window_end_hour": 8,
    "gpu_temp_threshold_c": 85.0,
    "gpu_alert_interval_minutes": 10,
    "gpu_alert_emails": [],
    "training_defaults": {},
}


class SettingsStore:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._read()

    def _read(self):
        merged = dict(DEFAULTS)
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                merged.update(data)
        return merged

    def _write(self):
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.data, indent=2))
        temp.replace(self.path)

    def update(self, patch: dict):
        self.data.update(patch)
        self._write()
        return self.data
