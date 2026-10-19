"""Real system metrics for the Dashboard.

Read from inside the orchestrator container — which on this machine is the
colima Linux VM Docker runs in, not raw macOS host stats. Docker containers
cannot reach host hardware sensors, so GPU utilization, temperature, and
NVRAM are reported as unavailable with an honest reason rather than faked.
CPU/RAM/disk/load are real, live numbers for the VM Docker actually schedules
containers on, which is what matters for "can this machine run more jobs."
"""
import os
import shutil
import subprocess
import time


def _read_proc_stat():
    with open("/proc/stat") as f:
        line = f.readline()
    fields = [int(x) for x in line.split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    total = sum(fields)
    return idle, total


def cpu_percent(sample_ms=120):
    try:
        idle1, total1 = _read_proc_stat()
        time.sleep(sample_ms / 1000)
        idle2, total2 = _read_proc_stat()
        delta_total = total2 - total1
        delta_idle = idle2 - idle1
        if delta_total <= 0:
            return 0.0
        return round((1 - delta_idle / delta_total) * 100, 1)
    except OSError:
        return None


def memory_info():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.strip().split()
                if parts:
                    info[key] = int(parts[0])
        total_kb = info.get("MemTotal", 0)
        available_kb = info.get("MemAvailable", info.get("MemFree", 0))
        used_kb = max(total_kb - available_kb, 0)
        return {
            "total_gb": round(total_kb / 1024 / 1024, 2),
            "used_gb": round(used_kb / 1024 / 1024, 2),
            "available_gb": round(available_kb / 1024 / 1024, 2),
            "percent": round(used_kb / total_kb * 100, 1) if total_kb else 0.0,
        }
    except OSError:
        return None


def disk_info(path="/"):
    try:
        usage = shutil.disk_usage(path)
        used = usage.total - usage.free
        return {
            "total_gb": round(usage.total / 1024 ** 3, 2),
            "used_gb": round(used / 1024 ** 3, 2),
            "free_gb": round(usage.free / 1024 ** 3, 2),
            "percent": round(used / usage.total * 100, 1) if usage.total else 0.0,
        }
    except OSError:
        return None


def load_average():
    try:
        one, five, fifteen = os.getloadavg()
        return {"1m": round(one, 2), "5m": round(five, 2), "15m": round(fifteen, 2)}
    except OSError:
        return None


def uptime_seconds():
    try:
        with open("/proc/uptime") as f:
            return float(f.readline().split()[0])
    except OSError:
        return None


def cpu_count():
    return os.cpu_count() or 1


def gpu_info(docker):
    has_gpu = False
    try:
        has_gpu = docker.has_gpu()
    except Exception:
        has_gpu = False
    if has_gpu:
        return {"available": True, "detail": "NVIDIA runtime detected."}
    return {
        "available": False,
        "detail": "No NVIDIA container runtime attached to this Docker daemon. "
                  "Training will run on CPU.",
    }


def gpu_utilization():
    """Direct nvidia-smi query. Only returns real numbers when this container
    itself has GPU access (runtime: nvidia / --gpus in compose) -- the toolkit
    mounts the host driver's nvidia-smi binary into any container regardless of
    base image. Returns None with no GPU/toolkit or any query failure; this is
    best-effort telemetry for the dashboard trend and alerting, not load-bearing."""
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) != 5:
        return None
    try:
        return {
            "gpu_percent": float(parts[0]),
            "power_watts": float(parts[1]),
            "memory_used_mb": float(parts[2]),
            "memory_total_mb": float(parts[3]),
            "temperature_c": float(parts[4]),
        }
    except ValueError:
        return None


def unavailable_sensor(name):
    return {
        "available": False,
        "reason": f"{name} is not exposed to this container. Docker cannot read host "
                  "hardware sensors from inside a container, and this VM does not "
                  "surface thermal_zone/NVRAM data.",
    }


def best_practice_insights(cpu_pct, mem, disk, gpu, load, cores):
    """Rule-based tips derived from the live readings above — not generic filler."""
    tips = []
    if mem and mem["percent"] >= 85:
        tips.append({
            "level": "warning",
            "text": f"RAM usage is {mem['percent']}% — training/prepare containers can be "
                    "SIGKILLed by the OOM killer above this level. Increase the colima VM's "
                    "memory allocation (colima stop && colima start --memory 8) or stop other jobs.",
        })
    elif mem and mem["percent"] >= 70:
        tips.append({
            "level": "info",
            "text": f"RAM usage is {mem['percent']}%. Fine for one training run; watch it if "
                    "you start a second concurrently.",
        })
    if disk and disk["percent"] >= 85:
        tips.append({
            "level": "warning",
            "text": f"Disk usage is {disk['percent']}% on the container's root filesystem. "
                    "Dataset downloads and checkpoints will fail once this fills up.",
        })
    if not gpu["available"]:
        tips.append({
            "level": "info",
            "text": "No GPU detected — training runs on CPU and will be significantly slower "
                    "than a GPU host, especially for ViT/Swin architectures.",
        })
    if load and cores and load["1m"] > cores:
        tips.append({
            "level": "warning",
            "text": f"1-minute load average ({load['1m']}) exceeds the {cores} CPUs available — "
                    "the system is CPU-saturated; new jobs will queue behind existing work.",
        })
    if cpu_pct is not None and cpu_pct >= 90:
        tips.append({
            "level": "info",
            "text": f"CPU usage is {cpu_pct}%. Expected during an active training run; if idle, "
                    "check for a stuck job in the Dashboard's Training jobs table.",
        })
    if not tips:
        tips.append({"level": "ok", "text": "No resource pressure detected."})
    return tips


def collect(docker):
    cpu_pct = cpu_percent()
    mem = memory_info()
    disk = disk_info()
    load = load_average()
    cores = cpu_count()
    gpu = gpu_info(docker)
    utilization = gpu_utilization()
    return {
        "cpu": {"percent": cpu_pct, "cores": cores},
        "memory": mem,
        "disk": disk,
        "load_average": load,
        "uptime_seconds": uptime_seconds(),
        "gpu": gpu,
        "gpu_utilization": utilization,
        "temperature": unavailable_sensor("Temperature") if utilization is None else {
            "available": True, "celsius": utilization["temperature_c"]},
        "nvram": unavailable_sensor("NVRAM"),
        "insights": best_practice_insights(cpu_pct, mem, disk, gpu, load, cores),
        "note": "Metrics reflect the Docker VM (colima) this container runs in, not raw macOS host stats.",
    }
