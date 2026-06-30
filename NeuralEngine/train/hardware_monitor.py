"""Background hardware-utilisation logger for long unattended training runs.

Runs in a daemon thread so it never blocks shutdown. Samples GPU utilisation,
VRAM, CPU utilisation, and system RAM **every second**, then at the end of the
configured interval takes the mean of all data points and logs one line. This
smooths out transient spikes/dips so the log reflects real sustained utilisation.

GPU stats come from nvidia-smi (the most reliable cross-driver source). CPU
reads /proc/stat deltas across 1s windows. RAM reads /proc/meminfo.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time


class _CpuReader:
    """Tracks CPU utilisation across calls. Prefers cgroup v2 cpu.stat (container-scoped,
    matches the pod dashboard) and falls back to /proc/stat (namespaced on modern kernels,
    host-scoped on older ones)."""

    def __init__(self) -> None:
        self._prev_ts: float | None = None
        self._prev_usec: int | None = None
        self._prev_proc: list[int] | None = None
        self._num_cpus: int | None = None
        # Try cgroup v2 first
        self._cg_stat_path = "/sys/fs/cgroup/cpu.stat"
        self._use_cgroup = os.path.exists(self._cg_stat_path)

    def _read_cgroup_usec(self) -> int | None:
        try:
            with open(self._cg_stat_path) as f:
                for line in f:
                    if line.startswith("usage_usec "):
                        return int(line.split()[1])
        except Exception:
            pass
        return None

    def sample(self) -> float | None:
        if self._use_cgroup:
            now = time.time()
            usec = self._read_cgroup_usec()
            if usec is not None and self._prev_usec is not None and self._prev_ts is not None:
                dt = now - self._prev_ts
                du = usec - self._prev_usec
                self._prev_ts, self._prev_usec = now, usec
                if dt > 0 and du >= 0:
                    if self._num_cpus is None:
                        try:
                            self._num_cpus = len(os.sched_getaffinity(0))
                        except (AttributeError, OSError):
                            self._num_cpus = max(1, os.cpu_count() or 1)
                    # usage_usec counts across all CPUs => fraction = cores_used / num_cpus
                    cores = du / (dt * 1_000_000)
                    return min(1.0, cores / self._num_cpus)
            else:
                self._prev_ts, self._prev_usec = now, usec
                return None

        # Fallback: /proc/stat
        try:
            with open("/proc/stat") as f:
                fields = f.readline().split()
            if fields[0] != "cpu":
                return None
            curr = [int(x) for x in fields[1:]]
            if self._prev_proc is None:
                self._prev_proc = curr
                return None
            idle_delta = curr[3] - self._prev_proc[3]
            total_delta = sum(curr) - sum(self._prev_proc)
            self._prev_proc = curr
            if total_delta <= 0:
                return None
            return 1.0 - idle_delta / total_delta
        except Exception:
            return None


def _sample_gpu() -> dict | None:
    """Query the first GPU via nvidia-smi. Returns {gpu_pct, vram_used_mib, vram_total_mib}
    or None when no GPU / smi is available."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        parts = [p.strip() for p in out.stdout.split(",")]
        if len(parts) < 3:
            return None
        return {
            "gpu_pct": float(parts[0]) / 100.0,
            "vram_used_mib": int(parts[1]),
            "vram_total_mib": int(parts[2]),
        }
    except Exception:
        return None


def _sample_ram() -> dict | None:
    """Returns {ram_used_pct, ram_total_gb} from cgroup memory stats (container-scoped,
    matches the pod dashboard). Falls back to /proc/meminfo (host-scoped, inaccurate in
    containers) when cgroup files are unavailable."""
    # ── cgroup v2 ──
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            used_bytes = int(f.read().strip())
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
        if raw != "max":
            total_bytes = int(raw)
            if total_bytes > 0 and used_bytes >= 0:
                return {
                    "ram_total_gb": total_bytes / (1024 ** 3),
                    "ram_used_pct": used_bytes / total_bytes,
                }
    except (OSError, ValueError):
        pass

    # ── cgroup v1 ──
    try:
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
            used_bytes = int(f.read().strip())
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            total_bytes = int(f.read().strip())
        # A limit near LONG_MAX (~9 EiB) means "unlimited"; treat as host RAM.
        if total_bytes > 0 and total_bytes < (1 << 50) and used_bytes >= 0:
            return {
                "ram_total_gb": total_bytes / (1024 ** 3),
                "ram_used_pct": used_bytes / total_bytes,
            }
    except (OSError, ValueError):
        pass

    # ── Fallback: /proc/meminfo (host view, inaccurate in containers) ──
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    parts = val.strip().split()
                    if parts:
                        mem[key.strip()] = int(parts[0])
        total_kb = mem.get("MemTotal", 0)
        avail_kb = mem.get("MemAvailable", 0)
        if total_kb <= 0:
            return None
        return {
            "ram_total_gb": total_kb / (1024 * 1024),
            "ram_used_pct": 1.0 - avail_kb / total_kb,
        }
    except Exception:
        return None


def _format(averages: dict) -> str:
    """Format averaged stats into a compact single-line string."""
    parts: list[str] = []
    if "cpu_pct" in averages and averages["cpu_pct"] is not None:
        parts.append(f"cpu={averages['cpu_pct']:.0%}")
    if "ram_used_pct" in averages:
        parts.append(f"ram={averages['ram_used_pct']:.0%}({averages.get('ram_total_gb', 0):.0f}G)")
    if "gpu_pct" in averages:
        gpu = f"gpu={averages['gpu_pct']:.0%}"
        if "vram_used_mib" in averages:
            gpu += f" vram={averages['vram_used_mib']:.0f}/{averages['vram_total_mib']:.0f}MiB"
        parts.append(gpu)
    return "  ".join(parts) if parts else "(no stats)"


# ── Public sampling API (used by test_logging.py for one-shot checks) ─────

def sample_now() -> dict:
    """Take one instantaneous sample (no averaging). Useful for smoke tests."""
    snap: dict = {}
    cpu = _CpuReader()
    cpu.sample()  # prime
    time.sleep(0.5)
    snap["cpu_pct"] = cpu.sample()
    ram = _sample_ram()
    if ram:
        snap.update(ram)
    gpu = _sample_gpu()
    if gpu:
        snap.update(gpu)
    return snap


class HardwareMonitor:
    """Background thread: sample hardware every 1s, log the mean every `interval_s` seconds.

    A 60s interval → 60 data points averaged per log line.  Short intervals
    (< 2s) fall back to a single non-averaged sample so tests run fast."""

    def __init__(self, interval_s: float = 60.0) -> None:
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        from train.clock import log
        self._thread = threading.Thread(target=self._run, args=(log,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self, log) -> None:
        # Wait one full interval before the first log line so the training
        # loop has settled and the log file is set up.
        if self._stop.wait(self._interval):
            return

        cpu = _CpuReader()

        while not self._stop.is_set():
            # Short interval: take one sample and log it (no averaging).
            if self._interval < 2.0:
                if self._stop.is_set():
                    break
                snap = sample_now()
                try:
                    log(f"[hw] {_format(snap)}")
                except Exception:
                    pass
                self._stop.wait(self._interval)
                continue

            # Long interval: sample every second, average, then log.
            gpu_points: list[dict] = []
            cpu_points: list[float] = []
            ram_points: list[dict] = []

            cpu.sample()  # prime the first /proc/stat reading
            deadline = time.time() + self._interval

            while time.time() < deadline and not self._stop.is_set():
                time.sleep(1.0)
                if self._stop.is_set():
                    break

                g = _sample_gpu()
                if g:
                    gpu_points.append(g)

                c = cpu.sample()
                if c is not None:
                    cpu_points.append(c)

                r = _sample_ram()
                if r:
                    ram_points.append(r)

            if self._stop.is_set():
                break

            # Compute means across all data points.
            averages: dict = {}
            if cpu_points:
                averages["cpu_pct"] = sum(cpu_points) / len(cpu_points)
            if ram_points:
                averages["ram_total_gb"] = ram_points[-1]["ram_total_gb"]
                averages["ram_used_pct"] = sum(r["ram_used_pct"] for r in ram_points) / len(ram_points)
            if gpu_points:
                averages["gpu_pct"] = sum(g["gpu_pct"] for g in gpu_points) / len(gpu_points)
                averages["vram_used_mib"] = sum(g["vram_used_mib"] for g in gpu_points) / len(gpu_points)
                averages["vram_total_mib"] = gpu_points[0]["vram_total_mib"]

            try:
                log(f"[hw] {_format(averages)}")
            except Exception:
                pass
