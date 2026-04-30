"""
power.py — CPU energy measurement

Priority:
  1. Linux Intel RAPL  (/sys/class/powercap/intel-rapl)
  2. macOS powermetrics (requires sudo — skipped silently if unavailable)
  3. psutil process CPU-percent proxy (always available, least accurate)

All read_energy() calls return Joules elapsed since the previous call
(or since start_session()).  Call start_session() once before the
benchmarking loop, then wrap each inference run:

    e0 = power.read_energy()
    run_inference()
    e1 = power.read_energy()
    joules = e1 - e0
"""

import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# RAPL helpers (Linux)
# ---------------------------------------------------------------------------

_RAPL_ROOT = Path("/sys/class/powercap/intel-rapl")


def _rapl_available() -> bool:
    return _RAPL_ROOT.exists()


def _rapl_read_uj() -> int:
    """Sum energy_uj across all RAPL package domains."""
    total = 0
    for domain in _RAPL_ROOT.glob("intel-rapl:*"):
        if ":" in domain.name[len("intel-rapl:"):]:
            continue  # skip sub-domains (cores, uncore) to avoid double-count
        ej = domain / "energy_uj"
        if ej.exists():
            try:
                total += int(ej.read_text().strip())
            except (ValueError, OSError):
                pass
    return total


def _rapl_max_uj() -> int:
    total = 0
    for domain in _RAPL_ROOT.glob("intel-rapl:*"):
        if ":" in domain.name[len("intel-rapl:"):]:
            continue
        mj = domain / "max_energy_range_uj"
        if mj.exists():
            try:
                total += int(mj.read_text().strip())
            except (ValueError, OSError):
                pass
    return total or (2**32)


# ---------------------------------------------------------------------------
# psutil proxy (cross-platform fallback)
# ---------------------------------------------------------------------------

class _PsutilProxy:
    """
    Estimates energy from CPU utilisation × TDP.
    Accuracy: rough (±30 %), but portable and always available.
    Default TDP = 15 W (laptop CPU).  Override via env TDP_WATTS.
    """
    TDP_W = float(os.environ.get("TDP_WATTS", 15.0))

    def __init__(self):
        import psutil
        self._psutil = psutil
        self._last_time = time.perf_counter()
        # Prime the pump — first call always returns 0.0
        self._psutil.cpu_percent(interval=None)

    def read_joules(self) -> float:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        cpu_frac = self._psutil.cpu_percent(interval=None) / 100.0
        return cpu_frac * self.TDP_W * dt


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

class PowerMeter:
    """
    Single object that picks the best available backend and exposes
    start() / joules_since_start() / sample() for a single run.
    """

    def __init__(self):
        self.backend: str
        self._rapl_wrap: int = 0
        self._rapl_base_uj: int = 0
        self._rapl_max: int = 1

        if platform.system() == "Linux" and _rapl_available():
            self.backend = "rapl"
            self._rapl_max = _rapl_max_uj()
        else:
            self.backend = "psutil"
            self._proxy = _PsutilProxy()

        self._session_joules: float = 0.0
        self._last_joules: float = 0.0

    # ---- low-level reads --------------------------------------------------

    def _read_rapl_joules(self) -> float:
        raw = _rapl_read_uj()
        # Handle counter wraparound
        if raw < self._rapl_base_uj:
            self._rapl_wrap += self._rapl_max
        self._rapl_base_uj = raw
        return (raw + self._rapl_wrap) / 1e6  # µJ → J

    def _current_joules(self) -> float:
        if self.backend == "rapl":
            return self._read_rapl_joules()
        else:
            # psutil proxy is differential — accumulate
            self._session_joules += self._proxy.read_joules()
            return self._session_joules

    # ---- public API -------------------------------------------------------

    def start(self):
        """Call once before the timed section."""
        if self.backend == "rapl":
            self._rapl_wrap = 0
            self._rapl_base_uj = _rapl_read_uj()
        else:
            self._session_joules = 0.0
            self._proxy.read_joules()  # reset dt clock
        self._last_joules = self._current_joules()

    def delta(self) -> float:
        """Joules consumed since last call to start() or delta()."""
        now = self._current_joules()
        d = now - self._last_joules
        self._last_joules = now
        return max(d, 0.0)

    def info(self) -> str:
        return f"PowerMeter(backend={self.backend})"


# Module-level singleton
_meter: Optional[PowerMeter] = None


def get_meter() -> PowerMeter:
    global _meter
    if _meter is None:
        _meter = PowerMeter()
    return _meter