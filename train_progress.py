"""
Live training progress, readable while training runs.

Every trainer attaches ProgressCallback, and train_all.py records which stages
are pending, running, done or failed. Everything is written to
logs/training_progress.json (current state) and logs/training_progress.log
(history), so progress can be followed from a second terminal with:

    uv run python scripts/watch_training.py

A slowdown is flagged when training becomes much slower than it was earlier in
the same stage. On a laptop GPU that is the usual symptom of GPU memory
spilling into system RAM, which makes training silently many times slower.
"""

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from transformers import TrainerCallback

from app.services.artifacts import REPO_ROOT

LOG_DIR = Path(os.getenv("TRAINING_LOG_DIR") or REPO_ROOT / "logs")
STATE_FILE = LOG_DIR / "training_progress.json"
HISTORY_FILE = LOG_DIR / "training_progress.log"

# A stage is flagged as slowed down when its recent speed drops below this
# fraction of the speed it reached early on.
SLOWDOWN_RATIO = 0.4


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(state: dict) -> None:
    """Atomic replace, so the watcher never reads a half-written file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    for _ in range(5):
        try:
            os.replace(tmp, STATE_FILE)
            return
        except PermissionError:
            # Windows: the watcher may have the file open for a moment.
            time.sleep(0.1)


def log_event(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()}  {message}\n")


def _gpu_memory() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return {
            "gpu_used_gb": round(used / 1024**3, 2),
            "gpu_total_gb": round(total / 1024**3, 2),
            "gpu_used_pct": round(100 * used / total, 1),
        }
    except Exception:
        return {}


# ------------------------------------------------------------- run / stages

def run_started(profile: str, stages: list[str]) -> None:
    state = {
        "profile": profile,
        "run_started_at": _now(),
        "stages": {s: {"status": "pending"} for s in stages},
        "current": None,
    }
    _write_state(state)
    log_event(f"run started: profile={profile} stages={','.join(stages)}")


def stage_started(task: str) -> None:
    state = _read_state()
    state.setdefault("stages", {})[task] = {"status": "running", "started_at": _now()}
    state["current"] = {"task": task, "phase": "loading"}
    _write_state(state)
    log_event(f"{task}: started")


def stage_finished(task: str, status: str, minutes: float, headline: dict | None = None,
                   error: str | None = None) -> None:
    state = _read_state()
    entry = state.setdefault("stages", {}).setdefault(task, {})
    entry.update({"status": status, "finished_at": _now(), "minutes": minutes})
    if headline:
        entry["result"] = headline
    if error:
        entry["error"] = error
    if (state.get("current") or {}).get("task") == task:
        state["current"] = None
    _write_state(state)
    log_event(f"{task}: {status} after {minutes} min"
              + (f" {headline}" if headline else "")
              + (f" ERROR {error}" if error else ""))


def run_finished(total_minutes: float) -> None:
    state = _read_state()
    state["run_finished_at"] = _now()
    state["total_minutes"] = total_minutes
    state["current"] = None
    _write_state(state)
    log_event(f"run finished after {total_minutes} min")


# ------------------------------------------------------------------ callback

class ProgressCallback(TrainerCallback):
    """Writes step, epoch, speed, ETA, loss and eval metrics while training."""

    def __init__(self, task: str, every_seconds: float = 5.0):
        self.task = task
        self.every = every_seconds
        self._last_write = 0.0
        self._start = None
        self._start_step = 0
        self._window = deque(maxlen=30)       # (time, step) samples
        self._best_rate = 0.0
        self._slow = False
        self._loss = None
        self._eval = None

    def _update(self, state, force=False, phase="training"):
        now = time.time()
        if not force and now - self._last_write < self.every:
            return
        self._last_write = now

        step, total = state.global_step, state.max_steps or 0
        self._window.append((now, step))
        elapsed = now - (self._start or now)

        rate = None
        if len(self._window) >= 2:
            (t0, s0), (t1, s1) = self._window[0], self._window[-1]
            if t1 > t0 and s1 > s0:
                rate = (s1 - s0) / (t1 - t0)

        if rate:
            # Judge "normal" speed only once the stage has warmed up.
            if step - self._start_step > 20:
                self._best_rate = max(self._best_rate, rate)
            slow_now = self._best_rate > 0 and rate < SLOWDOWN_RATIO * self._best_rate
            if slow_now and not self._slow:
                log_event(
                    f"{self.task}: SLOWDOWN - {rate:.2f} steps/s vs {self._best_rate:.2f} "
                    "earlier. On a laptop GPU this usually means GPU memory is "
                    "spilling into system RAM."
                )
            self._slow = slow_now

        remaining = (total - step) / rate if rate and total else None
        current = {
            "task": self.task,
            "phase": phase,
            "step": step,
            "max_steps": total,
            "percent": round(100 * step / total, 1) if total else None,
            "epoch": round(state.epoch or 0, 2),
            "num_epochs": state.num_train_epochs,
            "steps_per_second": round(rate, 3) if rate else None,
            "elapsed_minutes": round(elapsed / 60, 1),
            "eta_minutes": round(remaining / 60, 1) if remaining is not None else None,
            "last_loss": self._loss,
            "last_eval": self._eval,
            "best_metric": state.best_metric,
            "slowdown": self._slow,
            **_gpu_memory(),
        }
        progress = _read_state()
        progress["current"] = current
        _write_state(progress)

    def on_train_begin(self, args, state, control, **kwargs):
        self._start = time.time()
        self._start_step = state.global_step
        progress = _read_state()
        if progress.pop("run_finished_at", None):
            # A trainer launched on its own after an earlier full run.
            _write_state(progress)
        if state.global_step:
            log_event(f"{self.task}: resumed at step {state.global_step}")
        self._update(state, force=True)

    def on_step_end(self, args, state, control, **kwargs):
        self._update(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._loss = round(float(logs["loss"]), 4)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self._eval = {k: round(v, 4) for k, v in metrics.items()
                          if isinstance(v, float) and not k.endswith(("runtime", "per_second"))}
            log_event(f"{self.task}: epoch {state.epoch:.0f} eval {self._eval}")
        self._update(state, force=True, phase="evaluating")

    def on_save(self, args, state, control, **kwargs):
        log_event(f"{self.task}: checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        self._update(state, force=True, phase="final evaluation and export")
