"""
Live training progress: the state file and the watcher that reads it.

A run that silently slowed to a crawl for hours - GPU memory spilling into
system RAM - went unnoticed because nothing reported progress. These pin the
reporting so it keeps working.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import train_progress as progress

WATCHER = Path(__file__).resolve().parent.parent / "scripts" / "watch_training.py"


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "LOG_DIR", tmp_path)
    monkeypatch.setattr(progress, "STATE_FILE", tmp_path / "training_progress.json")
    monkeypatch.setattr(progress, "HISTORY_FILE", tmp_path / "training_progress.log")
    monkeypatch.setattr(progress, "_gpu_memory", lambda: {
        "gpu_used_gb": 5.9, "gpu_total_gb": 6.0, "gpu_used_pct": 98.3})
    return tmp_path


def state(logs):
    return json.loads((logs / "training_progress.json").read_text(encoding="utf-8"))


def trainer_state(step, max_steps=1000, epoch=0.5):
    return SimpleNamespace(global_step=step, max_steps=max_steps, epoch=epoch,
                           num_train_epochs=3, best_metric=None)


def test_stage_lifecycle_is_recorded(logs):
    progress.run_started("3050", ["ate", "apc"])
    assert state(logs)["stages"] == {"ate": {"status": "pending"},
                                     "apc": {"status": "pending"}}

    progress.stage_started("ate")
    assert state(logs)["stages"]["ate"]["status"] == "running"
    assert state(logs)["current"]["task"] == "ate"

    progress.stage_finished("ate", "done", 7.2, headline={"test_f1": 0.846})
    s = state(logs)
    assert s["stages"]["ate"] == {**s["stages"]["ate"], "status": "done",
                                  "minutes": 7.2, "result": {"test_f1": 0.846}}
    assert s["current"] is None

    progress.stage_started("apc")
    progress.stage_finished("apc", "failed", 0.3, error="CUDA out of memory")
    assert state(logs)["stages"]["apc"]["error"] == "CUDA out of memory"

    progress.run_finished(7.5)
    assert state(logs)["total_minutes"] == 7.5

    history = (logs / "training_progress.log").read_text(encoding="utf-8")
    assert "ate: done" in history and "apc: failed" in history


def test_callback_reports_percent_speed_and_eta(logs, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(progress.time, "time", lambda: clock["t"])
    cb = progress.ProgressCallback("sentiment", every_seconds=0)

    cb.on_train_begin(None, trainer_state(0), None)
    for step in (100, 200):
        clock["t"] += 50                      # 2 steps per second
        cb.on_step_end(None, trainer_state(step), None)

    current = state(logs)["current"]
    assert current["task"] == "sentiment"
    assert current["percent"] == 20.0
    assert current["steps_per_second"] == 2.0
    assert current["eta_minutes"] == pytest.approx(800 / 2 / 60, abs=0.1)
    assert current["gpu_used_pct"] == 98.3


def test_callback_flags_a_slowdown(logs, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(progress.time, "time", lambda: clock["t"])
    cb = progress.ProgressCallback("sentiment", every_seconds=0)
    cb._window = progress.deque(maxlen=2)     # judge speed over the last interval

    cb.on_train_begin(None, trainer_state(0), None)
    step = 0
    for _ in range(4):                        # healthy: 4 steps/s
        clock["t"] += 10
        step += 40
        cb.on_step_end(None, trainer_state(step), None)
    assert state(logs)["current"]["slowdown"] is False

    for _ in range(3):                        # memory spill: 0.2 steps/s
        clock["t"] += 10
        step += 2
        cb.on_step_end(None, trainer_state(step), None)
    assert state(logs)["current"]["slowdown"] is True
    assert "SLOWDOWN" in (logs / "training_progress.log").read_text(encoding="utf-8")


def test_callback_records_loss_and_evaluation(logs):
    cb = progress.ProgressCallback("apc", every_seconds=0)
    cb.on_train_begin(None, trainer_state(0), None)
    cb.on_log(None, trainer_state(50), None, logs={"loss": 0.52341})
    cb.on_evaluate(None, trainer_state(100, epoch=1.0), None,
                   metrics={"eval_macro_f1": 0.76361, "eval_runtime": 3.2})
    current = state(logs)["current"]
    assert current["last_loss"] == 0.5234
    assert current["last_eval"] == {"eval_macro_f1": 0.7636}
    assert current["phase"] == "evaluating"


def load_watcher():
    spec = importlib.util.spec_from_file_location("watch_training", WATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_watcher_renders_stages_progress_and_warnings():
    watcher = load_watcher()
    text = watcher.render({
        "profile": "4090",
        "stages": {"ate": {"status": "done", "minutes": 4.0, "result": {"test_f1": 0.85}},
                   "sentiment": {"status": "running"}},
        "current": {"task": "sentiment", "phase": "training", "step": 500,
                    "max_steps": 1000, "percent": 50.0, "epoch": 2.5, "num_epochs": 5,
                    "steps_per_second": 0.2, "elapsed_minutes": 70, "eta_minutes": 41.7,
                    "last_loss": 0.61, "gpu_used_gb": 5.9, "gpu_total_gb": 6.0,
                    "gpu_used_pct": 98.3, "slowdown": True},
    })
    assert "test_f1=0.85" in text
    assert "50.0%" in text and "step 500/1000" in text
    assert "1h10m" in text                    # elapsed formatting
    assert "nearly full" in text
    assert "SLOWDOWN" in text


def test_watcher_once_without_a_run(tmp_path, monkeypatch, capsys):
    watcher = load_watcher()
    monkeypatch.setattr(watcher, "STATE_FILE", tmp_path / "missing.json")
    monkeypatch.setattr("sys.argv", ["watch_training.py", "--once"])
    assert watcher.main() == 0
    assert "No training progress yet" in capsys.readouterr().out
