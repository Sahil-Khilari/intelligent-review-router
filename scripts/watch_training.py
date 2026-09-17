"""
Shows live training progress. Run it in a second terminal while train_all.py
(or any single trainer) is running:

    uv run python scripts/watch_training.py

Refreshes every few seconds and exits when the run finishes. Use --once to
print the current state a single time, and --history for the event log.
Needs only the standard library.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path(os.getenv("TRAINING_LOG_DIR") or REPO_ROOT / "logs")
STATE_FILE = LOG_DIR / "training_progress.json"
HISTORY_FILE = LOG_DIR / "training_progress.log"

ICONS = {"pending": "  ", "running": ">>", "done": "ok", "ok": "ok", "failed": "!!"}


def bar(percent, width=30):
    if percent is None:
        return "[" + "?" * width + "]"
    filled = int(width * percent / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def minutes(value):
    if value is None:
        return "--"
    if value >= 60:
        return f"{int(value // 60)}h{int(value % 60):02d}m"
    return f"{value:.1f}m"


def render(state):
    lines = []
    profile = state.get("profile", "?")
    lines.append(f"Training run  profile={profile}  started {state.get('run_started_at', '?')}"
                 f"  (updated {state.get('updated_at', '?')})")
    lines.append("")

    for task, info in state.get("stages", {}).items():
        status = info.get("status", "?")
        line = f"  {ICONS.get(status, '??')} {task:<10} {status:<8}"
        if info.get("minutes") is not None:
            line += f" {minutes(info['minutes'])}"
        if info.get("result"):
            line += "  " + ", ".join(f"{k}={v}" for k, v in info["result"].items())
        if info.get("error"):
            line += f"  ERROR: {info['error']}"
        lines.append(line)

    current = state.get("current")
    if current:
        lines.append("")
        lines.append(f"  Now: {current.get('task')} - {current.get('phase', '')}")
        if current.get("max_steps"):
            lines.append(f"  {bar(current.get('percent'))} {current.get('percent')}%  "
                         f"step {current.get('step')}/{current.get('max_steps')}  "
                         f"epoch {current.get('epoch')}/{current.get('num_epochs')}")
            speed = current.get("steps_per_second")
            lines.append(f"  speed {speed if speed is not None else '--'} steps/s   "
                         f"elapsed {minutes(current.get('elapsed_minutes'))}   "
                         f"remaining ~{minutes(current.get('eta_minutes'))}")
        if current.get("last_loss") is not None:
            lines.append(f"  last training loss {current['last_loss']}")
        if current.get("last_eval"):
            shown = {k: v for k, v in current["last_eval"].items()
                     if any(m in k for m in ("f1", "accuracy", "loss"))}
            lines.append(f"  last validation    {shown}")
        if current.get("gpu_used_gb") is not None:
            gpu = (f"  GPU memory {current['gpu_used_gb']}/{current['gpu_total_gb']} GB "
                   f"({current['gpu_used_pct']}%)")
            if current["gpu_used_pct"] >= 97:
                gpu += "  <- nearly full"
            lines.append(gpu)
        if current.get("slowdown"):
            lines.append("")
            lines.append("  !! SLOWDOWN: training is much slower than earlier in this stage.")
            lines.append("     On a laptop GPU this usually means memory is spilling into")
            lines.append("     system RAM. Close other GPU programs, or stop and use a")
            lines.append("     smaller profile.")

        # A state file that stops updating means the trainer died or hung.
        try:
            age = time.time() - STATE_FILE.stat().st_mtime
            if age > 300:
                lines.append("")
                lines.append(f"  !! No update for {int(age // 60)} min - is training still running?")
        except OSError:
            pass

    if state.get("run_finished_at"):
        lines.append("")
        lines.append(f"  Run finished at {state['run_finished_at']} "
                     f"after {minutes(state.get('total_minutes'))}.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between refreshes")
    parser.add_argument("--once", action="store_true", help="print once and exit")
    parser.add_argument("--history", action="store_true", help="print the event log and exit")
    args = parser.parse_args()

    if args.history:
        if HISTORY_FILE.exists():
            print(HISTORY_FILE.read_text(encoding="utf-8"))
        else:
            print(f"No history yet ({HISTORY_FILE}).")
        return 0

    while True:
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except FileNotFoundError:
            state = None
        except (OSError, json.JSONDecodeError):
            time.sleep(0.5)
            continue

        if not args.once:
            os.system("cls" if os.name == "nt" else "clear")
        if state is None:
            print(f"No training progress yet ({STATE_FILE}).\n"
                  "Start training with: uv run python train_all.py --profile <gpu>")
        else:
            print(render(state))

        if args.once or (state and state.get("run_finished_at")):
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
