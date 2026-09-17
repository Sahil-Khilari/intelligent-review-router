"""
Trains every model in one command.

This is the entrypoint for the GPU session. It is non-interactive and
sequential by design: the intended use is a single run on a borrowed 4090,
where a half-finished pipeline means another trip.

    uv run python train_all.py --profile 4090 --push-to-user <your-hf-username>

Each stage is independent. If one fails, the others still complete and their
weights are kept, and the failure is reported in the summary rather than
aborting everything after it.

Validate the pipeline first without burning GPU time on a real run:

    uv run python data/fetch_real_datasets.py --profile smoke
    uv run python train_all.py --profile smoke

Measured on this pipeline: the smoke profile takes under two minutes. A full
4090 run takes roughly 1.5-2 hours, almost all of it review sentiment
(5 epochs over ~320k Amazon reviews); a Colab T4 takes roughly 3 hours.

An interrupted run resumes from its last checkpoint when re-run with the same
arguments. Checkpoints are keyed to the exact configuration, so a smoke run can
never be resumed into a real one. Set CHECKPOINT_DIR to persistent storage (for
example Google Drive on Colab) so a disconnect does not lose them.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import train_apc  # noqa: E402
import train_ate  # noqa: E402
import train_sentiment  # noqa: E402
from train_common import config_for, output_root, report_environment  # noqa: E402

STAGES = [
    ("ate", train_ate),
    ("apc", train_apc),
    ("sentiment", train_sentiment),
]


def run(profile, hub_user=None, only=None, model=None, resume=True):
    report_environment()
    summary, started = {}, time.time()

    for task, module in STAGES:
        if only and task not in only:
            print(f"\n=== skipping {task} ===")
            continue

        print(f"\n{'=' * 70}\n=== {task} ({profile})\n{'=' * 70}")
        cfg = config_for(profile, task)
        if model:
            cfg.model_name = model
        cfg.resume = resume
        if hub_user:
            cfg.push_to_hub = True
            cfg.hub_model_id = f"{hub_user}/review-router-{task}"

        stage_started = time.time()
        try:
            result = module.train(cfg)
            summary[task] = {
                "status": "ok",
                "minutes": round((time.time() - stage_started) / 60, 1),
                "saved_to": result["saved_to"],
                "metrics": {
                    k: round(v, 4)
                    for k, v in result["metrics"].items()
                    if isinstance(v, float)
                },
            }
        except Exception as exc:
            # One stage failing must not discard the others.
            traceback.print_exc()
            summary[task] = {
                "status": "failed",
                "minutes": round((time.time() - stage_started) / 60, 1),
                "error": f"{type(exc).__name__}: {exc}",
            }

    summary["_total_minutes"] = round((time.time() - started) / 60, 1)
    summary["_profile"] = profile

    root = output_root(profile)
    root.mkdir(parents=True, exist_ok=True)
    (root / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 70}\n=== SUMMARY\n{'=' * 70}")
    for task, _ in STAGES:
        entry = summary.get(task)
        if not entry:
            continue
        if entry["status"] == "ok":
            headline = {
                k: v for k, v in entry["metrics"].items()
                if k.startswith("test_") and ("f1" in k or "accuracy" in k)
            }
            print(f"  {task:<10} ok      {entry['minutes']:>5} min  {headline}")
        else:
            print(f"  {task:<10} FAILED  {entry['minutes']:>5} min  {entry['error']}")
    print(f"  total      {summary['_total_minutes']} min")
    print(f"\nWrote {root / 'training_summary.json'}")
    if profile == "smoke":
        print("Smoke models are in models/_smoke/ and are never served. The "
              "pipeline works; run a real profile to produce servable models.")

    failed = [t for t, _ in STAGES if summary.get(t, {}).get("status") == "failed"]
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="4090",
                        choices=["4090", "t4", "3050", "smoke"])
    parser.add_argument("--push-to-user", default=None,
                        help="HF username; publishes <user>/review-router-<task>")
    parser.add_argument("--only", nargs="*", default=None,
                        choices=["ate", "apc", "sentiment"],
                        help="run a subset of stages")
    parser.add_argument("--model", default=None,
                        help="override the base model for every stage "
                             "(e.g. roberta-base if deberta-v3 misbehaves)")
    parser.add_argument("--no-resume", action="store_true",
                        help="start every stage fresh, ignoring checkpoints left "
                             "by an interrupted identical run")
    args = parser.parse_args()

    return run(args.profile, args.push_to_user, args.only, args.model,
               resume=not args.no_resume)


if __name__ == "__main__":
    sys.exit(main())
