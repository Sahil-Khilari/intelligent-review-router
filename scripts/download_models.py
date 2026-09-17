"""
Fetches trained models from the Hugging Face Hub into models/.

Training happens once on a GPU machine; everything else downloads. Run this
after cloning, or any time models/ is empty:

    uv run python scripts/download_models.py --user <your-hf-username>

Private repos need a token: `uv run hf auth login`, or set HF_TOKEN.
(`huggingface-cli` is deprecated in huggingface_hub 1.x and no longer works.)
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.artifacts import MODELS_DIR, has_weights, model_dir  # noqa: E402

TASKS = ["ate", "apc", "sentiment"]


def download(repo_id: str, target: Path, token: str | None) -> bool:
    from huggingface_hub import snapshot_download

    print(f"  {repo_id} -> {target}")
    try:
        cached = snapshot_download(repo_id=repo_id, token=token)
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return False

    target.mkdir(parents=True, exist_ok=True)
    for item in Path(cached).iterdir():
        if item.is_file():
            shutil.copy2(item, target / item.name)
    return has_weights(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="HF username or org")
    parser.add_argument("--token", default=None, help="HF token for private repos")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if weights are already present")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for task in TASKS:
        target = model_dir(task)
        if has_weights(target) and not args.force:
            print(f"  {task}: already present, skipping (--force to refetch)")
            results[task] = True
            continue
        results[task] = download(f"{args.user}/review-router-{task}", target, args.token)

    print("\nsummary:")
    for task, ok in results.items():
        print(f"  {task:<10} {'ok' if ok else 'MISSING'}")

    missing = [t for t, ok in results.items() if not ok]
    if missing:
        print(
            f"\n{len(missing)} model(s) unavailable: {missing}\n"
            f"The service will start on its fallbacks and report degraded=true "
            f"at /health. Train them with: "
            f"uv run python train_all.py --profile <gpu> --push-to-user {args.user}"
        )
        return 1

    print("\nAll models present. /health should now report degraded=false.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
