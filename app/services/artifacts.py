"""
Where trained model artifacts live.

This is the single definition shared by training (train_common.py), download
(scripts/download_models.py) and serving (sentiment.py, absa.py). They
previously disagreed: training and download wrote the sentiment model to
models/sentiment/ while SentimentAnalyzer only looked in
app/models/trained_model/sentiment/, so a freshly trained model was never
served and the service told the user to train again.

Paths are anchored to the repository root rather than the working directory,
so the API, the dashboard and the scripts resolve the same files wherever they
are launched from.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Override with MODELS_DIR to serve models from elsewhere (e.g. a mounted
# volume in Docker).
MODELS_DIR = Path(os.getenv("MODELS_DIR") or REPO_ROOT / "models")
DATA_DIR = REPO_ROOT / "data" / "real"

# Where the original DistilBERT (trained on the synthetic corpus) was exported.
# Still honoured as a fallback so older checkouts keep working.
LEGACY_SENTIMENT_DIR = REPO_ROOT / "app" / "models" / "trained_model" / "sentiment"
BASELINE_DIR = REPO_ROOT / "app" / "models" / "trained_model" / "baseline"

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def has_weights(path) -> bool:
    path = Path(path)
    return any((path / name).exists() for name in WEIGHT_FILES)


def model_dir(task: str) -> Path:
    """models/<task>, where training writes and download_models.py fetches."""
    return MODELS_DIR / task


def resolve_sentiment_dir() -> tuple[Path | None, list[Path]]:
    """
    Returns (directory to load, directories searched).

    The model trained by train_sentiment.py wins; the legacy DistilBERT export
    is used only when no new model exists. Returns None when neither has
    weights, along with every path that was checked so the caller can say
    exactly where it looked.
    """
    candidates = [model_dir("sentiment"), LEGACY_SENTIMENT_DIR]
    for candidate in candidates:
        if has_weights(candidate):
            return candidate, candidates
    return None, candidates
