"""
Where trained models are looked for.

Training and download wrote the sentiment model to models/sentiment/ while the
service only looked in app/models/trained_model/sentiment/, so a freshly
trained model was never served and the service told the user to train again.
These tests pin the resolution order so that cannot silently regress.
"""

import pytest

from app.services import artifacts
from app.services.sentiment import SentimentAnalyzer


def make_weights(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"")
    return path


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    models = tmp_path / "models"
    legacy = tmp_path / "legacy" / "sentiment"
    monkeypatch.setattr(artifacts, "MODELS_DIR", models)
    monkeypatch.setattr(artifacts, "LEGACY_SENTIMENT_DIR", legacy)
    return models, legacy


def test_has_weights(tmp_path):
    assert artifacts.has_weights(tmp_path) is False
    (tmp_path / "config.json").write_text("{}")
    assert artifacts.has_weights(tmp_path) is False, "metadata alone is not a model"
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert artifacts.has_weights(tmp_path) is True


def test_model_dir_is_under_models(isolated):
    models, _ = isolated
    assert artifacts.model_dir("ate") == models / "ate"


def test_newly_trained_sentiment_model_wins(isolated):
    """The regression: a model in models/sentiment must be the one served."""
    models, legacy = isolated
    trained = make_weights(models / "sentiment")
    make_weights(legacy)
    found, searched = artifacts.resolve_sentiment_dir()
    assert found == trained
    assert searched == [trained, legacy]


def test_legacy_export_is_the_fallback(isolated):
    _, legacy = isolated
    make_weights(legacy)
    found, _ = artifacts.resolve_sentiment_dir()
    assert found == legacy


def test_metadata_only_legacy_folder_is_not_a_model(isolated):
    """A fresh clone ships config.json without weights in the legacy folder."""
    models, legacy = isolated
    legacy.mkdir(parents=True)
    (legacy / "config.json").write_text("{}")
    found, searched = artifacts.resolve_sentiment_dir()
    assert found is None
    assert searched == [models / "sentiment", legacy]


def test_analyzer_serves_baseline_and_says_so_when_nothing_is_trained(isolated, capsys):
    analyzer = SentimentAnalyzer()
    assert analyzer.model_type == "baseline"
    assert analyzer.weights_missing is True
    message = capsys.readouterr().out
    assert "models" in message and "sentiment" in message
    assert "uv run python" in message, "the hint must be runnable as printed"


def test_analyzer_honours_an_explicit_empty_model_dir(tmp_path):
    analyzer = SentimentAnalyzer(model_dir=str(tmp_path))
    assert analyzer.model_type == "baseline"
    assert analyzer.weights_missing is True


def test_repo_root_contains_the_app():
    assert (artifacts.REPO_ROOT / "app" / "services" / "artifacts.py").exists()
