"""
Training configuration: where runs write, and when they may resume.

Both failure modes here are silent. A smoke run writing into models/ replaces a
servable model with a junk one while /health still reports healthy; a resume
keyed too loosely lets a real run continue from a smoke checkpoint.
"""

import pytest

import train_common
from train_common import (
    config_for,
    checkpoint_dir_for,
    output_root,
    resume_point,
    warmup_steps_for,
)


@pytest.fixture
def checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(train_common, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    return tmp_path / "checkpoints"


SPLITS = {"train": 3000, "val": 300, "test": 400}


def test_smoke_models_never_land_where_serving_reads():
    assert output_root("smoke").name == "_smoke"
    assert output_root("smoke").parent == train_common.MODELS_DIR
    for profile in ("4090", "t4", "3050"):
        assert output_root(profile) == train_common.MODELS_DIR


def test_config_output_dir_follows_the_profile():
    assert "_smoke" in config_for("smoke", "sentiment").output_dir
    assert "_smoke" not in config_for("4090", "sentiment").output_dir


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        config_for("a100", "ate")


def test_sentiment_gets_longer_sequences_and_its_own_batch():
    sentiment, ate = config_for("4090", "sentiment"), config_for("4090", "ate")
    assert sentiment.max_length == 256 and ate.max_length == 128
    assert sentiment.batch_size == 64 and ate.batch_size == 32


def test_identical_runs_share_a_checkpoint_dir(checkpoints):
    first = checkpoint_dir_for(config_for("4090", "ate"), "ate", SPLITS)
    second = checkpoint_dir_for(config_for("4090", "ate"), "ate", SPLITS)
    assert first == second
    assert (first / "run_identity.json").exists()


def test_smoke_can_never_be_resumed_into_a_real_run(checkpoints):
    """The failure this guards: a 4090 run inheriting a 200-row checkpoint."""
    smoke = checkpoint_dir_for(config_for("smoke", "sentiment"), "sentiment", SPLITS)
    real = checkpoint_dir_for(config_for("4090", "sentiment"), "sentiment", SPLITS)
    assert smoke != real


def test_different_data_means_a_different_run(checkpoints):
    cfg = config_for("4090", "apc")
    a = checkpoint_dir_for(cfg, "apc", SPLITS)
    b = checkpoint_dir_for(cfg, "apc", {**SPLITS, "train": 2999})
    assert a != b


def test_resume_point_is_none_without_checkpoints(checkpoints):
    cfg = config_for("4090", "ate")
    path = checkpoint_dir_for(cfg, "ate", SPLITS)
    assert resume_point(cfg, path) is None


def test_resume_point_finds_the_latest_checkpoint(checkpoints):
    cfg = config_for("4090", "ate")
    path = checkpoint_dir_for(cfg, "ate", SPLITS)
    for step in (100, 250):
        (path / f"checkpoint-{step}").mkdir()
    assert resume_point(cfg, path).endswith("checkpoint-250")


def test_no_resume_ignores_existing_checkpoints(checkpoints):
    cfg = config_for("4090", "ate")
    path = checkpoint_dir_for(cfg, "ate", SPLITS)
    (path / "checkpoint-100").mkdir()
    cfg.resume = False
    assert resume_point(cfg, path) is None


def test_warmup_is_ten_percent_of_optimizer_steps():
    cfg = config_for("4090", "ate")          # batch 32, accum 1, 5 epochs
    assert warmup_steps_for(3200, cfg) == 50  # 100 steps/epoch * 5 * 0.1
    cfg.grad_accum = 2
    assert warmup_steps_for(3200, cfg) == 25
