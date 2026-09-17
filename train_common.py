"""
Shared training configuration and utilities.

All training logic lives in the train_*.py modules. train_all.py and the Colab
notebook only sequence calls into them, so the two venues cannot drift apart.
"""

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path

from app.services.artifacts import DATA_DIR, MODELS_DIR, REPO_ROOT

EXPORT_FALLBACK = REPO_ROOT / "_export_fallback"

# Put this on persistent storage (e.g. Google Drive on Colab) so an interrupted
# run can resume instead of starting over.
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR") or REPO_ROOT / "checkpoints")

DEFAULT_BASE_MODEL = "microsoft/deberta-v3-base"

LABELS = ["Negative", "Neutral", "Positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# BIO tags for aspect term extraction.
ATE_LABELS = ["O", "B-ASP", "I-ASP"]
ATE_LABEL2ID = {label: i for i, label in enumerate(ATE_LABELS)}
ATE_ID2LABEL = {i: label for label, i in ATE_LABEL2ID.items()}


@dataclass
class TrainConfig:
    """
    One config per training run. Profiles set the GPU-dependent fields; see
    PROFILES below for the concrete values.
    """
    profile: str = "4090"
    model_name: str = DEFAULT_BASE_MODEL
    output_dir: str = "models/run"
    lr: float = 2e-5
    batch_size: int = 32
    grad_accum: int = 1
    epochs: int = 5
    max_length: int = 128
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    fp16: bool = True
    push_to_hub: bool = False
    hub_model_id: str | None = None
    resume: bool = True
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


# batch sizes: (ate_apc, sentiment); amazon rows are set in fetch_real_datasets
PROFILES = {
    "4090": {"batch": 32, "sentiment_batch": 64, "grad_accum": 1, "epochs": 5},
    "t4": {"batch": 16, "sentiment_batch": 32, "grad_accum": 2, "epochs": 5},
    "3050": {"batch": 8, "sentiment_batch": 8, "grad_accum": 4, "epochs": 3},
    "smoke": {"batch": 4, "sentiment_batch": 4, "grad_accum": 1, "epochs": 1},
}


SMOKE_DIR_NAME = "_smoke"


def output_root(profile: str) -> Path:
    """
    Where a profile's trained models are written.

    Smoke runs go to models/_smoke/, which serving never reads. Writing them to
    models/ replaced servable models with 200-row, one-epoch ones - a smoke
    sentiment model rated "top notch quality and works amazingly well" as
    Negative - while /health still reported the service as healthy.
    """
    return MODELS_DIR / SMOKE_DIR_NAME if profile == "smoke" else MODELS_DIR


def config_for(profile: str, task: str, **overrides) -> TrainConfig:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; expected one of {list(PROFILES)}")
    p = PROFILES[profile]
    batch = p["sentiment_batch"] if task == "sentiment" else p["batch"]
    cfg = TrainConfig(
        profile=profile,
        output_dir=str(output_root(profile) / task),
        batch_size=batch,
        grad_accum=p["grad_accum"],
        epochs=p["epochs"],
        max_length=256 if task == "sentiment" else 128,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def warmup_steps_for(n_examples: int, cfg: "TrainConfig") -> int:
    """
    transformers 5 removed warmup_ratio, keeping only warmup_steps, so the
    ratio is converted here once instead of in each trainer.
    """
    per_step = max(1, cfg.batch_size * cfg.grad_accum)
    steps_per_epoch = math.ceil(n_examples / per_step)
    total = max(1, steps_per_epoch * cfg.epochs)
    return max(0, int(total * cfg.warmup_ratio))


def load_model_fp32(auto_class, model_name: str, **kwargs):
    """
    Loads a model in float32, explicitly.

    transformers 5 honours the checkpoint's stored dtype, and
    microsoft/deberta-v3-base stores float16. Training that under AMP raises
    "Attempting to unscale FP16 gradients" the moment the gradient scaler runs.
    Master weights must be fp32; mixed precision is the Trainer's job.
    """
    import torch

    model = auto_class.from_pretrained(model_name, dtype=torch.float32, **kwargs)
    param_dtype = next(model.parameters()).dtype
    if param_dtype != torch.float32:
        raise RuntimeError(
            f"{model_name} loaded as {param_dtype}, expected float32. "
            f"Training under AMP requires fp32 master weights."
        )
    return model


def checkpoint_dir_for(cfg: "TrainConfig", task: str, split_sizes: dict) -> Path:
    """
    Checkpoints are keyed by a fingerprint of the exact run.

    Resuming must only ever continue an identical run. Keying by task alone
    would let `--profile 4090` "resume" from a `--profile smoke` checkpoint -
    inheriting 200-row weights, optimizer state and a learning-rate schedule
    built for a different number of steps - which would silently corrupt the
    one run that matters.
    """
    identity = {
        "task": task,
        "profile": cfg.profile,
        "model": cfg.model_name,
        "lr": cfg.lr,
        "batch": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
        "epochs": cfg.epochs,
        "max_length": cfg.max_length,
        "seed": cfg.seed,
        "splits": split_sizes,
    }
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    path = CHECKPOINT_DIR / f"{task}-{cfg.profile}-{digest}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return path


def resume_point(cfg: "TrainConfig", checkpoint_dir: Path) -> str | None:
    """Latest checkpoint of this exact run, or None to start fresh."""
    if not cfg.resume:
        return None
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(str(checkpoint_dir))
    if last:
        print(f"[resume] continuing from {last}")
    return last


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Build the corpus first: "
            f"uv run python data/fetch_real_datasets.py --profile <profile>"
        )
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def safe_save(model, tokenizer, output_dir: str, name: str) -> str:
    """
    Export with a retry to a fallback path.

    A previous run lost a completed 12-minute training to a Windows file lock
    inside a OneDrive-synced folder, at the final save_pretrained. On a
    one-shot GPU session that must never discard the run, so a failed save
    retries somewhere else and says loudly where the weights went.
    """
    target = Path(output_dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target)
        tokenizer.save_pretrained(target)
        return str(target)
    except OSError as exc:
        fallback = EXPORT_FALLBACK / name
        print(
            f"\n[safe_save] FAILED to write {target}: {exc}\n"
            f"[safe_save] Retrying at {fallback} - the training run is NOT lost."
        )
        fallback.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(fallback)
        tokenizer.save_pretrained(fallback)
        print(f"[safe_save] Weights written to {fallback.resolve()}")
        return str(fallback)


def write_model_card(output_dir: str, name: str, cfg: TrainConfig,
                     metrics: dict, dataset_info: dict, notes: list[str]) -> None:
    """Metrics ship with the weights, so 'which model is this' is answerable."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    (path / "metrics.json").write_text(
        json.dumps({"model": name, "config": cfg.to_json(), "metrics": metrics,
                    "dataset": dataset_info}, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# {name}",
        "",
        f"Base model: `{cfg.model_name}`",
        "",
        "## Training data",
        "",
        "```json",
        json.dumps(dataset_info, indent=2),
        "```",
        "",
        "## Hyperparameters",
        "",
        "```json",
        json.dumps(cfg.to_json(), indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
    ]
    if notes:
        lines += ["", "## Known limitations", ""]
        lines += [f"- {n}" for n in notes]
    (path / "MODEL_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def push_if_requested(cfg: TrainConfig, output_dir: str, name: str) -> str | None:
    """Publish to the Hub so the laptop and containers never need a GPU."""
    if not cfg.push_to_hub or not cfg.hub_model_id:
        return None
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(cfg.hub_model_id, exist_ok=True, private=True)
        api.upload_folder(folder_path=output_dir, repo_id=cfg.hub_model_id)
        print(f"[hub] pushed {name} -> {cfg.hub_model_id}")
        return cfg.hub_model_id
    except Exception as exc:
        # Never fail a completed training run because publication failed.
        print(f"[hub] push failed for {name}: {exc}\n"
              f"[hub] weights remain at {output_dir}")
        return None


def free_disk_gb(path: str = ".") -> float:
    return round(shutil.disk_usage(path).free / 1024**3, 1)


def report_environment() -> dict:
    import torch

    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "free_disk_gb": free_disk_gb(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
    print(f"[env] {json.dumps(info)}")
    return info
