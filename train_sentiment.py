"""
Review-level sentiment on real Amazon reviews.

Replaces train_transformer.py, which trained on 1500 template-generated reviews
where 31.7% of the held-out split appeared verbatim in train, and which selected
its best checkpoint on the test set and then reported that same score as the
result. Both are fixed here: real data with an enforced leakage gate, and a
three-way split where selection happens on validation and test is touched once.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.services.device import resolve_device  # noqa: E402
from train_progress import ProgressCallback  # noqa: E402
from train_common import (  # noqa: E402
    DATA_DIR,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    TrainConfig,
    checkpoint_dir_for,
    config_for,
    load_jsonl,
    load_model_fp32,
    push_if_requested,
    report_environment,
    resume_point,
    safe_save,
    warmup_steps_for,
    write_model_card,
)

TASK = "sentiment"


def make_compute_metrics():
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
            "weighted_f1": f1_score(labels, predictions, average="weighted"),
        }

    return compute_metrics


def train(cfg: TrainConfig) -> dict:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    device = resolve_device("Sentiment training")
    report_environment()
    set_seed(cfg.seed)

    limit = 200 if cfg.profile == "smoke" else None
    raw = {
        name: load_jsonl(DATA_DIR / f"amazon_{name}.jsonl", limit=limit)
        for name in ("train", "val", "test")
    }

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def encode(batch):
        # Dynamic padding via the collator. The old script padded every sample
        # to max_length, wasting 2-4x compute on a corpus of short reviews.
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    encoded, stats = {}, {}
    for name, rows in raw.items():
        for row in rows:
            row["label"] = LABEL2ID[row["label"]] if isinstance(row["label"], str) else row["label"]
        distribution = Counter(ID2LABEL[r["label"]] for r in rows)
        stats[name] = {"rows": len(rows), "distribution": dict(distribution)}
        print(f"[{TASK}] {name}: {stats[name]}")
        dataset = Dataset.from_list(
            [{"text": r["text"], "label": r["label"]} for r in rows]
        )
        encoded[name] = dataset.map(encode, batched=True, remove_columns=["text"])

    model = load_model_fp32(
        AutoModelForSequenceClassification,
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    checkpoint_dir = checkpoint_dir_for(
        cfg, TASK, {split: len(ds) for split, ds in encoded.items()}
    )

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps_for(len(encoded["train"]), cfg),
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        # Neutral (3-star) stays the hardest class even after balancing, and it
        # is the class that matters for routing: a mixed review is not praise.
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=cfg.fp16 and device.type == "cuda",
        # Real reviews have a median of 74 tokens but 7% hit the 256 cap, so
        # almost every random batch of 64 contains a long one and pads the
        # whole batch to 256. Grouping similar lengths removes that waste.
        train_sampling_strategy="group_by_length",
        seed=cfg.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["val"],          # selection on validation, not test
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_compute_metrics(),
        callbacks=[ProgressCallback(TASK)],
    )

    trainer.train(resume_from_checkpoint=resume_point(cfg, checkpoint_dir))

    val_metrics = trainer.evaluate(encoded["val"], metric_key_prefix="val")
    test_metrics = trainer.evaluate(encoded["test"], metric_key_prefix="test")
    metrics = {**val_metrics, **test_metrics}
    print(f"[{TASK}] {json.dumps({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})}")

    saved_to = safe_save(model, tokenizer, cfg.output_dir, TASK)
    write_model_card(
        saved_to,
        "review-router-sentiment",
        cfg,
        metrics,
        {"dataset": "McAuley-Lab/Amazon-Reviews-2023", "splits": stats},
        notes=[
            "Labels derive from star ratings (1-2 Negative, 3 Neutral, "
            "4-5 Positive). 3-star reviews are genuinely mixed rather than "
            "neutral, which is the main source of error and the reason "
            "3-class accuracy sits well below binary accuracy on this corpus.",
            "Classes were balanced to the scarcest (3-star) before splitting, "
            "so accuracy is not inflated by skew.",
            "Train/val/test overlap is asserted empty by "
            "data/fetch_real_datasets.py and fails the build if violated.",
        ],
    )
    push_if_requested(cfg, saved_to, TASK)
    return {"metrics": metrics, "stats": stats, "saved_to": saved_to}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="4090")
    parser.add_argument("--model", default=None)
    parser.add_argument("--push-to", default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore checkpoints from an interrupted identical run")
    args = parser.parse_args()

    cfg = config_for(args.profile, TASK)
    if args.model:
        cfg.model_name = args.model
    if args.push_to:
        cfg.push_to_hub, cfg.hub_model_id = True, args.push_to
    if args.no_resume:
        cfg.resume = False

    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
