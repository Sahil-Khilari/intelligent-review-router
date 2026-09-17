"""
Aspect Polarity Classification: sentence-pair classification.

Encodes [CLS] review [SEP] aspect_term [SEP], so the same sentence can carry
opposite polarities for different aspects. That is what replaces the clause
splitting heuristics in app/services/preprocessing.py - "the phone is great but
delivery was late" needs no splitting when polarity is conditioned on the
aspect being asked about.

Trained on SemEval-2014 Task 4 (laptops + restaurants).
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

TASK = "apc"


def build_pairs(rows, limit=None):
    """
    One training row per (sentence, aspect_term, polarity) triple.

    Sentences with no annotated aspect contribute nothing here - polarity is
    only defined relative to an aspect.
    """
    pairs = []
    for row in rows:
        post = row["post"]
        terms = row.get("aspect_terms") or []
        polarities = row.get("polarities") or []
        for term, polarity in zip(terms, polarities):
            if polarity not in LABEL2ID:
                continue
            pairs.append({
                "text": post,
                "aspect": term,
                "label": LABEL2ID[polarity],
                "source": row.get("source", "unknown"),
            })
            if limit and len(pairs) >= limit:
                return pairs
    return pairs


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

    device = resolve_device("APC training")
    report_environment()
    set_seed(cfg.seed)

    limit = 200 if cfg.profile == "smoke" else None
    raw = {
        name: load_jsonl(DATA_DIR / f"semeval_{name}.jsonl")
        for name in ("train", "val", "test")
    }
    pairs = {name: build_pairs(rows, limit=limit) for name, rows in raw.items()}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def encode(batch):
        return tokenizer(
            batch["text"],
            batch["aspect"],
            truncation=True,
            max_length=cfg.max_length,
        )

    encoded, stats = {}, {}
    for name, rows in pairs.items():
        distribution = Counter(ID2LABEL[r["label"]] for r in rows)
        stats[name] = {"pairs": len(rows), "distribution": dict(distribution)}
        print(f"[{TASK}] {name}: {stats[name]}")
        dataset = Dataset.from_list(rows)
        encoded[name] = dataset.map(
            encode, batched=True, remove_columns=["text", "aspect", "source"]
        )

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
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        # Macro, not accuracy: the corpus is skewed toward Positive
        # (3905 / 1700 / 1402), so accuracy rewards ignoring Neutral.
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=cfg.fp16 and device.type == "cuda",
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
        "review-router-apc",
        cfg,
        metrics,
        {"dataset": "FangornGuardian/semeval-2014-absa", "splits": stats},
        notes=[
            "Polarity is conditioned on the aspect term, so a sentence with "
            "opposing opinions yields a different label per aspect.",
            "Class distribution is skewed toward Positive; model selection "
            "uses macro-F1 so the minority Neutral class is not ignored.",
            "Domain is laptops and restaurants; e-commerce phrasing around "
            "delivery, refunds and packaging is under-represented.",
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
