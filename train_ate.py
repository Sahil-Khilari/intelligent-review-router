"""
Aspect Term Extraction: token classification with BIO tags.

Replaces the keyword lexicon in app/services/aspect_extraction.py, which could
only find aspects whose vocabulary someone had written down in advance. This
model finds aspect *spans* in arbitrary text, which is what lets the system
handle reviews it has never seen.

Trained on SemEval-2014 Task 4 (laptops + restaurants). Span recovery lives in
app/services/bio_alignment.py and is unit-tested separately.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.services.bio_alignment import (  # noqa: E402
    ATE_ID2LABEL,
    ATE_LABELS,
    ATE_LABEL2ID,
    IGNORE_INDEX,
    find_spans,
    tag_offsets,
)
from app.services.device import resolve_device  # noqa: E402
from train_progress import ProgressCallback  # noqa: E402
from train_common import (  # noqa: E402
    DATA_DIR,
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

TASK = "ate"


def build_examples(rows, tokenizer, max_length):
    """
    Tokenizes and BIO-tags each sentence.

    Returns (encodings, stats). Terms that cannot be located are counted rather
    than silently dropped, so label corruption is visible before training.
    """
    features = []
    unmatched_total = 0
    terms_total = 0
    sentences_with_spans = 0

    for row in rows:
        post = row["post"]
        terms = row.get("aspect_terms") or []
        terms_total += len(terms)

        spans, unmatched = find_spans(post, terms)
        unmatched_total += len(unmatched)
        if spans:
            sentences_with_spans += 1

        encoded = tokenizer(
            post,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        labels = tag_offsets(
            encoded["offset_mapping"],
            spans,
            special_mask=encoded["special_tokens_mask"],
        )
        features.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        })

    stats = {
        "sentences": len(rows),
        "sentences_with_spans": sentences_with_spans,
        "aspect_terms": terms_total,
        "terms_unmatched": unmatched_total,
        "terms_matched_pct": round(
            100 * (terms_total - unmatched_total) / terms_total, 4
        ) if terms_total else 0.0,
    }
    return features, stats


def make_compute_metrics():
    from seqeval.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_tags, pred_tags = [], []
        for pred_row, label_row in zip(predictions, labels):
            true_seq, pred_seq = [], []
            for p, l in zip(pred_row, label_row):
                if l == IGNORE_INDEX:
                    continue
                true_seq.append(ATE_ID2LABEL[int(l)])
                pred_seq.append(ATE_ID2LABEL[int(p)])
            true_tags.append(true_seq)
            pred_tags.append(pred_seq)

        return {
            "precision": precision_score(true_tags, pred_tags),
            "recall": recall_score(true_tags, pred_tags),
            "f1": f1_score(true_tags, pred_tags),
            "token_accuracy": accuracy_score(true_tags, pred_tags),
        }

    return compute_metrics


def train(cfg: TrainConfig) -> dict:
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    device = resolve_device("ATE training")
    report_environment()
    set_seed(cfg.seed)

    limit = 200 if cfg.profile == "smoke" else None
    splits = {
        name: load_jsonl(DATA_DIR / f"semeval_{name}.jsonl", limit=limit)
        for name in ("train", "val", "test")
    }

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if not tokenizer.is_fast:
        raise SystemExit(
            f"{cfg.model_name} did not provide a fast tokenizer. BIO alignment "
            f"needs offset mappings, which only fast tokenizers return."
        )

    encoded, stats = {}, {}
    for name, rows in splits.items():
        features, split_stats = build_examples(rows, tokenizer, cfg.max_length)
        encoded[name] = Dataset.from_list(features)
        stats[name] = split_stats
        print(f"[{TASK}] {name}: {split_stats}")

    model = load_model_fp32(
        AutoModelForTokenClassification,
        cfg.model_name,
        num_labels=len(ATE_LABELS),
        id2label=ATE_ID2LABEL,
        label2id=ATE_LABEL2ID,
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
        warmup_steps=warmup_steps_for(len(encoded['train']), cfg),
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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
        data_collator=DataCollatorForTokenClassification(tokenizer),
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
        "review-router-ate",
        cfg,
        metrics,
        {"dataset": "FangornGuardian/semeval-2014-absa", "splits": stats},
        notes=[
            "SemEval annotates specific aspect occurrences, but the dataset "
            "exposes terms as strings only, so every occurrence of a term is "
            "labelled. Measured at 218 affected terms in the full corpus.",
            f"{sum(s['terms_unmatched'] for s in stats.values())} aspect terms "
            "could not be located in their sentence and were dropped; these are "
            "annotation defects in the source (truncated terms, missing spaces).",
            "Domain is laptops and restaurants. E-commerce vocabulary such as "
            "delivery, refund and packaging is under-represented; the category "
            "mapper in app/services/aspect_mapping.py bridges this.",
        ],
    )
    push_if_requested(cfg, saved_to, TASK)
    return {"metrics": metrics, "stats": stats, "saved_to": saved_to}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="4090")
    parser.add_argument("--model", default=None, help="override base model")
    parser.add_argument("--push-to", default=None, help="HF repo id to publish to")
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
