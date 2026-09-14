"""
Per-stage evaluation on real held-out data.

Supersedes paper/evaluate_pipeline.py, which measured against a corpus where
31.7% of the test split appeared verbatim in train. That file is left in place
for the synthetic comparison; this one produces numbers that mean something
outside this repository, because SemEval-2014 is a published benchmark.

    python eval/evaluate_real.py

Writes eval/metrics.json.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.bio_alignment import ATE_ID2LABEL, find_spans  # noqa: E402
from train_common import DATA_DIR, MODELS_DIR, load_jsonl  # noqa: E402

OUT = Path("eval/metrics.json")


def banner(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def evaluate_ate(rows, pipeline):
    """Span-level P/R/F1, exact and partial, against gold aspect spans."""
    banner("ASPECT TERM EXTRACTION (span level)")
    exact_tp = exact_fp = exact_fn = 0
    partial_tp = 0
    latencies = []

    for row in rows:
        text = row["post"]
        gold_spans, _ = find_spans(text, row.get("aspect_terms") or [])
        gold = {(s, e) for s, e in gold_spans}

        started = time.perf_counter()
        predicted = {(s, e) for _, s, e in pipeline.extract_terms(text)}
        latencies.append((time.perf_counter() - started) * 1000)

        exact_tp += len(gold & predicted)
        exact_fp += len(predicted - gold)
        exact_fn += len(gold - predicted)

        for p_start, p_end in predicted:
            if any(p_start < g_end and p_end > g_start for g_start, g_end in gold):
                partial_tp += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    p, r, f = prf(exact_tp, exact_fp, exact_fn)
    pp, pr, pf = prf(partial_tp, exact_fp + exact_tp - partial_tp, exact_fn)
    latencies = np.array(latencies)

    print(f"exact    P {p:.4f}  R {r:.4f}  F1 {f:.4f}   (TP {exact_tp} FP {exact_fp} FN {exact_fn})")
    print(f"partial  P {pp:.4f}  R {pr:.4f}  F1 {pf:.4f}")
    print(f"latency ms: mean {latencies.mean():.1f}  p95 {np.percentile(latencies, 95):.1f}")

    return {
        "exact": {"precision": p, "recall": r, "f1": f,
                  "tp": exact_tp, "fp": exact_fp, "fn": exact_fn},
        "partial": {"precision": pp, "recall": pr, "f1": pf},
        "latency_ms": {"mean": float(latencies.mean()),
                       "p95": float(np.percentile(latencies, 95))},
    }


def evaluate_apc(rows, pipeline):
    """Accuracy and macro-F1 on gold (sentence, aspect) pairs."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    banner("ASPECT POLARITY CLASSIFICATION")
    y_true, y_pred = [], []
    for row in rows:
        terms = row.get("aspect_terms") or []
        polarities = row.get("polarities") or []
        if not terms:
            continue
        predicted = pipeline.classify_polarities(row["post"], terms)
        for gold, (label, _) in zip(polarities, predicted):
            y_true.append(gold)
            y_pred.append(label)

    if not y_true:
        print("no gold pairs")
        return {}

    accuracy = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"accuracy {accuracy:.4f} | macro F1 {macro:.4f}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    return {"accuracy": accuracy, "macro_f1": macro, "pairs": len(y_true)}


def evaluate_sentiment(rows, analyzer):
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    banner("REVIEW SENTIMENT")
    y_true, y_pred, latencies = [], [], []
    for row in rows:
        started = time.perf_counter()
        result = analyzer.predict(row["text"])
        latencies.append((time.perf_counter() - started) * 1000)
        y_true.append(row["label"])
        y_pred.append(result["sentiment"])

    accuracy = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    latencies = np.array(latencies)

    print(f"model_type = {analyzer.model_type}")
    print(f"accuracy {accuracy:.4f} | macro F1 {macro:.4f} | weighted F1 {weighted:.4f}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print(f"latency ms: mean {latencies.mean():.1f}  p95 {np.percentile(latencies, 95):.1f}")

    return {"accuracy": accuracy, "macro_f1": macro, "weighted_f1": weighted,
            "model_type": analyzer.model_type,
            "latency_ms": {"mean": float(latencies.mean()),
                           "p95": float(np.percentile(latencies, 95))}}


def evaluate_end_to_end(rows, ticket_service, limit=200):
    """
    Routing behaviour on real reviews.

    There is no gold department label for Amazon reviews, so this reports the
    distribution of decisions and the low-confidence rate rather than an
    accuracy. Grading against auto-generated labels would repeat the original
    mistake of measuring against a generator; see data/gold/ for the labelled
    set that makes an accuracy figure meaningful.
    """
    banner("END-TO-END ROUTING (distributional, not graded)")
    departments, priorities, latencies = Counter(), Counter(), []
    low_confidence = aspects = 0

    for row in rows[:limit]:
        started = time.perf_counter()
        result = ticket_service.analyze_and_route(row["text"])
        latencies.append((time.perf_counter() - started) * 1000)
        departments[result["primary_department"]] += 1
        priorities[result["overall_priority"]] += 1
        for issue in result["issues"]:
            aspects += 1
            if issue.get("low_confidence"):
                low_confidence += 1

    latencies = np.array(latencies)
    rate = low_confidence / aspects if aspects else 0.0
    print(f"reviews {len(latencies)} | aspects {aspects} | "
          f"low-confidence mappings {low_confidence} ({rate:.2%})")
    print(f"departments: {dict(departments)}")
    print(f"priorities : {dict(priorities)}")
    print(f"latency ms : mean {latencies.mean():.1f}  p95 {np.percentile(latencies, 95):.1f}")

    return {
        "reviews": len(latencies),
        "aspects": aspects,
        "low_confidence_rate": rate,
        "departments": dict(departments),
        "priorities": dict(priorities),
        "latency_ms": {"mean": float(latencies.mean()),
                       "p95": float(np.percentile(latencies, 95))},
        "note": "distributional only; no gold routing labels for Amazon reviews",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap rows per stage (for a quick check)")
    args = parser.parse_args()

    from app.services.absa import ABSAPipeline, artifacts_present
    from app.services.routing import SupportRouter
    from app.services.sentiment import SentimentAnalyzer
    from app.services.ticket_service import TicketService

    if not artifacts_present():
        raise SystemExit(
            f"No ABSA models in {MODELS_DIR}/. Run train_all.py, or fetch them "
            f"with scripts/download_models.py."
        )

    semeval_test = load_jsonl(DATA_DIR / "semeval_test.jsonl", limit=args.limit)
    amazon_test = load_jsonl(DATA_DIR / "amazon_test.jsonl", limit=args.limit)

    pipeline = ABSAPipeline()
    analyzer = SentimentAnalyzer()
    service = TicketService(analyzer, pipeline, SupportRouter())

    metrics = {
        "ate": evaluate_ate(semeval_test, pipeline),
        "apc": evaluate_apc(semeval_test, pipeline),
        "sentiment": evaluate_sentiment(amazon_test, analyzer),
        "routing": evaluate_end_to_end(amazon_test, service),
    }

    manifest = DATA_DIR / "manifest.json"
    if manifest.exists():
        metrics["corpus"] = json.loads(manifest.read_text(encoding="utf-8"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
