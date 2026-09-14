"""
Builds the real training corpora, replacing the synthetic generator.

The synthetic corpus (data/build_dataset.py) assembled 1500 reviews from ~100
fixed phrases, and 31.7% of its held-out split appeared verbatim in train. Any
score measured on it reflects template memorisation. This script sources:

  SemEval-2014 Task 4  -> aspect term extraction (ATE) and polarity (APC)
  Amazon Reviews 2023  -> review-level sentiment

Train/test overlap is asserted empty and raises on violation, rather than being
reported as a statistic after the fact.

Usage:
    python data/fetch_real_datasets.py --profile 4090
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

OUT_DIR = Path("data/real")
SEMEVAL_ID = "FangornGuardian/semeval-2014-absa"
AMAZON_ID = "McAuley-Lab/Amazon-Reviews-2023"

AMAZON_CATEGORIES = [
    "All_Beauty",
    "Electronics",
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
    "Cell_Phones_and_Accessories",
]

# Total Amazon rows per profile, split across the three sentiment classes.
PROFILE_AMAZON_ROWS = {
    "4090": 400_000,
    "t4": 150_000,
    "3050": 20_000,
    "smoke": 600,
}

LABELS = ("Negative", "Neutral", "Positive")
MIN_CHARS, MAX_CHARS = 20, 2000
SEED = 42
MAX_SCAN_PER_CONFIG = 400_000


def normalise(text: str) -> str:
    """Canonical form for duplicate and leakage detection."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()


def assert_no_leakage(splits, field, name):
    """
    Hard gate. Overlap between splits is the defect that made the previous
    0.9633 meaningless, so it fails the build instead of being logged.
    """
    hashes = {k: {text_hash(r[field]) for r in rows} for k, rows in splits.items()}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        if a in hashes and b in hashes:
            overlap = hashes[a] & hashes[b]
            if overlap:
                raise SystemExit(
                    f"[{name}] LEAKAGE: {len(overlap)} texts appear in both "
                    f"{a} and {b}. Refusing to write a corpus that cannot be "
                    f"honestly evaluated."
                )
    summary = ", ".join(f"{k}={len(v)}" for k, v in hashes.items())
    print(f"[{name}] leakage gate passed ({summary} unique)")


def dedupe(rows, field):
    seen, out = set(), []
    for r in rows:
        h = text_hash(r[field])
        if h not in seen:
            seen.add(h)
            out.append(r)
    return out


# ---------------------------------------------------------------- SemEval-2014

def build_semeval(smoke):
    from datasets import load_dataset

    print(f"Loading {SEMEVAL_ID} ...")
    ds = load_dataset(SEMEVAL_ID, split="train")
    print(f"  {len(ds)} rows, fields: {ds.column_names}")

    rows, dropped_conflict, multi_occurrence = [], 0, 0
    for r in ds:
        post = (r.get("post") or "").strip()
        if not post:
            continue

        terms = list(r.get("aspect_terms") or [])
        pols = list(r.get("polarities") or [])
        if len(terms) != len(pols):
            # Misaligned annotation: unusable for APC, still valid for ATE.
            shortest = min(len(terms), len(pols))
            terms, pols = terms[:shortest], pols[:shortest]

        kept_terms, kept_pols = [], []
        for term, pol in zip(terms, pols):
            pol = (pol or "").strip().lower()
            if pol == "conflict":
                # SemEval-2014 carries a 'conflict' class; not one of our three.
                dropped_conflict += 1
                continue
            if pol not in ("negative", "neutral", "positive"):
                continue
            kept_terms.append(term)
            kept_pols.append(pol.capitalize())
            occurrences = re.findall(rf"\b{re.escape(term)}\b", post, re.IGNORECASE)
            if len(occurrences) > 1:
                multi_occurrence += 1

        rows.append({
            "post": post,
            "aspect_terms": kept_terms,
            "polarities": kept_pols,
            "source": r.get("source") or "unknown",
            "orig_split": (r.get("split") or "train").strip().lower(),
        })

    rows = dedupe(rows, "post")
    train_pool = [r for r in rows if r["orig_split"] != "test"]
    test = [r for r in rows if r["orig_split"] == "test"]

    if not test:
        # No usable split column; carve our own.
        random.Random(SEED).shuffle(train_pool)
        cut = int(len(train_pool) * 0.2)
        test, train_pool = train_pool[:cut], train_pool[cut:]

    random.Random(SEED).shuffle(train_pool)
    val_cut = max(1, int(len(train_pool) * 0.1))
    val, train = train_pool[:val_cut], train_pool[val_cut:]

    if smoke:
        train, val, test = train[:200], val[:50], test[:50]

    splits = {"train": train, "val": val, "test": test}
    assert_no_leakage(splits, "post", "semeval")

    print(f"  dropped 'conflict' annotations: {dropped_conflict}")
    print(f"  terms occurring >1x in their sentence: {multi_occurrence} "
          f"(every occurrence gets labelled - documented in the model card)")
    for k, v in splits.items():
        pairs = sum(len(r["aspect_terms"]) for r in v)
        print(f"  {k}: {len(v)} sentences, {pairs} aspect pairs")

    return {
        "splits": splits,
        "stats": {
            "dropped_conflict": dropped_conflict,
            "multi_occurrence_terms": multi_occurrence,
            "counts": {k: len(v) for k, v in splits.items()},
            "aspect_pairs": {
                k: sum(len(r["aspect_terms"]) for r in v) for k, v in splits.items()
            },
        },
    }


# ------------------------------------------------------------ Amazon Reviews

def rating_to_label(rating):
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return None
    if value <= 2:
        return "Negative"
    if value < 4:
        return "Neutral"
    return "Positive"


def resolve_amazon_configs():
    """
    Config names are discovered, never assumed. A missing category reports what
    is available instead of failing on a KeyError deep in the stream.
    """
    from datasets import get_dataset_config_names

    available = set(get_dataset_config_names(AMAZON_ID))
    chosen, missing = [], []
    for category in AMAZON_CATEGORIES:
        name = f"raw_review_{category}"
        if name in available:
            chosen.append(name)
        else:
            missing.append(name)

    if not chosen:
        sample = sorted(n for n in available if n.startswith("raw_review_"))[:15]
        raise SystemExit(
            f"None of the expected configs exist on {AMAZON_ID}.\n"
            f"Looked for: {[f'raw_review_{c}' for c in AMAZON_CATEGORIES]}\n"
            f"Available raw_review_* configs include: {sample}"
        )
    if missing:
        print(f"  note: unavailable, skipping: {missing}")
    return chosen


def build_amazon(total_rows, smoke):
    from datasets import load_dataset

    configs = resolve_amazon_configs()
    per_class = max(1, total_rows // 3)
    print(f"Streaming {AMAZON_ID} from {len(configs)} configs "
          f"(target {per_class} per class) ...")

    buckets = {label: [] for label in LABELS}
    seen_hashes = set()

    for config in configs:
        if all(len(buckets[l]) >= per_class for l in LABELS):
            break
        print(f"  {config} ...")
        stream = load_dataset(AMAZON_ID, config, split="full", streaming=True)

        scanned = 0
        for record in stream:
            if all(len(buckets[l]) >= per_class for l in LABELS):
                break
            if scanned > MAX_SCAN_PER_CONFIG:
                break
            scanned += 1

            label = rating_to_label(record.get("rating"))
            if label is None or len(buckets[label]) >= per_class:
                continue

            title = (record.get("title") or "").strip()
            body = (record.get("text") or "").strip()
            text = (f"{title}. {body}" if title else body).strip()
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue

            digest = text_hash(text)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            buckets[label].append({"text": text, "label": label, "category": config})

        counts = ", ".join(f"{l}={len(buckets[l])}" for l in LABELS)
        print(f"    scanned {scanned}, have {counts}")

    # Balance to the scarcest class (3-star reviews), so accuracy is not
    # inflated by a skew the model can exploit.
    smallest = min(len(buckets[l]) for l in LABELS)
    if smallest == 0:
        empty = [l for l in LABELS if not buckets[l]]
        raise SystemExit(f"No usable rows for class(es) {empty}; aborting.")

    rng = random.Random(SEED)
    rows = []
    for label in LABELS:
        rng.shuffle(buckets[label])
        rows.extend(buckets[label][:smallest])
    rng.shuffle(rows)
    print(f"  balanced to {smallest} per class -> {len(rows)} rows")

    if smoke:
        rows = rows[:600]

    total = len(rows)
    test_cut, val_cut = int(total * 0.1), int(total * 0.2)
    splits = {
        "test": rows[:test_cut],
        "val": rows[test_cut:val_cut],
        "train": rows[val_cut:],
    }
    assert_no_leakage(splits, "text", "amazon")
    for k, v in splits.items():
        print(f"  {k}: {len(v)} ({dict(Counter(r['label'] for r in v))})")

    return {
        "splits": splits,
        "stats": {
            "configs": configs,
            "per_class_after_balance": smallest,
            "counts": {k: len(v) for k, v in splits.items()},
            "class_counts": {
                k: dict(Counter(r["label"] for r in v)) for k, v in splits.items()
            },
        },
    }


# --------------------------------------------------------------------------

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="4090",
                        choices=sorted(PROFILE_AMAZON_ROWS))
    parser.add_argument("--skip-amazon", action="store_true",
                        help="SemEval only; ATE/APC work needs no Amazon data")
    args = parser.parse_args()

    smoke = args.profile == "smoke"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "profile": args.profile,
        "filters": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS},
    }

    semeval = build_semeval(smoke)
    for split, rows in semeval["splits"].items():
        write_jsonl(OUT_DIR / f"semeval_{split}.jsonl", rows)
    manifest["semeval"] = dict(dataset=SEMEVAL_ID, **semeval["stats"])

    if not args.skip_amazon:
        amazon = build_amazon(PROFILE_AMAZON_ROWS[args.profile], smoke)
        for split, rows in amazon["splits"].items():
            write_jsonl(OUT_DIR / f"amazon_{split}.jsonl", rows)
        manifest["amazon"] = dict(dataset=AMAZON_ID, **amazon["stats"])

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_DIR}/ and manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
