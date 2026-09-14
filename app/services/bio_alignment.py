"""
Character-span recovery and BIO tagging for aspect term extraction.

SemEval-2014 supplies aspect terms as bare strings, not character offsets, so
spans have to be recovered by searching the sentence. This is the most
error-prone step in the training pipeline, which is why it lives here - free of
any model or tokenizer dependency - and is unit-tested directly.

Known approximation: the dataset does not record *which* occurrence was
annotated, so when a term appears more than once every occurrence is labelled.
Callers count these and the total is recorded in the model card.
"""

import re

O, B, I = "O", "B-ASP", "I-ASP"

ATE_LABELS = [O, B, I]
ATE_LABEL2ID = {label: index for index, label in enumerate(ATE_LABELS)}
ATE_ID2LABEL = {index: label for label, index in ATE_LABEL2ID.items()}

IGNORE_INDEX = -100


def find_spans(text, terms):
    """
    Returns (spans, unmatched_terms).

    Spans are character offsets into `text`: sorted, non-overlapping, and
    resolved longest-first so "keyboard functions" wins over "keyboard".
    """
    if not text or not terms:
        return [], []

    candidates = []
    unmatched = []

    for term in terms:
        term = (term or "").strip()
        if not term:
            continue

        # Exact match first, then a whitespace-flexible retry that covers
        # annotation spacing mismatches such as "battery  life".
        found = _search_exact(text, term) or _search_flexible(text, term)
        if found:
            candidates.extend(found)
        else:
            unmatched.append(term)

    return _resolve_overlaps(candidates), unmatched


def _search_exact(text, term):
    pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]


def _search_flexible(text, term):
    """Match the term allowing any run of whitespace between its words."""
    parts = [re.escape(part) for part in term.split()]
    if not parts:
        return []
    pattern = r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]


def _resolve_overlaps(spans):
    """Longest span wins; ties broken by earliest start."""
    if not spans:
        return []
    ordered = sorted(set(spans), key=lambda s: (-(s[1] - s[0]), s[0]))
    kept = []
    for start, end in ordered:
        if any(start < kept_end and end > kept_start for kept_start, kept_end in kept):
            continue
        kept.append((start, end))
    return sorted(kept)


def tag_offsets(offsets, spans, special_mask=None):
    """
    Converts tokenizer character offsets into BIO label ids.

    `offsets` is `offset_mapping` from a fast tokenizer. Special tokens (offset
    (0, 0), or flagged by `special_mask`) get IGNORE_INDEX so they are excluded
    from the loss. The first token overlapping a span is B-ASP; subsequent
    tokens of the same span are I-ASP.
    """
    labels = []
    started = set()

    for index, (start, end) in enumerate(offsets):
        is_special = (
            (special_mask is not None and special_mask[index] == 1)
            or (start == 0 and end == 0)
        )
        if is_special:
            labels.append(IGNORE_INDEX)
            continue

        tag = O
        for span in spans:
            span_start, span_end = span
            if start < span_end and end > span_start:
                if span in started:
                    tag = I
                else:
                    tag = B
                    started.add(span)
                break
        labels.append(ATE_LABEL2ID[tag])

    return labels


def spans_from_tags(tags, offsets):
    """Inverse of tag_offsets: decode predicted BIO tags back into char spans."""
    spans, current = [], None
    for tag, (start, end) in zip(tags, offsets):
        if start == 0 and end == 0:
            continue
        if tag == B:
            if current:
                spans.append(current)
            current = (start, end)
        elif tag == I and current is not None:
            current = (current[0], end)
        else:
            if current:
                spans.append(current)
            current = None
    if current:
        spans.append(current)
    return spans
