"""
Tests for the riskiest code in the training pipeline.

SemEval gives aspect terms as strings, not offsets. If span recovery or BIO
tagging is wrong, the ATE model trains on silently corrupted labels and the
error only shows up as mediocre F1 hours later on a GPU we get one shot at.
"""

import pytest

from app.services.bio_alignment import (
    ATE_ID2LABEL,
    ATE_LABEL2ID,
    IGNORE_INDEX,
    find_spans,
    spans_from_tags,
    tag_offsets,
)

B, I, O = "B-ASP", "I-ASP", "O"


def tags_of(labels):
    return [ATE_ID2LABEL.get(x, "IGN") if x != IGNORE_INDEX else "IGN" for x in labels]


# ----------------------------------------------------------------- find_spans

def test_finds_single_term():
    text = "The battery life is terrible."
    spans, unmatched = find_spans(text, ["battery life"])
    assert unmatched == []
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == "battery life"


def test_longest_span_wins_on_overlap():
    """'keyboard functions' must beat the nested 'keyboard'."""
    text = "It lost half the keyboard functions."
    spans, _ = find_spans(text, ["keyboard", "keyboard functions"])
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == "keyboard functions"


def test_does_not_match_inside_a_word():
    """The substring-matching bug that made 'late' match 'chocolate'."""
    spans, unmatched = find_spans("I ordered a chocolate bar.", ["late"])
    assert spans == []
    assert unmatched == ["late"]


def test_every_occurrence_is_labelled():
    """Documented approximation: the dataset does not say which one."""
    text = "The screen is bad and the screen flickers."
    spans, _ = find_spans(text, ["screen"])
    assert len(spans) == 2


def test_flexible_whitespace_retry():
    text = "The battery  life drains fast."
    spans, unmatched = find_spans(text, ["battery life"])
    assert unmatched == []
    assert len(spans) == 1


def test_unmatched_terms_are_reported_not_silently_dropped():
    spans, unmatched = find_spans("Nothing relevant here.", ["waiter", "sushi"])
    assert spans == []
    assert sorted(unmatched) == ["sushi", "waiter"]


def test_case_insensitive():
    text = "BATTERY LIFE is fine."
    spans, unmatched = find_spans(text, ["battery life"])
    assert unmatched == []
    assert len(spans) == 1


def test_empty_inputs():
    assert find_spans("", ["x"]) == ([], [])
    assert find_spans("text", []) == ([], [])
    assert find_spans("text", [""]) == ([], [])


# ---------------------------------------------------------------- tag_offsets

def test_special_tokens_are_ignored():
    offsets = [(0, 0), (0, 3), (0, 0)]
    labels = tag_offsets(offsets, [])
    assert labels[0] == IGNORE_INDEX
    assert labels[-1] == IGNORE_INDEX
    assert labels[1] == ATE_LABEL2ID[O]


def test_special_mask_is_honoured():
    offsets = [(0, 2), (2, 5)]
    labels = tag_offsets(offsets, [], special_mask=[1, 0])
    assert labels[0] == IGNORE_INDEX
    assert labels[1] == ATE_LABEL2ID[O]


def test_first_token_of_span_is_b_rest_are_i():
    # "battery life" split across two tokens at chars 4-11 and 12-16
    offsets = [(0, 0), (0, 3), (4, 11), (12, 16), (17, 19), (0, 0)]
    labels = tag_offsets(offsets, [(4, 16)])
    assert tags_of(labels) == ["IGN", O, B, I, O, "IGN"]


def test_multi_subword_span_gets_one_b():
    """A span split into four wordpieces must produce exactly one B-ASP."""
    offsets = [(0, 0), (0, 3), (4, 6), (6, 9), (9, 12), (13, 15), (0, 0)]
    labels = tag_offsets(offsets, [(4, 12)])
    tags = tags_of(labels)
    assert tags.count(B) == 1
    assert tags.count(I) == 2
    assert tags == ["IGN", O, B, I, I, O, "IGN"]


def test_two_separate_spans_each_get_a_b():
    offsets = [(0, 3), (4, 9), (10, 13), (14, 20)]
    labels = tag_offsets(offsets, [(4, 9), (14, 20)])
    assert tags_of(labels) == [O, B, O, B]


def test_no_spans_is_all_o():
    offsets = [(0, 3), (4, 9)]
    assert tags_of(tag_offsets(offsets, [])) == [O, O]


# ------------------------------------------------------------- round tripping

def test_round_trip_recovers_spans():
    offsets = [(0, 0), (0, 3), (4, 11), (12, 16), (17, 19), (0, 0)]
    spans = [(4, 16)]
    tags = [ATE_ID2LABEL.get(x, O) if x != IGNORE_INDEX else O
            for x in tag_offsets(offsets, spans)]
    recovered = spans_from_tags(tags, offsets)
    assert recovered == spans


def test_round_trip_two_spans():
    offsets = [(0, 3), (4, 9), (10, 13), (14, 20)]
    spans = [(4, 9), (14, 20)]
    tags = [ATE_ID2LABEL.get(x, O) if x != IGNORE_INDEX else O
            for x in tag_offsets(offsets, spans)]
    assert spans_from_tags(tags, offsets) == spans


@pytest.mark.parametrize("text,terms", [
    ("The waiter was rude but the sushi was fresh.", ["waiter", "sushi"]),
    ("Great screen, poor battery life, noisy fan.", ["screen", "battery life", "fan"]),
    ("Delivery was late and the box was crushed.", ["Delivery", "box"]),
])
def test_end_to_end_alignment_is_consistent(text, terms):
    """Spans found must survive tagging and decode back to the same text."""
    spans, unmatched = find_spans(text, terms)
    assert unmatched == []
    # Simulate a whitespace tokenizer with real offsets.
    offsets, cursor = [], 0
    for word in text.split():
        start = text.index(word, cursor)
        offsets.append((start, start + len(word)))
        cursor = start + len(word)
    tags = [ATE_ID2LABEL.get(x, O) if x != IGNORE_INDEX else O
            for x in tag_offsets(offsets, spans)]
    recovered = spans_from_tags(tags, offsets)
    recovered_text = {text[s:e].strip(",.").lower() for s, e in recovered}
    assert recovered_text == {t.lower() for t in terms}
