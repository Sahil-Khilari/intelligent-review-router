from app.services.aspect_extraction import polarity_scores
from app.services.preprocessing import clean_text, split_into_sentences


def aspects_of(extractor, text):
    return {i["aspect"] for i in extractor.extract_aspects(text)}


def sentiment_of(extractor, text, aspect):
    for issue in extractor.extract_aspects(text):
        if issue["aspect"] == aspect:
            return issue["sentiment"]
    raise AssertionError(f"aspect {aspect!r} not detected in {text!r}")


def test_single_aspect_extraction(extractor):
    issues = extractor.extract_aspects("The delivery was delayed by three days.")
    assert len(issues) >= 1
    assert issues[0]["aspect"] == "delivery"
    assert issues[0]["department"] == "Logistics"


def test_multi_aspect_extraction(extractor):
    """
    Both aspects must be found.

    The previous assertion was `refund_return in ... or product_quality in
    ...`, and product_quality is the fallback aspect injected when nothing is
    detected - so the test passed even with extraction completely broken.
    """
    found = aspects_of(extractor, "The phone display is clear, but the refund was never processed.")
    assert "refund_return" in found
    assert "product_quality" in found


def test_contrastive_clause_without_comma_is_split():
    """"great product but the refund never came" must not stay fused."""
    clauses = split_into_sentences("This is a great product but the refund never came")
    assert len(clauses) == 2


def test_complaint_after_praise_is_negative(extractor):
    """Regression: this returned refund_return/Positive, so no ticket was made."""
    text = "This is a great product but the refund never came"
    assert sentiment_of(extractor, text, "refund_return") == "Negative"


def test_repeated_aspect_keeps_worst_mention(extractor):
    """
    Regression: only the first mention of an aspect survived, so the
    pleasantry won and the complaint was discarded.
    """
    text = "Delivery was fast. However the delivery driver damaged my box."
    assert sentiment_of(extractor, text, "delivery") == "Negative"
    assert "packaging" in aspects_of(extractor, text)


def test_negation_is_honoured(extractor):
    """Regression: "not late"/"nothing was broken" scored as a complaint."""
    text = "The delivery was not late and nothing was broken."
    assert sentiment_of(extractor, text, "delivery") != "Negative"


def test_keywords_match_on_word_boundaries(extractor):
    """Regression: "late" matched inside "chocolate" and raised Logistics."""
    assert "delivery" not in aspects_of(extractor, "I ordered a chocolate bar.")
    assert "delivery" not in aspects_of(extractor, "Please translate the manual.")
    assert "payment" not in aspects_of(extractor, "The cardboard was sturdy.")


def test_inflections_still_match(extractor):
    """Word boundaries must not cost us ordinary morphology."""
    assert "delivery" in aspects_of(extractor, "The shipment was delayed again.")


def test_three_aspects_in_one_sentence(extractor):
    text = "The laptop is top notch quality, but the refund has not arrived and the box was crushed."
    found = aspects_of(extractor, text)
    assert {"product_quality", "refund_return", "packaging"} <= found


def test_polarity_scores_negation():
    assert polarity_scores("the delivery was late")[1] == 1
    assert polarity_scores("the delivery was not late")[1] == 0


def test_polarity_scores_negative_phrase():
    assert polarity_scores("the refund never came")[1] >= 1


def test_fallback_uses_model_verdict(extractor):
    """The fallback issue must not assert Neutral over a Negative verdict."""
    issues = extractor.extract_aspects(
        "Absolutely furious about this whole situation.",
        overall_sentiment="Negative",
        overall_confidence=0.97,
    )
    assert len(issues) == 1
    assert issues[0]["sentiment"] == "Negative"
    assert issues[0]["source"] == "fallback"


def test_clean_text_strips_html():
    assert clean_text("<p>Late   delivery</p>") == "Late delivery"


def test_empty_text_yields_no_clauses():
    assert split_into_sentences("") == []
    assert split_into_sentences("   ") == []
