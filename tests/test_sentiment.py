import pytest

# Reviews the model must get right. These assert the exact label: the previous
# versions accepted "Neutral" for both the positive and the negative case, so a
# model that answered Neutral to everything passed the whole suite - which is
# precisely what a silent fallback to an untrained model looks like.
POSITIVE_REVIEWS = [
    "This product is top notch quality and works amazingly well!",
    "Super fast delivery, arrived earlier than expected.",
    "Refund processed within 24 hours, hassle-free return experience.",
]

NEGATIVE_REVIEWS = [
    "Delivery was horrible and item was broken on arrival.",
    "Charged twice for a single order and the refund has not arrived.",
    "Support representative was useless and the call disconnected repeatedly.",
]


@pytest.mark.parametrize("text", POSITIVE_REVIEWS)
def test_positive_sentiment(analyzer, text):
    res = analyzer.predict(text)
    assert res["sentiment"] == "Positive"
    assert 0.0 <= res["confidence"] <= 1.0


@pytest.mark.parametrize("text", NEGATIVE_REVIEWS)
def test_negative_sentiment(analyzer, text):
    res = analyzer.predict(text)
    assert res["sentiment"] == "Negative"
    assert 0.0 <= res["confidence"] <= 1.0


def test_empty_string(analyzer):
    res = analyzer.predict("")
    assert res["sentiment"] == "Neutral"
    assert res["confidence"] == 0.50


def test_accuracy_floor(analyzer):
    """
    Guards the whole suite against a silent model downgrade.

    Any model serving this endpoint must clear a floor on a small labelled
    set, so swapping in an untrained or wrongly-loaded model fails CI rather
    than quietly degrading routing quality.
    """
    cases = [(t, "Positive") for t in POSITIVE_REVIEWS]
    cases += [(t, "Negative") for t in NEGATIVE_REVIEWS]
    correct = sum(analyzer.predict(t)["sentiment"] == expected for t, expected in cases)
    assert correct / len(cases) >= 0.80, (
        f"sentiment accuracy {correct}/{len(cases)} below floor; "
        f"active model is {analyzer.model_type!r}"
    )


def test_probabilities_agree_with_verdict(analyzer):
    """The reported label must be the argmax of the reported probabilities."""
    res = analyzer.predict("The package was crushed and the screen is broken.")
    probs = res.get("probabilities")
    if not probs:
        pytest.skip("active model does not expose probabilities")
    assert max(probs, key=probs.get) == res["sentiment"]


def test_describe_reports_active_model(analyzer):
    described = analyzer.describe()
    assert described["model_type"] == analyzer.model_type
    assert described["transformer_loaded"] is (analyzer.hf_model is not None)
