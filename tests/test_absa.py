"""
Structural tests for the ABSA pipeline.

These deliberately avoid asserting anything about model *quality* - the models
under test in CI are whatever happens to be on disk, and prediction quality is
measured by eval/evaluate_real.py instead. What is tested here is the wiring:
the issue contract, deduplication, fallback behaviour and span handling, all of
which must hold regardless of how well the models are trained.
"""

from app.services import absa
from app.services.absa import ABSAPipeline, _sentence_around, build_extractor

REQUIRED_KEYS = {"aspect", "opinion", "sentiment", "confidence", "department"}


class StubPipeline(ABSAPipeline):
    """Substitutes fixed model outputs so wiring can be tested deterministically."""

    def __init__(self, spans, polarities, mappings):
        super().__init__(mapper=_StubMapper(mappings))
        self._spans = spans
        self._polarities = polarities

    def extract_terms(self, text):
        return self._spans

    def classify_polarities(self, text, terms):
        return self._polarities


class _StubMapper:
    def __init__(self, mappings):
        self._mappings = mappings

    def map_terms(self, terms):
        return self._mappings


def test_sentence_around_picks_the_containing_sentence():
    text = "Delivery was fast. The box was crushed. Support helped."
    start = text.index("box")
    assert _sentence_around(text, start, start + 3) == "The box was crushed."


def test_sentence_around_handles_no_punctuation():
    text = "the refund never came"
    assert _sentence_around(text, 4, 10) == text


def test_sentence_around_handles_first_sentence():
    text = "Battery is poor. Delivery fine."
    assert _sentence_around(text, 0, 7) == "Battery is poor."


def test_issue_contract_is_preserved():
    """Downstream code was written against the lexicon's keys; they must hold."""
    pipeline = StubPipeline(
        spans=[("delivery", 0, 8)],
        polarities=[("Negative", 0.91)],
        mappings=[{"department": "delivery", "department_name": "Logistics",
                   "score": 0.9, "low_confidence": False}],
    )
    issues = pipeline.extract_aspects("delivery was late")
    assert len(issues) == 1
    assert REQUIRED_KEYS <= set(issues[0])
    assert issues[0]["aspect"] == "delivery"
    assert issues[0]["department"] == "Logistics"
    assert issues[0]["sentiment"] == "Negative"


def test_repeated_aspect_keeps_the_worst_mention():
    """Routing must surface the worst thing said about a department."""
    pipeline = StubPipeline(
        spans=[("delivery", 0, 8), ("shipping", 30, 38)],
        polarities=[("Positive", 0.9), ("Negative", 0.95)],
        mappings=[
            {"department": "delivery", "department_name": "Logistics",
             "score": 0.9, "low_confidence": False},
            {"department": "delivery", "department_name": "Logistics",
             "score": 0.9, "low_confidence": False},
        ],
    )
    issues = pipeline.extract_aspects("delivery was fast but shipping broke it")
    assert len(issues) == 1
    assert issues[0]["sentiment"] == "Negative"


def test_low_confidence_mapping_is_flagged_on_the_issue():
    pipeline = StubPipeline(
        spans=[("sushi", 0, 5)],
        polarities=[("Positive", 0.8)],
        mappings=[{"department": "customer_support",
                   "department_name": "Customer Support",
                   "score": 0.2, "low_confidence": True}],
    )
    issues = pipeline.extract_aspects("sushi was great")
    assert issues[0]["low_confidence"] is True


def test_no_terms_falls_back_to_the_model_verdict():
    """
    The lexicon used to assert Neutral product_quality here, overriding a
    confident Negative verdict from the sentiment model.
    """
    pipeline = StubPipeline(spans=[], polarities=[], mappings=[])
    issues = pipeline.extract_aspects(
        "Absolutely furious about all of this.",
        overall_sentiment="Negative",
        overall_confidence=0.97,
    )
    assert len(issues) == 1
    assert issues[0]["sentiment"] == "Negative"
    assert issues[0]["confidence"] == 0.97
    assert issues[0]["source"] == "fallback"


def test_empty_review_returns_a_single_fallback_issue():
    pipeline = StubPipeline(spans=[], polarities=[], mappings=[])
    issues = pipeline.extract_aspects("")
    assert len(issues) == 1
    assert issues[0]["source"] == "fallback"


def test_build_extractor_falls_back_when_artifacts_absent(monkeypatch):
    """The service must start without models, but say so."""
    monkeypatch.setattr(absa, "artifacts_present", lambda: False)
    extractor = build_extractor()
    assert extractor.__class__.__name__ == "AspectExtractor"


def test_build_extractor_uses_absa_when_artifacts_present(monkeypatch):
    monkeypatch.setattr(absa, "artifacts_present", lambda: True)
    assert isinstance(build_extractor(), ABSAPipeline)


def test_describe_reports_artifact_state():
    described = ABSAPipeline().describe()
    assert {"loaded", "artifacts_present", "ate_dir", "apc_dir"} <= set(described)
