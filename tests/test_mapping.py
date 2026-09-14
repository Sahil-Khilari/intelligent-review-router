"""
Tests for aspect term -> department mapping.

This is the least certain component in the pipeline: there is no published
benchmark for "aspect term -> support department", so these tests encode the
behaviour we actually measured and guard the threshold against regression.
"""

import pytest

from app.services.aspect_mapping import (
    DEPARTMENT_NAMES,
    DEPARTMENT_PROTOTYPES,
    THRESHOLD,
    AspectMapper,
)


@pytest.fixture(scope="module")
def mapper():
    return AspectMapper()


def test_every_department_has_prototypes_and_a_name():
    assert set(DEPARTMENT_PROTOTYPES) == set(DEPARTMENT_NAMES)
    for department, prototypes in DEPARTMENT_PROTOTYPES.items():
        assert prototypes, f"{department} has no prototypes"
        assert DEPARTMENT_NAMES[department]


@pytest.mark.parametrize("term,expected", [
    ("delivery", "delivery"),
    ("shipping time", "delivery"),
    ("the delivery guy", "delivery"),
    ("refund", "refund_return"),
    ("money back", "refund_return"),
    ("charged twice", "payment"),
    ("price", "payment"),
    ("battery life", "product_quality"),
    ("screen", "product_quality"),
    ("keyboard", "product_quality"),
    ("sound quality", "product_quality"),
    ("box", "packaging"),
    ("bubble wrap", "packaging"),
    ("customer service", "customer_support"),
    ("support agent", "customer_support"),
])
def test_ecommerce_terms_map_to_the_right_department(mapper, term, expected):
    result = mapper.map_term(term)
    assert result["department"] == expected, (
        f"{term!r} -> {result['department']} (score {result['score']})"
    )
    assert result["low_confidence"] is False


@pytest.mark.parametrize("term", ["sushi", "ambience", "the food"])
def test_out_of_domain_terms_are_flagged_not_forced(mapper, term):
    """
    SemEval is laptops and restaurants, so restaurant vocabulary will reach the
    mapper. Guessing a department for it is worse than admitting uncertainty:
    a flagged term goes to a review queue, a misrouted one goes to the wrong
    team silently.
    """
    result = mapper.map_term(term)
    assert result["low_confidence"] is True
    assert result["department"] == "customer_support"


def test_low_confidence_still_returns_a_usable_department(mapper):
    result = mapper.map_term("zzzz nonsense token")
    assert result["department"] in DEPARTMENT_NAMES
    assert result["department_name"]
    assert result["low_confidence"] is True


def test_empty_term_is_low_confidence(mapper):
    for term in ("", "   ", None):
        result = mapper.map_term(term)
        assert result["low_confidence"] is True


def test_batch_matches_single(mapper):
    terms = ["delivery", "refund", "box", "sushi"]
    batch = mapper.map_terms(terms)
    assert len(batch) == len(terms)
    for term, batched in zip(terms, batch):
        single = mapper.map_term(term)
        assert batched["department"] == single["department"]
        assert batched["low_confidence"] == single["low_confidence"]


def test_batch_of_nothing():
    assert AspectMapper().map_terms([]) == []


def test_threshold_is_above_the_measured_misroute_band():
    """
    At 0.35, "keyboard" mapped to packaging (0.358) and "the food" to delivery
    (0.384). Both were wrong and both sat just over the line, which is why the
    threshold was raised. Lowering it back below 0.40 reintroduces those.
    """
    assert THRESHOLD >= 0.40


def test_confident_mappings_clear_the_threshold_comfortably(mapper):
    """Correct e-commerce mappings should not be sitting on the boundary."""
    for term in ("delivery", "refund", "charged twice", "bubble wrap"):
        result = mapper.map_term(term)
        assert result["score"] >= THRESHOLD + 0.05, f"{term} is borderline"
