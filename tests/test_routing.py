def issue(aspect, sentiment, opinion="", department=None):
    return {
        "aspect": aspect,
        "sentiment": sentiment,
        "opinion": opinion,
        "department": department,
    }


def test_urgent_priority_routing(router):
    res = router.route_issues("Negative", [
        issue("payment", "Negative", "charged twice", "Payments"),
        issue("refund_return", "Negative", "refund not arrived", "Refunds"),
    ])
    assert res["overall_priority"] == "URGENT"
    assert res["primary_department"] in ["Payments", "Refunds"]


def test_high_priority_routing(router):
    res = router.route_issues("Negative", [
        issue("delivery", "Negative", "extremely late", "Logistics"),
        issue("product_quality", "Negative", "broken screen", "Product Support"),
    ])
    assert res["overall_priority"] in ["HIGH", "URGENT"]


def test_low_priority_routing(router):
    res = router.route_issues("Positive", [
        issue("product_quality", "Positive", "works great", "Product Support"),
    ])
    assert res["overall_priority"] == "LOW"


def test_praise_does_not_choose_the_department(router):
    """
    Regression: a positive aspect could become primary_department, filing a
    real complaint with the wrong team.
    """
    res = router.route_issues("Negative", [
        issue("delivery", "Positive", "arrived early", "Logistics"),
        issue("payment", "Negative", "charged twice", "Payments"),
    ])
    assert res["primary_department"] == "Payments"


def test_model_verdict_lifts_priority(router):
    """
    Regression: overall_sentiment was accepted and ignored entirely, so a
    confidently negative review the lexicon misread stayed LOW.
    """
    lexicon_saw_nothing = [issue("product_quality", "Neutral", "hmm", "Product Support")]
    low = router.route_issues("Neutral", lexicon_saw_nothing, overall_confidence=0.99)
    lifted = router.route_issues("Negative", lexicon_saw_nothing, overall_confidence=0.99)
    assert low["overall_priority"] == "LOW"
    assert lifted["overall_priority"] == "MEDIUM"


def test_unconfident_model_verdict_does_not_lift(router):
    res = router.route_issues(
        "Negative",
        [issue("product_quality", "Neutral", "hmm", "Product Support")],
        overall_confidence=0.41,
    )
    assert res["overall_priority"] == "LOW"


def test_duplicate_aspects_do_not_fabricate_escalation(router):
    """
    Regression: negative_aspects was a list and len()-ed, so the same aspect
    twice counted as two distinct complaints and escalated.
    """
    res = router.route_issues("Negative", [
        issue("delivery", "Negative", "late", "Logistics"),
        issue("delivery", "Negative", "still late", "Logistics"),
    ])
    assert res["overall_priority"] == "MEDIUM"


def test_no_issues_defaults_to_customer_support(router):
    res = router.route_issues("Neutral", [])
    assert res["primary_department"] == "Customer Support"
    assert res["overall_priority"] == "LOW"


def test_critical_aspect_wins_ties(router):
    res = router.route_issues("Negative", [
        issue("delivery", "Negative", "late", "Logistics"),
        issue("refund_return", "Negative", "no refund", "Refunds"),
    ])
    assert res["primary_department"] == "Refunds"
