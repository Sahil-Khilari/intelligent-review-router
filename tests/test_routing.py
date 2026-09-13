import pytest
from app.services.routing import SupportRouter

@pytest.fixture
def router():
    return SupportRouter()

def test_urgent_priority_routing(router):
    issues = [
        {"aspect": "payment", "sentiment": "Negative", "opinion": "charged twice", "department": "Payments"},
        {"aspect": "refund_return", "sentiment": "Negative", "opinion": "refund not arrived", "department": "Refunds"}
    ]
    res = router.route_issues("Negative", issues)
    assert res["overall_priority"] == "URGENT"
    assert res["primary_department"] in ["Payments", "Refunds"]

def test_high_priority_routing(router):
    issues = [
        {"aspect": "delivery", "sentiment": "Negative", "opinion": "extremely late", "department": "Logistics"},
        {"aspect": "product_quality", "sentiment": "Negative", "opinion": "broken screen", "department": "Product Support"}
    ]
    res = router.route_issues("Negative", issues)
    assert res["overall_priority"] in ["HIGH", "URGENT"]

def test_low_priority_routing(router):
    issues = [
        {"aspect": "product_quality", "sentiment": "Positive", "opinion": "works great", "department": "Product Support"}
    ]
    res = router.route_issues("Positive", issues)
    assert res["overall_priority"] == "LOW"
