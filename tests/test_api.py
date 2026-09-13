import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_health():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

def test_analyze_endpoint():
    payload = {
        "review_text": "The phone is excellent, but delivery was very late and my refund has not arrived.",
        "customer_id": "CUST-TEST",
        "product_id": "PROD-TEST"
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "ticket_id" in data
    assert "overall_sentiment" in data
    assert "primary_department" in data
    assert "overall_priority" in data
    assert len(data["issues"]) >= 1

def test_analytics_endpoint():
    res = client.get("/api/v1/analytics")
    assert res.status_code == 200
    data = res.json()
    assert "total_reviews_analyzed" in data
    assert "sentiments" in data
