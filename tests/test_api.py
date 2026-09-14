def test_root_health(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"


def test_health_reports_active_model(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "model_type" in body["model"]
    assert isinstance(body["degraded"], bool)


def test_analyze_endpoint(client):
    res = client.post("/api/v1/analyze", json={
        "review_text": "The phone is excellent, but delivery was very late and my refund has not arrived.",
        "customer_id": "CUST-TEST",
        "product_id": "PROD-TEST",
    })
    assert res.status_code == 200
    data = res.json()
    assert "ticket_id" in data
    assert data["overall_sentiment"] in ("Positive", "Neutral", "Negative")
    assert data["primary_department"]
    assert data["overall_priority"] in ("LOW", "MEDIUM", "HIGH", "URGENT")
    assert len(data["issues"]) >= 1


def test_analyze_detects_both_complaints(client):
    """The delivery and refund complaints must both reach the payload."""
    res = client.post("/api/v1/analyze", json={
        "review_text": "The phone is excellent, but delivery was very late and my refund has not arrived.",
    })
    aspects = {i["aspect"] for i in res.json()["issues"]}
    assert {"delivery", "refund_return"} <= aspects


def test_analyze_rejects_empty_review(client):
    assert client.post("/api/v1/analyze", json={"review_text": "   "}).status_code == 422


def test_analyze_rejects_oversized_review(client):
    res = client.post("/api/v1/analyze", json={"review_text": "x" * 20_001})
    assert res.status_code == 422


def test_analytics_endpoint(client):
    res = client.get("/api/v1/analytics")
    assert res.status_code == 200
    data = res.json()
    assert "total_reviews_analyzed" in data
    assert "sentiments" in data


def test_analytics_counts_reflect_writes(client):
    before = client.get("/api/v1/analytics").json()["total_reviews_analyzed"]
    client.post("/api/v1/analyze", json={"review_text": "Charged twice and no refund yet."})
    after = client.get("/api/v1/analytics").json()["total_reviews_analyzed"]
    assert after == before + 1


def test_create_and_list_ticket(client):
    created = client.post("/api/v1/tickets", json={
        "review_text": "Refund has not arrived and I was charged twice.",
        "customer_id": "CUST-9",
    })
    assert created.status_code == 200
    ticket_id = created.json()["ticket_id"]

    listed = client.get("/api/v1/tickets", params={"limit": 50}).json()
    assert any(t["ticket_id"] == ticket_id for t in listed)


def test_manual_ticket_is_counted_as_a_review(client):
    """Regression: POST /tickets analysed a review but never stored it."""
    before = client.get("/api/v1/analytics").json()["total_reviews_analyzed"]
    client.post("/api/v1/tickets", json={"review_text": "Box arrived crushed."})
    after = client.get("/api/v1/analytics").json()["total_reviews_analyzed"]
    assert after == before + 1


def test_update_ticket_status(client):
    ticket_id = client.post("/api/v1/tickets", json={
        "review_text": "Payment failed but money was deducted.",
    }).json()["ticket_id"]

    res = client.patch(f"/api/v1/tickets/{ticket_id}", json={"status": "resolved"})
    assert res.status_code == 200
    assert res.json()["status"] == "RESOLVED"


def test_update_unknown_ticket_is_404(client):
    assert client.patch("/api/v1/tickets/TICK-NOPE", json={"status": "OPEN"}).status_code == 404


def test_update_with_invalid_status_is_rejected(client):
    ticket_id = client.post("/api/v1/tickets", json={
        "review_text": "Box arrived torn.",
    }).json()["ticket_id"]
    res = client.patch(f"/api/v1/tickets/{ticket_id}", json={"status": "BANANA"})
    assert res.status_code in (400, 422)


def test_ticket_limit_is_validated(client):
    assert client.get("/api/v1/tickets", params={"limit": -5}).status_code == 422
    assert client.get("/api/v1/tickets", params={"limit": 10_000}).status_code == 422


def test_tickets_filter_by_department(client):
    client.post("/api/v1/tickets", json={"review_text": "Charged twice for one order."})
    rows = client.get("/api/v1/tickets", params={"department": "Payments"}).json()
    assert all(t["department"] == "Payments" for t in rows)
