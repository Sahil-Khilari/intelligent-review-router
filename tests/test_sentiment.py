import pytest
from app.services.sentiment import SentimentAnalyzer

@pytest.fixture
def analyzer():
    return SentimentAnalyzer()

def test_positive_sentiment(analyzer):
    res = analyzer.predict("This product is top notch quality and works amazingly well!")
    assert res["sentiment"] in ["Positive", "Neutral"]
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0

def test_negative_sentiment(analyzer):
    res = analyzer.predict("Delivery was horrible and item was broken on arrival.")
    assert res["sentiment"] in ["Negative", "Neutral"]
    assert "confidence" in res

def test_empty_string(analyzer):
    res = analyzer.predict("")
    assert res["sentiment"] == "Neutral"
    assert res["confidence"] == 0.50
