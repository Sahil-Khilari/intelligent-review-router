import uuid
from datetime import datetime

class TicketService:
    def __init__(self, sentiment_analyzer, aspect_extractor, support_router):
        self.sentiment_analyzer = sentiment_analyzer
        self.aspect_extractor = aspect_extractor
        self.support_router = support_router

    def analyze_and_route(self, review_text: str, customer_id: str = "CUST-GUEST", product_id: str = "PROD-GENERIC") -> dict:
        # Step 1: Overall Sentiment
        sent_res = self.sentiment_analyzer.predict(review_text)
        overall_sentiment = sent_res["sentiment"]
        overall_confidence = sent_res["confidence"]

        # Step 2: Aspect & Opinion Extraction
        issues = self.aspect_extractor.extract_aspects(review_text)

        # Step 3: Support Routing & Priority
        route_res = self.support_router.route_issues(overall_sentiment, issues)

        ticket_id = f"TICK-{uuid.uuid4().hex[:8].upper()}"

        return {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "product_id": product_id,
            "review_text": review_text,
            "overall_sentiment": overall_sentiment,
            "overall_confidence": overall_confidence,
            "model_type": sent_res.get("model_type", "baseline"),
            "issues": issues,
            "primary_department": route_res["primary_department"],
            "overall_priority": route_res["overall_priority"],
            "routing": route_res["routing"],
            "created_at": datetime.now().isoformat()
        }
