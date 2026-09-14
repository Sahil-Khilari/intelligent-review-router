import re
from app.services.preprocessing import split_into_sentences

ASPECT_DICTIONARY = {
    "delivery": {
        "keywords": ["delivery", "delivered", "shipping", "courier", "delayed", "late", "dispatch", "postage", "tracking", "transit"],
        "department": "Logistics"
    },
    "product_quality": {
        "keywords": ["quality", "broken", "screen", "defective", "working", "sound", "material", "build", "camera", "battery", "display", "performance", "durable"],
        "department": "Product Support"
    },
    "payment": {
        "keywords": ["payment", "charged", "deducted", "fee", "gateway", "transaction", "upi", "card", "billing", "double charge"],
        "department": "Payments"
    },
    "refund_return": {
        "keywords": ["refund", "return", "money back", "reimbursement", "pickup", "exchange"],
        "department": "Refunds"
    },
    "packaging": {
        "keywords": ["packaging", "box", "package", "envelope", "wrapped", "cushioning", "unpacked", "bubble wrap", "torn"],
        "department": "Returns / Replacement"
    },
    "customer_support": {
        "keywords": ["support", "customer service", "representative", "agent", "email", "response", "call", "helpdesk", "staff"],
        "department": "Customer Support"
    }
}

POSITIVE_INDICATORS = ["good", "great", "excellent", "fast", "prompt", "amazing", "premium", "helpful", "polite", "secure", "quick", "perfect", "love", "satisfied", "smooth", "durable"]
NEGATIVE_INDICATORS = ["bad", "terrible", "horrible", "late", "delayed", "broken", "defective", "failed", "useless", "rejected", "crushed", "damaged", "disappointed", "unhappy", "disconnected", "wrongful", "deducted"]

class AspectExtractor:
    def __init__(self, sentiment_analyzer=None):
        self.sentiment_analyzer = sentiment_analyzer

    def extract_aspects(self, review_text: str, overall_sentiment: str = None,
                        overall_confidence: float = None) -> list[dict]:
        clauses = split_into_sentences(review_text)
        detected_issues = []
        seen_aspects = set()

        for clause in clauses:
            lower_c = clause.lower()
            best_aspect = None
            max_kw_count = 0

            for aspect, info in ASPECT_DICTIONARY.items():
                count = sum(1 for kw in info["keywords"] if kw in lower_c)
                if count > max_kw_count:
                    max_kw_count = count
                    best_aspect = aspect

            if best_aspect and best_aspect not in seen_aspects:
                seen_aspects.add(best_aspect)
                
                # Aspect-level sentiment score
                pos_hits = sum(1 for w in POSITIVE_INDICATORS if w in lower_c)
                neg_hits = sum(1 for w in NEGATIVE_INDICATORS if w in lower_c)

                if neg_hits > pos_hits:
                    sent = "Negative"
                    conf = min(0.75 + 0.08 * neg_hits, 0.98)
                elif pos_hits > neg_hits:
                    sent = "Positive"
                    conf = min(0.75 + 0.08 * pos_hits, 0.98)
                else:
                    # If clause sentiment ambiguous, query overall sentiment analyzer if available
                    if self.sentiment_analyzer:
                        sub_res = self.sentiment_analyzer.predict(clause)
                        sent = sub_res["sentiment"]
                        conf = sub_res["confidence"]
                    else:
                        sent = "Neutral"
                        conf = 0.75

                detected_issues.append({
                    "aspect": best_aspect,
                    "opinion": clause,
                    "sentiment": sent,
                    "confidence": round(conf, 2),
                    "department": ASPECT_DICTIONARY[best_aspect]["department"]
                })

        # Fallback if no specific aspect detected. Reports the sentiment
        # model's own verdict when one is available, rather than asserting
        # Neutral over a confident Negative.
        if not detected_issues:
            detected_issues.append({
                "aspect": "product_quality",
                "opinion": review_text,
                "sentiment": overall_sentiment or "Neutral",
                "confidence": round(float(overall_confidence), 2)
                if overall_confidence is not None else 0.70,
                "department": "Product Support",
                "source": "fallback",
            })

        return detected_issues
