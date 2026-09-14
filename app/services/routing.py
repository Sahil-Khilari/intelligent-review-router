ASPECT_DEPARTMENT_MAP = {
    "delivery": "Logistics",
    "product_quality": "Product Support",
    "payment": "Payments",
    "refund_return": "Refunds",
    "packaging": "Returns / Replacement",
    "customer_support": "Customer Support"
}

CRITICAL_ASPECTS = {"payment", "refund_return"}

PRIORITY_WEIGHTS = {"URGENT": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

# A confident Negative verdict from the sentiment model is evidence in its own
# right. When the lexicon finds no negative aspect but the model is sure the
# review is negative, the review is still a complaint.
OVERRIDE_CONFIDENCE = 0.80


class SupportRouter:
    def route_issues(self, overall_sentiment: str, issues: list[dict],
                     overall_confidence: float = None) -> dict:
        routing_list = []
        negative_aspects = set()

        for issue in issues:
            aspect = issue.get("aspect", "product_quality")
            sent = issue.get("sentiment", "Neutral")
            dept = issue.get("department") or ASPECT_DEPARTMENT_MAP.get(aspect, "Customer Support")

            if sent == "Negative":
                negative_aspects.add(aspect)
                if aspect in CRITICAL_ASPECTS:
                    priority = "HIGH"
                else:
                    priority = "MEDIUM"
                reason = f"Customer reported negative issue regarding {aspect} ('{issue.get('opinion', '')}')"
            elif sent == "Neutral":
                priority = "LOW"
                reason = f"Neutral feedback on {aspect}"
            else:
                priority = "LOW"
                reason = f"Positive feedback on {aspect}"

            routing_list.append({
                "department": dept,
                "aspect": aspect,
                "priority": priority,
                "reason": reason
            })

        neg_count = len(negative_aspects)
        has_critical = bool(negative_aspects & CRITICAL_ASPECTS)

        # The sentiment model used to be computed, passed in, and ignored.
        # A confident Negative verdict now lifts a review the lexicon read as
        # harmless to at least MEDIUM, which is where under-escalation lived.
        model_says_negative = (
            overall_sentiment == "Negative"
            and (overall_confidence is None or overall_confidence >= OVERRIDE_CONFIDENCE)
        )

        if has_critical and neg_count >= 2:
            overall_priority = "URGENT"
        elif has_critical or neg_count >= 2:
            overall_priority = "HIGH"
        elif neg_count == 1:
            overall_priority = "MEDIUM"
        elif model_says_negative:
            overall_priority = "MEDIUM"
        else:
            overall_priority = "LOW"

        # Route on complaints. Praise stays in the payload as insight data but
        # must not decide which team owns the ticket, and must not become the
        # primary department of an otherwise negative review.
        actionable = [r for r in routing_list if r["priority"] != "LOW"]
        candidates = actionable or routing_list

        if candidates:
            ranked = sorted(
                candidates,
                key=lambda x: (
                    PRIORITY_WEIGHTS.get(x["priority"], 0),
                    x["aspect"] in CRITICAL_ASPECTS,
                ),
                reverse=True,
            )
            primary_department = ranked[0]["department"]
        else:
            primary_department = "Customer Support"

        return {
            "primary_department": primary_department,
            "overall_priority": overall_priority,
            "routing": routing_list
        }
