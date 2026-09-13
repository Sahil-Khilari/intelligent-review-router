ASPECT_DEPARTMENT_MAP = {
    "delivery": "Logistics",
    "product_quality": "Product Support",
    "payment": "Payments",
    "refund_return": "Refunds",
    "packaging": "Returns / Replacement",
    "customer_support": "Customer Support"
}

CRITICAL_ASPECTS = {"payment", "refund_return"}

class SupportRouter:
    def route_issues(self, overall_sentiment: str, issues: list[dict]) -> dict:
        routing_list = []
        negative_aspects = []
        
        for issue in issues:
            aspect = issue.get("aspect", "product_quality")
            sent = issue.get("sentiment", "Neutral")
            dept = issue.get("department") or ASPECT_DEPARTMENT_MAP.get(aspect, "Customer Support")
            
            if sent == "Negative":
                negative_aspects.append(aspect)
                # Compute aspect-specific priority
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
            
        # Calculate overall system priority
        neg_count = len(negative_aspects)
        has_critical = any(a in CRITICAL_ASPECTS for a in negative_aspects)
        
        if has_critical and neg_count >= 2:
            overall_priority = "URGENT"
        elif has_critical or neg_count >= 2:
            overall_priority = "HIGH"
        elif neg_count == 1:
            overall_priority = "MEDIUM"
        else:
            overall_priority = "LOW"
            
        # Select primary department
        if routing_list:
            # Prefer department with highest priority issue
            priority_weights = {"URGENT": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            sorted_routing = sorted(routing_list, key=lambda x: priority_weights.get(x["priority"], 0), reverse=True)
            primary_department = sorted_routing[0]["department"]
        else:
            primary_department = "Customer Support"
            
        return {
            "primary_department": primary_department,
            "overall_priority": overall_priority,
            "routing": routing_list
        }
