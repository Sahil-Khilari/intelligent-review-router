"""
Maps an extracted aspect term to one of the six support departments.

This bridges a domain gap. SemEval-2014 supplies real aspect-term annotations
but in laptop and restaurant domains, while this project routes e-commerce
support categories. Extracting the *term* generalises across domains; deciding
which department owns it is a separate question.

The mapping is deliberately training-free: embedding similarity against
per-department prototype phrases. That means it can be retuned on a CPU without
another GPU session, which matters because it is the least certain component in
the pipeline - there is no published benchmark for "aspect term -> support
department" to anchor it against.

Terms whose best match falls below THRESHOLD are routed to customer_support and
flagged low_confidence, rather than being forced into a department the model is
not actually confident about.
"""

import functools

DEPARTMENT_PROTOTYPES = {
    "delivery": [
        "delivery", "shipping", "courier", "arrived late", "tracking",
        "dispatch", "shipment delay", "postage", "never arrived", "in transit",
    ],
    "product_quality": [
        "build quality", "broken screen", "defective item", "battery life",
        "performance", "material", "durability", "stopped working",
        "sound quality", "display", "faulty product", "keyboard", "touchpad",
        "buttons", "camera", "fits well", "size", "colour", "taste",
        "the product itself", "hardware", "software",
    ],
    "payment": [
        "payment", "charged twice", "billing", "transaction", "card declined",
        "checkout", "invoice", "overcharged", "payment gateway", "price",
    ],
    "refund_return": [
        "refund", "return", "money back", "replacement", "exchange", "pickup",
        "reimbursement", "return policy", "cancel order", "warranty claim",
    ],
    "packaging": [
        "packaging", "box", "bubble wrap", "sealed", "damaged box", "carton",
        "wrapping", "package condition", "crushed box", "torn packet",
    ],
    "customer_support": [
        "customer service", "support agent", "helpline", "response time",
        "representative", "email support", "call centre", "staff",
        "complaint handling", "after sales service",
    ],
}

# Department key -> the team name used by SupportRouter and the UI.
DEPARTMENT_NAMES = {
    "delivery": "Logistics",
    "product_quality": "Product Support",
    "payment": "Payments",
    "refund_return": "Refunds",
    "packaging": "Returns / Replacement",
    "customer_support": "Customer Support",
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cosine similarity below which a term is not confidently any department.
#
# Raised from 0.35 after measurement: at 0.35, "keyboard" mapped to packaging
# (0.358) and "the food" to delivery (0.384) - both wrong, and both only just
# over the line. Misrouting a ticket just above a threshold is worse than
# admitting uncertainty, and the low-confidence flag feeds a review queue
# rather than being discarded. Every correct e-commerce mapping measured scores
# above 0.50, so 0.45 costs nothing on the cases that matter.
#
# Tunable on CPU against the gold set; no GPU session required.
THRESHOLD = 0.45

FALLBACK_DEPARTMENT = "customer_support"


class AspectMapper:
    """
    Embedding-similarity mapper.

    Prototype embeddings are computed once and cached, so mapping a term is a
    single encode plus a matrix multiply.
    """

    def __init__(self, model_name=EMBEDDING_MODEL, threshold=THRESHOLD, device=None):
        self.model_name = model_name
        self.threshold = threshold
        self._device = device
        self._model = None
        self._prototype_matrix = None
        self._prototype_owner = []

    def _ensure_loaded(self):
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        from app.services.device import resolve_device

        device = self._device or resolve_device("Aspect mapping")
        self._model = SentenceTransformer(self.model_name, device=str(device))

        phrases, owners = [], []
        for department, prototypes in DEPARTMENT_PROTOTYPES.items():
            phrases.extend(prototypes)
            owners.extend([department] * len(prototypes))

        self._prototype_matrix = self._model.encode(
            phrases, convert_to_tensor=True, normalize_embeddings=True
        )
        self._prototype_owner = owners

    def map_term(self, term: str) -> dict:
        """
        Returns {department, department_name, score, low_confidence}.

        Max-pools similarity per department rather than averaging: a term only
        has to match one prototype well, and averaging would penalise
        departments with broader prototype sets.
        """
        term = (term or "").strip()
        if not term:
            return self._fallback(0.0)

        self._ensure_loaded()
        from sentence_transformers import util

        embedding = self._model.encode(
            term, convert_to_tensor=True, normalize_embeddings=True
        )
        scores = util.cos_sim(embedding, self._prototype_matrix)[0]

        best_per_department = {}
        for owner, score in zip(self._prototype_owner, scores.tolist()):
            if score > best_per_department.get(owner, -1.0):
                best_per_department[owner] = score

        department, score = max(best_per_department.items(), key=lambda kv: kv[1])
        if score < self.threshold:
            return self._fallback(score)

        return {
            "department": department,
            "department_name": DEPARTMENT_NAMES[department],
            "score": round(float(score), 4),
            "low_confidence": False,
        }

    def map_terms(self, terms: list[str]) -> list[dict]:
        """Batch form; one encode call for the whole list."""
        terms = [(t or "").strip() for t in terms]
        if not terms:
            return []

        self._ensure_loaded()
        from sentence_transformers import util

        embeddings = self._model.encode(
            terms, convert_to_tensor=True, normalize_embeddings=True
        )
        all_scores = util.cos_sim(embeddings, self._prototype_matrix)

        results = []
        for row in all_scores:
            best_per_department = {}
            for owner, score in zip(self._prototype_owner, row.tolist()):
                if score > best_per_department.get(owner, -1.0):
                    best_per_department[owner] = score
            department, score = max(best_per_department.items(), key=lambda kv: kv[1])
            if score < self.threshold:
                results.append(self._fallback(score))
            else:
                results.append({
                    "department": department,
                    "department_name": DEPARTMENT_NAMES[department],
                    "score": round(float(score), 4),
                    "low_confidence": False,
                })
        return results

    @staticmethod
    def _fallback(score: float) -> dict:
        return {
            "department": FALLBACK_DEPARTMENT,
            "department_name": DEPARTMENT_NAMES[FALLBACK_DEPARTMENT],
            "score": round(float(score), 4),
            "low_confidence": True,
        }


@functools.cache
def get_mapper() -> AspectMapper:
    """Process-wide singleton; the prototype matrix is built once."""
    return AspectMapper()
