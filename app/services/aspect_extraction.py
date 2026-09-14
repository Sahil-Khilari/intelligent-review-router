import re

from app.services.preprocessing import split_into_sentences

ASPECT_DICTIONARY = {
    "delivery": {
        "keywords": ["delivery", "delivered", "shipping", "courier", "delayed", "late", "dispatch", "postage", "tracking", "transit", "arrive", "arrived", "shipped"],
        "department": "Logistics"
    },
    "product_quality": {
        "keywords": ["quality", "broken", "screen", "defective", "working", "sound", "material", "build", "camera", "battery", "display", "performance", "durable", "item"],
        "department": "Product Support"
    },
    "payment": {
        "keywords": ["payment", "charged", "deducted", "fee", "gateway", "transaction", "upi", "card", "billing", "double charge", "checkout", "paid", "invoice"],
        "department": "Payments"
    },
    "refund_return": {
        "keywords": ["refund", "return", "money back", "reimbursement", "pickup", "exchange", "replacement"],
        "department": "Refunds"
    },
    "packaging": {
        "keywords": ["packaging", "box", "package", "envelope", "wrapped", "wrap", "cushioning", "unpacked", "bubble wrap", "torn", "carton"],
        "department": "Returns / Replacement"
    },
    "customer_support": {
        "keywords": ["support", "customer service", "representative", "agent", "email", "response", "call", "helpdesk", "staff", "service team"],
        "department": "Customer Support"
    }
}

POSITIVE_INDICATORS = [
    "good", "great", "excellent", "fast", "prompt", "amazing", "premium", "helpful",
    "polite", "secure", "quick", "perfect", "love", "satisfied", "smooth", "durable",
    "top", "notch", "sturdy", "beautiful", "pristine", "hassle-free", "exceeded",
    "impressed", "recommend", "instant", "easy", "clear", "crystal", "happy", "nice",
    "solid", "worth", "reliable", "early", "cheap",
]

NEGATIVE_INDICATORS = [
    "bad", "terrible", "horrible", "late", "delayed", "broken", "defective", "failed",
    "useless", "rejected", "crushed", "damaged", "disappointed", "unhappy",
    "disconnected", "wrongful", "deducted", "poor", "rude", "missing", "torn",
    "error", "unauthorized", "cracked", "scratched", "faulty", "worst", "awful",
    "cheated", "fraud", "stuck", "pending", "unresponsive", "slow", "wrong",
    "expired", "leaking", "melted", "smashed", "dented",
]

# Multi-word complaints that carry no single negative token. Without these,
# "the refund never came" scores zero negative hits and reads as Positive.
NEGATIVE_PHRASES = [
    r"never (?:came|arrived|received|showed|shipped|responded|worked|refunded)",
    r"(?:not|n't) (?:arrived|received|working|resolved|delivered|shipped|refunded|scheduled|processed)",
    r"(?:has|have|had) (?:not|n't) \w+",
    r"stopped working",
    r"no response",
    r"still (?:waiting|pending|not)",
    r"charged twice",
    r"double[- ]charged",
    r"money (?:was )?deducted",
    r"waiting \w+ (?:days|weeks|months)",
    r"want(?:ed)? (?:a|my) refund",
    r"worse than",
]

# A cue flips the polarity of an indicator appearing shortly after it, so
# "not late" and "nothing was broken" stop reading as complaints.
NEGATION_CUES = {
    "not", "no", "never", "nothing", "none", "cannot", "cant", "didnt", "doesnt",
    "dont", "isnt", "wasnt", "werent", "arent", "without", "hardly", "barely",
    "neither", "nor",
}

NEGATION_WINDOW = 3

# Match a keyword at a word boundary, allowing ordinary inflection. This is the
# difference between matching "delayed"/"lately" (wanted) and matching "late"
# inside "chocolate", "translate" or "related" (a false Logistics ticket).
_INFLECTION = r"(?:s|es|ed|d|ing|ly|er|est)?"

_WORD = re.compile(r"[a-z']+")


def _term_pattern(terms):
    alternatives = "|".join(
        re.escape(t) for t in sorted(terms, key=len, reverse=True)
    )
    return re.compile(rf"\b(?:{alternatives}){_INFLECTION}\b", re.IGNORECASE)


_ASPECT_PATTERNS = {
    aspect: _term_pattern(info["keywords"])
    for aspect, info in ASPECT_DICTIONARY.items()
}
_POSITIVE_SET = {w.lower() for w in POSITIVE_INDICATORS}
_NEGATIVE_SET = {w.lower() for w in NEGATIVE_INDICATORS}
_NEGATIVE_PHRASE_RE = re.compile("|".join(NEGATIVE_PHRASES), re.IGNORECASE)
_STEM_SUFFIXES = ("ly", "ing", "ed", "es", "s")

_SENTIMENT_RANK = {"Negative": 0, "Neutral": 1, "Positive": 2}


def _stems(token):
    yield token
    for suffix in _STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            yield token[: -len(suffix)]


def _lookup(token, vocabulary):
    return any(stem in vocabulary for stem in _stems(token))


def polarity_scores(text: str) -> tuple[int, int]:
    """
    Counts positive and negative opinion cues, honouring negation.

    Returns (positive_hits, negative_hits).
    """
    tokens = _WORD.findall(text.lower().replace("'", ""))
    positive = negative = 0

    for index, token in enumerate(tokens):
        is_positive = _lookup(token, _POSITIVE_SET)
        is_negative = _lookup(token, _NEGATIVE_SET)
        if not (is_positive or is_negative):
            continue

        window = tokens[max(0, index - NEGATION_WINDOW):index]
        negated = any(w in NEGATION_CUES for w in window)

        if is_negative and not is_positive:
            negative += 0 if negated else 1
            positive += 1 if negated else 0
        else:
            positive += 0 if negated else 1
            negative += 1 if negated else 0

    negative += len(_NEGATIVE_PHRASE_RE.findall(text))
    return positive, negative


class AspectExtractor:
    def __init__(self, sentiment_analyzer=None):
        self.sentiment_analyzer = sentiment_analyzer

    def _clause_polarity(self, span_text, clause):
        positive, negative = polarity_scores(span_text)
        if negative > positive:
            return "Negative", min(0.75 + 0.08 * negative, 0.98)
        if positive > negative:
            return "Positive", min(0.75 + 0.08 * positive, 0.98)

        # Ambiguous clause: defer to the sentiment model rather than guessing.
        if self.sentiment_analyzer is not None:
            result = self.sentiment_analyzer.predict(clause)
            return result["sentiment"], result["confidence"]
        return "Neutral", 0.70

    def extract_aspects(self, review_text: str, overall_sentiment: str = None,
                        overall_confidence: float = None) -> list[dict]:
        """
        Returns one issue per aspect mentioned anywhere in the review.

        Every clause is scanned for every aspect, and an aspect seen more than
        once keeps its most negative mention — support routing has to surface
        the worst thing said about delivery, not the first thing.
        """
        clauses = split_into_sentences(review_text)
        detected: dict[str, dict] = {}

        for clause in clauses:
            matches = {
                aspect: pattern.search(clause)
                for aspect, pattern in _ASPECT_PATTERNS.items()
            }
            present = {a: m for a, m in matches.items() if m is not None}

            for aspect, match in present.items():
                # With several aspects in one clause, score each against the
                # text around its own mention instead of the whole clause.
                if len(present) > 1:
                    start = max(0, match.start() - 40)
                    span_text = clause[start:match.end() + 40]
                else:
                    span_text = clause

                sentiment, confidence = self._clause_polarity(span_text, clause)
                issue = {
                    "aspect": aspect,
                    "opinion": clause,
                    "sentiment": sentiment,
                    "confidence": round(confidence, 2),
                    "department": ASPECT_DICTIONARY[aspect]["department"],
                }

                existing = detected.get(aspect)
                if existing is None or (
                    _SENTIMENT_RANK[sentiment] < _SENTIMENT_RANK[existing["sentiment"]]
                ):
                    detected[aspect] = issue

        if detected:
            return list(detected.values())

        # Nothing matched. Fall back to the model's own verdict on the review
        # rather than asserting a Neutral product_quality issue.
        sentiment = overall_sentiment or "Neutral"
        confidence = overall_confidence if overall_confidence is not None else 0.70
        return [{
            "aspect": "product_quality",
            "opinion": " ".join(clauses) if clauses else review_text,
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "department": ASPECT_DICTIONARY["product_quality"]["department"],
            "source": "fallback",
        }]
