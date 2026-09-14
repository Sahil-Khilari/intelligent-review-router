"""
Aspect-based sentiment analysis pipeline.

Replaces the keyword lexicon in aspect_extraction.py as the primary extractor:

    ATE model      -> aspect term spans in arbitrary text
    APC model      -> polarity conditioned on each extracted term
    AspectMapper   -> which of the six departments owns that term

Deliberately returns the same issue contract the lexicon did - aspect, opinion,
sentiment, confidence, department - so TicketService, SupportRouter, the API
response models and the Streamlit UI are unchanged by this swap.

Falls back to the lexicon when model artifacts are absent, so the service still
starts on a checkout that has not downloaded models.
"""

import os
from pathlib import Path

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
ATE_DIR = MODELS_DIR / "ate"
APC_DIR = MODELS_DIR / "apc"

MAX_ASPECTS = 8
MIN_SPAN_CHARS = 2

# Worst sentiment wins when an aspect is mentioned more than once: support
# routing has to surface the worst thing said about delivery, not the first.
SENTIMENT_RANK = {"Negative": 0, "Neutral": 1, "Positive": 2}


def artifacts_present() -> bool:
    def has_weights(path: Path) -> bool:
        return any((path / name).exists()
                   for name in ("model.safetensors", "pytorch_model.bin"))
    return has_weights(ATE_DIR) and has_weights(APC_DIR)


class ABSAPipeline:
    """
    Models are loaded lazily on first use so importing this module is cheap and
    a missing artifact fails at the call site with a clear message.
    """

    def __init__(self, ate_dir=ATE_DIR, apc_dir=APC_DIR, mapper=None,
                 device=None, max_aspects=MAX_ASPECTS):
        self.ate_dir = Path(ate_dir)
        self.apc_dir = Path(apc_dir)
        self.max_aspects = max_aspects
        self._mapper = mapper
        self._device = device
        self._ate_model = self._ate_tokenizer = None
        self._apc_model = self._apc_tokenizer = None

    # ----------------------------------------------------------------- loading

    def _ensure_loaded(self):
        if self._ate_model is not None:
            return

        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        from app.services.device import resolve_device

        if not artifacts_present():
            raise FileNotFoundError(
                f"ABSA models not found in {MODELS_DIR}/. Train them with "
                f"`python train_all.py` or fetch them with "
                f"`python scripts/download_models.py`."
            )

        self._device = self._device or resolve_device("ABSA inference")

        self._ate_tokenizer = AutoTokenizer.from_pretrained(self.ate_dir)
        self._ate_model = AutoModelForTokenClassification.from_pretrained(
            self.ate_dir, dtype=torch.float32
        ).to(self._device).eval()

        self._apc_tokenizer = AutoTokenizer.from_pretrained(self.apc_dir)
        self._apc_model = AutoModelForSequenceClassification.from_pretrained(
            self.apc_dir, dtype=torch.float32
        ).to(self._device).eval()

    @property
    def mapper(self):
        if self._mapper is None:
            from app.services.aspect_mapping import get_mapper

            self._mapper = get_mapper()
        return self._mapper

    def describe(self) -> dict:
        return {
            "loaded": self._ate_model is not None,
            "artifacts_present": artifacts_present(),
            "ate_dir": str(self.ate_dir),
            "apc_dir": str(self.apc_dir),
            "device": str(self._device) if self._device else None,
        }

    # --------------------------------------------------------------- inference

    def extract_terms(self, text: str) -> list[tuple[str, int, int]]:
        """Returns [(term, start, end)] for each aspect span the ATE model finds."""
        import torch

        from app.services.bio_alignment import ATE_ID2LABEL, spans_from_tags

        self._ensure_loaded()
        encoded = self._ate_tokenizer(
            text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self._ate_model(**encoded).logits

        tags = [ATE_ID2LABEL[i] for i in logits.argmax(-1)[0].tolist()]
        results = []
        for start, end in spans_from_tags(tags, offsets):
            # Tokenizer offsets include the preceding space, so the raw slice
            # would carry a leading blank into the ticket summary.
            term = text[start:end].strip()
            if len(term) < MIN_SPAN_CHARS:
                continue
            adjusted_start = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            results.append((term, adjusted_start, adjusted_start + len(term)))
        return results[: self.max_aspects]

    def classify_polarities(self, text: str, terms: list[str]) -> list[tuple[str, float]]:
        """Polarity per term, conditioned on the term. One batched forward pass."""
        import torch

        if not terms:
            return []

        self._ensure_loaded()
        encoded = self._apc_tokenizer(
            [text] * len(terms),
            terms,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self._apc_model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu()

        id2label = self._apc_model.config.id2label
        out = []
        for row in probabilities:
            best = int(row.argmax())
            out.append((id2label.get(best, str(best)), float(row[best])))
        return out

    def extract_aspects(self, review_text: str, overall_sentiment: str = None,
                        overall_confidence: float = None) -> list[dict]:
        """
        The contract the lexicon used, so nothing downstream changes.

        An aspect mentioned more than once keeps its most negative mention.
        """
        text = (review_text or "").strip()
        if not text:
            return self._fallback_issue(text, overall_sentiment, overall_confidence)

        spans = self.extract_terms(text)
        if not spans:
            return self._fallback_issue(text, overall_sentiment, overall_confidence)

        terms = [term for term, _, _ in spans]
        polarities = self.classify_polarities(text, terms)
        mappings = self.mapper.map_terms(terms)

        detected: dict[str, dict] = {}
        for (term, start, end), (sentiment, confidence), mapping in zip(
            spans, polarities, mappings
        ):
            issue = {
                "aspect": mapping["department"],
                "opinion": _sentence_around(text, start, end),
                "sentiment": sentiment,
                "confidence": round(float(confidence), 2),
                "department": mapping["department_name"],
                "term": term,
                "mapping_score": mapping["score"],
            }
            if mapping["low_confidence"]:
                issue["low_confidence"] = True

            existing = detected.get(issue["aspect"])
            if existing is None or (
                SENTIMENT_RANK.get(sentiment, 1)
                < SENTIMENT_RANK.get(existing["sentiment"], 1)
            ):
                detected[issue["aspect"]] = issue

        return list(detected.values())

    @staticmethod
    def _fallback_issue(text, overall_sentiment, overall_confidence):
        """
        No aspect found. Report the model's own verdict rather than asserting a
        Neutral product_quality issue, which is what the lexicon used to do.
        """
        return [{
            "aspect": "product_quality",
            "opinion": text,
            "sentiment": overall_sentiment or "Neutral",
            "confidence": round(float(overall_confidence), 2)
            if overall_confidence is not None else 0.70,
            "department": "Product Support",
            "source": "fallback",
        }]


def _sentence_around(text: str, start: int, end: int) -> str:
    """
    The sentence containing the span, so the ticket shows the complaint in
    context rather than a bare noun phrase.
    """
    left = max(text.rfind(".", 0, start), text.rfind("!", 0, start),
               text.rfind("?", 0, start))
    right_candidates = [p for p in (text.find(".", end), text.find("!", end),
                                    text.find("?", end)) if p != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return text[left + 1:right].strip() or text.strip()


def build_extractor(sentiment_analyzer=None):
    """
    Returns the ABSA pipeline when artifacts exist, else the keyword lexicon.

    Keeps the service startable on a checkout with no models downloaded, while
    making the degraded state visible through /health rather than silent.
    """
    if artifacts_present():
        return ABSAPipeline()

    from app.services.aspect_extraction import AspectExtractor

    print(
        f"[ABSA] No models in {MODELS_DIR}/ - falling back to the keyword "
        f"lexicon. Fetch them with: python scripts/download_models.py"
    )
    return AspectExtractor(sentiment_analyzer=sentiment_analyzer)
