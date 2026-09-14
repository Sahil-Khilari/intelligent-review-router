import os
import joblib
import json
import torch
import numpy as np

from app.services.device import resolve_device
from app.services.preprocessing import clean_text

class SentimentAnalyzer:
    def __init__(self, model_dir: str = None):
        self.model_type = "baseline"
        self.vectorizer = None
        self.model = None
        self.tokenizer = None
        self.hf_model = None
        self.label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.model_path = None
        self.weights_missing = False
        self.device = None
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        transformer_path = model_dir or os.path.join(base_dir, "models", "trained_model", "sentiment")
        baseline_path = os.path.join(base_dir, "models", "trained_model", "baseline")
        
        # Presence is decided by the weights file, not config.json. Gating on the
        # metadata meant an untrained checkout entered the transformer branch,
        # raised inside from_pretrained, and degraded to TF-IDF in silence.
        self.weights_missing = (
            os.path.exists(os.path.join(transformer_path, "config.json"))
            and not self._has_weights(transformer_path)
        )
        if self.weights_missing:
            print(
                f"[SentimentAnalyzer] No trained weights in {transformer_path} - "
                "serving the TF-IDF baseline. Run: python train_all.py"
            )

        if self._has_weights(transformer_path):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.device = resolve_device("Sentiment inference")
                self.tokenizer = AutoTokenizer.from_pretrained(transformer_path)
                self.hf_model = AutoModelForSequenceClassification.from_pretrained(transformer_path)
                self.hf_model.to(self.device)
                self.hf_model.eval()
                self.model_type = "transformer"
                self.model_path = transformer_path
                print(f"[SentimentAnalyzer] Loaded Transformer from {transformer_path} on {self.device}")
                return
            except Exception as e:
                print(f"[SentimentAnalyzer] Failed to load Transformer: {e}. Falling back to baseline.")
                
        # Fallback to TF-IDF Baseline model
        vec_file = os.path.join(baseline_path, "tfidf_vectorizer.joblib")
        model_file = os.path.join(baseline_path, "baseline_model.joblib")
        
        if os.path.exists(vec_file) and os.path.exists(model_file):
            try:
                self.vectorizer = joblib.load(vec_file)
                self.model = joblib.load(model_file)
                self.model_type = "baseline"
                print(f"[SentimentAnalyzer] Loaded TF-IDF Baseline model from {baseline_path}")
            except Exception as e:
                print(f"[SentimentAnalyzer] Error unpickling model: {e}")
                self._build_inmemory_fallback()
        else:
            self._build_inmemory_fallback()

    @staticmethod
    def _has_weights(path: str) -> bool:
        return any(
            os.path.exists(os.path.join(path, name))
            for name in ("model.safetensors", "pytorch_model.bin", "tf_model.h5")
        )

    def describe(self) -> dict:
        """Which model is answering, and on what device. Surfaced by /health."""
        return {
            "model_type": self.model_type,
            "transformer_loaded": self.hf_model is not None,
            "weights_missing": self.weights_missing,
            "model_path": self.model_path,
            "device": str(self.device) if self.device is not None else None,
        }

    def _build_inmemory_fallback(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        print("[SentimentAnalyzer] Initializing fast in-memory fallback model...")
        train_texts = [
            "excellent top notch quality works great amazing love it",
            "great fast delivery super happy pristine packaging",
            "terrible horrible broken defective screen stopped working",
            "delivery was delayed very late refund has not arrived charged twice",
            "product quality average okay standard delivery"
        ]
        train_labels = ["Positive", "Positive", "Negative", "Negative", "Neutral"]
        
        self.vectorizer = TfidfVectorizer().fit(train_texts)
        self.model = LogisticRegression().fit(self.vectorizer.transform(train_texts), train_labels)
        self.model_type = "rule_fallback"

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)
        if not cleaned:
            return {"sentiment": "Neutral", "confidence": 0.50, "model_type": self.model_type}
            
        if self.model_type == "transformer" and self.hf_model is not None:
            inputs = self.tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
                pred_id = int(np.argmax(probs))
                conf = float(probs[pred_id])
                
            id2label = self.hf_model.config.id2label
            sentiment = id2label.get(pred_id, self.label_map.get(pred_id, "Neutral"))
            # Label names come from the model config rather than a hardcoded
            # 0/1/2 ordering, so `sentiment` and `probabilities` cannot disagree
            # if a model is trained with a different label order.
            return {
                "sentiment": sentiment,
                "confidence": round(conf, 4),
                "probabilities": {
                    id2label.get(i, self.label_map.get(i, str(i))): round(float(p), 4)
                    for i, p in enumerate(probs)
                },
                "model_type": "transformer"
            }
            
        elif self.vectorizer is not None and self.model is not None:
            try:
                features = self.vectorizer.transform([cleaned])
                probs = self.model.predict_proba(features)[0]
                classes = self.model.classes_
                
                best_idx = np.argmax(probs)
                sentiment = classes[best_idx]
                conf = float(probs[best_idx])
                
                prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
                
                return {
                    "sentiment": sentiment,
                    "confidence": round(conf, 4),
                    "probabilities": prob_dict,
                    "model_type": self.model_type
                }
            except Exception as e:
                # Rule-based pattern matching fallback on error
                lower = cleaned.lower()
                pos_count = sum(1 for w in ["great", "good", "excellent", "love", "amazing", "top", "perfect"] if w in lower)
                neg_count = sum(1 for w in ["bad", "horrible", "terrible", "late", "broken", "defective", "refund"] if w in lower)
                if neg_count > pos_count:
                    sent = "Negative"
                elif pos_count > neg_count:
                    sent = "Positive"
                else:
                    sent = "Neutral"
                return {"sentiment": sent, "confidence": 0.80, "model_type": "heuristic_fallback"}
            
        return {"sentiment": "Neutral", "confidence": 0.50, "model_type": "none"}
