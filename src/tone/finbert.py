"""FinBERT (ProsusAI/finbert) sentence-level tone, aggregated to one score per document.

Runs locally on Apple MPS when available, else CPU. Sentences are batched and sorted by
length so padding is minimal; peak memory stays well under 1 GB.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def aggregate(probs: np.ndarray, label_index: dict[str, int]) -> dict:
    """Collapse an (n_sentences, 3) probability matrix into document-level tone.

    finbert_tone   = mean over sentences of P(positive) - P(negative), in [-1, 1]
    finbert_pos/neg = share of sentences whose top label is positive / negative
    """
    if len(probs) == 0:
        return {"finbert_tone": np.nan, "finbert_pos": np.nan, "finbert_neg": np.nan, "n_sents": 0}
    p, n = label_index["positive"], label_index["negative"]
    top = probs.argmax(axis=1)
    return {
        "finbert_tone": float(np.mean(probs[:, p] - probs[:, n])),
        "finbert_pos": float(np.mean(top == p)),
        "finbert_neg": float(np.mean(top == n)),
        "n_sents": int(len(probs)),
    }


class FinBert:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.set_num_threads(4)
        self.tok = AutoTokenizer.from_pretrained(C.FINBERT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(C.FINBERT_MODEL).to(self.device).eval()
        self.label_index = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}

    def predict(self, sentences: list[str]) -> np.ndarray:
        """Return (n, 3) softmax probabilities in model label order."""
        order = np.argsort([len(s) for s in sentences])
        out = np.zeros((len(sentences), 3), dtype=np.float32)
        with self.torch.inference_mode():
            for i in range(0, len(order), C.FINBERT_BATCH):
                idx = order[i:i + C.FINBERT_BATCH]
                enc = self.tok([sentences[j] for j in idx], padding=True, truncation=True,
                               max_length=C.FINBERT_MAX_LEN, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                out[idx] = self.torch.softmax(logits, dim=-1).float().cpu().numpy()
        return out
