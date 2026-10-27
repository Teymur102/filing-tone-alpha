"""Loughran-McDonald (2011) dictionary tone, using the official Master Dictionary from SRAF (Notre Dame)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import config as C
from .text import words

# Official CSV, linked from https://sraf.nd.edu/loughranmcdonald-master-dictionary/
LM_URL = "https://drive.usercontent.google.com/download?id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF&export=download&confirm=t"
LM_PATH = C.DATA / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
NEGATORS = {"NO", "NOT", "NONE", "NEITHER", "NEVER", "NOBODY"}


def download() -> None:
    if LM_PATH.exists():
        return
    import requests
    r = requests.get(LM_URL, timeout=120, headers={"User-Agent": C.USER_AGENT})
    r.raise_for_status()
    if not r.content.startswith(b"Word,"):
        raise RuntimeError("LM download did not return the expected CSV")
    LM_PATH.parent.mkdir(parents=True, exist_ok=True)
    LM_PATH.write_bytes(r.content)


@lru_cache(maxsize=1)
def word_lists() -> tuple[frozenset, frozenset]:
    download()
    df = pd.read_csv(LM_PATH, usecols=["Word", "Negative", "Positive"], keep_default_na=False)
    # a positive entry is the year the word was added; negative means it was later removed
    pos = frozenset(df.loc[df.Positive > 0, "Word"].str.upper())
    neg = frozenset(df.loc[df.Negative > 0, "Word"].str.upper())
    return pos, neg


def count(tokens: list[str], pos: frozenset, neg: frozenset) -> tuple[int, int]:
    """Count positive and negative words. A positive word within three words after a
    negator ('not', 'no', ...) is not counted as positive (LM 2011 convention)."""
    n_pos = n_neg = 0
    for i, w in enumerate(tokens):
        if w in neg:
            n_neg += 1
        elif w in pos and not any(t in NEGATORS for t in tokens[max(0, i - 3):i]):
            n_pos += 1
    return n_pos, n_neg


def doc_tone(sentences: list[str], pos: frozenset | None = None, neg: frozenset | None = None) -> dict:
    if pos is None:
        pos, neg = word_lists()
    n_pos = n_neg = n_words = 0
    for s in sentences:
        toks = words(s)
        p, n = count(toks, pos, neg)
        n_pos, n_neg, n_words = n_pos + p, n_neg + n, n_words + len(toks)
    return {
        "lm_pos": n_pos, "lm_neg": n_neg, "n_words": n_words,
        # net tone in [-1, 1]; 0 when the document has no sentiment words
        "lm_tone": (n_pos - n_neg) / (n_pos + n_neg) if n_pos + n_neg else 0.0,
        "lm_neg_frac": n_neg / n_words if n_words else 0.0,
    }
