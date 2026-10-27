import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tone import analysis as A  # noqa: E402
from tone import lm, text  # noqa: E402
from tone.finbert import aggregate  # noqa: E402

LABELS = {"positive": 0, "negative": 1, "neutral": 2}
POS = frozenset({"STRONG", "GROWTH", "RECORD"})
NEG = frozenset({"DECLINE", "LOSS", "IMPAIRMENT"})


def test_finbert_aggregate_mean_pos_minus_neg():
    probs = np.array([[0.8, 0.1, 0.1],    # positive
                      [0.1, 0.7, 0.2],    # negative
                      [0.2, 0.1, 0.7]])   # neutral
    out = aggregate(probs, LABELS)
    assert out["finbert_tone"] == pytest.approx(np.mean([0.7, -0.6, 0.1]))
    assert out["finbert_pos"] == pytest.approx(1 / 3)
    assert out["finbert_neg"] == pytest.approx(1 / 3)
    assert out["n_sents"] == 3


def test_finbert_aggregate_respects_label_order():
    probs = np.array([[0.1, 0.9, 0.0]])
    assert aggregate(probs, {"negative": 0, "positive": 1, "neutral": 2})["finbert_tone"] == pytest.approx(0.8)


def test_finbert_aggregate_empty_doc():
    out = aggregate(np.zeros((0, 3)), LABELS)
    assert np.isnan(out["finbert_tone"]) and out["n_sents"] == 0


def test_lm_counts_and_net_tone():
    sents = ["Record revenue and strong growth this quarter.", "We booked an impairment and a loss."]
    out = lm.doc_tone(sents, POS, NEG)
    assert (out["lm_pos"], out["lm_neg"]) == (3, 2)
    assert out["lm_tone"] == pytest.approx(1 / 5)
    assert out["lm_neg_frac"] == pytest.approx(2 / out["n_words"])


def test_lm_negation_cancels_positive_words():
    assert lm.count(text.words("Results were not strong this quarter"), POS, NEG) == (0, 0)
    # negation more than three words back does not apply
    assert lm.count(text.words("Not that we expected it, but growth"), POS, NEG) == (1, 0)
    # negation never flips negative words
    assert lm.count(text.words("no decline"), POS, NEG) == (0, 1)


def test_lm_tone_zero_without_sentiment_words():
    assert lm.doc_tone(["The meeting will be held in Boston on Tuesday."], POS, NEG)["lm_tone"] == 0.0


def test_text_drops_numeric_tables_and_boilerplate():
    html = """<html><body><p>Acme Corp reported record revenue of $5 billion, up 10 percent from last year.</p>
    <table><tr><td>Revenue</td><td>5,000</td><td>4,500</td><td>3,900</td></tr></table>
    <p>This release contains forward-looking statements within the meaning of the safe harbor provisions.</p>
    <table><tr><td>Our customers continued to adopt the new platform at a faster pace than expected.</td></tr></table>
    </body></html>"""
    sents = text.narrative_sentences(html)
    assert sents == ["Acme Corp reported record revenue of $5 billion, up 10 percent from last year.",
                     "Our customers continued to adopt the new platform at a faster pace than expected."]


def test_sentence_split_keeps_abbreviations_together():
    s = text.split_sentences("Sales at Acme Inc. grew in the U.S. market. Margins fell. Outlook is weak.")
    assert s == ["Sales at Acme Inc. grew in the U.S. market.", "Margins fell.", "Outlook is weak."]


def test_tone_change_uses_only_prior_releases():
    df = pd.DataFrame({"ticker": ["A"] * 5, "day0": pd.date_range("2020-01-01", periods=5, freq="90D"),
                       "lm_tone": [0.0, 0.2, 0.4, 0.6, 1.0], "finbert_tone": [0.0] * 5})
    out = A.add_tone_changes(df)
    assert out["lm_tone_chg"].iloc[:2].isna().all()                 # needs two prior releases
    assert out["lm_tone_chg"].iloc[2] == pytest.approx(0.4 - 0.1)
    assert out["lm_tone_chg"].iloc[4] == pytest.approx(1.0 - 0.3)   # mean of the previous four
