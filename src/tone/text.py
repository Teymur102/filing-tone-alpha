"""Turn an earnings-release exhibit (HTML) into clean narrative sentences.

Financial-statement tables are dropped: we want the words management chose, not the numbers.
"""
from __future__ import annotations

import html as htmllib
import re

from . import config as C

_DROP = re.compile(r"<(script|style|head)\b.*?</\1\s*>", re.S | re.I)
_TABLE = re.compile(r"<table\b.*?</table\s*>", re.S | re.I)
_BLOCK = re.compile(r"</?(p|div|br|tr|td|li|h[1-6])\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v\xa0]+")
# split after . ! ? when followed by whitespace + capital/quote/digit; avoid common abbreviations
_SPLIT = re.compile(r"(?<!\b[A-Z])(?<!\bInc)(?<!\bCo)(?<!\bCorp)(?<!\bLtd)(?<!\bNo)(?<!\bU\.S)(?<!\bvs)(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<=[.!?])[\"”']?\s+(?=[\"“A-Z0-9])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
# Legal boilerplate is identical quarter to quarter and says nothing about the quarter.
_BOILER = re.compile(r"forward-looking|safe harbor|private securities litigation|non-gaap|"
                     r"reconciliation|webcast|conference call|investor relations|www\.|http", re.I)


def _table_filter(m: re.Match) -> str:
    """Drop numeric tables (financial statements); keep layout tables that hold prose."""
    txt = htmllib.unescape(_TAG.sub(" ", m.group(0)))
    letters = sum(ch.isalpha() for ch in txt)
    digits = sum(ch.isdigit() for ch in txt)
    return m.group(0) if letters > 0 and digits / letters < 0.08 else " "


def html_to_text(raw: str) -> str:
    if "<" not in raw[:5000]:  # plain-text exhibit: paragraphs are separated by blank lines
        paras = re.split(r"\n\s*\n", raw)
        return "\n".join(_WS.sub(" ", p.replace("\n", " ")).strip() for p in paras if p.strip())
    s = _DROP.sub(" ", raw.replace("\r", " ").replace("\n", " "))
    s = _TABLE.sub(_table_filter, s)
    s = _BLOCK.sub("\n", s)
    s = _TAG.sub(" ", s)
    s = htmllib.unescape(s)
    s = _WS.sub(" ", s)
    lines = [ln.strip() for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def split_sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n"):
        out.extend(p.strip() for p in _SPLIT.split(para) if p.strip())
    return out


def is_narrative(sent: str) -> bool:
    """Keep prose sentences: enough words, mostly alphabetic tokens, not boilerplate."""
    toks = sent.split()
    if not (C.MIN_SENT_WORDS <= len(toks) <= C.MAX_SENT_WORDS):
        return False
    alpha = sum(1 for t in toks if _WORD.fullmatch(t.strip(".,;:()\"“”'’")))
    if alpha / len(toks) < 0.6:
        return False
    return not _BOILER.search(sent)


def narrative_sentences(raw_html: str) -> list[str]:
    return [s for s in split_sentences(html_to_text(raw_html)) if is_narrative(s)]


def words(text: str) -> list[str]:
    return [w.upper() for w in _WORD.findall(text)]
