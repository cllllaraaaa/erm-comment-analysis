"""
Objective stance-intensity signal — HOW STRONGLY a comment is expressed,
computed from the text alone (no API): ALL-CAPS words, exclamation marks,
intensity vocabulary, and length. Used on its own, and to cross-validate the
LLM's intensity rating (the two should correlate). Exploratory — no gold set.
"""
from __future__ import annotations
import re

INTENSE_WORDS = {
    "absolutely", "never", "must", "demand", "urge", "urgently", "outraged",
    "horrified", "strongly", "unacceptable", "catastrophic", "devastating",
    "refuse", "reject", "vehemently", "appalled", "disgusted", "urgent",
    "insist", "imperative", "dangerous", "destroy", "destruction", "irreversible",
    "crisis", "unequivocally", "deeply", "gravely", "beg", "plead", "shameful",
}
_WORD = re.compile(r"[A-Za-z']+")


def signal(text: str) -> dict:
    t = str(text or "")
    words = _WORD.findall(t)
    n_words = max(len(words), 1)
    n_caps = sum(1 for w in words if len(w) >= 3 and w.isupper())
    n_excl = t.count("!")
    low = t.lower()
    n_int = sum(low.count(w) for w in INTENSE_WORDS)
    s_caps = min(n_caps / n_words * 4, 1.0)
    s_excl = min(n_excl / 3, 1.0)
    s_int = min(n_int / 5, 1.0)
    s_len = min(len(t) / 1500, 1.0)
    score = min(0.30 * s_caps + 0.25 * s_excl + 0.35 * s_int + 0.10 * s_len, 1.0)
    return {"score": round(score, 3), "caps": n_caps, "excl": n_excl, "intense": n_int}


def score(text: str) -> float:
    return signal(text)["score"]


def label(s: float) -> str:
    try:
        s = float(s)
    except (TypeError, ValueError):
        return "mild"
    if s >= 0.45:
        return "strong"
    if s >= 0.20:
        return "moderate"
    return "mild"
