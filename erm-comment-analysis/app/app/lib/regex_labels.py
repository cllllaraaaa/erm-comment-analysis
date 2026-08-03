"""
Regex baseline labels (topics + stance), ported from the offline notebooks.

Used as a cheap, transparent CROSS-CHECK against the LLM, not as a competing
labeller. Where the keyword layer catches an explicit term the LLM didn't tag,
or the two disagree on stance, the comment is surfaced for a human — a guard
against relying on a single method (the bias concern).

Custom-schema keyword patterns come from the active domain pack and are passed
in EXPLICITLY (patterns argument) — never stored as module state, because
Streamlit shares this module across all user sessions in one process.
"""
from __future__ import annotations
import re

TOPIC_PATTERNS = {
    "oil_spill_risk": [r"\boil spill", r"\bdeepwater horizon", r"\bspill risk", r"\bcatastrophic"],
    "climate_emissions": [r"\bclimate change", r"\bgreenhouse gas", r"\bemissions?", r"\bfossil fuel"],
    "air_quality": [r"\bair quality", r"\bair pollution", r"\bozone", r"\bnox\b"],
    "environmental_justice": [r"\benvironmental justice", r"\benvironmental racism",
                              r"\blow-income", r"\bminority communities"],
    "marine_wildlife": [r"\bsea turtle", r"\bmarine mammals?", r"\bdolphin", r"\bwhale",
                        r"\bblack rail", r"\bhabitats?"],
    "fisheries": [r"\bfisher(?:y|ies)", r"\bshrimp", r"\bcommercial fishing", r"\bfishermen"],
    "wetlands_coast": [r"\bwetlands?", r"\bcoastline", r"\berosion", r"\bstorm surge",
                       r"\bsea level rise", r"\bflood"],
    "national_interest": [r"\bnational interest", r"\bnot in the national interest"],
}

# 'concerned' removed — far too broad, it flipped ordinary worried-but-supportive
# comments to oppose and inflated the disagreement flag.
# 'speaking against' / 'against the proposed site' added after real misses on
# MARAD-2019-0093-0021/-0022 (opposing comments containing the literal word
# 'support', e.g. 'please support us and vote NO').
_OPP = re.compile(
    r"\b(?:oppose|opposed|opposition|deny|reject"
    r"|speak(?:s|ing)? (?:out )?against"
    r"|against (?:the|this) (?:proposed )?"
    r"(?:project|proposal|site|location|plan|permit|facility|rule|action)"
    r"|not in the national interest)\b", re.I)
_SUP = re.compile(r"\b(?:support|approve|in favor|in favour|favou?r (?:the|this))\b", re.I)


def topics_set(text: str, patterns: dict | None = None) -> set:
    """Topic codes whose patterns fire in `text`. `patterns` defaults to the
    built-in GulfLink set; a domain pack passes its own (see lib/domain)."""
    pats = patterns if patterns is not None else TOPIC_PATTERNS
    t = str(text or "")
    out = set()
    for code, plist in pats.items():
        if any(re.search(p, t, re.I) for p in plist):
            out.add(code)
    return out


def stance(text: str) -> str:
    t = str(text or "")
    if _OPP.search(t):
        return "oppose"
    if _SUP.search(t):
        return "support"
    return "unclear"
