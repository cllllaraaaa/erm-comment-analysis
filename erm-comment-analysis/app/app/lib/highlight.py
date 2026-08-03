"""
In-card term highlighting.

Marks three kinds of terms inside a comment's text, so a reviewer can see at a
glance WHY the comment carries its labels:

  topic keywords     (from the active topic pack, or the built-in patterns)
  stance phrases     (oppose wording / support wording)
  cited documents    (statutes, agencies, CFR sections, EIS documents)

Pure string work: find spans with the same regexes the pipeline itself uses,
resolve overlaps by priority (documents > stance > topics), HTML-escape all
non-highlighted text, and wrap matches in coloured <span>s. No network, no
model involvement; what is highlighted is exactly what the code matched.
"""
from __future__ import annotations
import html
import re

from lib import citations, regex_labels

STYLES = {
    "doc":     "background:#F5E9C9;border-radius:3px;padding:0 2px",
    "oppose":  "background:#F2DCCD;border-radius:3px;padding:0 2px",
    "support": "background:#D9E8DE;border-radius:3px;padding:0 2px",
    "topic":   "background:#E2E8F1;border-radius:3px;padding:0 2px",
}
_PRIORITY = {"doc": 0, "oppose": 1, "support": 1, "topic": 2}

LEGEND_HTML = (
    "<span style='font-size:12px;color:#5b6b63'>"
    f"<span style='{STYLES['topic']}'>topic keyword</span> "
    f"<span style='{STYLES['oppose']}'>opposing wording</span> "
    f"<span style='{STYLES['support']}'>supporting wording</span> "
    f"<span style='{STYLES['doc']}'>cited document</span>"
    "</span>"
)


class Highlighter:
    """Build once per page render, call .render(text) per card."""

    def __init__(self, keyword_map: dict | None = None, extra_whitelist=None):
        # cited documents: same pattern families as citations.extract
        self._doc_pats = [citations._ACT, citations._SEC, citations._CFR,
                          citations._EO, citations._EIS]
        wl = citations.WHITELIST_ACRO | set(extra_whitelist or ())
        if wl:
            alts = "|".join(sorted(map(re.escape, wl), key=len, reverse=True))
            self._doc_pats.append(re.compile(r"\b(?:" + alts + r")\b"))
        # topic keywords: active pack if it has any, else the built-in patterns
        pats = []
        if keyword_map:
            for kws in keyword_map.values():
                for kw in kws:
                    kw = str(kw).strip()
                    if len(kw) >= 3:
                        pats.append(r"\b" + re.escape(kw))
        else:
            for plist in regex_labels.TOPIC_PATTERNS.values():
                pats.extend(plist)
        self._topic_pat = re.compile("|".join(pats), re.I) if pats else None

    def _spans(self, text: str):
        found = []
        for pat in self._doc_pats:
            for m in pat.finditer(text):
                found.append((m.start(), m.end(), "doc"))
        for m in regex_labels._OPP.finditer(text):
            found.append((m.start(), m.end(), "oppose"))
        for m in regex_labels._SUP.finditer(text):
            found.append((m.start(), m.end(), "support"))
        if self._topic_pat is not None:
            for m in self._topic_pat.finditer(text):
                found.append((m.start(), m.end(), "topic"))
        # resolve overlaps: higher priority first, then longer match first
        found.sort(key=lambda s: (_PRIORITY[s[2]], s[0], -(s[1] - s[0])))
        taken: list = []
        for s, e, c in found:
            if all(e <= ts or s >= te for ts, te, _ in taken):
                taken.append((s, e, c))
        taken.sort(key=lambda s: s[0])
        return taken

    def render(self, text: str) -> str:
        """HTML for the text with highlighted terms; safe to pass to
        st.markdown(..., unsafe_allow_html=True)."""
        text = str(text or "")
        out, last = [], 0
        for s, e, cat in self._spans(text):
            out.append(html.escape(text[last:s]))
            out.append(f"<span style='{STYLES[cat]}'>{html.escape(text[s:e])}</span>")
            last = e
        out.append(html.escape(text[last:]))
        return "".join(out)
