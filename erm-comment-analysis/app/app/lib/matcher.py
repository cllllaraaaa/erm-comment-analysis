"""
Keyword matching layer — finds which topics' keywords literally appear in a text.

Uses spaCy's PhraseMatcher when spaCy is installed (token-based matching, so
'port' won't fire inside 'important'); otherwise falls back to word-boundary
regex, which behaves almost identically. No spaCy model download is needed —
a blank English pipeline (tokeniser only) is enough.

    pip install spacy      # optional

The matcher is built per keyword-map and cached, so page reruns don't pay the
construction cost again.
"""
from __future__ import annotations
import re

try:  # optional dependency
    import spacy
    from spacy.matcher import PhraseMatcher
    _NLP = spacy.blank("en")
    _NLP.max_length = 2_000_000
    HAVE_SPACY = True
except Exception:  # pragma: no cover - environment without spacy
    _NLP = None
    HAVE_SPACY = False


class KeywordMatcher:
    """topics_in(text) -> set of topic codes whose keywords appear in the text."""

    def __init__(self, keyword_map: dict[str, list[str]]):
        # keyword_map: {code: [keyword/phrase, ...]}
        self.keyword_map = {
            code: [str(k).strip() for k in kws if str(k).strip()]
            for code, kws in (keyword_map or {}).items()
        }
        self.keyword_map = {c: kws for c, kws in self.keyword_map.items() if kws}
        self._pm = None
        self._rx = None
        if HAVE_SPACY:
            self._pm = PhraseMatcher(_NLP.vocab, attr="LOWER")
            for code, kws in self.keyword_map.items():
                self._pm.add(code, [_NLP.make_doc(k) for k in kws])
        else:
            self._rx = {
                code: re.compile(
                    "|".join(r"\b" + re.escape(k) for k in kws), re.IGNORECASE)
                for code, kws in self.keyword_map.items()
            }

    def topics_in(self, text) -> set:
        t = str(text or "")
        if not t or not self.keyword_map:
            return set()
        if self._pm is not None:
            doc = _NLP.make_doc(t)
            return {_NLP.vocab.strings[mid] for mid, _, _ in self._pm(doc)}
        return {code for code, rx in self._rx.items() if rx.search(t)}


_CACHE: dict[str, KeywordMatcher] = {}


def get(keyword_map: dict[str, list[str]]) -> KeywordMatcher:
    """Cached constructor — the same keyword map returns the same matcher."""
    key = repr(sorted((c, tuple(k)) for c, k in (keyword_map or {}).items()))
    m = _CACHE.get(key)
    if m is None:
        m = _CACHE[key] = KeywordMatcher(keyword_map or {})
        if len(_CACHE) > 8:  # keep the cache tiny
            _CACHE.pop(next(iter(_CACHE)))
    return m


def engine() -> str:
    return "spaCy PhraseMatcher" if HAVE_SPACY else "regex (word-boundary)"
