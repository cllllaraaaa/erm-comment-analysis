"""
Cited-document extraction (regex-first) + canonicalisation.

Ported from the offline notebook so the APP can compute cited official
documents itself on any uploaded file — no pre-computed column required.
Same four pattern families + acronym whitelist, then a canonical map that
merges acronym / full-name / typo variants into one labelled document with a
category (statute / agency / EIS document / regulation / ...).

Domain packs (lib/domain) extend recognition to other subject areas by passing
an explicit `extra` config into extract()/category(). This is a plain argument,
NOT module state: Streamlit runs every user session in one Python process, so
module-level registration would leak one consultant's docket into another's.
The built-in defaults below always stay active (NEPA, CFR, EO... are generic).
"""
from __future__ import annotations
import re
from collections import Counter

WHITELIST_ACRO = {
    "NEPA", "DWPA", "EIS", "DEIS", "SDEIS", "FEIS", "ESA", "CWA", "CAA", "CZMA",
    "NHPA", "ROD", "MARAD", "USACE", "USCG", "EPA", "NOAA", "USFWS", "MMPA",
    "APA", "OCSLA", "CFR", "MBTA",
}

_ACT = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\s+Act)\b")
_ACRO = re.compile(r"\b([A-Z]{2,6})\b")
_SEC = re.compile(r"\b(Section\s+\d+[A-Za-z]?)\b")
_CFR = re.compile(r"\b(\d+\s+CFR\s+\d+)\b")
_EO = re.compile(r"\b(Executive Order\s+\d+)\b")
_EIS = re.compile(r"\b((?:Draft|Final|Supplemental)\s+Environmental Impact Statement)\b")

CANON = {
    "NEPA": "NEPA (National Environmental Policy Act)",
    "National Environmental Policy Act": "NEPA (National Environmental Policy Act)",
    "DWPA": "DWPA (Deepwater Port Act)",
    "Deepwater Port Act": "DWPA (Deepwater Port Act)",
    "The Deepwater Port Act": "DWPA (Deepwater Port Act)",
    "Deepwater Ports Act": "DWPA (Deepwater Port Act)",
    "Deepwater Water Port Act": "DWPA (Deepwater Port Act)",
    "FEIS": "FEIS (Final EIS)",
    "Final Environmental Impact Statement": "FEIS (Final EIS)",
    "DEIS": "DEIS (Draft EIS)",
    "Draft Environmental Impact Statement": "DEIS (Draft EIS)",
    "SDEIS": "SDEIS (Supplemental EIS)",
    "Supplemental Environmental Impact Statement": "SDEIS (Supplemental EIS)",
    "EIS": "EIS (Environmental Impact Statement)",
    "ESA": "ESA (Endangered Species Act)",
    "Endangered Species Act": "ESA (Endangered Species Act)",
    "CWA": "CWA (Clean Water Act)",
    "Clean Water Act": "CWA (Clean Water Act)",
    "CAA": "CAA (Clean Air Act)",
    "Clean Air Act": "CAA (Clean Air Act)",
    "CZMA": "CZMA (Coastal Zone Management Act)",
    "Coastal Zone Management Act": "CZMA (Coastal Zone Management Act)",
    "NHPA": "NHPA (National Historic Preservation Act)",
    "National Historic Preservation Act": "NHPA (National Historic Preservation Act)",
    "MMPA": "MMPA (Marine Mammal Protection Act)",
    "Marine Mammal Protection Act": "MMPA (Marine Mammal Protection Act)",
    "MBTA": "MBTA (Migratory Bird Treaty Act)",
    "Migratory Bird Treaty Act": "MBTA (Migratory Bird Treaty Act)",
    "OCSLA": "OCSLA (Outer Continental Shelf Lands Act)",
    "Outer Continental Shelf Lands Act": "OCSLA (Outer Continental Shelf Lands Act)",
    "APA": "APA (Administrative Procedure Act)",
    "Administrative Procedure Act": "APA (Administrative Procedure Act)",
    "MARAD": "MARAD (Maritime Administration)",
    "EPA": "EPA (Environmental Protection Agency)",
    "NOAA": "NOAA",
    "USACE": "USACE (Army Corps of Engineers)",
    "USCG": "USCG (Coast Guard)",
    "USFWS": "USFWS (Fish & Wildlife Service)",
    "ROD": "ROD (Record of Decision)",
    "CFR": "CFR (Code of Federal Regulations)",
}

_STATUTES = {"NEPA", "DWPA", "ESA", "CWA", "CAA", "CZMA", "NHPA", "MMPA", "MBTA", "OCSLA", "APA"}
_AGENCIES = {"MARAD", "EPA", "NOAA", "USACE", "USCG", "USFWS"}


# --------------------------------------------------------------- domain config
class DomainConfig:
    """Extra whitelist / canon / category from the active domain pack.
    Passed explicitly per call — no shared module state between sessions."""

    __slots__ = ("whitelist", "canon", "category")

    def __init__(self, whitelist=None, canon=None, category=None):
        self.whitelist = set(whitelist or ())
        self.canon = dict(canon or {})
        self.category = dict(category or {})


EMPTY = DomainConfig()


def canonical(x: str, extra: DomainConfig = EMPTY) -> str:
    return extra.canon.get(x) or CANON.get(x, x)


def category(canon_name: str, extra: DomainConfig = EMPTY) -> str:
    head = canon_name.split(" (")[0]
    if head in extra.category:
        return extra.category[head]
    if head in _STATUTES:
        return "Statute"
    if "EIS" in canon_name:
        return "EIS document"
    if head in _AGENCIES:
        return "Agency"
    if canon_name.startswith("Section"):
        return "Statute section"
    if canon_name.startswith("Executive Order"):
        return "Executive order"
    if "CFR" in canon_name:
        return "Regulation"
    if canon_name.startswith("ROD"):
        return "Decision document"
    return "Other"


def extract(text: str, extra: DomainConfig = EMPTY) -> list[str]:
    """Return the sorted set of canonical documents cited in a piece of text."""
    if not isinstance(text, str) or not text:
        return []
    wl = WHITELIST_ACRO | extra.whitelist
    found = set()
    for m in _ACT.findall(text):
        found.add(m.strip())
    for m in _ACRO.findall(text):
        if m in wl:
            found.add(m)
    for m in _SEC.findall(text):
        found.add(m)
    for m in _CFR.findall(text):
        found.add(m)
    for m in _EO.findall(text):
        found.add(m)
    for m in _EIS.findall(text):
        found.add(m)
    return sorted({canonical(d, extra) for d in found})


def frequency(list_series, weight_series=None):
    """Counter of canonical docs. If weight_series given (e.g. dedup mask) it is
    ignored here; callers pass the already-filtered series they want counted."""
    c = Counter()
    for lst in list_series:
        for d in lst:
            c[d] += 1
    return c
