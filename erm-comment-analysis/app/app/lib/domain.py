"""
Domain pack — the thing that makes the tool work on ANY docket, not just
maritime / oil & gas.

A pack bundles, for one subject area, everything that used to be hard-coded to
GulfLink:

  topics    [{code, label, keywords[], description}]   -> LLM schema + keyword
                                                           cross-check + display
  acronyms  {ACRO: {full, category}}                   -> cited-document
                                                           recognition + canon

It can be:
  * the built-in default (GulfLink oil/gas NEPA), assembled from the offline
    schema so nothing changes for the existing case study;
  * pasted straight from the EIS's own topic-area list (one line per area —
    lines without a colon work fine);
  * proposed by the LLM from a sample of the user's own comments (build_pack);
  * hand-edited, saved to JSON, and re-loaded for the next docket.

The pack is carried in st.session_state and its derived configs are passed
EXPLICITLY into citations / regex_labels / matcher (see citation_config and
keyword_patterns). There is deliberately no module-level registration:
Streamlit shares modules across user sessions in one process, so global state
would leak one consultant's docket into another's.
"""
from __future__ import annotations
import json
import re
import time
import requests

from lib import labeling, citations
from lib.config import GEMINI_MODEL, endpoint


# --------------------------------------------------------------------- default
_DEFAULT_KEYWORDS = {
    "oil_spill_risk": ["oil spill", "spill", "deepwater horizon", "blowout", "leak", "catastrophic"],
    "climate_emissions": ["climate change", "greenhouse gas", "emissions", "co2", "fossil fuel", "carbon"],
    "air_quality": ["air quality", "air pollution", "ozone", "smog", "nox"],
    "environmental_justice": ["environmental justice", "low-income", "minority communities", "frontline"],
    "marine_wildlife": ["sea turtle", "marine mammal", "dolphin", "whale", "bird", "habitat", "wildlife"],
    "fisheries": ["fishery", "fisheries", "shrimp", "commercial fishing", "fishermen", "fishing"],
    "wetlands_coast": ["wetlands", "coastline", "erosion", "storm surge", "sea level rise", "flooding"],
    "national_interest": ["national interest", "not in the national interest", "public interest"],
}


def default_pack() -> dict:
    """The built-in GulfLink pack — derived from the existing offline schema so
    the case study behaves exactly as before (backward compatible). Presented
    in the UI as an EXAMPLE TEMPLATE, not an assumption about your project."""
    topics = []
    for code in labeling.TOPIC_CODES:
        topics.append({
            "code": code,
            "label": _title(code),
            "keywords": _DEFAULT_KEYWORDS.get(code, []),
            "description": labeling.TOPIC_DESC.get(code, code),
        })
    acro = {}
    for a in sorted(citations.WHITELIST_ACRO):
        disp = citations.CANON.get(a, a)
        full = disp.split("(", 1)[1].rstrip(")") if "(" in disp else a
        acro[a] = {"full": full, "category": citations.category(disp)}
    return {"name": "Oil & gas / maritime (example)", "topics": topics, "acronyms": acro}


def _title(code: str) -> str:
    return code.replace("_", " ").capitalize()


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return s or "topic"


# ------------------------------------------------------------------ accessors
def topic_codes(pack: dict) -> list[str]:
    return [t["code"] for t in pack.get("topics", [])]


def topic_desc(pack: dict) -> dict:
    return {t["code"]: t.get("description", t["code"]) for t in pack.get("topics", [])}


def topic_labels(pack: dict) -> dict:
    return {t["code"]: t.get("label") or _title(t["code"]) for t in pack.get("topics", [])}


def keyword_map(pack: dict) -> dict:
    """{code: [keyword phrases]} for the matcher layer (spaCy / regex)."""
    out = {}
    for t in pack.get("topics", []):
        kws = [str(k).strip() for k in t.get("keywords", []) if len(str(k).strip()) >= 3]
        if kws:
            out[t["code"]] = kws
    return out


def keyword_patterns(pack: dict) -> dict:
    """{code: [regex strings]} built from each topic's keywords, for the
    regex-based cross-check path (lib/regex_labels.topics_set)."""
    return {code: [r"\b" + re.escape(kw) for kw in kws]
            for code, kws in keyword_map(pack).items()}


def has_keyword_coverage(pack: dict, active_codes) -> bool:
    km = keyword_map(pack)
    return any(c in km for c in active_codes)


def citation_config(pack: dict | None) -> citations.DomainConfig:
    """DomainConfig for the citations module (passed per call, no globals)."""
    if not pack:
        return citations.EMPTY
    wl, canon, cat = set(), {}, {}
    for a, meta in pack.get("acronyms", {}).items():
        a = str(a).strip()
        if not a:
            continue
        full = str(meta.get("full", "")).strip()
        disp = f"{a} ({full})" if full else a
        wl.add(a)
        canon[a] = disp
        if full:
            canon[full] = disp
        cat[disp.split(" (")[0]] = meta.get("category", "Other")
    return citations.DomainConfig(wl, canon, cat)


# ----------------------------------------------------------------- schema text
def to_schema_text(pack: dict) -> str:
    """`code: description` lines for the topic editor (keeps the existing UI)."""
    return "\n".join(f"{t['code']}: {t.get('description', t['code'])}"
                     for t in pack.get("topics", []))


def pack_from_schema_text(text: str, base: dict | None = None) -> dict:
    """Rebuild a pack after the user edits the schema box. Accepts both
    `code: description` lines AND bare lines pasted from an EIS topic-area list
    (e.g. just `Air Quality`). Keeps keywords/acronyms from `base` for topics
    whose code is unchanged."""
    base = base or {}
    base_topics = {t["code"]: t for t in base.get("topics", [])}
    topics = []
    for line in str(text).splitlines():
        line = line.strip().lstrip("-•*0123456789. ").strip()
        if not line or line.startswith("#"):
            continue
        name, d = (line.split(":", 1) + [""])[:2]
        code = _slug(name)
        if any(t["code"] == code for t in topics):
            continue
        prev = base_topics.get(code, {})
        topics.append({
            "code": code,
            # a bare EIS line like "Air Quality" keeps its own capitalisation
            "label": prev.get("label") or (name.strip() if ":" not in line else _title(code)),
            "keywords": prev.get("keywords", []),
            "description": d.strip() or name.strip(),
        })
    return {"name": base.get("name", "Custom pack"), "topics": topics,
            "acronyms": base.get("acronyms", {})}


# --------------------------------------------------------------------- build
def _sample(texts, n=45):
    """Spread the sample across the data instead of taking the first N rows —
    Regulations.gov files are time-ordered, so the head is often one campaign."""
    xs = [str(t) for t in texts if t is not None]
    xs = [t for t in xs if len(t.strip()) > 40]
    if len(xs) <= n:
        return xs
    step = len(xs) / n
    return [xs[int(i * step)] for i in range(n)]


def build_pack(texts, key: str, model: str = GEMINI_MODEL) -> dict:
    """Ask the LLM to propose a whole domain pack from a sample of comments:
    topics (with keywords) AND the statutes / agencies people cite. This is what
    lets citations + cross-check generalise, not just the topic list.

    Non-retryable API errors (bad key, bad request...) are raised immediately
    with the real status + message; only rate-limit / transient errors retry."""
    sample = "\n---\n".join(t[:500] for t in _sample(texts))
    prompt = (
        "You are configuring a tool that classifies public comments on a proposed "
        "government action. From the sample comments below, infer the subject area "
        "and return a JSON 'domain pack'.\n\n"
        "Return ONLY raw JSON, no prose, no code fences, in exactly this shape:\n"
        '{\n'
        '  "name": "<short subject-area name>",\n'
        '  "topics": [\n'
        '    {"code": "lower_snake_case", "label": "Human label",\n'
        '     "keywords": ["distinctive word or phrase", "..."],\n'
        '     "description": "one line"}\n'
        '  ],\n'
        '  "acronyms": {\n'
        '    "ACRONYM": {"full": "Full name", "category": '
        '"Statute|Agency|Regulation|EIS document|Other"}\n'
        '  }\n'
        '}\n\n'
        "Rules: 6-10 non-overlapping topics that capture what people actually "
        "discuss; 3-8 concrete keywords per topic (lowercase, the terms a keyword "
        "search would use); include any laws, agencies or official documents the "
        "commenters cite in 'acronyms' (leave {} if none are apparent).\n\n"
        "SAMPLE COMMENTS:\n" + sample
    )
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}
    last_err = "unknown error"
    for k in range(3):
        try:
            r = requests.post(endpoint(model), params={"key": key},
                              headers={"Content-Type": "application/json"},
                              data=json.dumps(body), timeout=90)
        except Exception as e:            # network problem — worth retrying
            last_err = f"request failed: {e}"
            time.sleep(min(2 ** k, 8))
            continue
        if r.status_code in (429, 503):   # transient — retry with backoff
            last_err = f"{r.status_code}: model busy / rate-limited"
            time.sleep(min(2 ** k, 8))
            continue
        if r.status_code != 200:          # real error — surface it immediately
            raise RuntimeError(f"API error {r.status_code}: {r.text[:200]}")
        try:
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            raw = raw.strip().replace("```json", "").replace("```", "").strip()
            return normalise(json.loads(raw))
        except Exception as e:            # malformed reply — retry once or twice
            last_err = f"could not parse the model reply: {e}"
            time.sleep(min(2 ** k, 8))
    raise RuntimeError(f"Could not build a domain pack — last error: {last_err}")


def normalise(pack: dict) -> dict:
    """Coerce a raw / user-supplied pack into the shape the app expects."""
    topics = []
    for t in pack.get("topics", []):
        if not isinstance(t, dict):
            continue
        code = _slug(t.get("code") or t.get("label") or "")
        if not code or any(x["code"] == code for x in topics):
            continue
        kws = t.get("keywords") or []
        if isinstance(kws, str):
            kws = [k.strip() for k in kws.split(",")]
        topics.append({
            "code": code,
            "label": t.get("label") or _title(code),
            "keywords": [str(k).strip() for k in kws if str(k).strip()],
            "description": t.get("description") or t.get("label") or code,
        })
    acro = {}
    for a, meta in (pack.get("acronyms") or {}).items():
        a = str(a).strip().upper()
        if not a:
            continue
        if isinstance(meta, str):
            meta = {"full": meta, "category": "Other"}
        acro[a] = {"full": str(meta.get("full", "")).strip(),
                   "category": meta.get("category", "Other") or "Other"}
    return {"name": pack.get("name", "Custom pack"), "topics": topics, "acronyms": acro}


# ----------------------------------------------------------------- (de)serialise
def to_json(pack: dict) -> str:
    return json.dumps(pack, indent=2, ensure_ascii=False)


def from_json(text) -> dict:
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", "ignore")
    return normalise(json.loads(text))
