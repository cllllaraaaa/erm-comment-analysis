"""
In-app analysis: label raw comments (stance + multi-label topic + evidence)
with Gemini, using the user's own API key. Used ONLY when the uploaded file
has no labels.

Design mirrors the offline pipeline: GROUP FIRST, LABEL ONCE, PROPAGATE.
1. Exact de-duplication collapses identical copies of a form letter.
2. Near-duplicate FAMILY grouping (MinHash, lib/dedupe) then collapses the
   lightly-personalised variants of the same template — the '~85% form
   letters' the consultants describe — so the LLM labels ONE representative
   per letter family (the most-signed variant), and the labels propagate to
   the whole family. Only the one-of-a-kind letters get individual calls.
3. The per-comment keyword/regex cross-check in lib/data still runs on each
   comment's own full text, so a personal addition that changes topic or
   stance is flagged for human review rather than silently inheriting the
   template's labels. `llm_propagated` marks every propagated row.
This cuts LLM work from 'one call per unique text' to 'one call per distinct
letter' (~6,500 -> ~1,600 on MARAD-2019-0093), on top of request batching.

Schema editing / suggestion moved to lib/domain.py (domain packs); the old
suggest_schema / parse_schema helpers were removed with it.
"""
from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import requests

from lib.config import GEMINI_MODEL, endpoint
from lib.dedupe import near_dup_groups

TOPIC_CODES = [
    "oil_spill_risk", "climate_emissions", "air_quality", "environmental_justice",
    "marine_wildlife", "fisheries", "wetlands_coast", "national_interest",
]
TOPIC_DESC = {
    "oil_spill_risk": "oil spills, leaks, blowouts, spill response, Deepwater Horizon, catastrophic accident",
    "climate_emissions": "climate change, greenhouse gas, CO2, fossil fuel expansion, emissions",
    "air_quality": "air pollution, ozone, NOx, local air health",
    "environmental_justice": "impact on low-income / minority / coastal communities, fairness, EJ",
    "marine_wildlife": "sea turtles, marine mammals, dolphins, whales, birds, habitat, ecosystems",
    "fisheries": "fishing industry, shrimp, commercial fishing, fishermen, fishery impacts",
    "wetlands_coast": "wetlands, coastline, erosion, storm surge, sea-level rise, flooding",
    "national_interest": "whether the project is / isn't in the national interest (general opinion)",
}


def _system(topic_codes, topic_desc) -> str:
    block = "\n".join(f"- {c}: {topic_desc.get(c, c)}" for c in topic_codes)
    return (
        "You label public comments submitted to a US federal agency during a public "
        "consultation (an environmental review, a proposed rule, or a policy action).\n"
        "Assign ALL applicable TOPIC codes (multi-label) from this fixed list ONLY:\n"
        f"{block}\n\n"
        "Also give STANCE = the commenter's overall position on the PROPOSED PROJECT / "
        "RULE / ACTION itself (infer the subject from the comments themselves):\n"
        "- oppose = against it, or wants it rejected, denied, weakened or stopped. "
        "A politely-worded comment that asks for the project to be denied, raises "
        "objections, or lists harms it would cause is OPPOSE — do not mark it "
        "unclear or support just because the tone is calm or courteous.\n"
        "- support = in favour of it, or wants it approved or strengthened. Praise "
        "for the environment or the review process alone is NOT support for the "
        "project.\n"
        "- unclear = genuinely neutral, evenly mixed, a question only, or no "
        "position at all. Use this sparingly — most commenters have a position.\n\n"
        "Also give INTENSITY = how strongly the position is expressed:\n"
        "- strong = forceful / emotional / urgent / demanding\n"
        "- moderate = clear but measured\n"
        "- mild = brief / tentative / lukewarm\n\n"
        "Also say whether the comment PROVIDES EVIDENCE for its position — any of: "
        "data or figures, a study or report, a law or official document, first-hand "
        "local/professional experience, or expert credentials. A bare opinion "
        "(\"I don't like this project\") is evidence=false.\n\n"
        'Return ONLY raw JSON, no prose, no code fences:\n'
        '{"topics": ["code", ...], "stance": "oppose|support|unclear", '
        '"confidence": 0.0, "intensity": "strong|moderate|mild", '
        '"intensity_score": 0.0, "provides_evidence": true, '
        '"evidence_type": "data|study|law_or_document|personal_experience|expertise|none"}\n'
        "Rules: topics must be from the list; if none apply -> []."
    )


def label_one(text: str, key: str, topic_codes, topic_desc,
              model: str = GEMINI_MODEL) -> dict:
    system = _system(topic_codes, topic_desc)
    body = {
        "contents": [{"parts": [{"text": system + "\n\nCOMMENT:\n" + str(text)[:8000]}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    valid = set(topic_codes)
    last = "unknown"
    for k in range(3):
        try:
            r = requests.post(endpoint(model), params={"key": key},
                              headers={"Content-Type": "application/json"},
                              data=json.dumps(body), timeout=60)
            if r.status_code != 200:
                if r.status_code in (429, 503):
                    time.sleep(min(2 ** k, 8)); continue
                return {"topics": [], "stance": "unclear", "confidence": 0.0,
                        "failed": True, "err": f"{r.status_code}: {r.text[:120]}"}
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            raw = raw.strip().replace("```json", "").replace("```", "").strip()
            o = json.loads(raw)
            o["topics"] = [t for t in o.get("topics", []) if t in valid]
            o["failed"] = False; o["err"] = ""
            return o
        except Exception as e:  # noqa
            last = str(e)
            time.sleep(min(2 ** k, 8))
    return {"topics": [], "stance": "unclear", "confidence": 0.0, "failed": True,
            "err": last}


_FAILED = {"topics": [], "stance": "unclear", "confidence": 0.0, "failed": True}

BATCH_SIZE = 8       # comments per request: ~8x fewer round-trips than one-by-one


def _parse_batch(raw: str, n: int, valid: set):
    """Parse a batch reply into {index: label}. Returns None if the reply is
    structurally unusable (caller falls back to one-by-one labelling)."""
    try:
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        o = json.loads(raw)
    except Exception:
        return None
    items = o.get("results") if isinstance(o, dict) else o
    if not isinstance(items, list):
        return None
    out: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            i = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        if not (0 <= i < n) or i in out:
            continue
        item["topics"] = [t for t in item.get("topics", []) if t in valid]
        item["failed"] = False
        item["err"] = ""
        out[i] = item
    return out or None


def label_batch(texts: list, key: str, topic_codes, topic_desc,
                model: str = GEMINI_MODEL):
    """Label up to BATCH_SIZE comments in ONE request. Returns {index: label}
    (possibly missing some indices) or None on failure."""
    system = _system(topic_codes, topic_desc)
    numbered = "\n\n".join(f"COMMENT {i}:\n{str(t)[:4000]}"
                           for i, t in enumerate(texts))
    prompt = (
        system
        + "\n\nYou will now label SEVERAL comments at once. For EACH numbered "
          "comment below, produce one result object with the fields above plus "
          '"i" = the comment number. Return ONLY raw JSON of the form '
          '{"results": [{"i": 0, "topics": [...], "stance": "...", '
          '"confidence": 0.0, "intensity": "...", "intensity_score": 0.0, '
          '"provides_evidence": true, "evidence_type": "..."}, ...]} with '
          "exactly one entry per comment.\n\n" + numbered)
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    for k in range(3):
        try:
            r = requests.post(endpoint(model), params={"key": key},
                              headers={"Content-Type": "application/json"},
                              data=json.dumps(body), timeout=120)
            if r.status_code != 200:
                if r.status_code in (429, 503):
                    time.sleep(min(2 ** k, 8)); continue
                return None
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return _parse_batch(raw, len(texts), set(topic_codes))
        except Exception:
            time.sleep(min(2 ** k, 8))
    return None


def _families(uniq: list, counts=None, threshold: float = 0.7) -> dict:
    """Map each unique text to its family REPRESENTATIVE.

    Families are near-duplicate groups (same template, small personal edits;
    see lib/dedupe). The representative is the family's most-signed variant
    (`counts` = exact-copy count per text, e.g. Series.value_counts()), which
    is normally the clean, unpersonalised template; ties break to the longest
    text. Texts too short/distinct to group map to themselves. Pure function,
    no network — unit-testable."""
    if len(uniq) < 2:
        return {t: t for t in uniq}
    gids = near_dup_groups(uniq, threshold=threshold)
    fam: dict = {}
    for t, g in zip(uniq, gids):
        fam.setdefault(g, []).append(t)

    def _weight(t):
        c = 0
        if counts is not None:
            try:
                c = int(counts.get(t, 0))
            except Exception:
                c = 0
        return (c, len(t))

    rep_of: dict = {}
    for members in fam.values():
        rep = max(members, key=_weight)
        for m in members:
            rep_of[m] = rep
    return rep_of


def _label_chunk(chunk: list, key: str, topic_codes, topic_desc,
                 model: str) -> list:
    """Label one chunk: batch request first, one-by-one fallback for anything
    the batch reply missed or garbled."""
    got = label_batch(chunk, key, topic_codes, topic_desc, model) or {}
    out = []
    for i, t in enumerate(chunk):
        if i in got:
            out.append(got[i])
        else:
            out.append(label_one(t, key, topic_codes, topic_desc, model))
    return out


def analyse(df: pd.DataFrame, text_col: str, key: str,
            topic_codes=None, topic_desc=None, progress=None,
            model: str = GEMINI_MODEL, workers: int = 8,
            family_grouping: bool = True) -> pd.DataFrame:
    """Label the frame: returns a copy with llm_* columns.

    Group first, label once, propagate: exact duplicates collapse, then
    near-duplicate letter families collapse to one representative each
    (family_grouping=True; see _families). Representatives are labelled in
    batches of BATCH_SIZE with `workers` batches in flight; labels propagate
    to every family member, and `llm_propagated` marks the rows that
    inherited their labels rather than being labelled directly.
    `progress(done, total)` is called from the main thread for a live bar."""
    topic_codes = topic_codes or TOPIC_CODES
    topic_desc = topic_desc or TOPIC_DESC
    out = df.copy()
    texts = out[text_col].fillna("").astype(str)

    uniq = list(pd.Index(texts[texts.str.strip() != ""].unique()))
    if family_grouping:
        rep_of = _families(uniq, counts=texts.value_counts())
    else:
        rep_of = {t: t for t in uniq}
    _order = {t: i for i, t in enumerate(uniq)}
    reps = sorted(set(rep_of.values()), key=_order.get)
    total = len(reps)
    chunks = [reps[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    cache: dict[str, dict] = {}
    done = 0
    lock = Lock()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_label_chunk, c, key, topic_codes, topic_desc,
                               model): c for c in chunks}
        for fut in as_completed(futures):
            chunk = futures[fut]
            try:
                labels = fut.result()
            except Exception as e:  # pragma: no cover - defensive
                labels = [dict(_FAILED, err=str(e)) for _ in chunk]
            for t, lab in zip(chunk, labels):
                cache[t] = lab
            with lock:
                done += len(chunk)
                n = done
            if progress:
                progress(n, total)   # called from this (main) thread — safe for st.*

    def lab(t):
        return cache.get(rep_of.get(t, t), _FAILED)

    labs = texts.map(lab)
    # True where the row inherited its labels from a family representative
    # (near-duplicate of a template) rather than being labelled directly
    out["llm_propagated"] = texts.map(lambda t: rep_of.get(t, t) != t)
    out["llm_topics"] = labs.map(lambda o: "|".join(o["topics"]) or "NONE")
    out["llm_stance"] = labs.map(lambda o: o.get("stance", "unclear"))
    out["llm_conf"] = labs.map(lambda o: o.get("confidence", 0.0))
    out["llm_intensity"] = labs.map(lambda o: o.get("intensity", ""))
    out["llm_intensity_score"] = labs.map(lambda o: o.get("intensity_score", 0.0))
    out["llm_evidence"] = labs.map(lambda o: bool(o.get("provides_evidence", False)))
    out["llm_evidence_type"] = labs.map(lambda o: o.get("evidence_type", "") or "")
    out["llm_failed"] = labs.map(lambda o: bool(o.get("failed", False)))
    # also expand into 0/1 topic flag columns the rest of the app understands
    for c in topic_codes:
        out[c] = labs.map(lambda o, c=c: int(c in o["topics"]))
    return out
