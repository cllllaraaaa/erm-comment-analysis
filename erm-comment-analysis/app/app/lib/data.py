"""
Shared data layer.

Reads an uploaded (or API-fetched) comment file and normalises it into one
internal schema the whole app understands, regardless of how it arrived:

  stance_norm   Oppose / Support / Unclear
  <topic flags> 0/1 topic columns (expanded from llm_topics if needed)
  evidence      does the comment provide supporting evidence? (from the LLM)
  cited_list    canonical official documents cited (computed here if absent)
  _group        form-letter group key (for de-duplication / weighting)
  auto_flag     needs a human eye, with categorized reasons

The active domain pack lives in st.session_state (per user session) and its
derived configs are passed EXPLICITLY into citations / regex_labels / matcher.
No module-level registration — Streamlit shares modules across sessions.

If the file has NO labels at all, `has_labels()` returns False and the
Dashboard offers to run the analysis in-app (see lib/labeling.py).
"""
from __future__ import annotations
import io
import pandas as pd
import streamlit as st

from lib import citations, dedupe, intensity, regex_labels, matcher
from lib import domain as domainmod

KNOWN_TOPICS = [
    "oil_spill_risk", "climate_emissions", "air_quality", "environmental_justice",
    "marine_wildlife", "fisheries", "wetlands_coast", "national_interest",
]
TOPIC_LABELS = {
    "oil_spill_risk": "Oil spill risk", "climate_emissions": "Climate & emissions",
    "air_quality": "Air quality", "environmental_justice": "Environmental justice",
    "marine_wildlife": "Marine wildlife", "fisheries": "Fisheries",
    "wetlands_coast": "Wetlands & coast", "national_interest": "National interest",
}
STANCE_MAP = {   # keys matched case-insensitively (see enrich)
    "oppose": "Oppose", "support": "Support", "unclear": "Unclear",
    "attachment only / unknown": "Unclear", "attachment_only": "Unclear",
    "for": "Support", "against": "Oppose", "favour": "Support", "favor": "Support",
    "in favour": "Support", "in favor": "Support", "in support": "Support",
    "positive": "Support", "negative": "Oppose", "neutral": "Unclear",
    "opposed": "Oppose", "supportive": "Support", "mixed": "Unclear",
}

LOW_CONF = 0.60   # LLM confidence below this -> review queue

# palette
INK = "#12261F"; LEAF = "#2F6B4F"; SAGE = "#7BA894"
CLAY = "#C0673B"; SAND = "#EDE7DA"; MIST = "#DCE5DF"; GOLD = "#C9A227"
STANCE_COLOUR = {"Oppose": CLAY, "Support": LEAF, "Unclear": SAGE}


# --------------------------------------------------------- session-scoped pack
def active_pack():
    """The domain pack in play for THIS user session (custom / suggested), or
    None -> the built-in GulfLink defaults."""
    try:
        return st.session_state.get("pack")
    except Exception:      # outside a Streamlit run (tests)
        return None


def label_for(topic: str) -> str:
    pack = active_pack()
    if pack:
        lab = domainmod.topic_labels(pack).get(topic)
        if lab:
            return lab
    return TOPIC_LABELS.get(topic) or topic.replace("_", " ").title()


def _cit_cfg() -> citations.DomainConfig:
    return domainmod.citation_config(active_pack())


def strength_dots(strength: str) -> str:
    return {"strong": "●●●", "moderate": "●●○", "mild": "●○○"}.get(strength, "")


def read_upload(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded, engine="openpyxl")
    return pd.read_csv(uploaded, low_memory=False)


# ---------------------------------------------------------------- detection
def _has_topic_cols(df) -> bool:
    return any(t in df.columns for t in KNOWN_TOPICS) or "llm_topics" in df.columns


def _has_stance(df) -> bool:
    return any(c in df.columns for c in ["stance", "llm_stance", "stance_norm"])


def has_labels(df: pd.DataFrame) -> bool:
    """True if the file already carries stance + topics (no need to run analysis)."""
    return _has_stance(df) and _has_topic_cols(df)


def has_signal(df: pd.DataFrame) -> bool:
    """True if there's actually something to show — some non-Unclear stance or at
    least one topic tagged. A raw / unlabelled file has no signal → analyse first."""
    if (df["stance_norm"] != "Unclear").any():
        return True
    topics = detect_topics(df)
    if topics and int(df[topics].fillna(0).to_numpy().sum()) > 0:
        return True
    return False


def topic_vocab(df: pd.DataFrame) -> list[str]:
    """Active topic codes discovered from llm_topics content — schema-agnostic, so
    the app works with the default 8 codes OR any custom / LLM-suggested schema."""
    if "llm_topics" in df.columns:
        codes = sorted({c for s in df["llm_topics"].fillna("").astype(str)
                        for c in s.split("|") if c and c != "NONE"})
        if codes:
            return codes
    return []


def detect_topics(df: pd.DataFrame) -> list[str]:
    codes = [c for c in topic_vocab(df) if c in df.columns]
    if codes:
        # include active domain-pack topics present as columns — a keyword-only
        # topic (added by keyword priority) won't show up in the llm_topics vocab
        pack = active_pack()
        if pack:
            for c in domainmod.topic_codes(pack):
                if c in df.columns and c not in codes:
                    codes.append(c)
        return codes
    present = [t for t in KNOWN_TOPICS if t in df.columns]
    if present:
        return present
    flags = []
    skip = {"has_attachment", "has_inline_text", "exact_form_letter", "llm_failed",
            "template_family", "n_topics", "n_cited", "auto_flag", "strength",
            "regex_stance", "cited_str", "topics_str", "stance_norm", "evidence",
            "llm_evidence", "human_reviewed"}
    for c in df.columns:
        cl = c.lower()
        if (c.startswith("_") or c.startswith("flag_") or cl in skip
                or cl.endswith(("_score", "_signal", "_conf", "_len"))):
            continue          # never treat the app's own internal/derived columns as topics
        s = df[c].dropna()
        if len(s) and s.isin([0, 1, True, False]).all() and df[c].nunique() <= 2:
            flags.append(c)
    return flags


def _ci(df: pd.DataFrame, names):
    """Case-insensitive column lookup by a list of candidate names."""
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _longest_text_col(df: pd.DataFrame) -> str:
    """The string column with the greatest average length — real comment bodies are
    long, IDs / metadata are short. Used when no column is named comment_text."""
    best, blen = df.columns[0], -1.0
    for c in df.columns:
        if df[c].dtype == object:
            try:
                avg = float(df[c].astype(str).str.len().mean())
            except Exception:
                continue
            if avg > blen:
                blen, best = avg, c
    return best


def text_col(df: pd.DataFrame) -> str:
    return _ci(df, ["comment_text", "comment", "text", "body", "comment text"]) \
        or _longest_text_col(df)


def best_text_col(df: pd.DataFrame) -> str:
    return _ci(df, ["llm_input_text", "comment_text", "comment", "text", "body"]) \
        or _longest_text_col(df)


def id_col(df: pd.DataFrame) -> str:
    return _ci(df, ["Document ID", "comment_id", "documentid", "id"]) or df.columns[0]


def attach_col(df: pd.DataFrame) -> str | None:
    return _ci(df, ["Attachment Files", "attachment_files", "Content Files",
                    "attachment url", "attachment_url"])


def apply_columns(raw: pd.DataFrame, text_c: str, id_c: str) -> pd.DataFrame:
    """Re-map a raw upload so the chosen columns become comment_text / Document ID."""
    df = raw.copy()
    if text_c and text_c != "comment_text" and "comment_text" in df.columns:
        df = df.drop(columns=["comment_text"])
    if id_c and id_c != "Document ID" and "Document ID" in df.columns:
        df = df.drop(columns=["Document ID"])
    ren = {}
    if text_c and text_c != "comment_text":
        ren[text_c] = "comment_text"
    if id_c and id_c != "Document ID":
        ren[id_c] = "Document ID"
    return df.rename(columns=ren)


def attachment_url(row, col) -> str | None:
    if not col:
        return None
    v = row.get(col)
    if pd.isna(v) or not str(v).strip():
        return None
    url = str(v).split("|")[0].split(",")[0].strip()
    return url if url.startswith("http") else None


# ---------------------------------------------------------------- enrich
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pack = active_pack()
    cit_cfg = domainmod.citation_config(pack)
    kw_map = domainmod.keyword_map(pack) if pack else {}
    kw_pats = domainmod.keyword_patterns(pack) if pack else None

    # expand llm_topics "a|b|c" -> 0/1 columns. When llm_topics exists it is the
    # SOURCE OF TRUTH and columns are rebuilt every enrich — that's what makes the
    # keyword-priority toggle reversible after analysis.
    if "llm_topics" in df.columns:
        lt = df["llm_topics"].fillna("").astype(str)
        codes = sorted({c for s in lt for c in s.split("|") if c and c != "NONE"})
        codes = sorted(set(codes) | set(kw_map.keys()))
        for c in codes:
            df[c] = lt.apply(lambda s, c=c: int(c in s.split("|")))
    else:
        # pre-labelled file without llm_topics: snapshot the original topic flags
        # once, so keyword priority can be toggled off again later
        for c in detect_topics(df):
            if f"_base_{c}" not in df.columns:
                df[f"_base_{c}"] = df[c]
            df[c] = df[f"_base_{c}"]

    # stance
    def _mapstance(col):
        return df[col].astype(str).str.strip().str.lower().map(STANCE_MAP)
    if "stance_norm" in df.columns:
        df["stance_norm"] = _mapstance("stance_norm").fillna(df["stance_norm"])
    elif "llm_stance" in df.columns:
        df["stance_norm"] = _mapstance("llm_stance").fillna("Unclear")
    elif "stance" in df.columns:
        df["stance_norm"] = _mapstance("stance").fillna("Unclear")
    else:
        df["stance_norm"] = "Unclear"

    # evidence provided? (from the LLM labelling run, if present)
    if "llm_evidence" in df.columns:
        df["evidence"] = df["llm_evidence"].fillna(False).astype(bool)
        df["evidence_type"] = (df.get("llm_evidence_type", pd.Series("", index=df.index))
                               .fillna("").astype(str))
        df["_has_evidence_info"] = True
    else:
        df["evidence"] = False
        df["evidence_type"] = ""
        df["_has_evidence_info"] = False

    # keyword strong-match priority (optional): if a comment literally contains a
    # topic's keyword, force that topic ON even when the LLM missed it — a recall
    # boost the reviewer can toggle on the Dashboard. Reversible: topic columns
    # are rebuilt from llm_topics / _base_ snapshots above on every enrich.
    kw_hits = None
    disp_for_kw = None
    try:
        _kw_priority = bool(st.session_state.get("keyword_priority"))
    except Exception:
        _kw_priority = False
    if kw_map:
        m = matcher.get(kw_map)
        disp_for_kw = df[best_text_col(df)].fillna("").astype(str)
        kw_hits = disp_for_kw.map(m.topics_in)
        if _kw_priority:
            for code in kw_map:
                hit = kw_hits.map(lambda s, code=code: code in s)
                base = df[code].fillna(0).astype(bool) if code in df.columns else False
                df[code] = (base | hit).astype(int)

    topics = detect_topics(df)
    if topics:
        def topics_of(row):
            return ", ".join(label_for(t) for t in topics if row.get(t, 0) in (1, True))
        df["topics_str"] = df.apply(topics_of, axis=1)
        df["n_topics"] = df[topics].fillna(0).astype(bool).sum(axis=1)
    else:
        df["topics_str"] = ""; df["n_topics"] = 0

    # cited documents: use existing column or compute now (domain-aware)
    if "cited_documents" in df.columns:
        df["cited_list"] = df["cited_documents"].fillna("").astype(str).apply(
            lambda s: [x for x in s.split("|") if x])
    else:
        tcol = best_text_col(df)
        df["cited_list"] = df[tcol].fillna("").astype(str).apply(
            lambda t: citations.extract(t, cit_cfg))
    df["cited_str"] = df["cited_list"].apply(lambda l: " · ".join(l))
    df["n_cited"] = df["cited_list"].apply(len)

    # form-letter group key: use the offline pipeline's column when present;
    # otherwise group EXACT duplicate texts (normalised), so form letters in a
    # raw upload still collapse in 'distinct letters' views
    gcol = next((c for c in ["exact_form_letter_group", "template_family_id"]
                 if c in df.columns), None)
    if gcol:
        df["_group"] = df[gcol].fillna("__u__").astype(str)
    else:
        tc0 = text_col(df)
        norm = (df[tc0].fillna("").astype(str).str.lower()
                .str.replace(r"\s+", " ", regex=True).str.strip())
        empty = norm.str.len() < 20
        df["_group"] = ("t" + pd.util.hash_pandas_object(norm, index=False).astype(str)
                        ).where(~empty, "u" + df.index.astype(str))

    # near-duplicate template grouping (default on): merge letter groups whose
    # representative texts are the same template with small personal edits.
    # Toggle under Grouping settings on the Dashboard; results are cached per
    # dataset so reruns and toggles don't recompute the clustering.
    try:
        _exact_only = bool(st.session_state.get("group_exact_only"))
    except Exception:
        _exact_only = False
    if not _exact_only and len(df):
        tc0 = text_col(df)
        disp0 = df[tc0].fillna("").astype(str)
        reps = (pd.DataFrame({"g": df["_group"], "t": disp0})
                .groupby("g")["t"].agg(lambda s2: max(s2, key=len)))
        fp = int(pd.util.hash_pandas_object(reps, index=True).sum())
        merge_map = None
        try:
            cache = st.session_state.get("_ndcache")
            if cache and cache.get("fp") == fp:
                merge_map = cache["map"]
        except Exception:
            pass
        if merge_map is None:
            gids = dedupe.near_dup_groups(reps.tolist())
            merge_map = {g: f"nd{cid}" for g, cid in zip(reps.index, gids)}
            try:
                st.session_state["_ndcache"] = {"fp": fp, "map": merge_map}
            except Exception:
                pass
        df["_group"] = df["_group"].map(merge_map).fillna(df["_group"])

    # attachment / OCR state
    ac = attach_col(df)
    tc = text_col(df)
    inline_len = df[tc].fillna("").astype(str).str.strip().str.len()
    has_att = (df["has_attachment"].fillna(False).astype(bool)
               if "has_attachment" in df.columns
               else (df[ac].fillna("").astype(str).str.strip().ne("") if ac
                     else pd.Series(False, index=df.index)))
    ocr_len = (df["ocr_text"].fillna("").astype(str).str.len()
               if "ocr_text" in df.columns else pd.Series(0, index=df.index))
    if "submission_source" in df.columns:
        att_only = df["submission_source"].fillna("").astype(str).str.contains(
            "attachment only", case=False)
    else:
        att_only = has_att & (inline_len < 20)
    df["_attachment_only"] = att_only
    df["_needs_ocr"] = att_only & (ocr_len == 0)
    df["_has_attachment"] = has_att

    # display text (inline, else OCR) for intensity + keyword cross-check
    ocr_series = (df["ocr_text"].fillna("").astype(str)
                  if "ocr_text" in df.columns else pd.Series("", index=df.index))
    disp = df[tc].fillna("").astype(str).where(inline_len >= 20, ocr_series)

    # objective stance intensity (no API); unify with LLM intensity if present
    df["intensity_signal"] = disp.map(intensity.score)
    if "llm_intensity_score" in df.columns:
        df["strength_score"] = (pd.to_numeric(df["llm_intensity_score"], errors="coerce")
                                .fillna(df["intensity_signal"]))
    else:
        df["strength_score"] = df["intensity_signal"]
    df["strength"] = df["strength_score"].map(intensity.label)

    # keyword/regex cross-check (bias guard): the keyword layer caught a topic the
    # LLM missed, or the two disagree on a committed stance
    active = set(topics)   # only cross-check codes in the active schema
    if kw_map and disp_for_kw is not None:
        # domain-pack keywords via the matcher (spaCy if installed)
        r_topics = disp.map(lambda t: matcher.get(kw_map).topics_in(t) & active)
    else:
        r_topics = disp.map(lambda t: regex_labels.topics_set(t, kw_pats) & active)
    r_stance = disp.map(regex_labels.stance)
    df["regex_stance"] = r_stance
    if topics:
        tvals = df[topics].fillna(0).astype(bool).values
        prim = [set(t for t, fl in zip(topics, row) if fl) for row in tvals]
    else:
        prim = [set()] * len(df)
    regex_missed = [bool(rt - pt) for rt, pt in zip(r_topics, prim)]
    stance_conflict = [(rs in ("oppose", "support")) and (str(sn).lower() in ("oppose", "support"))
                       and (str(sn).lower() != rs)
                       for rs, sn in zip(r_stance, df["stance_norm"])]
    df["flag_disagree"] = [a or b for a, b in zip(regex_missed, stance_conflict)]

    # does the cross-check actually have keyword coverage for the active schema?
    try:
        st.session_state["_crosscheck_covered"] = bool(
            topics and (any(c in (kw_map or {}) for c in topics)
                        or (not pack and any(c in regex_labels.TOPIC_PATTERNS
                                             for c in topics))))
    except Exception:
        pass

    # categorized review flags
    unclear = df["stance_norm"] == "Unclear"
    no_topic = df["n_topics"] == 0
    failed = (df["llm_failed"].fillna(False).astype(bool)
              if "llm_failed" in df.columns else pd.Series(False, index=df.index))
    if "llm_conf" in df.columns:
        low_conf = (pd.to_numeric(df["llm_conf"], errors="coerce").fillna(1.0)
                    < LOW_CONF) & ~failed
    else:
        low_conf = pd.Series(False, index=df.index)
    hr = (df["human_reviewed"].fillna(False).astype(bool)
          if "human_reviewed" in df.columns else pd.Series(False, index=df.index))
    df["flag_unclear_stance"] = unclear & ~hr
    df["flag_no_topic"] = no_topic & ~hr
    df["flag_needs_ocr"] = df["_needs_ocr"] & ~hr
    df["flag_not_labelled"] = failed & ~hr
    df["flag_low_conf"] = low_conf & ~hr
    df["flag_disagree"] = df["flag_disagree"] & ~hr
    df["auto_flag"] = (unclear | no_topic | df["_needs_ocr"] | failed | low_conf) & ~hr

    def reasons(row):
        rs = []
        if row["flag_unclear_stance"]: rs.append("unclear stance")
        if row["flag_no_topic"]: rs.append("no topic detected")
        if row["flag_needs_ocr"]:
            why = {"download_failed": "attachment could not be downloaded",
                   "unsupported": "attachment is not a PDF or image",
                   "empty": "attachment was blank or illegible",
                   "no_url": "no attachment link in the file",
                   }.get(str(row.get("ocr_status", "")),
                         "attachment only, content not read yet")
            rs.append(why)
        if row["flag_not_labelled"]: rs.append("not labelled")
        if row["flag_low_conf"]: rs.append("low model confidence")
        return ", ".join(rs)
    df["flag_reason"] = df.apply(reasons, axis=1)
    return df


def read_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes); buf.name = filename
    return read_upload(buf)


def set_data(df: pd.DataFrame):
    st.session_state["df"] = df


def get_data() -> pd.DataFrame | None:
    return st.session_state.get("df")


def csv_bytes(frame: pd.DataFrame, cols=None) -> bytes:
    """CSV for download — utf-8-sig so Excel opens non-ASCII text correctly."""
    if cols:
        frame = frame[[c for c in cols if c in frame.columns]]
    return frame.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------- summaries
def unique_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("_group")


def topic_counts(df: pd.DataFrame, weighted: bool = True) -> dict:
    topics = detect_topics(df)
    frame = df if weighted else unique_frame(df)
    counts = {label_for(t): int(frame[t].fillna(0).astype(bool).sum()) for t in topics}
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def stance_counts(df: pd.DataFrame, weighted: bool = True) -> pd.Series:
    frame = df if weighted else unique_frame(df)
    return frame["stance_norm"].value_counts()


def cited_frequency(df: pd.DataFrame, weighted: bool = True) -> pd.DataFrame:
    frame = df if weighted else unique_frame(df)
    cfg = _cit_cfg()
    c = citations.frequency(frame["cited_list"])
    rows = [{"document": d, "category": citations.category(d, cfg), "count": n}
            for d, n in c.most_common()]
    return pd.DataFrame(rows)


def all_cited_documents(df: pd.DataFrame) -> list[str]:
    docs = set()
    for lst in df["cited_list"]:
        docs.update(lst)
    return sorted(docs)


def crosstab_topic_stance(df: pd.DataFrame, weighted: bool = True) -> pd.DataFrame:
    """Topic x stance counts — used by the AI assistant's context."""
    topics = detect_topics(df)
    frame = df if weighted else unique_frame(df)
    rows = []
    for t in topics:
        m = frame[t].fillna(0).astype(bool)
        rows.append({
            "topic": label_for(t),
            "Oppose": int((m & (frame["stance_norm"] == "Oppose")).sum()),
            "Support": int((m & (frame["stance_norm"] == "Support")).sum()),
            "Unclear": int((m & (frame["stance_norm"] == "Unclear")).sum()),
        })
    return pd.DataFrame(rows)


def data_summary(df: pd.DataFrame) -> str:
    topics = detect_topics(df)
    lines = [f"Total comments: {len(df)}",
             "Distinct comments (form letters grouped): "
             f"{unique_frame(df).shape[0]} of {len(df)}"]
    sb = stance_counts(df)
    lines.append("Stance (by record): " + ", ".join(f"{k} {v}" for k, v in sb.items()))
    if topics:
        tc = topic_counts(df)
        lines.append("Topics: " + ", ".join(f"{k} {v}" for k, v in tc.items()))
        ct = crosstab_topic_stance(df)
        if len(ct):
            lines.append("Topic x stance (Oppose/Support/Unclear): " + "; ".join(
                f"{r.topic} {r.Oppose}/{r.Support}/{r.Unclear}"
                for r in ct.itertuples()))
    freq = cited_frequency(df, weighted=False)
    if len(freq):
        top = ", ".join(f"{r.document.split(' (')[0]} {r.count}"
                        for r in freq.head(6).itertuples())
        lines.append("Most-cited documents (distinct letters): " + top)
    if bool(df.get("_has_evidence_info", pd.Series(dtype=bool)).any()):
        n_ev = int(df["evidence"].sum())
        lines.append(f"Comments providing supporting evidence: {n_ev} of {len(df)}")
    ones = int((df.groupby("_group")["_group"].transform("size") == 1).sum())
    lines.append(f"One-of-a-kind letters (no template match): {ones}")
    strong_opp = int(((df["stance_norm"] == "Oppose") & (df["strength"] == "strong")).sum())
    lines.append(f"Strongly-worded opposition (high intensity): {strong_opp}")
    lines.append(f"Flagged for review: {int(df['auto_flag'].sum())}")
    return "\n".join(lines)
