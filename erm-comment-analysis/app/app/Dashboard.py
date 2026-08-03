"""
Consultant Dashboard, home page.

Load comments (fetch a docket by API or upload a file), define the topics for
the project, run the analysis, then see the whole picture. All user-facing
copy is deliberately short; longer explanations live in the "?" help tooltips.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data as D
from lib import labeling, regs, ocr, domain
from lib.config import GEMINI_MODEL

st.set_page_config(page_title="Comment Analysis", layout="wide")

st.markdown(f"""
<style>
  .stApp {{ background: #FBFAF6; }}
  h1, h2, h3 {{ color: {D.INK}; font-family: "Georgia", serif; }}
  .metric-card {{ background: white; border: 1px solid {D.MIST}; border-radius: 14px;
     padding: 18px 20px; box-shadow: 0 1px 3px rgba(18,38,31,.06);
     height: 100%; min-height: 116px; display: flex; flex-direction: column;
     justify-content: center; }}
  .metric-num {{ font-size: 34px; font-weight: 700; color: {D.LEAF}; line-height: 1; }}
  .metric-lab {{ font-size: 12.5px; color: #5b6b63; margin-top: 6px; }}
  .metric-num.alert {{ color: {D.CLAY}; }}
  .metric-num.gold {{ color: {D.GOLD}; }}
</style>
""", unsafe_allow_html=True)

st.title("Public Comment Analysis")

MODEL_CHOICES = {
    "Fast (gemini-2.5-flash-lite)": "gemini-2.5-flash-lite",
    "Accurate (gemini-2.5-flash)": "gemini-2.5-flash",
}
MODEL_HELP = ("The model used to label the comments. Fast is the cheapest and "
              "quickest. Accurate costs more per comment and is slower, but is "
              "noticeably better on difficult stance calls; recommended for "
              "client deliverables.")


def _ingest(raw, name):
    st.session_state["raw_df"] = raw
    st.session_state["filename"] = name
    st.session_state.pop("_just_labelled", None)
    D.set_data(D.enrich(raw))


# ---------------------------------------------------------------- data source UI
def upload_form(key: str):
    up = st.file_uploader(
        "Upload a comments file (CSV or Excel)", type=["csv", "xlsx", "xls"], key=key,
        help="One row per comment; only a text column is required. A labelled "
             "dataset saved from this tool can be uploaded here and its labels "
             "are reused without re-running the analysis.")
    if up is not None:
        _sig = getattr(up, "file_id", None) or f"{up.name}:{up.size}"
        if st.session_state.get("_upsig_" + key) != _sig:
            try:
                st.session_state["_upsig_" + key] = _sig
                _ingest(D.read_bytes(up.getvalue(), up.name), up.name)
                st.rerun()
            except Exception as e:
                st.error(f"Could not read that file: {e}")


def fetch_form(key: str):
    c1, c2 = st.columns(2)
    docket = c1.text_input(
        "Docket ID", placeholder="e.g. MARAD-2019-0093", key=f"{key}_docket",
        help="The Regulations.gov docket whose public comments to pull.")
    reg_key = c2.text_input(
        "API key", type="password", key=f"{key}_key",
        help="Your Regulations.gov API key. Used only to fetch the comments; "
             "never stored. Once fetched, the data can be saved as a CSV.")
    maxn = st.slider("Comments to pull", 50, 2000, 300, step=50, key=f"{key}_maxn",
                     help="Bounded by the API rate limit of 1,000 requests per hour.")
    if st.button("Fetch comments", type="primary", key=f"{key}_go"):
        if not (docket and reg_key):
            st.warning("Enter both a docket ID and your API key.")
        else:
            bar = st.progress(0.0, "Fetching")
            try:
                raw = regs.fetch_docket_comments(
                    reg_key, docket.strip(), max_records=maxn,
                    progress=lambda i, n: bar.progress(i / max(n, 1), f"{i}/{n}"))
                _ingest(raw, f"{docket} (Regulations.gov)")
                st.rerun()
            except Exception as e:
                st.error(f"Fetch failed: {e}")


def source_tabs(suffix: str):
    t_api, t_up = st.tabs(["Fetch by API", "Upload a file"])
    with t_api:
        fetch_form(f"fetch{suffix}")
    with t_up:
        upload_form(f"up{suffix}")


# ---------------------------------------------------------------- topic pack UI
def pack_editor(df: pd.DataFrame, api_key: str):
    """Topic editor. Returns (pack, codes, desc-with-keyword-examples)."""
    tcol = D.best_text_col(df)
    st.session_state.setdefault("pack", domain.default_pack())

    cbld, cload = st.columns([1, 1])
    if cbld.button("Suggest topics from the data", key="build_pack",
                   help="Reads a sample of the comments and proposes topics with "
                        "keywords, plus the laws and agencies commenters cite. "
                        "Needs your Gemini API key."):
        if not api_key:
            st.warning("Add your Gemini API key first.")
        else:
            sample = df[tcol].dropna().astype(str)
            sample = sample[sample.str.len() > 40].tolist()
            try:
                with st.spinner("Reading a sample of the comments"):
                    st.session_state["pack"] = domain.build_pack(sample, api_key)
                st.session_state["schema_text"] = domain.to_schema_text(st.session_state["pack"])
                st.rerun()
            except Exception as e:
                st.error(f"Could not suggest topics: {e}")
    up_pack = cload.file_uploader(
        "Load a topic pack (.json)", type=["json"], key="packup",
        help="A topic pack saved earlier with the Save button below. It carries "
             "the topics, their keywords, and the cited-document list for a "
             "subject area, so it can be reused on the next project.")
    if up_pack is not None:
        # process each uploaded file once; the uploader returns the same file on
        # every rerun and reloading it would overwrite the user's edits
        _sig = getattr(up_pack, "file_id", None) or f"{up_pack.name}:{up_pack.size}"
        if st.session_state.get("_packsig") != _sig:
            try:
                st.session_state["pack"] = domain.from_json(up_pack.getvalue())
                st.session_state["schema_text"] = domain.to_schema_text(st.session_state["pack"])
                st.session_state["_packsig"] = _sig
            except Exception as e:
                st.error(f"Could not read that pack: {e}")

    st.session_state.setdefault("schema_text", domain.to_schema_text(st.session_state["pack"]))
    schema_text = st.text_area(
        "Topics (one per line)", key="schema_text", height=180,
        help="Either 'code: description' or just a topic-area name such as "
             "Air Quality. The topic-area list from the project's EIS can be "
             "pasted straight in, one area per line. The pre-filled list is an "
             "editable example; replace it with the topics for this project.")
    # rebuild the pack from the (possibly edited) schema, keeping keywords/acronyms
    pack = domain.pack_from_schema_text(schema_text, base=st.session_state["pack"])
    st.session_state["pack"] = pack
    codes = domain.topic_codes(pack)
    desc = domain.topic_desc(pack)
    # inject each topic's keywords into the label prompt as examples
    for t in pack.get("topics", []):
        if t.get("keywords"):
            desc[t["code"]] = (t.get("description", "").rstrip(" .") +
                               " (e.g. " + ", ".join(t["keywords"][:6]) + ")").strip()

    st.caption(f"{len(codes)} topics: " + ", ".join(codes[:14])
               + ("..." if len(codes) > 14 else ""))
    if not domain.has_keyword_coverage(pack, codes):
        st.caption("These topics have no keywords yet, so the keyword cross-check "
                   "only covers stance. Suggest topics from the data to add "
                   "keywords automatically.")

    def _reapply_priority():
        cur = D.get_data()
        if cur is not None:
            D.set_data(D.enrich(cur))
    st.checkbox(
        "Keyword priority", key="keyword_priority", value=False,
        on_change=_reapply_priority,
        help="When a comment contains one of a topic's keywords, that topic is "
             "always applied, even where the model missed it. Improves recall. "
             "Takes effect immediately and can be switched off again.")
    st.download_button("Save topic pack (.json)", domain.to_json(pack),
                       file_name="topic_pack.json", mime="application/json",
                       key="save_pack",
                       help="Saves the topics, keywords and cited-document list "
                            "for reuse on another docket.")
    return pack, codes, desc


def model_select(key: str) -> str:
    choice = st.selectbox("Model", list(MODEL_CHOICES.keys()), index=0,
                          key=key, help=MODEL_HELP)
    return MODEL_CHOICES.get(choice, GEMINI_MODEL)


def run_analysis(df: pd.DataFrame, api_key: str, codes, desc, do_ocr: bool, AC,
                 model: str):
    """OCR (optional) + labelling + enrich; sets the fresh frame in session."""
    tcol = D.best_text_col(df)
    # drop topic columns from a previous schema so stale flags don't linger
    old = [c for c in D.detect_topics(df) if c in df.columns]
    work = df.drop(columns=[c for c in old + [f"_base_{t}" for t in old]
                            if c in df.columns], errors="ignore")
    if do_ocr and AC:
        obar = st.progress(0.0, "Reading attachments")
        mask = work["_attachment_only"].values
        texts, statuses = ocr.ocr_attachments(
            work, AC, api_key, rows_mask=mask,
            progress=lambda i, n: obar.progress(i / max(n, 1), f"{i}/{n}"))
        work = work.copy()
        work["ocr_text"] = texts
        work["ocr_status"] = statuses
        work["llm_input_text"] = (
            work[D.text_col(work)].fillna("").astype(str)
            + "\n\n" + work["ocr_text"].fillna("")).str.strip()
    run_tcol = "llm_input_text" if "llm_input_text" in work.columns else tcol
    bar = st.progress(0.0, "Labelling")
    labelled = labeling.analyse(
        work, run_tcol, api_key, topic_codes=codes, topic_desc=desc, model=model,
        progress=lambda i, n: bar.progress(i / max(n, 1), f"{i}/{n} unique comments"))
    D.set_data(D.enrich(labelled))
    st.session_state["_just_labelled"] = True
    st.rerun()


df = D.get_data()

# ============================================================ INPUT (no data yet)
if df is None:
    st.subheader("Load comments")
    source_tabs("0")
    st.stop()

# ============================================================ DATA SOURCE (loaded)
src1, src2 = st.columns([3, 1])
src1.success(f"Loaded {st.session_state.get('filename', 'file')}: {len(df):,} comments.")
src2.download_button("Save loaded data (CSV)", D.csv_bytes(df),
                     file_name="comments_dataset.csv", mime="text/csv",
                     help="The dataset as currently loaded and labelled, for "
                          "example to keep a CSV of comments fetched by API.")

with st.expander("Data source"):
    source_tabs("1")

# --- column settings: which column holds the comment text? (non-standard files) ---
_raw = st.session_state.get("raw_df")
if _raw is not None and len(_raw.columns) > 1:
    _cur = D.text_col(df)
    _avg = df[_cur].astype(str).str.len().mean() if _cur in df.columns else 0
    _bad = _avg < 25          # the "text" looks like IDs / metadata, not comments
    with st.expander("Column settings", expanded=bool(_bad)):
        st.caption("Only needed if the wrong columns were detected: choose which "
                   "column holds the comment text and which holds the comment ID.")
        if _bad:
            st.warning("The detected text column looks like IDs or metadata "
                       f"('{_cur}', average length {_avg:.0f} characters). "
                       "Pick the column with the real comment text.")
        _cols = list(_raw.columns)
        _gt, _gi = D.text_col(_raw), D.id_col(_raw)
        _t = st.selectbox("Comment text column", _cols,
                          index=_cols.index(_gt) if _gt in _cols else 0, key="map_text")
        _i = st.selectbox("Comment ID column", _cols,
                          index=_cols.index(_gi) if _gi in _cols else 0, key="map_id")
        if st.button("Apply columns"):
            D.set_data(D.enrich(D.apply_columns(_raw, _t, _i)))
            st.rerun()

# ============================================================ RUN ANALYSIS (if unlabelled)
_stance_present = (df["stance_norm"] != "Unclear").any()
if not D.has_signal(df) or not _stance_present:
    st.warning("This file has no usable labels yet. Set the topics, then run the "
               "analysis to generate stance, topics, evidence and intensity.")
    key = st.text_input(
        "API key", type="password", key="label_key",
        help="Your Gemini API key, used only to label the comments; never "
             "stored. This is a different key from the Regulations.gov one.")
    mchoice = model_select("label_model")
    pack, codes, desc = pack_editor(df, key)

    # optional OCR for 'see attached' comments
    AC = D.attach_col(df)
    n_seeatt = int(df["_attachment_only"].sum()) if "_attachment_only" in df.columns else 0
    do_ocr = False
    if AC and n_seeatt:
        do_ocr = st.checkbox(
            f"Also read the {n_seeatt:,} attachment-only comments", value=True,
            help="These comments say 'see attached'; their content is in a PDF. "
                 "Each PDF is downloaded and transcribed before labelling. "
                 "Slower, and uses your key.")

    tcol = D.best_text_col(df)
    n_unique = df[tcol].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique()
    st.caption(f"About {n_unique:,} unique comments to label; identical form "
               "letters are labelled once.")
    if st.button("Run analysis", type="primary"):
        if not key:
            st.warning("Add your API key first.")
        elif not codes:
            st.warning("Add at least one topic.")
        else:
            run_analysis(df, key, codes, desc, do_ocr, AC, mchoice)
    st.stop()

# ============================================================ TOPICS (always available)
with st.expander("Topics"):
    st.caption("Upload or edit your own topic list, then re-run the analysis "
               "against it. The current labels stay until the re-run finishes.")
    key_r = st.text_input(
        "API key", type="password", key="relabel_key",
        help="Your Gemini API key, used only to label the comments; never stored.")
    mchoice_r = model_select("relabel_model")
    pack_r, codes_r, desc_r = pack_editor(df, key_r)
    AC_r = D.attach_col(df)
    tcol_r = D.best_text_col(df)
    n_unique_r = (df[tcol_r].fillna("").astype(str).str.strip()
                  .replace("", pd.NA).dropna().nunique())
    st.caption(f"About {n_unique_r:,} unique comments would be relabelled; "
               "identical form letters are labelled once.")
    if st.button("Re-run analysis with these topics", type="primary", key="rerun_btn"):
        if not key_r:
            st.warning("Add your API key first.")
        elif not codes_r:
            st.warning("Add at least one topic.")
        else:
            run_analysis(df, key_r, codes_r, desc_r, False, AC_r, mchoice_r)

# ============================================================ SAVE YOUR LABELS
if st.session_state.get("_just_labelled"):
    with st.container(border=True):
        st.markdown("**Analysis finished. Download the labelled dataset now.** "
                    "Results are kept in this browser session only; uploading "
                    "this file again later reuses the labels at no API cost.")
        st.download_button("Save labelled dataset (CSV)", D.csv_bytes(df),
                           file_name="comments_labelled.csv", mime="text/csv",
                           type="primary")
        if "ocr_status" in df.columns:
            _bad = df["ocr_status"].isin(["download_failed", "unsupported", "empty"])
            if int(_bad.sum()):
                _bd = df.loc[_bad, "ocr_status"].value_counts().to_dict()
                st.caption(f"{int(_bad.sum()):,} attachments could not be read "
                           f"({', '.join(f'{k}: {v}' for k, v in _bd.items())}). "
                           "They stay in the review queue; the reason is shown "
                           "on each comment there and in exports.")

# ============================================================ OVERVIEW
weighted = st.radio(
    "Count comments by",
    ["Every submission (by volume)", "Distinct letters (form letters grouped)"],
    horizontal=True, index=0,
    help="Form letters are the same message sent by many people. Distinct "
         "letters counts each unique letter once, so a template with 2,000 "
         "signers is not counted as 2,000 separate arguments.",
) == "Every submission (by volume)"

with st.expander("Grouping settings"):
    def _regroup():
        cur = D.get_data()
        if cur is not None:
            D.set_data(D.enrich(cur))
    st.checkbox(
        "Group exact copies only", key="group_exact_only", value=False,
        on_change=_regroup,
        help="By default, near-identical letters (the same template with small "
             "personal edits, such as an added sentence or a different "
             "salutation) are grouped as one form letter. Tick this to group "
             "only letters that match word for word.")

if st.session_state.get("_crosscheck_covered") is False:
    st.caption("The keyword cross-check currently covers stance only, because "
               "the active topics have no keywords. Add keywords under Topics "
               "to enable the full cross-check.")

df = D.get_data()   # the keyword-priority toggle may have refreshed the frame

frame = df if weighted else D.unique_frame(df)
n_total = len(frame)
sb = D.stance_counts(df, weighted)
n_oppose = int(sb.get("Oppose", 0)); n_support = int(sb.get("Support", 0))
n_flag = int(df["auto_flag"].sum())
n_unique = D.unique_frame(df).shape[0]
n_cited = int((frame["n_cited"] > 0).sum())
has_ev_info = bool(frame.get("_has_evidence_info", pd.Series(dtype=bool)).any())
n_evid = int(frame["evidence"].sum()) if has_ev_info else 0

def card(col, num, lab, cls=""):
    col.markdown(f"<div class='metric-card'><div class='metric-num {cls}'>{num}</div>"
                 f"<div class='metric-lab'>{lab}</div></div>", unsafe_allow_html=True)

cols = st.columns(6 if has_ev_info else 5)
card(cols[0], f"{n_total:,}", "Comments" if weighted else "Distinct letters")
op = f"{n_oppose / n_total * 100:.0f}%" if n_total else "0%"
card(cols[1], f"{n_oppose:,}", f"Oppose ({op})", "alert")
card(cols[2], f"{n_support:,}", "Support")
card(cols[3], f"{n_cited:,}", "Cite a document", "gold")
if has_ev_info:
    card(cols[4], f"{n_evid:,}", "Provide evidence", "gold")
card(cols[-1], f"{n_flag:,}", "Need review", "alert")

st.divider()

# ---------- charts row 1: topics + stance ----------
left, right = st.columns([1.4, 1])

with left:
    st.subheader("What are people talking about?")
    tc = D.topic_counts(df, weighted)
    tc = dict(sorted(tc.items(), key=lambda x: x[1]))
    vals = list(tc.values())
    fig = go.Figure(go.Bar(
        x=vals, y=list(tc.keys()), orientation="h",
        marker_color=D.LEAF, text=[f"{v:,}" for v in vals],
        textposition="auto", textfont=dict(color="white", size=13), cliponaxis=False))
    fig.update_layout(
        height=440, margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#EEE", range=[0, max(vals) * 1.18] if vals else None),
        font=dict(family="Georgia, serif", color=D.INK, size=13))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Where do people stand?")
    fig = go.Figure(go.Pie(
        labels=sb.index.tolist(), values=sb.values.tolist(), hole=0.58,
        marker=dict(colors=[D.STANCE_COLOUR.get(k, D.MIST) for k in sb.index]),
        textinfo="label+percent", textfont=dict(size=13)))
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      font=dict(family="Georgia, serif", color=D.INK, size=13))
    st.plotly_chart(fig, use_container_width=True)

# ---------- charts row 2: stance-by-topic + cited documents ----------
left2, right2 = st.columns(2)

with left2:
    st.subheader("Stance within each topic")
    topics = D.detect_topics(df)
    order = [t for t in topics]
    labels = [D.label_for(t) for t in order]
    fig = go.Figure()
    for stance in ["Oppose", "Unclear", "Support"]:
        counts = []
        for t in order:
            m = frame[t].fillna(0).astype(bool) & (frame["stance_norm"] == stance)
            counts.append(int(m.sum()))
        fig.add_bar(y=labels, x=counts, orientation="h", name=stance,
                    marker_color=D.STANCE_COLOUR[stance])
    fig.update_layout(barmode="stack", height=430,
                      margin=dict(l=10, r=20, t=10, b=10), plot_bgcolor="white",
                      paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1),
                      font=dict(family="Georgia, serif", color=D.INK, size=12),
                      xaxis=dict(showgrid=True, gridcolor="#EEE"))
    st.plotly_chart(fig, use_container_width=True)

with right2:
    st.subheader("Most-cited official documents")
    freq = D.cited_frequency(df, weighted)
    if len(freq):
        top = freq.head(10).iloc[::-1]
        catcol = {"Statute": D.LEAF, "Agency": D.SAGE, "EIS document": D.GOLD,
                  "Regulation": D.CLAY}
        fig = go.Figure(go.Bar(
            x=top["count"], y=[d.split(" (")[0] for d in top["document"]],
            orientation="h", text=[f"{v:,}" for v in top["count"]],
            textposition="auto", textfont=dict(color="white", size=12), cliponaxis=False,
            marker_color=[catcol.get(c, D.MIST) for c in top["category"]]))
        fig.update_layout(height=430, margin=dict(l=10, r=55, t=10, b=10),
                          plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=True, gridcolor="#EEE",
                                     range=[0, top["count"].max() * 1.18]),
                          font=dict(family="Georgia, serif", color=D.INK, size=12))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No official documents were detected in this dataset.")

# ---------- evidence ----------
if has_ev_info:
    st.subheader("Do commenters back up their position?")
    ev1, ev2 = st.columns([1.4, 1])
    with ev1:
        rows = []
        for sn in ["Oppose", "Support", "Unclear"]:
            m = frame["stance_norm"] == sn
            rows.append({"stance": sn,
                         "With evidence": int((m & frame["evidence"]).sum()),
                         "Opinion only": int((m & ~frame["evidence"]).sum())})
        fig = go.Figure()
        fig.add_bar(y=[r["stance"] for r in rows], x=[r["With evidence"] for r in rows],
                    orientation="h", name="With evidence", marker_color=D.LEAF)
        fig.add_bar(y=[r["stance"] for r in rows], x=[r["Opinion only"] for r in rows],
                    orientation="h", name="Opinion only", marker_color=D.MIST)
        fig.update_layout(barmode="stack", height=300, margin=dict(l=10, r=20, t=10, b=10),
                          plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=1.18),
                          font=dict(family="Georgia, serif", color=D.INK, size=12),
                          xaxis=dict(showgrid=True, gridcolor="#EEE"))
        st.plotly_chart(fig, use_container_width=True)
    with ev2:
        pct = f"{n_evid / n_total * 100:.0f}%" if n_total else "0%"
        st.markdown(f"<div class='metric-card'><div class='metric-num gold'>{pct}</div>"
                    f"<div class='metric-lab'>of comments give supporting evidence "
                    f"(data, studies, laws, first-hand or professional experience)</div></div>",
                    unsafe_allow_html=True)
        st.caption("Evidence-backed comments usually deserve a substantive "
                   "response. Filter to them on the Browse and Export pages.")

# ---------- intensity ----------
st.subheader("How strongly do people feel?")
ic1, ic2 = st.columns([1.4, 1])
with ic1:
    scol = {"strong": D.CLAY, "moderate": D.GOLD, "mild": D.SAGE}
    fig = go.Figure()
    for s in ["strong", "moderate", "mild"]:
        counts = [int(((frame["stance_norm"] == sn) & (frame["strength"] == s)).sum())
                  for sn in ["Oppose", "Support", "Unclear"]]
        fig.add_bar(y=["Oppose", "Support", "Unclear"], x=counts, orientation="h",
                    name=s, marker_color=scol[s])
    fig.update_layout(barmode="stack", height=300, margin=dict(l=10, r=20, t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.18),
                      font=dict(family="Georgia, serif", color=D.INK, size=12),
                      xaxis=dict(showgrid=True, gridcolor="#EEE"))
    st.plotly_chart(fig, use_container_width=True)
with ic2:
    strong_opp = int(((frame["stance_norm"] == "Oppose") & (frame["strength"] == "strong")).sum())
    st.markdown(f"<div class='metric-card'><div class='metric-num alert'>{strong_opp:,}</div>"
                f"<div class='metric-lab'>strongly-worded opposition</div></div>",
                unsafe_allow_html=True)
    st.caption("Intensity reflects how forcefully a comment is written: capital "
               "letters, exclamation marks, urgent wording. It is indicative, "
               "not a validated measure.")

# ---------- regions ----------
region_col = next((c for c in ["state_clean", "State/Province", "state", "region"]
                   if c in df.columns), None)
if region_col:
    st.subheader("Which regions support the project most?")
    reg = frame.copy()
    reg[region_col] = (reg[region_col].fillna("Unknown").astype(str).str.strip()
                       .replace("", "Unknown"))
    vol = reg[region_col].value_counts()
    keep = vol[vol >= 10].head(15).index
    reg = reg[reg[region_col].isin(keep)]
    if len(reg):
        piv = reg.pivot_table(index=region_col, columns="stance_norm",
                              aggfunc="size", fill_value=0)
        for s in ["Oppose", "Support", "Unclear"]:
            if s not in piv.columns:
                piv[s] = 0
        piv["_tot"] = piv[["Oppose", "Support", "Unclear"]].sum(axis=1)
        piv["_share"] = piv["Support"] / piv["_tot"].replace(0, 1)
        piv = piv.sort_values("_share").tail(12)
        fig = go.Figure()
        for s in ["Oppose", "Unclear", "Support"]:
            fig.add_bar(y=piv.index.tolist(), x=piv[s].tolist(), orientation="h",
                        name=s, marker_color=D.STANCE_COLOUR[s])
        fig.update_layout(barmode="stack", height=440,
                          margin=dict(l=10, r=20, t=10, b=10), plot_bgcolor="white",
                          paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1),
                          font=dict(family="Georgia, serif", color=D.INK, size=12),
                          xaxis=dict(showgrid=True, gridcolor="#EEE"))
        st.plotly_chart(fig, use_container_width=True)
        _cap = ("Regions with at least 10 comments, ranked by share of support, "
                "most supportive at the top.")
        if n_total and n_support / n_total < 0.15:
            _cap += (" Support is uncommon in this docket, so the bars are "
                     "mostly opposition.")
        st.caption(_cap)

# ---------- repeated form letters ----------
if n_unique < len(df):
    st.subheader("Repeated form letters")
    grp = (df.groupby("_group").size().sort_values(ascending=False))
    biggest = grp[grp > 1].head(12)
    if len(biggest):
        fig = go.Figure(go.Bar(
            x=[f"#{i+1}" for i in range(len(biggest))], y=biggest.values,
            marker_color=D.SAGE, text=[f"{v:,}" for v in biggest.values],
            textposition="auto", textfont=dict(color="white", size=12), cliponaxis=False))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(title="copies", showgrid=True, gridcolor="#EEE"),
                          xaxis=dict(title="largest letter templates"),
                          font=dict(family="Georgia, serif", color=D.INK, size=12))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(df) - n_unique:,} of {len(df):,} comments are copies of "
                   "a template letter sent by many people, usually from an "
                   "advocacy campaign. Each bar is one template; its height is "
                   "how many people sent that letter.")
