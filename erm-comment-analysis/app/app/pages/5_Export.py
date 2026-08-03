"""
Export — download comments as CSV, comprehensively: the full labelled dataset,
a custom filtered slice (topic + stance + documents + attachments), or one of the
ready-made lists (review queue, attachment-only, form-letter representatives,
cited-document frequency).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from lib import data as D

st.set_page_config(page_title="Export", layout="wide")
st.title("Export")

df = D.get_data()
if df is None:
    st.info("No data loaded yet. Go to the Dashboard page and load comments first.")
    st.stop()

ID = D.id_col(df); TXT = D.text_col(df); AC = D.attach_col(df)
topics = D.detect_topics(df)

BASE_COLS = [c for c in [ID, "stance_norm", "human_reviewed", "strength",
                         "strength_score", "topics_str",
                         "evidence", "evidence_type", "cited_str", "n_cited",
                         "flag_reason", "flag_disagree", "llm_conf",
                         "regex_stance", "_has_attachment", AC, "ocr_status", "ocr_text", TXT]
             if c and c in df.columns]


def to_csv(frame, cols=None):
    cols = cols or [c for c in BASE_COLS if c in frame.columns]
    # utf-8-sig so Excel opens non-ASCII characters correctly
    return frame[cols].to_csv(index=False).encode("utf-8-sig")


# ------------------------------------------------------------ 1. custom slice
st.subheader("1 · Build a custom export")
c1, c2 = st.columns(2)
sel_labels = c1.multiselect("Topics (any of)", [D.label_for(t) for t in topics])
sel_codes = [{D.label_for(t): t for t in topics}[l] for l in sel_labels]
sel_stance = c2.multiselect("Stance", df["stance_norm"].unique().tolist(),
                            help="Leave empty for all stances.")
c3, c4 = st.columns(2)
sel_docs = c3.multiselect("Cites official document", D.all_cited_documents(df))
att_mode = c4.selectbox("Attachments", ["All comments", "Only with attachment",
                                        "Only 'see attached'"])
ev_mode = "All comments"
if bool(df.get("_has_evidence_info", pd.Series(dtype=bool)).any()):
    ev_mode = st.selectbox("Supporting evidence",
                           ["All comments", "Only with evidence", "Only opinion-only"])
sel_strength = st.multiselect("Intensity (degree of support/oppose)",
                             ["strong", "moderate", "mild"],
                             default=["strong", "moderate", "mild"])
dedup = st.checkbox("Distinct letters only (group form letters)", value=False)

sub = df.copy()
if sel_stance:
    sub = sub[sub["stance_norm"].isin(sel_stance)]
if sel_codes:
    sub = sub[sub[sel_codes].fillna(0).astype(bool).sum(axis=1) > 0]
if sel_docs:
    sub = sub[sub["cited_list"].apply(lambda l: any(d in l for d in sel_docs))]
if att_mode == "Only with attachment":
    sub = sub[sub["_has_attachment"]]
elif att_mode == "Only 'see attached'":
    sub = sub[sub["_attachment_only"]]
if ev_mode == "Only with evidence":
    sub = sub[sub["evidence"].fillna(False).astype(bool)]
elif ev_mode == "Only opinion-only":
    sub = sub[~sub["evidence"].fillna(False).astype(bool)]
if sel_strength:
    sub = sub[sub["strength"].isin(sel_strength)]
if dedup:
    sub = D.unique_frame(sub)

st.write(f"**{len(sub):,}** comments in this slice.")
st.download_button("Download custom slice (CSV)", to_csv(sub),
                   "comments_custom.csv", "text/csv", type="primary")

st.divider()

# ------------------------------------------------------------ 2. ready-made
st.subheader("2 · Ready-made exports")
g1, g2 = st.columns(2)

with g1:
    st.markdown("**Everything (full labelled dataset)**")
    st.download_button(f"All {len(df):,} comments",
                       to_csv(df), "comments_all.csv", "text/csv")

    rev = df[df["auto_flag"]]
    st.markdown("**Comments needing review**")
    st.download_button(f"{len(rev):,} to review",
                       to_csv(rev), "comments_to_review.csv", "text/csv")

    att = df[df["_has_attachment"]]
    st.markdown("**Comments with attachments (incl. links)**")
    st.download_button(f"{len(att):,} with attachments",
                       to_csv(att), "comments_with_attachments.csv", "text/csv")

with g2:
    seen = df[df["_attachment_only"]]
    st.markdown("**'See attached' comments only**")
    st.download_button(f"{len(seen):,} see-attached",
                       to_csv(seen), "comments_see_attached.csv", "text/csv")

    reps = D.unique_frame(df)
    st.markdown("**Distinct letters (form-letter representatives)**")
    st.download_button(f"{len(reps):,} distinct letters",
                       to_csv(reps), "comments_distinct.csv", "text/csv")

    _gsize = df.groupby("_group")["_group"].transform("size")
    ones = df[_gsize == 1]
    st.markdown("**One-of-a-kind letters (no template match)**")
    st.download_button(f"{len(ones):,} one-of-a-kind letters",
                       to_csv(ones), "comments_one_of_a_kind.csv", "text/csv")

    freq = D.cited_frequency(df, weighted=True)
    if len(freq):
        st.markdown("**Cited-document frequency table**")
        st.download_button(f"{len(freq):,} documents",
                           freq.to_csv(index=False).encode("utf-8-sig"),
                           "cited_documents_frequency.csv", "text/csv")

st.divider()

# ------------------------------------------------------------ 3. by-topic bundle
st.subheader("3 · One CSV per topic (quick pick)")
if topics:
    choice = st.selectbox("Topic", [D.label_for(t) for t in topics])
    code = {D.label_for(t): t for t in topics}[choice]
    sub = df[df[code].fillna(0).astype(bool)]
    st.write(f"**{len(sub):,}** comments on {choice} "
             f"({int((sub['stance_norm'] == 'Oppose').sum()):,} oppose, "
             f"{int((sub['stance_norm'] == 'Support').sum()):,} support).")
    st.download_button(f"Download {choice} comments",
                       to_csv(sub), f"comments_{code}.csv", "text/csv")
else:
    st.info("No topic columns in this dataset.")
