"""
Browse Comments — filter (topic, stance, official document cited, attachments),
search by text or comment ID, read comments as cards with attachment links,
flag for review. Pagination sits at the bottom.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from lib import data as D
from lib import domain, highlight

st.set_page_config(page_title="Browse Comments", layout="wide")
st.title("Browse comments")

df = D.get_data()
if df is None:
    st.info("No data loaded yet. Go to the Dashboard page and load comments first.")
    st.stop()

st.session_state.setdefault("flagged_ids", set())
ID = D.id_col(df); TXT = D.text_col(df); AC = D.attach_col(df)
topics = D.detect_topics(df)

# ------------------------------------------------------------ filters
st.sidebar.header("Filter")
sel_labels = st.sidebar.multiselect("Topic", [D.label_for(t) for t in topics],
                                    help="Leave empty for all topics.")
sel_codes = [{D.label_for(t): t for t in topics}[l] for l in sel_labels]

stance_opts = df["stance_norm"].unique().tolist()
sel_stance = st.sidebar.multiselect("Stance", stance_opts,
                                    help="Leave empty to show all stances.")

cited_opts = D.all_cited_documents(df)
sel_docs = st.sidebar.multiselect(
    "Cites official document", cited_opts,
    help="Show only comments that reference a specific statute / EIS / agency, e.g. NEPA.")

ev_mode = "All"
if bool(df.get("_has_evidence_info", pd.Series(dtype=bool)).any()):
    ev_mode = st.sidebar.selectbox(
        "Supporting evidence", ["All", "With evidence", "Opinion only"],
        help="Whether the comment backs its position with data, studies, laws, "
             "first-hand or professional experience (AI-labelled).")

only_unique = st.sidebar.checkbox(
    "Only one-of-a-kind letters", value=False,
    help="Letters that match no template: not part of any form-letter group. "
         "These are usually the individually-written comments worth reading "
         "closely.")
only_attach = st.sidebar.checkbox("Only comments with an attachment", value=False)
only_seeattach = st.sidebar.checkbox("Only 'see attached' comments", value=False)
search = st.sidebar.text_input("Text contains…", "")
id_search = st.sidebar.text_input("Find by comment ID", "",
                                 help="Type a full or partial comment ID.")
hl_on = st.sidebar.checkbox(
    "Highlight key terms", value=True,
    help="Marks topic keywords, opposing/supporting wording, and cited "
         "documents inside each comment, using the same matching rules the "
         "analysis itself uses.")
if hl_on:
    st.sidebar.markdown(highlight.LEGEND_HTML, unsafe_allow_html=True)

sort_by = st.sidebar.selectbox("Sort by",
                              ["Default", "Strongest first", "Most documents cited",
                               "Most unclear first"])

# ------------------------------------------------------------ apply
f = df.copy()
if sel_stance:                        # empty = all stances
    f = f[f["stance_norm"].isin(sel_stance)]
if sel_codes:
    f = f[f[sel_codes].fillna(0).astype(bool).sum(axis=1) > 0]
if sel_docs:
    f = f[f["cited_list"].apply(lambda l: any(d in l for d in sel_docs))]
if only_unique:
    _gsize = df.groupby("_group")["_group"].transform("size")
    f = f[f.index.map(_gsize) == 1]
if ev_mode == "With evidence":
    f = f[f["evidence"].fillna(False).astype(bool)]
elif ev_mode == "Opinion only":
    f = f[~f["evidence"].fillna(False).astype(bool)]
if only_attach and "_has_attachment" in f.columns:
    f = f[f["_has_attachment"]]
if only_seeattach and "_attachment_only" in f.columns:
    f = f[f["_attachment_only"]]
if search.strip() and TXT in f.columns:
    f = f[f[TXT].astype(str).str.contains(search, case=False, na=False)]
if id_search.strip() and ID in f.columns:
    f = f[f[ID].astype(str).str.contains(id_search.strip(), case=False, na=False)]

if sort_by == "Strongest first":
    f = f.sort_values("strength_score", ascending=False)
elif sort_by == "Most documents cited":
    f = f.sort_values("n_cited", ascending=False)
elif sort_by == "Most unclear first":
    f = (f.assign(_unc=(f["stance_norm"] == "Unclear").astype(int))
           .sort_values(["_unc", "strength_score"], ascending=[False, True])
           .drop(columns="_unc"))

st.write(f"**{len(f):,}** comments match your filter.")

_pack = D.active_pack()
_hl = highlight.Highlighter(
    keyword_map=domain.keyword_map(_pack) if _pack else None,
    extra_whitelist=domain.citation_config(_pack).whitelist if _pack else None)

def show_text(txt: str):
    if hl_on:
        st.markdown(_hl.render(txt), unsafe_allow_html=True)
    else:
        st.write(txt)

# ------------------------------------------------------------ paging (state)
PAGE = 25
total_pages = max(1, (len(f) + PAGE - 1) // PAGE)
if st.session_state.get("browse_page", 1) > total_pages:
    st.session_state["browse_page"] = 1
page = int(st.session_state.get("browse_page", 1))
chunk = f.iloc[(page - 1) * PAGE: page * PAGE]


def display_text(row) -> str:
    t = str(row.get(TXT) or "").strip()
    if len(t) >= 20:
        return t[:420] + ("…" if len(t) > 420 else "")
    ocr = str(row.get("ocr_text") or "").strip()
    if ocr:
        return "(see attached) " + ocr[:380] + ("…" if len(ocr) > 380 else "")
    return "(see attached: content is in the attachment)"


for _, row in chunk.iterrows():
    rid = str(row[ID])
    with st.container(border=True):
        c1, c2 = st.columns([6, 1])
        with c1:
            stance = row["stance_norm"]
            colour = D.STANCE_COLOUR.get(stance, D.SAGE)
            strg = row.get("strength", "")
            head = (f"**{rid}** · <span style='color:{colour};font-weight:600'>{stance}</span>"
                    f" <span style='color:{colour}' title='intensity'>{D.strength_dots(strg)}</span>"
                    f" <span style='color:#8a8a8a;font-size:12px'>{strg}</span>")
            if row.get("topics_str"):
                head += f" · {row['topics_str']}"
            if bool(row.get("evidence")):
                head += (f" · <span style='color:{D.GOLD};font-size:12px' "
                         f"title='backs its position with evidence'>evidence</span>")
            st.markdown(head, unsafe_allow_html=True)
            show_text(display_text(row))
            _full = str(row.get(TXT) or "").strip()
            if len(_full) < 20:
                _full = str(row.get("ocr_text") or "").strip()
            if len(_full) > 420:
                with st.expander("Show full comment"):
                    show_text(_full)
            if row.get("cited_str"):
                st.markdown(f"<span style='color:{D.GOLD};font-size:13px'>"
                            f"Cites: {row['cited_str']}</span>", unsafe_allow_html=True)
            url = D.attachment_url(row, AC)
            if url:
                st.markdown(f"[Open attachment]({url})")
        with c2:
            if rid in st.session_state["flagged_ids"]:
                if st.button("Unflag", key=f"unflag_{rid}"):
                    st.session_state["flagged_ids"].discard(rid); st.rerun()
            else:
                if st.button("Flag", key=f"flag_{rid}"):
                    st.session_state["flagged_ids"].add(rid); st.rerun()
            if bool(row.get("auto_flag")):
                st.caption("auto-flagged")

# ------------------------------------------------------------ paging (bottom)
if total_pages > 1:
    st.divider()
    a, b, c = st.columns([1, 2, 1])
    if a.button("← Previous", disabled=page <= 1, use_container_width=True):
        st.session_state["browse_page"] = page - 1; st.rerun()
    b.markdown(f"<div style='text-align:center;padding-top:6px'>Page "
               f"<b>{page}</b> of {total_pages}</div>", unsafe_allow_html=True)
    if c.button("Next →", disabled=page >= total_pages, use_container_width=True):
        st.session_state["browse_page"] = page + 1; st.rerun()
    st.number_input("Jump to page", 1, total_pages, key="browse_page")
