"""
Review: comments that need a human eye, shown as cards.

One card per DISTINCT letter by default: a flagged template sent by 500 people
appears once, with a "sent by N people" note, and any decision you make covers
every copy. Set the stance (or keep it) and mark the letter reviewed; reviewed
letters leave the queue and stay resolved, including in exports.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from lib import data as D
from lib import domain, highlight

st.set_page_config(page_title="Review", layout="wide")
st.title("Comments to review")

df = D.get_data()
if df is None:
    st.info("No data loaded yet. Go to the Dashboard page and load comments first.")
    st.stop()

ID = D.id_col(df); TXT = D.text_col(df); AC = D.attach_col(df)
manual = st.session_state.get("flagged_ids", set())

# ------------------------------------------------------------ reason filter
REASONS = {
    "Unclear stance": "flag_unclear_stance",
    "No topic detected": "flag_no_topic",
    "Attachment only (content not read)": "flag_needs_ocr",
    "Not labelled": "flag_not_labelled",
    "Low model confidence": "flag_low_conf",
}
st.sidebar.header("Why flagged")
chosen = [col for lab, col in REASONS.items()
          if col in df.columns and st.sidebar.checkbox(lab, value=True)]
inc_disagree = st.sidebar.checkbox("Keyword and model disagree", value=True,
                                  help="The keyword layer caught a term the model did not tag, "
                                       "or the two disagree on stance.")
inc_manual = st.sidebar.checkbox("Flagged by you", value=True)
one_per_letter = st.sidebar.checkbox(
    "One card per distinct letter", value=True,
    help="A flagged template letter sent by many people appears once. Your "
         "decision on it covers every copy.")
id_search = st.sidebar.text_input("Find by comment ID", "")
hl_on = st.sidebar.checkbox(
    "Highlight key terms", value=True,
    help="Marks topic keywords, opposing/supporting wording, and cited "
         "documents inside each comment.")
if hl_on:
    st.sidebar.markdown(highlight.LEGEND_HTML, unsafe_allow_html=True)

_pack = D.active_pack()
_hl = highlight.Highlighter(
    keyword_map=domain.keyword_map(_pack) if _pack else None,
    extra_whitelist=domain.citation_config(_pack).whitelist if _pack else None)

def show_text(txt: str):
    if hl_on:
        st.markdown(_hl.render(txt), unsafe_allow_html=True)
    else:
        st.write(txt)

mask = pd.Series(False, index=df.index)
for col in chosen:
    mask = mask | df[col]
if inc_disagree:
    mask = mask | df["flag_disagree"]
if inc_manual and manual:
    mask = mask | df[ID].astype(str).isin(manual)
r = df[mask].copy()
if id_search.strip():
    r = r[r[ID].astype(str).str.contains(id_search.strip(), case=False, na=False)]


def _why(row):
    if row["flag_reason"]:
        return row["flag_reason"]
    if row.get("flag_disagree"):
        return "keyword and model disagree"
    if str(row[ID]) in manual:
        return "flagged by you"
    return ""
r["why"] = r.apply(_why, axis=1)

_gsize = df.groupby("_group")["_group"].transform("size")
n_match = len(r)
n_letters = r["_group"].nunique()
if one_per_letter:
    r = r.drop_duplicates("_group")

n_reviewed = int(df.get("human_reviewed", pd.Series(False, index=df.index))
                 .astype(bool).groupby(df["_group"]).any().sum())
st.write(f"**{n_match:,}** flagged comments, **{n_letters:,}** distinct letters "
         f"to review. {n_reviewed:,} letters reviewed so far.")

# breakdown chips
b1, b2, b3, b4, b5, b6 = st.columns(6)
b1.metric("Unclear stance", int(df["flag_unclear_stance"].sum()))
b2.metric("No topic", int(df["flag_no_topic"].sum()))
b3.metric("Needs reading", int(df["flag_needs_ocr"].sum()))
b4.metric("Not labelled", int(df["flag_not_labelled"].sum()))
b5.metric("Low confidence", int(df.get("flag_low_conf", pd.Series(False, index=df.index)).sum()))
b6.metric("Keyword conflict", int(df["flag_disagree"].sum()))

st.divider()
if len(r) == 0:
    st.success("Nothing matches those filters.")
    st.stop()

# download (comprehensive)
dl_cols = [c for c in [ID, "stance_norm", "topics_str", "cited_str", "flag_reason",
                       "human_reviewed", AC, TXT] if c and c in r.columns]
st.download_button("Download this review list (CSV)",
                   r[dl_cols].to_csv(index=False).encode("utf-8-sig"),
                   "comments_to_review.csv", "text/csv")


# ------------------------------------------------------------ review actions
def _apply_review(gid, new_stance: str):
    cur = D.get_data()
    m = cur["_group"] == gid
    if new_stance in ("Oppose", "Support", "Unclear"):
        cur.loc[m, "stance_norm"] = new_stance
    if "human_reviewed" not in cur.columns:
        cur["human_reviewed"] = False
    cur.loc[m, "human_reviewed"] = True
    for c in ["flag_unclear_stance", "flag_no_topic", "flag_needs_ocr",
              "flag_not_labelled", "flag_low_conf", "flag_disagree"]:
        if c in cur.columns:
            cur.loc[m, c] = False
    cur.loc[m, "auto_flag"] = False
    cur.loc[m, "flag_reason"] = ""
    D.set_data(cur)


# ------------------------------------------------------------ paging (bottom)
PAGE = 25
total_pages = max(1, (len(r) + PAGE - 1) // PAGE)
if st.session_state.get("review_page", 1) > total_pages:
    st.session_state["review_page"] = 1
page = int(st.session_state.get("review_page", 1))
chunk = r.iloc[(page - 1) * PAGE: page * PAGE]

for _, row in chunk.iterrows():
    rid = str(row[ID])
    gid = row["_group"]
    copies = int(_gsize.get(row.name, 1))
    with st.container(border=True):
        c1, c2 = st.columns([4.2, 1])
        with c1:
            stance = row["stance_norm"]
            colour = D.STANCE_COLOUR.get(stance, D.SAGE)
            strg = row.get("strength", "")
            head = (f"**{rid}** · <span style='color:{colour};font-weight:600'>{stance}</span>"
                    f" <span style='color:{colour}'>{D.strength_dots(strg)}</span>"
                    f" · <span style='color:{D.CLAY}'>{row['why']}</span>"
                    + (f" · {row['topics_str']}" if row.get("topics_str") else ""))
            if copies > 1:
                head += (f" · <span style='color:#5b6b63;font-size:12.5px'>"
                         f"sent by {copies:,} people</span>")
            st.markdown(head, unsafe_allow_html=True)
            t = str(row.get(TXT) or "").strip()
            if len(t) >= 20:
                st.write(t[:400] + ("…" if len(t) > 400 else ""))
            else:
                ocr = str(row.get("ocr_text") or "").strip()
                st.write("(see attached) " + ocr[:360] if ocr
                         else "(see attached: content is in the attachment)")
            _full = t if len(t) >= 20 else str(row.get("ocr_text") or "").strip()
            if len(_full) > 400:
                with st.expander("Show full comment"):
                    st.write(_full)
            if row.get("cited_str"):
                st.markdown(f"<span style='color:{D.GOLD};font-size:13px'>"
                            f"Cites: {row['cited_str']}</span>", unsafe_allow_html=True)
            url = D.attachment_url(row, AC)
            if url:
                st.markdown(f"[Open attachment]({url})")
        with c2:
            choice = st.selectbox(
                "Stance", ["Keep as is", "Oppose", "Support", "Unclear"],
                key=f"rv_{rid}", label_visibility="collapsed",
                help="Correct the stance if the model got it wrong. Applies to "
                     "every copy of this letter.")
            if st.button("Mark reviewed", key=f"mr_{rid}", use_container_width=True,
                         help="Removes this letter and all its copies from the "
                              "review queue. The correction is kept in exports."):
                _apply_review(gid, choice)
                st.rerun()

if total_pages > 1:
    st.divider()
    a, b, c = st.columns([1, 2, 1])
    if a.button("Previous", disabled=page <= 1, use_container_width=True):
        st.session_state["review_page"] = page - 1; st.rerun()
    b.markdown(f"<div style='text-align:center;padding-top:6px'>Page "
               f"<b>{page}</b> of {total_pages}</div>", unsafe_allow_html=True)
    if c.button("Next", disabled=page >= total_pages, use_container_width=True):
        st.session_state["review_page"] = page + 1; st.rerun()
    st.number_input("Jump to page", 1, total_pages, key="review_page")
