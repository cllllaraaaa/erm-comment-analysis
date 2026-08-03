"""
AI assistant — one chat box. Ask questions AND generate reports here (e.g.
"write a report on opposition about oil spill risk"); no separate report tool.

What's new in v11:
* the model receives the CHAT HISTORY, so follow-up questions work;
* the model can query the REAL dataset (count keyword mentions, pull example
  comments for a topic/stance) through a tiny tool step — answers like
  "how many comments mention sea turtles" come from the dataframe, not a guess;
* reports quote real comments with their IDs, so every claim is traceable.

Bring your own API key (never hardcoded, never saved to disk).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import requests
import streamlit as st
from lib import data as D
from lib.config import endpoint

st.set_page_config(page_title="Assistant", layout="wide")
st.title("Assistant")

df = D.get_data()
if df is None:
    st.info("No data loaded yet. Go to the Dashboard page and load comments first.")
    st.stop()

st.sidebar.header("Assistant")
api_key = st.sidebar.text_input(
    "API key", type="password",
    help="Your Gemini API key, used only for this conversation; never stored.")
st.session_state["api_key"] = api_key

TXT = D.text_col(df)
ID = D.id_col(df)

summary = D.data_summary(df)
SYSTEM = (
    "You are an assistant helping an environmental consultant understand public "
    "comments submitted on a proposed project during its federal review. Answer "
    "clearly and plainly. You can answer questions AND write reports on request "
    "(for a chosen topic, stance, or the whole dataset). Base every answer on the "
    "dataset summary and on tool results; if something isn't in them, say so.\n\n"
    "TOOLS: when you need an exact number or real examples from the dataset, "
    "reply with ONLY a JSON object (no prose), one of:\n"
    '  {"tool": "count", "keyword": "<word or phrase>"}\n'
    '      -> how many comments contain that text (case-insensitive)\n'
    '  {"tool": "examples", "stance": "Oppose|Support|Unclear|null", '
    '"topic_label": "<topic label or null>", "keyword": "<text or null>", "n": 5}\n'
    '      -> up to n real comments (ID + excerpt) matching those filters\n'
    "The tool result will be sent back to you; then answer the user in plain "
    "language. Use at most a few tool calls per question. When you quote a "
    "comment in an answer or report, include its comment ID so the consultant "
    "can find it.\n\n"
    f"DATASET SUMMARY:\n{summary}\n"
)


# ------------------------------------------------------------------ tools
def _tool_count(keyword: str) -> dict:
    m = df[TXT].fillna("").astype(str).str.contains(str(keyword), case=False,
                                                    na=False, regex=False)
    by_stance = df.loc[m, "stance_norm"].value_counts().to_dict()
    return {"keyword": keyword, "comments_matching": int(m.sum()),
            "of_total": len(df), "by_stance": by_stance}


def _tool_examples(stance=None, topic_label=None, keyword=None, n=5) -> dict:
    sub = df
    if stance and str(stance) != "null":
        sub = sub[sub["stance_norm"] == stance]
    if topic_label and str(topic_label) != "null":
        codes = {D.label_for(t): t for t in D.detect_topics(df)}
        code = codes.get(topic_label)
        if code:
            sub = sub[sub[code].fillna(0).astype(bool)]
    if keyword and str(keyword) != "null":
        sub = sub[sub[TXT].fillna("").astype(str).str.contains(
            str(keyword), case=False, na=False, regex=False)]
    n = max(1, min(int(n or 5), 8))
    out = []
    for _, row in sub.head(n).iterrows():
        out.append({"id": str(row[ID]), "stance": row["stance_norm"],
                    "evidence": bool(row.get("evidence", False)),
                    "excerpt": str(row.get(TXT) or "")[:350]})
    return {"matching_total": len(sub), "examples": out}


def _maybe_tool(reply: str):
    """If the model replied with a tool JSON, run it; else return None."""
    raw = reply.strip().replace("```json", "").replace("```", "").strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        return None
    try:
        o = json.loads(raw)
    except Exception:
        return None
    if not isinstance(o, dict):
        return None
    if o.get("tool") == "count" and o.get("keyword"):
        return _tool_count(o["keyword"])
    if o.get("tool") == "examples":
        return _tool_examples(o.get("stance"), o.get("topic_label"),
                              o.get("keyword"), o.get("n", 5))
    return None


# ------------------------------------------------------------------ model call
def _call(contents: list, key: str) -> str:
    body = {"contents": contents,
            "systemInstruction": {"parts": [{"text": SYSTEM}]}}
    for k in range(3):
        try:
            resp = requests.post(endpoint(), params={"key": key},
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps(body), timeout=60)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            if resp.status_code in (429, 503):   # busy / rate-limited → retry
                time.sleep(2 ** k + 1)
                continue
            return f"[Error {resp.status_code}] {resp.text[:300]}"
        except Exception as e:
            if k == 2:
                return f"[Request failed] {e}"
            time.sleep(2 ** k)
    return "[The model is busy right now. Please try again in a moment.]"


def ask(question: str, key: str) -> str:
    """Send history + question; run up to 3 tool rounds; return the final text."""
    contents = []
    for role, msg in st.session_state["chat"][-8:]:   # last turns as context
        contents.append({"role": "user" if role == "user" else "model",
                         "parts": [{"text": msg}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    for _ in range(3):
        reply = _call(contents, key)
        result = _maybe_tool(reply)
        if result is None:
            return reply
        contents.append({"role": "model", "parts": [{"text": reply}]})
        contents.append({"role": "user", "parts": [{"text":
            "TOOL RESULT (from the real dataset):\n" + json.dumps(result) +
            "\nNow answer the user's question in plain language (or call another "
            "tool if you still need something)."}]})
    return _call(contents, key)


st.caption("Ask about the loaded comments, or ask for a report. Counts and "
           "quotes come from the dataset itself.")

# quick starters
examples = [
    "How many comments mentioned sea turtles?",
    "Write a report on opposition about oil spill risk.",
    "Which official documents do people cite most, and why?",
    "Show me evidence-backed comments that oppose the project.",
]
st.session_state.setdefault("chat", [])
cols = st.columns(len(examples))
pending = None
for i, ex in enumerate(examples):
    if cols[i].button(ex, key=f"ex_{i}"):
        pending = ex

for role, msg in st.session_state["chat"]:
    with st.chat_message(role):
        st.markdown(msg)

typed = st.chat_input("Ask about these comments, or ask for a report…")
q = typed or pending
if q:
    if not api_key:
        st.warning("Add your API key in the sidebar first.")
    else:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                ans = ask(q, api_key)
            st.markdown(ans)
        st.session_state["chat"].append(("user", q))
        st.session_state["chat"].append(("assistant", ans))

if st.session_state["chat"]:
    last = st.session_state["chat"][-1][1]
    st.download_button("Download last answer (text)",
                       last.encode("utf-8"), "assistant_report.txt", "text/plain")
    if st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()
