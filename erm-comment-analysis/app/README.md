# Public Comment Analysis — Consultant Dashboard

Streamlit app for analysing public comments on a proposed project.
Bring comments in (upload a CSV/Excel **or** pull a docket from Regulations.gov
by API), and the app works out **topics, stance, supporting evidence, and cited
official documents** . Then explore charts,
browse and review comments, export slices, and ask an AI assistant that answers
from the real data.

## Setup
```bash
poetry config virtualenvs.in-project true
poetry install
```
Optional: `pip install spacy` upgrades the keyword-matching layer from regex to
spaCy's PhraseMatcher (no model download needed). Optional: `pip install pytest`
then `pytest` runs the unit tests in `tests/`.

## Run
```bash
poetry run streamlit run app/Dashboard.py
```
Opens at http://localhost:8501.

## Getting data in (Dashboard page)
The first screen offers two tabs, and after data is loaded the same two stay
available under **Data source**, so you can always switch:

- **Fetch by API** — the normal route: enter a docket ID and your free
  Regulations.gov API key and pull the comments directly, no CSV needed. Use
  **Save loaded data (CSV)** afterwards to keep a copy.
- **Upload a file** — CSV/Excel, one row per comment. Only a text column is
  required. If stance/topics are present they're used; if not, click
  **Run analysis** (needs your  API key) and the app labels them. Form
  letters are labelled once, then propagated; comments are labelled in batches
  of eight with several batches in flight, so a 10,000-comment docket takes
  minutes rather than an hour. A model picker offers a fast tier and a more
  accurate tier; use the accurate one for client deliverables. Attachment
  reading covers every attachment link per comment, PDFs and images alike,
  and every unread attachment carries a reason (download failed, unsupported
  format, blank scan) shown on the Review page and in exports. When the run
  finishes, download the labelled CSV immediately; labels live in the browser
  session only, and re-uploading that CSV later skips the API cost entirely.
Cited documents and form-letter grouping are always computed automatically.
Grouping catches near-identical letters, not just exact copies: the same
template with a personal sentence added still counts as one letter. Switch to
exact-only matching under Grouping settings on the Dashboard. The letters that
match no template at all are available as a filter on Browse (one-of-a-kind
letters) and as a ready-made export; those are typically the individually
written comments that deserve close reading.

## Working on ANY project 
Your own topics are always one click away: the topic editor appears on the
analysis screen for raw files, and under **Topics** on the Dashboard for
already-labelled files, with a re-run button that re-labels everything against
your list. You define topics in any of three ways:

1. **Paste the EIS's own topic-area list** — one line per area (`Air Quality`,
   `Noise`, `Cultural Resources`…). Public comments respond to those areas, so
   they're the natural grouping for a response document.
2. **✨ Build a domain pack from this data** — the AI reads a spread sample of
   the comments and proposes topics *with keywords* and the *laws/agencies
   commenters cite*, so cited-document detection and the keyword cross-check
   follow the new subject area too.
3. **Load a saved pack (.json)** from a previous docket, or edit by hand.

A domain pack bundles `topics` (code, label, keywords, description) and
`acronyms` (cited statutes/agencies with categories). Save it with the
**⬇ Save this domain pack** button and reuse it next time. The pre-filled
oil/gas schema is only an example template.

**Keyword priority (optional toggle)** — when ON, a comment that literally
contains one of a topic's keywords is guaranteed that topic label even if the
AI missed it (a recall boost). It can be toggled from the Dashboard after
analysis too, and turning it off restores the AI-only labels.

## Pages
- **Dashboard** — metrics + charts (topics, stance, stance-by-topic, most-cited
  documents, evidence, intensity, regions, form-letter concentration). Toggle
  *by volume* vs *distinct letters*.
- **Browse Comments** — filter by topic, stance, **supporting evidence**,
  **official document cited**, one-of-a-kind letters, and attachments; search
  by text or comment ID. Key terms are highlighted inside each comment (topic
  keywords, opposing/supporting wording, cited documents), with a toggle and
  colour legend in the sidebar, and long comments expand to full text.
- **Review** — letters needing a human eye, one card per distinct letter: a
  flagged template sent by hundreds of people appears once, marked "sent by N
  people", and your decision covers every copy. Correct the stance on the card
  and mark it reviewed; reviewed letters leave the queue, stay resolved, and
  carry a human_reviewed column in exports. Filter by why letters are flagged:
  unclear stance, no topic, attachment not read, not labelled, low model
  confidence, keyword/model disagreement.
- **Export** — build a custom slice (topic × stance × evidence × document ×
  attachment) or grab ready-made lists. CSVs are `utf-8-sig`, so they open
  cleanly in Excel.
- **Assistant** — one chat box; ask questions **and** generate reports. The
  assistant remembers the conversation and can query the loaded dataset (count
  keyword mentions, pull real example comments with their IDs), so counts and
  quotes are grounded, not guessed. Bring your own API key.

## What the analysis produces per comment
- **Stance** — Oppose / Support / Unclear toward the proposed action itself
  (politely-worded objections count as opposition).
- **Topics** — multi-label, restricted to your schema.
- **Evidence** — does the comment back its position with data, a study, a law
  or document, first-hand experience, or professional expertise? Filterable
  everywhere; evidence-backed comments usually deserve a substantive response.
- **Intensity** — how forcefully it's written (text signal, cross-checked with
  the AI's rating). Exploratory — no gold standard.
- **Cited documents** — statutes / agencies / EIS documents, canonicalised.
- **Review flags** — see the Review page list above.

## Notes
- The de-duplication convention: the unit is the submission record, with form
  letters grouped. "N people sent this letter" is shown as popularity, never summed
  into topic/stance distributions. Switch views on the Dashboard.
- API keys are entered in the app, kept in session only, and never saved to disk.
- The model is configured in one place: `app/lib/config.py`.
