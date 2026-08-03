


## Repository map

```
erm-comment-analysis/
├── app/                  Streamlit dashboard (the deliverable)
│   ├── app/              Dashboard.py, pages/, lib/
│   ├── tests/            24 unit tests
│   └── pyproject.toml    poetry dependencies (+ poetry.lock)
├── notebooks/            11 analysis notebooks, 01–11, outputs preserved
├── data/                 gold standard, demo slice, domain pack — see data/README.md
├── outputs/              17 figures reported in the dissertation
└── requirements.txt      dependencies for the notebooks
```

---

## Quick start — run the dashboard

```bash
cd erm-comment-analysis/app
poetry config virtualenvs.in-project true
poetry install
poetry run streamlit run app/Dashboard.py
```

Opens at <http://localhost:8501>. To try it with no API key, choose **Upload a
file** on the Dashboard page and upload `data/sample_labelled_comments.csv` —
434 pre-labelled comments, so every page is populated immediately.

Optional: `pip install spacy` upgrades keyword matching from regex to spaCy's
PhraseMatcher (no model download needed). `pip install pytest && pytest` runs
the unit tests.

---

## Data

The corpus is the public comment docket **MARAD-2019-0093**, 10,267 public
submissions downloaded from Regulations.gov. Comments to a US federal docket are
public records; the files published here are nevertheless pseudonymised, with
commenter names and home towns removed. No private individual is named in this
repository or in the dissertation.

Three files are committed — the 150-comment hand-labelled gold standard, a
434-comment demo slice, and the domain pack. The raw snapshot and the
intermediate CSVs are not, to keep the repository light.
**See [`data/README.md`](data/README.md)** for the full manifest, the
producer/consumer chain, and how to handle the large files.

## Reproducibility

Every notebook is committed **with its cell outputs preserved**, so all reported
figures and tables can be read directly from the notebooks without executing
anything. Re-execution is a separate matter, and depends on which inputs you have:

| Notebook | Re-runs from a clean clone? | Needs |
|---|---|---|
| 01 — raw data EDA | ✗ | `raw_docket_metadata.csv` (download from Regulations.gov) |
| 02 — attachment pipeline | ✗ | output of 01, plus network access to fetch attachments |
| 03 — NLP baseline labelling | ✗ | `appendix_c_*.csv` |
| 04 — topic comparison | ✗ | outputs of 01 and 03 |
| 05 — LLM labelling + evaluation | partial | `cleaned_comments_ver2.csv` + `GEMINI_KEY`. The gold standard it evaluates against **is** committed. |
| 06 — regex scenario | partial | `cleaned_comments_ver2.csv`; gold standard committed |
| 07 — BERTopic | ✗ | `cleaned_comments_ver2.csv` |
| 08 — cited documents | ✗ | `cleaned_comments_ver2.csv` |
| 09 — attachment OCR | ✗ | `cleaned_comments_ver2.csv` + `GEMINI_KEY` |
| 10 — weighting comparison | ✗ | `cleaned_comments_ver2.csv`, `full_llm_UNIQUE.csv` |
| 11 — relabel with OCR | ✗ | outputs of 09 |

In short: the **application** runs from a clean clone; the **notebooks** are
readable from a clean clone but need the docket snapshot to re-execute.
`data/README.md` documents how to obtain and place it.

## Method notes

- **LLM labeller:** `gemini-2.5-flash-lite`, zero-shot, temperature 0, JSON mode;
  the full system prompt is reproduced in the dissertation appendix. The model is
  configured in one place, `app/lib/config.py`.
- **Grouping:** exact-duplicate hashing, then MinHash/LSH template families
  verified by Jaccard similarity; one label per distinct letter, propagated to
  family members and marked as propagated.
- **BERTopic:** all-MiniLM-L6-v2 embeddings; UMAP (n_neighbors 15, n_components 5,
  cosine, random_state 42); HDBSCAN (min_cluster_size 15); unigrams and bigrams;
  fitted on unique texts with sizes weighted back to comment counts.
- **Evaluation:** hand-labelled gold standard of 150 comments (single annotator,
  stated as a limitation); accuracy for stance with McNemar's test on paired
  predictions; per-class precision/recall/F1 and micro-averaged F1 for
  multi-label topics.
- **API keys:** entered in the app per session and held in memory only, or
  supplied to notebooks via the `GEMINI_KEY` environment variable.
  **No key is stored in this repository.**

---

## Using the application

**Getting data in (Dashboard page).** Two routes, always switchable under
**Data source**: *Fetch by API* (enter a docket ID and a free Regulations.gov API
key; save a CSV copy afterwards) or *Upload a file* (CSV/Excel, one row per
comment, only a text column required). If uploaded data already carry labels they
are used; otherwise **Run analysis** labels them. Form letters are labelled once
and propagated; comments are labelled in batches of eight with several batches in
flight, so a 10,000-comment docket takes minutes rather than an hour. Attachment
reading covers every attachment link per comment, PDFs and images alike, and
every unread attachment carries a reason shown on the Review page and in exports.
Labels live in the browser session only; download the labelled CSV when the run
finishes, and re-uploading it later skips the API cost entirely. Cited documents
and form-letter grouping are always computed automatically; near-identical
letters count as one template, and one-of-a-kind letters are available as a
filter and a ready-made export.

**Generalising beyond this docket (domain packs).** Topics are defined by pasting
an EIS topic-area list, by asking the model to build a domain pack from a sample
of the data (topics with keywords plus the laws/agencies commenters cite), or by
loading a saved pack (`.json`). The keyword cross-check and citation recogniser
follow the pack, so the transparency layer travels with the domain; the
pre-filled oil/gas schema is only an example. The **keyword-priority toggle**,
when on, guarantees a topic label to any comment containing that topic's literal
keyword, a recall floor under analyst control (dissertation Section 3.9).

**Pages.** *Dashboard* (metrics and charts with a by-volume versus
distinct-letters toggle), *Browse* (filters, text search, and in-text
highlighting of topic keywords, stance wording and cited documents), *Review*
(one card per distinct letter, flagged by unclear stance, no topic, unread
attachment, labelling failure, low confidence, or keyword/model disagreement; a
decision covers every copy and exports carry a `human_reviewed` column), *Export*
(custom slices, `utf-8-sig` CSVs), and *Assistant* (grounded querying and report
drafting that quotes real comment IDs).

**Per-comment outputs.** Stance (oppose/support/unclear, with politely worded
objections counting as opposition), multi-label topics restricted to the schema,
an evidence flag (data, study, law or document, first-hand experience,
expertise), an exploratory intensity signal (no gold standard; no dissertation
finding rests on it), canonicalised cited documents, and review flags.

**Counting convention.** The unit is the submission record, with form letters
grouped. "N people sent this letter" is shown as popularity, never summed into
topic or stance distributions; the by-volume and distinct-letters views are both
reported, as in the dissertation.

