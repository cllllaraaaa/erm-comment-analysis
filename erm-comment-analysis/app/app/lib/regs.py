"""
Optional input: pull public comments straight from Regulations.gov (API v4)
using the user's own Regulations.gov API key + a docket ID.

Notes / limits (surfaced in the UI):
- 1,000 requests/hour; list pagination is capped at 20 pages x 250 = 5,000 per query.
- The comment LIST only carries metadata; the full comment text + attachments
  need one detail call each, so a full text pull is bounded by a record cap.
"""
from __future__ import annotations
import time
import pandas as pd
import requests

BASE = "https://api.regulations.gov/v4"


def _headers(key: str) -> dict:
    return {"X-Api-Key": key, "Content-Type": "application/json"}


def list_comment_ids(api_key: str, docket_id: str, max_records: int = 1000,
                     progress=None) -> list[str]:
    """Page through the comment list for a docket, return comment IDs (metadata)."""
    ids: list[str] = []
    page = 1
    while len(ids) < max_records and page <= 20:
        params = {
            "filter[docketId]": docket_id,
            "page[size]": 250,
            "page[number]": page,
            "sort": "postedDate",
        }
        r = requests.get(f"{BASE}/comments", headers=_headers(api_key),
                         params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Regulations.gov error {r.status_code}: {r.text[:200]}")
        data = r.json().get("data", [])
        if not data:
            break
        ids.extend(d["id"] for d in data)
        if progress:
            progress(min(len(ids), max_records), max_records)
        page += 1
        time.sleep(0.2)
    return ids[:max_records]


def fetch_comment(api_key: str, comment_id: str) -> dict:
    """One comment's full text + submitter + attachment URLs."""
    r = requests.get(f"{BASE}/comments/{comment_id}", headers=_headers(api_key),
                     params={"include": "attachments"}, timeout=60)
    if r.status_code != 200:
        return {"Document ID": comment_id, "comment_text": "", "posted_date": None,
                "organization_clean": None, "state_clean": None,
                "Attachment Files": "", "has_attachment": False}
    j = r.json()
    a = j.get("data", {}).get("attributes", {})
    urls = []
    for inc in j.get("included", []):
        for fmt in (inc.get("attributes", {}).get("fileFormats") or []):
            if fmt.get("fileUrl"):
                urls.append(fmt["fileUrl"])
    return {
        "Document ID": comment_id,
        "comment_text": a.get("comment") or "",
        "posted_date": a.get("postedDate"),
        "organization_clean": a.get("organization"),
        "state_clean": a.get("stateProvinceRegion"),
        "Attachment Files": "|".join(urls),
        "has_attachment": bool(urls),
    }


def fetch_docket_comments(api_key: str, docket_id: str, max_records: int = 500,
                          progress=None) -> pd.DataFrame:
    """List + detail-fetch up to `max_records` comments into a raw DataFrame."""
    ids = list_comment_ids(api_key, docket_id, max_records=max_records)
    rows = []
    for i, cid in enumerate(ids, 1):
        rows.append(fetch_comment(api_key, cid))
        if progress:
            progress(i, len(ids))
        time.sleep(0.2)
    return pd.DataFrame(rows)
