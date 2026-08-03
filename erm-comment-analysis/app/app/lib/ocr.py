"""
In-app OCR for comment attachments.

For a raw upload the 'see attached' comments have their real content in an
attachment. For each such comment we download EVERY attachment URL (not just
the first), recognise PDFs and images by their file signature (not just the
extension), transcribe them with the model, and join the texts. Every row gets
an `ocr_status`, so the app can say WHY an attachment could not be read
instead of leaving a silent gap:

  ok               at least one attachment transcribed
  no_url           the attachment cell held no usable http link
  download_failed  every link timed out / was refused
  unsupported      downloads worked but none were a PDF or an image
  empty            transcription returned no text (blank or illegible scan)

Downloads and transcriptions run in a small thread pool; per-URL failures are
retried once. Uses the user's own key.
"""
from __future__ import annotations
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from lib.config import GEMINI_MODEL, endpoint

_BROWSER = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/137.0 Safari/537.36"),
    "Referer": "https://www.regulations.gov/",
    "Accept": "application/pdf,image/*,application/octet-stream,*/*",
}
_OCR_PROMPT = ("You are an OCR system. Transcribe ALL text in this document exactly as "
               "written, including handwriting, letterhead and signatures. Return ONLY "
               "the transcribed text, no commentary. If it is blank or illegible, return "
               "an empty string.")

MAX_URLS_PER_COMMENT = 5
MAX_BYTES = 20 * 1024 * 1024   # skip files larger than 20 MB


def all_urls(cell) -> list[str]:
    """Every http(s) link in an attachment cell ('a | b, c' separators)."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    parts = []
    for chunk in str(cell).split("|"):
        parts.extend(chunk.split(","))
    urls = [p.strip() for p in parts]
    return [u for u in urls if u.startswith("http")][:MAX_URLS_PER_COMMENT]


def first_url(cell) -> str | None:      # kept for backward compatibility
    urls = all_urls(cell)
    return urls[0] if urls else None


def sniff_mime(data: bytes) -> str | None:
    """Recognise PDFs and common image formats by file signature."""
    if not data:
        return None
    if data[:4] == b"%PDF":
        return "application/pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return None


def ocr_bytes(data: bytes, mime: str, key: str, model: str = GEMINI_MODEL) -> str:
    b64 = base64.b64encode(data).decode()
    body = {"contents": [{"parts": [
        {"text": _OCR_PROMPT},
        {"inline_data": {"mime_type": mime, "data": b64}},
    ]}], "generationConfig": {"temperature": 0}}
    for k in range(3):
        try:
            r = requests.post(endpoint(model), params={"key": key},
                              headers={"Content-Type": "application/json"},
                              data=json.dumps(body), timeout=120)
            if r.status_code == 200:
                return (r.json()["candidates"][0]["content"]["parts"][0]["text"]).strip()
            if r.status_code in (429, 503):
                time.sleep(2 ** k + 1); continue
            return ""
        except Exception:
            time.sleep(2 ** k)
    return ""


# kept for backward compatibility with earlier callers/tests
def ocr_pdf_bytes(pdf_bytes: bytes, key: str, model: str = GEMINI_MODEL) -> str:
    return ocr_bytes(pdf_bytes, "application/pdf", key, model)


def _download(url: str) -> bytes | None:
    for attempt in range(2):               # one retry on transient failures
        try:
            resp = requests.get(url, headers=_BROWSER, timeout=60,
                                allow_redirects=True)
            if resp.status_code == 200 and resp.content:
                if len(resp.content) > MAX_BYTES:
                    return b""             # too large: treated as unsupported
                return resp.content
        except Exception:
            pass
        time.sleep(1 + attempt)
    return None


def _read_one_comment(urls: list[str], key: str, model: str) -> tuple[str, str]:
    """(joined transcription, status) for one comment's attachment URLs."""
    if not urls:
        return "", "no_url"
    texts, any_download, any_supported = [], False, False
    for u in urls:
        data = _download(u)
        if data is None:
            continue
        any_download = True
        mime = sniff_mime(data)
        if not mime:
            continue
        any_supported = True
        t = ocr_bytes(data, mime, key, model)
        if t:
            texts.append(t)
    if texts:
        return "\n\n".join(texts), "ok"
    if not any_download:
        return "", "download_failed"
    if not any_supported:
        return "", "unsupported"
    return "", "empty"


def ocr_attachments(df: pd.DataFrame, url_col: str, key: str, rows_mask=None,
                    progress=None, model: str = GEMINI_MODEL, workers: int = 4):
    """Read the chosen rows' attachments concurrently.

    Returns (text_series, status_series); both empty-string for rows outside
    the mask. `progress(done, total)` is called from the calling thread."""
    out_text = pd.Series("", index=df.index, dtype=object)
    out_status = pd.Series("", index=df.index, dtype=object)
    idx = df.index[rows_mask] if rows_mask is not None else df.index
    targets = [(i, all_urls(df.at[i, url_col])) for i in idx]
    total = len(targets)
    if not total:
        return out_text, out_status
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_read_one_comment, urls, key, model): i
                   for i, urls in targets}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                text, status = fut.result()
            except Exception:
                text, status = "", "download_failed"
            out_text.at[i] = text
            out_status.at[i] = status
            done += 1
            if progress:
                progress(done, total)
    return out_text, out_status
