#!/usr/bin/env python3
"""Build source_metadata.json for the local corpus.

The output is bibliography-oriented metadata, not extracted document content.
It is safe to publish when source titles/citation labels may be public, but the
PDFs, chunks, and embeddings themselves are not redistributed.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "documents"
CHUNKS_FILE = ROOT / "chunks.json"
OUT_FILE = ROOT / "source_metadata.json"


def load_app_helpers():
    import sys

    sys.path.insert(0, str(ROOT))
    import app  # noqa: PLC0415

    return app


def infer_year(*values):
    for value in values:
        if not value:
            continue
        matches = re.findall(r"\b(19\d{2}|20\d{2})\b", str(value))
        if matches:
            return matches[-1]
    return None


def infer_doi(*values):
    doi_re = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
    for value in values:
        if not value:
            continue
        match = doi_re.search(str(value))
        if match:
            return match.group(0).rstrip(".")
    return None


def normalize_pdf_metadata(raw):
    if not raw:
        return {}
    clean = {}
    for key, value in raw.items():
        if value is None:
            continue
        key = str(key).strip().lstrip("/")
        if not key:
            continue
        value = str(value).strip()
        if value:
            clean[key] = value
    return clean


def pdf_stats(path):
    try:
        with pdfplumber.open(path) as pdf:
            chars = 0
            for page in pdf.pages[:3]:
                chars += len(page.extract_text(x_tolerance=1, y_tolerance=3) or "")
            metadata = normalize_pdf_metadata(pdf.metadata)
            return {
                "pdf_filename": path.name,
                "page_count": len(pdf.pages),
                "text_extractable": chars > 0,
                "pdf_metadata": metadata,
            }
    except Exception as exc:
        return {
            "pdf_filename": path.name,
            "page_count": None,
            "text_extractable": False,
            "pdf_metadata": {},
            "notes": f"PDF metadata extraction failed: {exc}",
        }


def load_chunk_counts(path):
    if not path.exists():
        return Counter()
    with path.open(encoding="utf-8") as f:
        chunks = json.load(f)
    return Counter(chunk.get("source", "") for chunk in chunks)


def build_entry(source_id, app, chunk_count, pdf_info=None):
    pdf_info = pdf_info or {}
    citation = app.friendly_source_name(source_id)
    title = app.full_source_title(source_id)
    raw_pdf_meta = pdf_info.get("pdf_metadata", {})
    year = infer_year(citation, title, raw_pdf_meta.get("Title"), raw_pdf_meta.get("CreationDate"))
    entry = {
        "source_id": source_id,
        "citation": citation,
        "title": title,
        "year": year,
        "authors": [],
        "publisher_or_institution": None,
        "venue_or_series": None,
        "doi": infer_doi(citation, title, raw_pdf_meta.get("Title"), raw_pdf_meta.get("Subject")),
        "url": None,
        "source_tier": app.source_tier(source_id),
        "document_type": "corpus PDF" if pdf_info else "indexed source",
        "pdf_filename": pdf_info.get("pdf_filename"),
        "page_count": pdf_info.get("page_count"),
        "text_extractable": pdf_info.get("text_extractable"),
        "chunk_count": chunk_count,
        "redistribution_status": "not included in public repository",
        "notes": pdf_info.get("notes", ""),
    }

    author = raw_pdf_meta.get("Author")
    if author and author.lower() not in {"anonymous", "unknown"}:
        entry["authors"] = [part.strip() for part in re.split(r";|,|\band\b", author) if part.strip()]
    return entry


def main():
    parser = argparse.ArgumentParser(description="Build source_metadata.json from local corpus files")
    parser.add_argument("--documents-dir", type=Path, default=DOCUMENTS_DIR)
    parser.add_argument("--chunks-file", type=Path, default=CHUNKS_FILE)
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    args = parser.parse_args()

    app = load_app_helpers()
    chunk_counts = load_chunk_counts(args.chunks_file)
    pdfs = sorted(args.documents_dir.glob("*.pdf")) if args.documents_dir.exists() else []
    pdf_info_by_source = {pdf.stem: pdf_stats(pdf) for pdf in pdfs}
    source_ids = sorted(set(chunk_counts) | set(pdf_info_by_source) | set(app.FRIENDLY_SOURCE_NAMES))

    entries = [
        build_entry(
            source_id,
            app,
            chunk_counts.get(source_id, 0),
            pdf_info_by_source.get(source_id),
        )
        for source_id in source_ids
        if source_id
    ]

    payload = {
        "schema_version": 1,
        "description": (
            "Bibliography/source metadata for the EMERGE bot corpus. "
            "This file intentionally excludes document text, chunks, embeddings, and chat logs."
        ),
        "source_count": len(entries),
        "sources": entries,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} source metadata records to {args.output}")


if __name__ == "__main__":
    main()
