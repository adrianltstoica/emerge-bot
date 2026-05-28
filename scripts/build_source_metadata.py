#!/usr/bin/env python3
"""Build source_metadata.json for the local corpus.

The output is bibliography-oriented metadata, not extracted document content.
It is safe to publish when source titles/citation labels may be public, but the
PDFs, chunks, and embeddings themselves are not redistributed.
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "documents"
CHUNKS_FILE = ROOT / "chunks.json"
OUT_FILE = ROOT / "source_metadata.json"
CSV_OUT_FILE = ROOT / "source_metadata.csv"


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


def infer_url(*values):
    url_re = re.compile(r"https?://[^\s)>\]]+", re.I)
    for value in values:
        if not value:
            continue
        match = url_re.search(str(value))
        if match:
            return match.group(0).rstrip(".,;")
    return None


def source_url(source_id, doi=None):
    if doi:
        return f"https://doi.org/{doi}"
    arxiv_match = re.fullmatch(r"(\d{4}\.\d{4,5})(?:v\d+)?", source_id)
    if arxiv_match:
        return f"https://arxiv.org/abs/{arxiv_match.group(1)}"
    return None


def clean_author_list(author):
    if not author:
        return []
    author = re.sub(r"\s+", " ", author).strip()
    if author.lower() in {"anonymous", "unknown"}:
        return []
    if "," in author and " and " not in author.lower() and ";" not in author:
        return [part.strip() for part in author.split(",") if part.strip()]
    return [part.strip() for part in re.split(r";|\band\b", author) if part.strip()]


def bibliography_entry(entry):
    authors = entry.get("authors") or []
    author_part = ", ".join(authors) if authors else entry["citation"]
    citation_has_year = bool(entry.get("year") and str(entry["year"]) in author_part)
    year_part = "" if citation_has_year else (f" ({entry['year']})" if entry.get("year") else "")
    title_part = "" if entry["title"] == author_part else f". {entry['title']}"
    venue = entry.get("venue_or_series") or entry.get("publisher_or_institution")
    venue_part = f". {venue}" if venue else ""
    url_part = f". {entry['url']}" if entry.get("url") else ""
    return f"{author_part}{year_part}{title_part}{venue_part}{url_part}.".replace("..", ".").strip()


def completeness_score(entry):
    fields = [
        "citation", "title", "year", "authors", "publisher_or_institution",
        "venue_or_series", "doi", "url", "source_tier", "document_type",
        "pdf_filename", "page_count", "chunk_count",
    ]
    present = 0
    for field in fields:
        value = entry.get(field)
        if value not in (None, "", []):
            present += 1
    return round(present / len(fields), 2)


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
            first_pages_text = []
            for page in pdf.pages[:3]:
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                chars += len(text)
                first_pages_text.append(text)
            metadata = normalize_pdf_metadata(pdf.metadata)
            return {
                "pdf_filename": path.name,
                "page_count": len(pdf.pages),
                "text_extractable": chars > 0,
                "pdf_metadata": metadata,
                "first_pages_text": "\n".join(first_pages_text),
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
    first_pages_text = pdf_info.get("first_pages_text", "")
    year = infer_year(citation, title, raw_pdf_meta.get("Title"), first_pages_text, raw_pdf_meta.get("CreationDate"))
    doi = infer_doi(citation, title, raw_pdf_meta.get("Title"), raw_pdf_meta.get("Subject"), first_pages_text)
    entry = {
        "source_id": source_id,
        "citation": citation,
        "title": title,
        "year": year,
        "authors": [],
        "publisher_or_institution": None,
        "venue_or_series": None,
        "doi": doi,
        "url": source_url(source_id, doi),
        "source_tier": app.source_tier(source_id),
        "document_type": "corpus PDF" if pdf_info else "indexed source",
        "pdf_filename": pdf_info.get("pdf_filename"),
        "page_count": pdf_info.get("page_count"),
        "text_extractable": pdf_info.get("text_extractable"),
        "chunk_count": chunk_count,
        "redistribution_status": "included in repository with project clearance",
        "notes": pdf_info.get("notes", ""),
    }

    entry["authors"] = clean_author_list(raw_pdf_meta.get("Author"))
    entry["bibliography_entry"] = bibliography_entry(entry)
    entry["metadata_completeness"] = completeness_score(entry)
    return entry


def write_csv(path, entries):
    columns = [
        "source_id", "citation", "title", "year", "authors",
        "publisher_or_institution", "venue_or_series", "doi", "url",
        "source_tier", "document_type", "pdf_filename", "page_count",
        "text_extractable", "chunk_count", "metadata_completeness",
        "bibliography_entry", "redistribution_status", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            row["authors"] = "; ".join(row.get("authors") or [])
            writer.writerow({col: row.get(col) for col in columns})


def main():
    parser = argparse.ArgumentParser(description="Build source_metadata.json from local corpus files")
    parser.add_argument("--documents-dir", type=Path, default=DOCUMENTS_DIR)
    parser.add_argument("--chunks-file", type=Path, default=CHUNKS_FILE)
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    parser.add_argument("--csv-output", type=Path, default=CSV_OUT_FILE)
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
    write_csv(args.csv_output, entries)
    print(f"Wrote {len(entries)} source metadata records to {args.output}")
    print(f"Wrote CSV source register to {args.csv_output}")


if __name__ == "__main__":
    main()
