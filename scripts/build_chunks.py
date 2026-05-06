#!/usr/bin/env python3
"""Rebuild chunks.json from every PDF in documents/.

This is the source-of-truth corpus build step for the Render bot.
"""

import argparse
import json
import re
from pathlib import Path

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "documents"
OUT_FILE = ROOT / "chunks.json"


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def words(text):
    return re.findall(r"\S+", text)


def chunk_words(source, tokens, chunk_size, overlap):
    step = max(1, chunk_size - overlap)
    chunk_id = 0
    for start in range(0, len(tokens), step):
        window = tokens[start:start + chunk_size]
        if len(window) < 80:
            continue
        yield {
            "text": f"[Source: {source}]\n" + " ".join(window),
            "source": source,
            "chunk_id": chunk_id,
        }
        chunk_id += 1


def extract_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text(x_tolerance=1, y_tolerance=3) or "")
    return clean_text("\n\n".join(pages))


def main():
    parser = argparse.ArgumentParser(description="Build chunks.json from PDFs")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=80)
    args = parser.parse_args()

    pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    all_chunks = []
    failures = []

    for pdf in pdfs:
        source = pdf.stem
        try:
            text = extract_pdf(pdf)
            tokens = words(text)
            doc_chunks = list(chunk_words(source, tokens, args.chunk_size, args.overlap))
            if not doc_chunks:
                failures.append({"file": pdf.name, "reason": "no text chunks extracted"})
            all_chunks.extend(doc_chunks)
            print(f"{pdf.name}: {len(tokens)} words -> {len(doc_chunks)} chunks", flush=True)
        except Exception as exc:
            failures.append({"file": pdf.name, "reason": str(exc)})
            print(f"{pdf.name}: FAILED: {exc}", flush=True)

    OUT_FILE.write_text(json.dumps(all_chunks, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(all_chunks)} chunks to {OUT_FILE.relative_to(ROOT)}")
    print(f"Documents processed: {len(pdfs)}")
    if failures:
        print("\nFiles needing attention:")
        for failure in failures:
            print(f"- {failure['file']}: {failure['reason']}")


if __name__ == "__main__":
    main()
