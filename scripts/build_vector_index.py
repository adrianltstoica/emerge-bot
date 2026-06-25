#!/usr/bin/env python3
"""Build an embedding vector index for chunks.json.

Requires OPENAI_API_KEY. The output is committed/deployed with the app, while
runtime query embeddings use the same embedding model.
"""

import argparse
import gzip
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "chunks.json"
OUT_FILE = Path(os.environ.get("VECTOR_INDEX_FILE", ROOT / "vector_index.json.gz"))
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_TEXT_CHAR_LIMIT = int(os.environ.get("EMBEDDING_TEXT_CHAR_LIMIT", "12000"))


def prepare_embedding_text(text, limit=DEFAULT_TEXT_CHAR_LIMIT):
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit]


def embed_texts(texts, api_key, model):
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Embedding request failed ({exc.code}): {detail}") from exc
    data = json.loads(body)
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


def batched(items, size):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


def main():
    parser = argparse.ArgumentParser(description="Build vector_index.json.gz from chunks.json")
    parser.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required to build the vector index.")

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = []
    texts = [prepare_embedding_text(chunk["text"]) for chunk in chunks]
    for start, batch in batched(texts, args.batch_size):
        for attempt in range(1, 4):
            try:
                embeddings.extend(embed_texts(batch, api_key, args.model))
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                print(f"Batch {start} failed ({exc}); retrying in {wait}s", flush=True)
                time.sleep(wait)
        print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)} chunks", flush=True)

    index = {
        "meta": {
            "model": args.model,
            "chunk_count": len(chunks),
            "source_file": CHUNKS_FILE.name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "embeddings": embeddings,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as f:
        json.dump(index, f)
    try:
        display_path = args.output.relative_to(ROOT)
    except ValueError:
        display_path = args.output
    print(f"Wrote {len(embeddings)} vectors to {display_path}")


if __name__ == "__main__":
    main()
