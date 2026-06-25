#!/usr/bin/env python3
"""Build vector_index.json.gz when it is missing or stale.

Render runs this before Gunicorn starts. If OPENAI_API_KEY is not configured,
the script exits successfully so the app can still serve with TF-IDF fallback.
"""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = Path(os.environ.get("CHUNKS_FILE", ROOT / "chunks.json"))
VECTOR_INDEX_FILE = Path(os.environ.get("VECTOR_INDEX_FILE", ROOT / "vector_index.json.gz"))


def chunk_count():
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return len(json.load(f))


def existing_index_count():
    if not VECTOR_INDEX_FILE.exists():
        return None
    opener = gzip.open if VECTOR_INDEX_FILE.suffix == ".gz" else open
    with opener(VECTOR_INDEX_FILE, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    return len(payload.get("embeddings") or [])


def main():
    if not CHUNKS_FILE.exists():
        print(f"chunks file missing at {CHUNKS_FILE}; skipping vector index build", flush=True)
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not configured; skipping vector index build", flush=True)
        return 0

    expected = chunk_count()
    try:
        actual = existing_index_count()
    except Exception as exc:
        print(f"Existing vector index is unreadable ({exc}); rebuilding", flush=True)
        actual = None

    if actual == expected:
        print(f"Vector index already current at {VECTOR_INDEX_FILE} ({actual} vectors)", flush=True)
        return 0

    print(
        f"Building vector index at {VECTOR_INDEX_FILE} "
        f"({actual or 0} existing vectors, {expected} chunks)",
        flush=True,
    )
    VECTOR_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_vector_index.py"),
        "--output",
        str(VECTOR_INDEX_FILE),
    ]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
