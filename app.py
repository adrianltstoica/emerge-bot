"""
EMERGE AI Ethics Information Bot - RAG Backend
Flask server with PDF upload, chunking, retrieval, and Claude API.
"""

import os
import json
import re
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic

try:
    import pdfplumber
    PDF_BACKEND = "pdfplumber"
except ImportError:
    PDF_BACKEND = None

app = Flask(__name__, static_folder="static")
CORS(app)

DOCS_FOLDER = Path("documents")
CHUNKS_CACHE = Path("chunks_cache.json")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 6

app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB per file

chunk_store = []

SYSTEM_PROMPT = """## Role

You are the EMERGE AI Ethics Information Bot, developed as part of the EMERGE project (EU Horizon Europe, Grant No. 101070918). Your role is to present what researchers, ethicists, and policymakers have found and said about AI ethics — based exclusively on the EMERGE project corpus and its referenced sources provided to you below.

You are an information tool, not an advisor. You surface findings and positions from the literature. You do not make recommendations, give opinions, or tell users what to do.

---

## Sourcing Rules

These rules are non-negotiable and apply to every response:

- Every claim you make must be traceable to a named source in the retrieved documents provided.
- Always name sources in full.
  CORRECT: "According to EMERGE D2.3…" or "Jobin et al. (2019) found…" or "The EU Ethics Guidelines for Trustworthy AI (HLEG, 2019) state…"
  INCORRECT: "According to [1]…" or "Research shows…" or "The corpus indicates…"
- Never use numbered citations like [1], [2], [3], or any bracketed reference. These are forbidden.
- If you cannot find a named source for a claim in the retrieved documents, do not make the claim.
- If the retrieved documents do not contain a sourced answer to the question, say clearly: "I don't have a sourced answer to this in the current corpus." Do not guess, infer, or fill gaps with general knowledge.
- Do not use vague attributions such as "researchers have found," "policymakers argue," or "studies show" without naming who.

---

## Priority Sources by Topic

- Trustworthy AI: EU Ethics Guidelines for Trustworthy AI (HLEG, 2019), EMERGE D2.4
- AI ethics principles (transparency, fairness, accountability): Jobin et al. (2019), Correa et al. (2023), Hagendorff (2022)
- Aware and collective AI systems: EMERGE D2.2, D2.3, D2.5, The Ethics of Aware and Collective AI Systems (Karpus & Bahrami)
- Collaborative awareness: EMERGE D1.1, D1.2, D1.3
- EU AI Act / regulation: EU AI Act (OJ L 2024/1689), EMERGE D2.2
- Trust in human-AI collaboration: EMERGE D2.4, Vereschak et al.
- Ethical resilience: EMERGE D2.5

---

## Scope Boundaries

Always outside scope — decline politely:
- Legal compliance questions
- Product or tool recommendations
- Implementation or technical advice
- Drafting of policies, contracts, or governance documents

For out-of-scope questions respond: "This falls outside the scope of this tool, which covers AI ethics research findings and policy positions — not legal, implementation, or product advice."

---

## Tone Rules

- Present findings and positions from the corpus. Never recommend or instruct.
- When the corpus contains opposing views, present both sides with named sources. Do not resolve disagreements.
- Use neutral, informative language.
- Do not add a closing summary or "in summary" section. Stop when the content is complete.
- Keep responses proportionate. A simple definitional question needs 2–4 sentences, not 8 paragraphs.

---

## What This Bot Does Not Do

- Does not give legal advice
- Does not recommend products, tools, or vendors
- Does not write policies or implementation plans
- Does not take sides in ongoing ethical debates
- Does not use numbered citations like [1] or [2]
- Does not answer questions it cannot source from the corpus"""


def chunk_text(text, source_name):
    words = text.split()
    chunks = []
    i = 0
    cid = 0
    while i < len(words):
        chunk_words = words[i:i + CHUNK_SIZE]
        chunks.append({
            "text": f"[Source: {source_name}]\n{' '.join(chunk_words)}",
            "source": source_name,
            "chunk_id": cid
        })
        cid += 1
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_from_bytes(pdf_bytes, source_name):
    if PDF_BACKEND != "pdfplumber":
        return ""
    try:
        import pdfplumber, io
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading {source_name}: {e}")
        return ""


def extract_from_path(pdf_path):
    if PDF_BACKEND != "pdfplumber":
        return ""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def save_cache():
    with open(CHUNKS_CACHE, "w") as f:
        json.dump(chunk_store, f)


def load_documents():
    global chunk_store
    DOCS_FOLDER.mkdir(exist_ok=True)

    if CHUNKS_CACHE.exists():
        print("Loading from cache...")
        with open(CHUNKS_CACHE) as f:
            chunk_store = json.load(f)
        print(f"  {len(chunk_store)} chunks from {len(set(c['source'] for c in chunk_store))} docs")
        return

    pdfs = list(DOCS_FOLDER.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found — upload via the web interface.")
        return

    print(f"Processing {len(pdfs)} PDFs...")
    for p in pdfs:
        text = extract_from_path(p)
        if text:
            chunks = chunk_text(text, p.stem)
            chunk_store.extend(chunks)
            print(f"  {p.stem}: {len(chunks)} chunks")
    save_cache()
    print(f"Total: {len(chunk_store)} chunks")


def retrieve_chunks(query):
    if not chunk_store:
        return []
    stopwords = {'the','a','an','is','are','was','were','be','been','have','has',
                 'had','do','does','did','will','would','shall','should','may','might',
                 'can','could','of','in','on','at','to','for','with','by','from',
                 'and','or','but','not','what','how','why','when','where','who',
                 'which','that','this','it','its'}
    qwords = set(re.findall(r'\b\w+\b', query.lower())) - stopwords
    scored = []
    for chunk in chunk_store:
        cwords = set(re.findall(r'\b\w+\b', chunk["text"].lower()))
        overlap = len(qwords & cwords)
        if overlap > 0:
            scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:TOP_K]]


# ── Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/status")
def status():
    docs = sorted(set(c["source"] for c in chunk_store))
    return jsonify({"chunks_loaded": len(chunk_store), "documents": docs, "pdf_backend": PDF_BACKEND})


@app.route("/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    results = []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"name": f.filename, "status": "skipped — not a PDF"})
            continue

        source_name = Path(f.filename).stem
        pdf_bytes = f.read()

        text = extract_from_bytes(pdf_bytes, source_name)
        if not text:
            results.append({"name": f.filename, "status": "error — could not extract text"})
            continue

        # Remove old chunks for this source
        global chunk_store
        chunk_store = [c for c in chunk_store if c["source"] != source_name]

        new_chunks = chunk_text(text, source_name)
        chunk_store.extend(new_chunks)

        # Save PDF to disk
        DOCS_FOLDER.mkdir(exist_ok=True)
        with open(DOCS_FOLDER / f.filename, "wb") as out:
            out.write(pdf_bytes)

        save_cache()
        results.append({"name": f.filename, "source": source_name, "chunks": len(new_chunks), "status": "ok"})

    return jsonify({
        "results": results,
        "total_chunks": len(chunk_store),
        "total_docs": len(set(c["source"] for c in chunk_store))
    })


@app.route("/delete-doc", methods=["POST"])
def delete_doc():
    global chunk_store
    source = request.json.get("source")
    if not source:
        return jsonify({"error": "No source provided"}), 400
    chunk_store = [c for c in chunk_store if c["source"] != source]
    save_cache()
    p = DOCS_FOLDER / f"{source}.pdf"
    if p.exists():
        p.unlink()
    return jsonify({"message": f"Removed {source}", "total_chunks": len(chunk_store)})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    api_key = data.get("api_key", "")
    stakeholder = data.get("stakeholder", "")

    if not api_key:
        return jsonify({"error": "No API key provided"}), 400
    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    chunks = retrieve_chunks(last_user_msg)

    if chunks:
        context = "\n\n---\n\n".join(c["text"] for c in chunks)
        context_block = f"\n\n## Retrieved Corpus Excerpts\n\n{context}\n\n---\n\nAnswer using only the above excerpts. Name the source document for every claim."
    else:
        context_block = "\n\n## Retrieved Corpus Excerpts\n\nNo relevant excerpts found. Respond: 'I don't have a sourced answer to this in the current corpus.'"

    stakeholder_note = f"\n\nUser identified as: {stakeholder.upper()}. Tailor framing to their concerns while maintaining sourcing standards." if stakeholder else ""
    full_system = SYSTEM_PROMPT + stakeholder_note + context_block

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=full_system,
            messages=messages
        )
        return jsonify({"reply": response.content[0].text, "chunks_used": len(chunks)})
    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid API key."}), 401
    except anthropic.RateLimitError:
        return jsonify({"error": "Rate limit reached. Please wait a moment."}), 429
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("EMERGE AI Ethics Information Bot")
    print("=" * 50)
    load_documents()
    port = int(os.environ.get("PORT", 5000))
    print(f"\nStarting at http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
