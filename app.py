"""
EMERGE AI Ethics Information Bot - Backend
Pure-Python TF-IDF retrieval, Haiku-driven query expansion (paraphrase + counter-query
for two-sided coverage), scope-gated refusals, vignette endpoints. No new deps.
"""

import os
import json
import re
import math
from collections import Counter, defaultdict
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic

app = Flask(__name__, static_folder="static")
CORS(app)

CHUNKS_FILE = Path("chunks.json")
VIGNETTES_FILE = Path("vignettes.json")
DOCUMENTS_DIR = Path("documents")
TOP_K = 18
PER_SOURCE_LIMIT = 3

# Empirical thresholds on the *mean cosine of top-10 chunks* — a far cleaner signal
# than max alone, because off-topic queries usually surface 1–2 lucky matches but the
# rest of top-10 trails off, while real ethics queries pull a coherent block of chunks.
# Calibrated against in-scope (collab awareness, accountability, duplicates → 0.13–0.27)
# vs out-of-scope (pizza, taxes, joke, weather → 0.05–0.10) on this corpus.
SCOPE_GATE_MIN = 0.07   # below this → hard refuse, return redirect
SCOPE_GATE_WEAK = 0.10  # below this → tell the model retrieval is weak

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

chunk_store = []
chunk_tf = []     # list[dict[token, weight]]
doc_norms = []    # list[float]
idf = {}          # dict[token, idf weight]

vignettes = []
corpus_stats = {
    "pdf_documents": 0,
    "indexed_sources": 0,
    "missing_pdf_sources": [],
}

STOPWORDS = {
    'the','a','an','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','shall','should','may','might','must',
    'can','could','of','in','on','at','to','for','with','by','from','as','into',
    'and','or','but','not','no','nor','what','how','why','when','where','who','whom',
    'which','that','this','these','those','it','its','they','them','their','there',
    'we','our','us','i','you','your','he','she','his','her','him','me','my',
    'if','than','then','so','too','very','also','about','over','under','between',
    'such','some','any','all','more','most','less','least','one','two','three',
    'just','only','same','other','another','each','both','either','neither',
    'because','while','until','during','through','against','before','after',
    'up','down','out','off','here','says','say','said','use','used','using','etc'
}

SYSTEM_PROMPT = """## Role

You are the EMERGE AI Ethics Information Bot, developed as part of the EMERGE project (EU Horizon Europe, Grant No. 101070918). Your role is to present what researchers, ethicists, and policymakers have found and said about AI ethics — based exclusively on the EMERGE project corpus and its referenced sources provided to you below.

You are an information tool, not an advisor. You surface findings and positions from the literature. You do not make recommendations, give opinions, or tell users what to do.

---

## Sourcing Rules — non-negotiable

- Every claim must be traceable to a named source in the retrieved excerpts.
- Always name sources in full, using the friendly name. Examples:
  CORRECT: "According to EMERGE D2.3…", "Jobin et al. (2019) found…", "The EU Ethics Guidelines for Trustworthy AI (HLEG, 2019) state…", "Hagendorff (2022) argues…"
  INCORRECT: "[1]", "[Source: D2.3 Map…]", "research shows", "the corpus indicates"
- Filename-to-friendly-name conventions:
    D1.1 Local awareness criteria → EMERGE D1.1
    D1.2 Demarcating collaborative awareness → EMERGE D1.2
    D1.3 Dimensions of collaborative awareness → EMERGE D1.3
    D2.2 Map of risks in AI systems → EMERGE D2.2
    D2.3 Map of risks and potentials for humans → EMERGE D2.3
    D2.4 Map of Ethical Virtues → EMERGE D2.4
    D2.5_Ethical_Resilience → EMERGE D2.5
    Jobin_etal2019 → Jobin et al. (2019)
    Hagendorff2022 / hagendorf2019 → Hagendorff (2022) / Hagendorff (2019)
    Correa_etal2023 → Correa et al. (2023)
    Floridi_etal2018 → Floridi et al. (2018)
    Lange_etal_NPJCommunication_2025 → Lange et al. (2025)
    Haas_etal_nature2026 → Haas et al. (2026)
    VallorVierkant2024 → Vallor and Vierkant (2024)
    winfield-et-al-2025 → Winfield et al. (2025)
    Danaher and Nyholm — 2025… → Danaher and Nyholm (2025)
    OJ_L_202401689_EN_TXT → EU AI Act (Regulation (EU) 2024/1689)
    ai-ethics-guidelines → EU Ethics Guidelines for Trustworthy AI (HLEG, 2019)
    OECD-LEGAL-0449-en → OECD AI Principles
    For any other source, use a sensible short academic citation form.
- If you cannot find a sourced answer in the retrieved excerpts, say so plainly. Do not invent.

---

## Two-sidedness on contested questions

The corpus contains multiple positions on most ethical questions. Your job is to surface them, not to resolve them.

- When the retrieved excerpts contain disagreement, present the contrasting positions side by side, each attributed to its source.
- Do not take a side, do not present synthesis as consensus, do not soften disagreement into agreement.
- When the excerpts contain only one position on a clearly contested question, say so explicitly: "The retrieved excerpts present one position; this is a contested area in the broader literature."
- Use language like "On one view… On another view…" or "X argues… Y, by contrast, argues…"

---

## Scope Boundaries

This bot covers AI ethics findings and policy positions from the EMERGE corpus. Always outside scope — decline:
- Legal compliance advice
- Product or vendor recommendations
- Implementation or technical advice
- Drafting of policies, contracts, or governance documents
- Topics not covered by the corpus

For out-of-scope questions, decline plainly and, where relevant, point the user toward where they could look (a relevant authority, professional, or category of resource). Do not invent facts to fill gaps.

---

## Tone and Length

- Present findings and positions. Never recommend or instruct.
- Neutral, informative language.
- FIRST response on any topic: 3–4 sentences. High-level overview, named source(s), one short follow-up offer ("Want me to expand on a specific position?").
- Only give a longer response when the user explicitly asks for depth ("go deeper", "expand", "tell me more", "elaborate").
- Never front-load detail. Never use bullet points on first responses — prose only.
- Do not add a closing summary.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Indexing & retrieval
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text):
    toks = re.findall(r'\b[a-zA-Z][a-zA-Z\-]+\b', text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def load_chunks():
    global chunk_store, chunk_tf, doc_norms, idf, corpus_stats
    if not CHUNKS_FILE.exists():
        print("WARNING: chunks.json not found.")
        return
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunk_store = json.load(f)

    N = len(chunk_store)
    df = defaultdict(int)
    chunk_tokens_list = []
    for chunk in chunk_store:
        toks = tokenize(chunk["text"])
        chunk_tokens_list.append(toks)
        for t in set(toks):
            df[t] += 1

    idf = {t: math.log((N + 1) / (df_t + 1)) + 1 for t, df_t in df.items()}

    chunk_tf.clear()
    doc_norms.clear()
    for toks in chunk_tokens_list:
        tf = Counter(toks)
        weighted = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in weighted.values())) or 1.0
        chunk_tf.append(weighted)
        doc_norms.append(norm)

    sources = sorted(set(c["source"] for c in chunk_store))
    pdf_stems = sorted(p.stem for p in DOCUMENTS_DIR.glob("*.pdf")) if DOCUMENTS_DIR.exists() else []
    missing = [
        stem for stem in pdf_stems
        if not any(stem.lower() in src.lower() or src.lower() in stem.lower() for src in sources)
    ]
    corpus_stats = {
        "pdf_documents": len(pdf_stems),
        "indexed_sources": len(sources),
        "missing_pdf_sources": missing,
    }
    print(f"Loaded {len(chunk_store)} chunks from {len(sources)} sources; vocab={len(idf)}")
    if missing:
        print(f"WARNING: {len(missing)} PDFs do not have an obvious source match in chunks.json")


def load_vignettes():
    global vignettes
    if VIGNETTES_FILE.exists():
        with open(VIGNETTES_FILE, encoding="utf-8") as f:
            vignettes = json.load(f)
        print(f"Loaded {len(vignettes)} vignettes")
    else:
        vignettes = []
        print("No vignettes.json found.")


def score_chunks(query):
    qtoks = tokenize(query)
    if not qtoks:
        return []
    qtf = Counter(qtoks)
    qweighted = {t: (1 + math.log(c)) * idf[t] for t, c in qtf.items() if t in idf}
    if not qweighted:
        return []
    qnorm = math.sqrt(sum(v * v for v in qweighted.values())) or 1.0

    scored = []
    for i, ctf in enumerate(chunk_tf):
        dot = 0.0
        for t, w in qweighted.items():
            cw = ctf.get(t)
            if cw:
                dot += w * cw
        if dot > 0:
            scored.append((dot / (qnorm * doc_norms[i]), i))
    scored.sort(reverse=True)
    return scored


def expand_query(query, client):
    """One Haiku call → JSON with paraphrase, counter-query, classification, redirect hint.
    Returns dict with safe defaults on any failure."""
    instructions = (
        "You help retrieve relevant excerpts from an AI ethics corpus.\n"
        "Given a user question, output STRICT JSON with these keys:\n"
        '  "paraphrase": one alternative phrasing using different vocabulary, same topic\n'
        '  "counter":    a search query that would surface OPPOSING views on the same question\n'
        '  "classification": one of "ai_ethics", "legal", "medical", "technical_implementation", "product", "general_news", "other"\n'
        '  "redirect":   short hint of where this user should look if the corpus cannot answer (e.g. "consult a lawyer", "see your platform documentation", "see medical professionals")\n'
        "Output ONLY the JSON object, no prose, no code fences."
    )
    fallback = {"paraphrase": "", "counter": "", "classification": "ai_ethics", "redirect": ""}
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=instructions,
            messages=[{"role": "user", "content": query}],
        )
        text = msg.content[0].text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        data = json.loads(text)
        for k, v in fallback.items():
            data.setdefault(k, v)
        return data
    except Exception as e:
        print(f"expand_query failed: {e}")
        return fallback


def retrieve(query, client):
    """Three-pass retrieval: original + paraphrase + counter-query.
    Returns (top_chunks, gate_score, expansion_dict).
    gate_score is the mean cosine of the top-10 chunks — used for scope gating."""
    expansion = expand_query(query, client) if client else {
        "paraphrase": "", "counter": "", "classification": "ai_ethics", "redirect": ""
    }

    queries = [query]
    if expansion.get("paraphrase"):
        queries.append(expansion["paraphrase"])
    if expansion.get("counter"):
        queries.append(expansion["counter"])

    best_score = {}
    for q in queries:
        for sim, i in score_chunks(q):
            if sim > best_score.get(i, 0):
                best_score[i] = sim

    if not best_score:
        return [], 0.0, expansion

    ranked = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
    top_for_gate = ranked[:10]
    gate_score = sum(s for _, s in top_for_gate) / len(top_for_gate)
    top_chunks = []
    source_counts = defaultdict(int)
    for i, _ in ranked:
        source = chunk_store[i].get("source", "")
        if source_counts[source] >= PER_SOURCE_LIMIT:
            continue
        top_chunks.append(chunk_store[i])
        source_counts[source] += 1
        if len(top_chunks) >= TOP_K:
            break
    return top_chunks, gate_score, expansion


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/status")
def status():
    docs = sorted(set(c["source"] for c in chunk_store))
    return jsonify({
        "chunks_loaded": len(chunk_store),
        "documents": docs,
        "pdf_documents": corpus_stats["pdf_documents"],
        "indexed_sources": corpus_stats["indexed_sources"],
        "missing_pdf_sources": corpus_stats["missing_pdf_sources"],
        "vignettes": len(vignettes),
        "ready": len(chunk_store) > 0,
        "runtime": "render-flask",
    })


@app.route("/vignettes")
def list_vignettes():
    summaries = [
        {"id": v["id"], "title": v["title"], "topic": v.get("topic", "")}
        for v in vignettes
    ]
    return jsonify(summaries)


@app.route("/vignettes/<vid>")
def get_vignette(vid):
    for v in vignettes:
        if v["id"] == vid:
            return jsonify(v)
    return jsonify({"error": "Vignette not found"}), 404


@app.route("/chat", methods=["POST"])
def chat():
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "API key not configured on server."}), 500

    data = request.json or {}
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    retrieval_query = " ".join(user_msgs[-3:])

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        chunks, gate_score, expansion = retrieve(retrieval_query, client)
    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid API key on server."}), 401
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {e}"}), 500

    # Hard scope gate: top-10 chunks have weak coherent overlap with the query.
    if gate_score < SCOPE_GATE_MIN:
        redirect = expansion.get("redirect", "").strip()
        msg = ("This question doesn't appear to be covered by the EMERGE corpus, "
               "which focuses on AI ethics research findings and policy positions.")
        if redirect:
            msg += f" For this kind of question, {redirect}."
        return jsonify({
            "reply": msg,
            "chunks_used": 0,
            "scope": "out_of_scope",
            "gate_score": round(gate_score, 4),
            "classification": expansion.get("classification", "other"),
        })

    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    weak_note = ""
    if gate_score < SCOPE_GATE_WEAK:
        weak_note = ("\n\nNote: retrieval signal for this question is weak. "
                     "If the excerpts above don't actually address the question, say so and "
                     "decline rather than reaching.")
    context_block = (
        f"\n\n## Retrieved Corpus Excerpts\n\n{context}\n\n---\n\n"
        f"Answer using only the excerpts above. Name the source for every claim using its friendly "
        f"name (e.g. EMERGE D2.4, Jobin et al. 2019). If the excerpts contain disagreement, present "
        f"both sides with sources — do not resolve.{weak_note}"
    )

    full_system = SYSTEM_PROMPT + context_block

    depth_keywords = ["go deeper", "expand", "tell me more", "explain further",
                      "what else", "more detail", "elaborate", "in depth", "give me more"]
    last_user = user_msgs[-1].lower() if user_msgs else ""
    wants_depth = any(kw in last_user for kw in depth_keywords)
    model = "claude-sonnet-4-20250514"
    max_tokens = 1400 if wants_depth else 800

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=full_system,
            messages=messages,
        )
        return jsonify({
            "reply": response.content[0].text,
            "chunks_used": len(chunks),
            "scope": "in_scope" if gate_score >= SCOPE_GATE_WEAK else "weak_retrieval",
            "gate_score": round(gate_score, 4),
            "sources_used": sorted(set(c.get("source", "") for c in chunks)),
        })
    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid API key on server."}), 401
    except anthropic.RateLimitError:
        return jsonify({"error": "Rate limit reached. Please wait a moment."}), 429
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("EMERGE AI Ethics Information Bot")
    print("=" * 50)
    if not ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY not set!")
    load_chunks()
    load_vignettes()
    port = int(os.environ.get("PORT", 5000))
    print(f"\nStarting at http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
