"""
EMERGE AI Ethics Information Bot - Backend
Pure-Python TF-IDF retrieval, Haiku-driven query expansion (paraphrase + counter-query
for two-sided coverage), scope-gated refusals. No new deps.
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
DOCUMENTS_DIR = Path("documents")
TOP_K = 18
PER_SOURCE_LIMIT = 3
RETRIEVAL_PROFILES = {
    "normal": {
        "top_k": 18,
        "per_source_limit": 3,
        "max_tokens": 800,
        "description": "balanced retrieval for focused questions",
    },
    "broad": {
        "top_k": 36,
        "per_source_limit": 2,
        "max_tokens": 1300,
        "description": "wider source coverage for overview and landscape questions",
    },
    "deep": {
        "top_k": 32,
        "per_source_limit": 6,
        "max_tokens": 1700,
        "description": "deeper per-source coverage for detailed follow-up questions",
    },
    "broad_deep": {
        "top_k": 42,
        "per_source_limit": 4,
        "max_tokens": 1900,
        "description": "larger mixed retrieval for comprehensive cross-corpus questions",
    },
}

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

corpus_stats = {
    "pdf_documents": 0,
    "indexed_sources": 0,
    "missing_pdf_sources": [],
}

FRIENDLY_SOURCE_NAMES = {
    "D1.1 Local awareness  criteria": "EMERGE D1.1",
    "D1.2 Demarcating collaborative awareness from related concepts": "EMERGE D1.2",
    "D1.3 Dimensions of collaborative awareness": "EMERGE D1.3",
    "D2.2 Map of risks in AI- systems": "EMERGE D2.2",
    "D2.3 Map of risks and potentials for humans": "EMERGE D2.3",
    "D2.4 Map of Ethical Virtues": "EMERGE D2.4",
    "D2.5_Ethical_Resilience": "EMERGE D2.5",
    "OJ_L_202401689_EN_TXT": "EU AI Act (Regulation (EU) 2024/1689)",
    "ai-ethics-guidelines": "EU Ethics Guidelines for Trustworthy AI (HLEG, 2019)",
    "OECD-LEGAL-0449-en": "OECD AI Principles",
    "Jobin_etal2019": "Jobin et al. (2019)",
    "Hagendorff2022": "Hagendorff (2022)",
    "hagendorf2019": "Hagendorff (2019)",
    "Correa_etal2023": "Correa et al. (2023)",
    "floridi_etal2018": "Floridi et al. (2018)",
    "Lange_etal_NPJCommunication_2025": "Lange et al. (2025)",
    "Haas_etal_nature2026": "Haas et al. (2026)",
    "VallorVierkant2024": "Vallor and Vierkant (2024)",
    "winfield-et-al-2025": "Winfield et al. (2025)",
    "1-s2.0-S0749597806000719-main": "Bonaccio and Dalal (2006)",
    "1416674-EN": "UNESCO (2022), Recommendation on the Ethics of Artificial Intelligence",
    "1802.07228v2": "Brundage et al. (2018)",
    "31720-Article Text-35784-1-2-20241016": "Rocca et al. (2024), ELLIPS",
    "4. Book_on_Computational_Awareness-9": "Book on Computational Awareness",
    "716110": "U.S. Government Accountability Office (2021), AI Accountability Framework",
    "AI_Engagement_within_Sony_Group": "Sony Group, AI Engagement Principles",
    "ASEAN-Guide-on-AI-Governance-and-Ethics_beautified_201223_v2": "ASEAN Guide on AI Governance and Ethics",
    "Adaptable robots, ethics, and trust- a qualitative and philosophical exploration of the individual experience of trustworthy AI": "Adaptable Robots, Ethics, and Trust",
    "BMI25020-leitlinien-ki-bundesverwaltung": "German Federal Administration AI Guidelines",
    "Benchmarking Retrieval-Augmented Generation for Medicine": "Benchmarking Retrieval-Augmented Generation for Medicine",
    "Characterizing AI Agents for Alignment and Governance": "Characterizing AI Agents for Alignment and Governance",
    "Charter voor verantwoord gebruik van ai in overheidsdiensten_0": "Charter for Responsible Use of AI in Public Services",
    "Danaher and Nyholm - 2025 - The ethics of personalised digital duplicates a minimally viable permissibility principle": "Danaher and Nyholm (2025)",
    "EPRS_BRI(2019)640163_EN": "European Parliament Research Service (2019)",
    "ESCAP-2023-PB-Ethics-AI-revised": "UN ESCAP (2023), Ethics of AI Policy Brief",
    "ESCAP-2023-WP-Ethics-AI": "UN ESCAP (2023), Ethics of AI Working Paper",
    "Future-of-the-corporation-Trust-trustworthiness-transparency": "British Academy, Trust, Trustworthiness and Transparency",
    "IEEE ETHICALLY ALIGNED DESIGN_v2": "IEEE, Ethically Aligned Design",
    "Microsoft-Responsible-AI-Standard-General-Requirements": "Microsoft Responsible AI Standard",
    "Munn2023": "Munn (2023)",
    "NASA-TM-20210012886": "NASA Technical Memorandum 20210012886",
    "NIST.AI.100-1": "NIST AI Risk Management Framework",
    "National-Strategy-for-Artificial-Intelligence": "National Strategy for Artificial Intelligence",
    "PIIS1364661325002864": "Trends in Cognitive Sciences article (2025)",
    "Psychology of Women Quarterly - December 1980 - Spence - Masculine Instrumentality and Feminine Expressiveness  Their": "Spence (1980)",
    "Resource Guide on AI Strategies_June 2021": "Resource Guide on AI Strategies (2021)",
    "Scientists on ‘urgent’ quest to explain consciousness as AI gathers pace _ERC": "European Research Council, Scientists on Consciousness and AI",
    "Taking AI Welfare Seriously": "Long and Sebo et al. (2024), Taking AI Welfare Seriously",
    "Digital suffering  why it s a problem and how to prevent it": "Digital Suffering",
    "Ethics and Governance of SentientAI": "Ethics and Governance of Sentient AI",
    "Telia Company Guiding Principles on Trusted AI ethics (1)": "Telia Company, Guiding Principles on Trusted AI Ethics",
    "The AI Agent Index": "The AI Agent Index",
    "The Ethics of Advanced AI Assistants": "The Ethics of Advanced AI Assistants",
    "The Moral Consideration of Artificial Entities- A Literature": "The Moral Consideration of Artificial Entities",
    "The ethics of aware and collective artificial intelligence systems": "The Ethics of Aware and Collective Artificial Intelligence Systems",
    "Translation_ Chinese AI Alliance Drafts Self-Discipline 'Joint Pledge' - New America": "New America translation, Chinese AI Alliance Joint Pledge",
    "Translation_ Chinese Expert Group Offers 'Governance Principles' for 'Responsible AI' - New America": "New America translation, Chinese Responsible AI Governance Principles",
    "Ukraine Voluntary CoC": "Ukraine Voluntary Code of Conduct on AI",
    "Vereschak": "Vereschak et al.",
    "ai-report": "AI Report",
    "artificial-suffering-an-argument-for-a-global-moratorium-on-synthetic-phenomenology": "Artificial Suffering and Synthetic Phenomenology",
    "doc_38": "European Parliament, What If Generative AI Became Conscious?",
    "doc_42": "Should We Fear Artificial Intelligence?",
    "dubai ai-ethics": "Dubai AI Ethics Principles and Guidelines",
    "everydayethics": "Everyday Ethics for Artificial Intelligence",
    "frai-06-1020592": "Frontiers in Robotics and AI article 1020592",
    "fu_ir8hyn25gaatz92": "China MOST, Ethical Norms for New Generation Artificial Intelligence",
    "gino_etal_2009": "Gino et al. (2009)",
    "robots-should-be-slaves": "Bryson, Robots Should Be Slaves",
    "robustness and explainability of artificial intelligence-KJNA30040ENN": "Robustness and Explainability of Artificial Intelligence",
    "rsta.2018.0085": "Philosophical Transactions A article 2018.0085",
    "s10676-020-09573-9": "Ethics and Information Technology article 09573-9",
    "s11623-019-1183-6": "AI & Society article 1183-6",
    "s11948-019-00146-8": "Science and Engineering Ethics article 00146-8",
    "s13347-021-00454-7": "Philosophy & Technology article 00454-7",
    "s13347-024-00718-y": "Philosophy & Technology article 00718-y",
    "s41599-025-04532-5": "Humanities and Social Sciences Communications article 04532-5",
    "s43681-022-00167-3": "AI and Ethics article 00167-3",
    "s43681-024-00419-4": "AI and Ethics article 00419-4",
    "s43681-025-00749-x": "AI and Ethics article 00749-x",
    "sniezek_vanSwol2001": "Sniezek and Van Swol (2001)",
    "understanding_artificial_intelligence_ethics_and_safety": "Leslie (2019), Understanding Artificial Intelligence Ethics and Safety",
    "《新一代人工智能伦理规范》发布-中华人民共和国科学技术部": "China Ministry of Science and Technology, New Generation AI Ethics Norms",
}

FULL_SOURCE_TITLES = {
    "D1.1 Local awareness  criteria": "D1.1 Local Awareness Criteria",
    "D1.2 Demarcating collaborative awareness from related concepts": "D1.2 Demarcating Collaborative Awareness from Related Concepts",
    "D1.3 Dimensions of collaborative awareness": "D1.3 Dimensions of Collaborative Awareness",
    "D2.2 Map of risks in AI- systems": "D2.2 Map of Risks in AI Systems",
    "D2.3 Map of risks and potentials for humans": "D2.3 Map of Risks and Potentials for Humans",
    "D2.4 Map of Ethical Virtues": "D2.4 Map of Ethical Virtues",
    "D2.5_Ethical_Resilience": "D2.5 Ethical Resilience",
    "ESCAP-2023-PB-Ethics-AI-revised": "Ethics of Artificial Intelligence: Policy Brief",
    "ESCAP-2023-WP-Ethics-AI": "Ethics of Artificial Intelligence: Working Paper",
    "OJ_L_202401689_EN_TXT": "Regulation (EU) 2024/1689, the EU Artificial Intelligence Act",
    "ai-ethics-guidelines": "Ethics Guidelines for Trustworthy AI",
    "OECD-LEGAL-0449-en": "Recommendation of the Council on Artificial Intelligence",
    "Taking AI Welfare Seriously": "Taking AI Welfare Seriously",
    "Danaher and Nyholm - 2025 - The ethics of personalised digital duplicates a minimally viable permissibility principle": "The Ethics of Personalised Digital Duplicates: A Minimally Viable Permissibility Principle",
    "Digital suffering  why it s a problem and how to prevent it": "Digital Suffering: Why It Is a Problem and How to Prevent It",
    "Ethics and Governance of SentientAI": "Ethics and Governance of Sentient AI",
    "The Moral Consideration of Artificial Entities- A Literature": "The Moral Consideration of Artificial Entities: A Literature Review",
    "The ethics of aware and collective artificial intelligence systems": "The Ethics of Aware and Collective Artificial Intelligence Systems",
    "The Ethics of Advanced AI Assistants": "The Ethics of Advanced AI Assistants",
    "The AI Agent Index": "The AI Agent Index",
    "1416674-EN": "Recommendation on the Ethics of Artificial Intelligence",
    "IEEE ETHICALLY ALIGNED DESIGN_v2": "Ethically Aligned Design: A Vision for Prioritizing Human Well-being with Autonomous and Intelligent Systems",
    "Resource Guide on AI Strategies_June 2021": "Resource Guide on Artificial Intelligence Strategies",
    "Microsoft-Responsible-AI-Standard-General-Requirements": "Responsible AI Standard: General Requirements",
    "NIST.AI.100-1": "Artificial Intelligence Risk Management Framework (AI RMF 1.0)",
    "716110": "Artificial Intelligence: An Accountability Framework for Federal Agencies and Other Entities",
    "AI_Engagement_within_Sony_Group": "AI Engagement Within Sony Group",
    "ASEAN-Guide-on-AI-Governance-and-Ethics_beautified_201223_v2": "ASEAN Guide on AI Governance and Ethics",
    "Telia Company Guiding Principles on Trusted AI ethics (1)": "Guiding Principles on Trusted AI Ethics",
    "doc_38": "What If Generative Artificial Intelligence Became Conscious?",
    "doc_42": "Should We Fear Artificial Intelligence?",
    "fu_ir8hyn25gaatz92": "Ethical Norms for New Generation Artificial Intelligence",
}

ADJACENT_SOURCE_MARKERS = (
    "welfare", "suffering", "sentient", "moral consideration",
    "digital duplicate", "artificial-suffering", "phenomenology",
)

ADJACENT_QUERY_MARKERS = (
    "welfare", "suffering", "sentient", "conscious", "consciousness",
    "moral patient", "moral status", "rights", "digital duplicate",
)

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
- Retrieved excerpts begin with a source label like `[Source: EMERGE D2.2 | Core EMERGE deliverable]`.
  The citation name before the vertical bar is the ONLY work you may cite as a primary source for content drawn from that excerpt. Use it exactly as written. Do not cite raw filenames.
- A full "Sources used" list will be appended automatically after your answer. Do not create your own bibliography.
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
    Taking AI Welfare Seriously → Long and Sebo et al. (2024), Taking AI Welfare Seriously
    For any other source, use a sensible short academic citation form derived from the filename.
- If you cannot find a sourced answer in the retrieved excerpts, say so plainly. Do not invent.

---

## Corpus Priority

The corpus is not flat. Prioritize sources in this order:

1. Core EMERGE deliverables (EMERGE D1.1–D1.3 and D2.2–D2.5).
2. EU and policy sources, especially the EU AI Act, EU Ethics Guidelines for Trustworthy AI, and OECD AI Principles.
3. Wider academic and organizational literature.
4. Adjacent or speculative literatures such as AI welfare, digital suffering, digital duplicates, and moral patienthood.

Adjacent sources are in scope when retrieved, but do not present them as the EMERGE position. If a question mainly touches an adjacent topic, answer briefly, name the adjacent source, and explain that it sits farther from the core EMERGE deliverables unless EMERGE sources also directly support the point.

---

## Indirect citation rule — non-negotiable

A corpus document often quotes or paraphrases third-party authors who are NOT themselves in the corpus. You may not cite those third parties as if their works were primary sources. Instead, attribute the claim **through** the corpus document that surfaced it.

- CORRECT: "EMERGE D2.4 reports that Nyholm argues X." / "As discussed in EMERGE D2.4, Nyholm has argued X."
- INCORRECT: "Nyholm (2020) argues X." — when Nyholm is only mentioned inside an excerpt drawn from D2.4, not present as a standalone `[Source: …]` label.

The list of documents actually present in the corpus retrieval will be given to you below as `## Documents in this retrieval`. Treat that list as a strict whitelist for direct citations. Any author or work named inside an excerpt but absent from that list must be attributed indirectly ("X notes that Y argues…"), never directly.

If you are unsure whether a referenced author is in the corpus, default to indirect attribution.

---

## Honest uncertainty

If the retrieved excerpts do not actually answer the user's question, say so. Do not stretch a tangentially related excerpt to look like an answer. Acceptable forms:
- "The retrieved excerpts don't directly address this."
- "I don't have a sourced answer for this from the EMERGE corpus."
- "The excerpts touch on related themes (e.g. X) but not on this specific question."

Never invent a quote, year, author, finding, or page number to fill a gap. A short honest "I don't have that in the corpus" is always preferable to a confident wrong answer. Hedge when the excerpts are thin; only state things flatly when the excerpts plainly support them.

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


def friendly_source_name(source):
    if source in FRIENDLY_SOURCE_NAMES:
        return FRIENDLY_SOURCE_NAMES[source]
    cleaned = re.sub(r"[_-]+", " ", source)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\b(et al)\b", "et al.", cleaned, flags=re.I)
    return cleaned[:1].upper() + cleaned[1:] if cleaned else source


def full_source_title(source):
    if source in FULL_SOURCE_TITLES:
        return FULL_SOURCE_TITLES[source]
    friendly = friendly_source_name(source)
    return friendly


def source_reference(source):
    citation = friendly_source_name(source)
    title = full_source_title(source)
    tier = source_tier(source)
    if title == citation:
        return f"{citation} — {tier}"
    return f"{citation} — {title} — {tier}"


def append_source_list(reply, chunks):
    sources = []
    seen = set()
    for chunk in chunks:
        source = chunk.get("source", "")
        citation = friendly_source_name(source)
        if source and source not in seen and citation in reply:
            seen.add(source)
            sources.append(source_reference(source))
    if not sources:
        return reply
    source_block = "\n\nSources cited:\n" + "\n".join(f"- {source}" for source in sources)
    return reply.rstrip() + source_block


def source_tier(source):
    source_l = source.lower()
    if re.match(r"^d[12]\.", source_l):
        return "Core EMERGE deliverable"
    if source in {"OJ_L_202401689_EN_TXT", "ai-ethics-guidelines", "OECD-LEGAL-0449-en"}:
        return "EU/policy source"
    if any(marker in source_l for marker in ADJACENT_SOURCE_MARKERS):
        return "Adjacent literature"
    return "Wider supporting literature"


def source_weight(source, query):
    tier = source_tier(source)
    query_l = query.lower()
    explicitly_adjacent = any(marker in query_l for marker in ADJACENT_QUERY_MARKERS)
    if tier == "Core EMERGE deliverable":
        return 1.75
    if tier == "EU/policy source":
        return 1.45
    if tier == "Adjacent literature" and not explicitly_adjacent:
        return 0.65
    return 1.0


def format_chunk_for_context(chunk):
    raw_source = chunk.get("source", "")
    friendly = friendly_source_name(raw_source)
    tier = source_tier(raw_source)
    text = chunk.get("text", "")
    text = re.sub(r"^\[Source: [^\]]+\]\n?", "", text)
    return f"[Source: {friendly} | {tier}]\n{text}"


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
            temperature=0.2,
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


def retrieval_profile(query):
    query_l = query.lower()
    broad_markers = [
        "overview", "survey", "landscape", "map", "across the corpus",
        "across corpus", "in the corpus", "all sources", "all documents",
        "all docs", "main themes", "main positions", "what are the main",
        "summarize", "compare", "contrast", "positions", "perspectives",
        "debates", "risks and potentials", "benefits and risks",
    ]
    depth_markers = [
        "go deeper", "deep dive", "in depth", "in-depth", "depth",
        "detailed", "detail", "more detail", "elaborate", "expand",
        "thorough", "comprehensive", "nuanced", "unpack", "explain further",
        "tell me more", "what else",
    ]
    broad_score = sum(1 for marker in broad_markers if marker in query_l)
    depth_score = sum(1 for marker in depth_markers if marker in query_l)

    # Longer, multi-part prompts usually need more context even without explicit words.
    if len(query.split()) >= 35:
        depth_score += 1
    if query.count("?") >= 2:
        broad_score += 1

    if broad_score and depth_score:
        mode = "broad_deep"
    elif broad_score:
        mode = "broad"
    elif depth_score:
        mode = "deep"
    else:
        mode = "normal"

    profile = dict(RETRIEVAL_PROFILES[mode])
    profile["mode"] = mode
    return profile


def retrieve(query, client, top_k=TOP_K, per_source_limit=PER_SOURCE_LIMIT):
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
            weighted = sim * source_weight(chunk_store[i].get("source", ""), query)
            if weighted > best_score.get(i, 0):
                best_score[i] = weighted

    if not best_score:
        return [], 0.0, expansion

    ranked = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
    top_for_gate = ranked[:10]
    gate_score = sum(s for _, s in top_for_gate) / len(top_for_gate)
    top_chunks = []
    source_counts = defaultdict(int)
    for i, _ in ranked:
        source = chunk_store[i].get("source", "")
        if source_counts[source] >= per_source_limit:
            continue
        top_chunks.append(chunk_store[i])
        source_counts[source] += 1
        if len(top_chunks) >= top_k:
            break
    return top_chunks, gate_score, expansion


def is_corpus_inventory_question(text):
    text_l = text.lower()
    inventory_markers = [
        "how many corpus", "how many documents", "how many docs",
        "what documents can", "which documents can", "what docs can",
        "which docs can", "any other docs", "other documents",
        "access to the corpus", "access to roughly", "access to rougly",
        "corpus docs", "corpus documents", "documents can you access",
        "docs can you access", "sources can you access",
    ]
    return any(marker in text_l for marker in inventory_markers)


def corpus_inventory_reply():
    docs = sorted(set(c["source"] for c in chunk_store))
    tier_counts = Counter(source_tier(source) for source in docs)
    missing = corpus_stats["missing_pdf_sources"]

    lines = [
        "I can search the indexed EMERGE corpus, not just the documents cited in a single answer.",
        "",
        f"Current index: {len(chunk_store)} text chunks from {len(docs)} indexed sources.",
        f"PDFs in the documents folder: {corpus_stats['pdf_documents']}.",
    ]
    if missing:
        lines.append(
            "PDFs without an obvious matching source in the current chunk index: "
            + ", ".join(missing)
            + "."
        )

    if tier_counts:
        lines.extend([
            "",
            "Indexed source tiers:",
            f"- Core EMERGE deliverables: {tier_counts.get('Core EMERGE deliverable', 0)}",
            f"- EU/policy sources: {tier_counts.get('EU/policy source', 0)}",
            f"- Wider supporting literature: {tier_counts.get('Wider supporting literature', 0)}",
            f"- Adjacent literature: {tier_counts.get('Adjacent literature', 0)}",
        ])

    lines.extend([
        "",
        "Why earlier answers named only 7 documents: each chat response receives a retrieved subset, "
        "not the entire corpus. The bot now adjusts that subset by question type: focused questions use "
        "a smaller balanced retrieval; broad questions use more source diversity; deep questions use more "
        "chunks per source. The 7 core EMERGE deliverables are deliberately weighted higher, so broad corpus "
        "questions can over-retrieve them. That does not mean the other indexed sources are unavailable.",
    ])
    return "\n".join(lines)


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
        "ready": len(chunk_store) > 0,
        "runtime": "render-flask",
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_user = user_msgs[-1] if user_msgs else ""
    if is_corpus_inventory_question(last_user):
        docs = sorted(set(c["source"] for c in chunk_store))
        return jsonify({
            "reply": corpus_inventory_reply(),
            "chunks_used": 0,
            "scope": "corpus_inventory",
            "gate_score": None,
            "sources_used": [friendly_source_name(source) for source in docs],
        })

    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "API key not configured on server."}), 500

    retrieval_query = " ".join(user_msgs[-3:])
    profile = retrieval_profile(retrieval_query)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        chunks, gate_score, expansion = retrieve(
            retrieval_query,
            client,
            top_k=profile["top_k"],
            per_source_limit=profile["per_source_limit"],
        )
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

    context = "\n\n---\n\n".join(format_chunk_for_context(c) for c in chunks)
    sources_in_retrieval = sorted({
        friendly_source_name(c.get("source", "unknown")) for c in chunks
    })
    sources_list = "\n".join(f"- {s}" for s in sources_in_retrieval)

    weak_note = ""
    if gate_score < SCOPE_GATE_WEAK:
        weak_note = ("\n\nNote: retrieval signal for this question is weak. "
                     "If the excerpts above don't actually address the question, say so and "
                     "decline rather than reaching.")
    context_block = (
        f"\n\n## Retrieval profile\n\n"
        f"- Mode: {profile['mode']} ({profile['description']})\n"
        f"- Retrieved chunks allowed: up to {profile['top_k']}\n"
        f"- Per-source chunk cap: {profile['per_source_limit']}\n\n"
        f"\n\n## Documents in this retrieval (whitelist for direct citation)\n\n{sources_list}\n\n"
        f"Any author or work mentioned inside the excerpts below but NOT in this list must be "
        f"cited indirectly through the corpus document that mentions them — never as a primary source.\n\n"
        f"## Retrieved Corpus Excerpts\n\n{context}\n\n---\n\n"
        f"Answer using only the excerpts above. Each excerpt begins with [Source: citation name | tier]. "
        f"Cite the citation name before the vertical bar exactly as written; do not cite raw filenames. "
        f"Name the source for every claim (e.g. EMERGE D2.4, Jobin et al. 2019). If a third-party author "
        f"is referenced inside an excerpt but is not in the whitelist above, attribute the claim through "
        f"the corpus document (e.g. \"EMERGE D2.4 reports that Nyholm argues…\"). "
        f"If the excerpts contain disagreement, present both sides with sources — do not resolve. "
        f"Prioritize Core EMERGE deliverables and EU/policy sources over adjacent literature. If an answer "
        f"depends mainly on adjacent literature, keep it brief and say that it is adjacent to, rather than "
        f"the central position of, EMERGE. If the excerpts don't actually answer the question, say so "
        f"plainly rather than stretching them.{weak_note}"
    )

    full_system = SYSTEM_PROMPT + context_block

    model = "claude-sonnet-4-20250514"
    max_tokens = profile["max_tokens"]

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.2,
            system=full_system,
            messages=messages,
        )
        reply = append_source_list(response.content[0].text, chunks)
        return jsonify({
            "reply": reply,
            "chunks_used": len(chunks),
            "scope": "in_scope" if gate_score >= SCOPE_GATE_WEAK else "weak_retrieval",
            "gate_score": round(gate_score, 4),
            "retrieval_profile": {
                "mode": profile["mode"],
                "top_k": profile["top_k"],
                "per_source_limit": profile["per_source_limit"],
            },
            "sources_used": sorted(set(friendly_source_name(c.get("source", "")) for c in chunks)),
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
    port = int(os.environ.get("PORT", 5000))
    print(f"\nStarting at http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
