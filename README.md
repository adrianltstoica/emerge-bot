# EMERGE Ethics Toolkit

Research software for source-grounded exploration of the ethics of aware and collective AI systems.

[Live toolkit](https://emerge-bot.onrender.com/) · [Source register](https://emerge-bot.onrender.com/#sources) · [Evaluation endpoint](https://emerge-bot.onrender.com/evaluation)

## Overview

The EMERGE Ethics Toolkit is a public-facing web application built for the EMERGE project and MI3 at Ludwig-Maximilians-Universität München. It provides a structured, source-grounded interface for exploring ethical questions raised by aware, collective, and high-impact AI systems.

The application combines five publication-facing surfaces:

| Surface | Purpose |
| --- | --- |
| Ethics Wiki | Concise syntheses of recurring concepts in the EMERGE corpus, including collaborative awareness, trust, benevolence, explainability, responsibility gaps, risks of aware AI, ethical resilience, and the EU AI Act. |
| Source Register | Bibliographic and corpus metadata for indexed sources, including source tier, citation label, DOI or stable URL where available, PDF page count, chunk count, and extractability flags. |
| Methodology & Limits | A transparent account of corpus preprocessing, retrieval, generation, scope boundaries, logging, and evaluation hooks. |
| Stakeholder Checklist | Role-specific review prompts for regulators, developers, researchers, and product users, with evidence expectations, risk notes, and source links. |
| Ethics Bot | A natural-language retrieval-augmented assistant that answers only from the indexed EMERGE corpus and returns named sources. |

This repository is intended to support an academic/public deliverable, not to provide legal advice or replace reading the underlying EMERGE deliverables and policy documents.

## Project Context

The toolkit supports EMERGE WP2 Deliverable 2.6 by making project findings easier to inspect, cite, and discuss. The corpus centers on EMERGE deliverables and selected policy or scholarly sources relevant to:

- local and collaborative awareness;
- risks and potentials of aware AI;
- trust, explainability, benevolence, and ethical resilience;
- responsibility gaps in distributed human-AI and AI-AI systems;
- trustworthy AI guidelines, including the EU AI Act and HLEG framework.

At the current indexed snapshot, the repository contains 84 source metadata records and 4,486 retrieval chunks from 83 text-indexed sources.

## Methodological Design

The system is a retrieval-augmented generation application with explicit scope control.

| Stage | Implementation |
| --- | --- |
| Corpus preparation | PDF files in `documents/` are processed offline into `chunks.json` using `pdfplumber`. |
| Chunking | Documents are split into sliding windows of approximately 400 words with 80-word overlap. |
| Source metadata | `source_metadata.json` and `source_metadata.csv` record citation labels, source tiers, document metadata, DOI/URL fields, page counts, chunk counts, and extractability flags. |
| Retrieval | Vector retrieval uses OpenAI `text-embedding-3-small` when `vector_index.json.gz` is available. The app falls back to TF-IDF if the vector index or API key is unavailable. |
| Query expansion | The app can generate paraphrase and counter-query variants before retrieval to improve recall and surface contrasting material. |
| Source diversity | Retrieval limits over-concentration from one source and broadens context around selected chunks. |
| Scope gating | Low-similarity or explicitly out-of-scope prompts are refused or redirected instead of receiving invented corpus-grounded answers. |
| Answer generation | User-facing answers are generated through the configured Anthropic model with source and scope rules in the system prompt. |
| Evaluation | `/evaluation` exposes a seed validation set with expected concepts, reference sources, and metric definitions. |

A diagnostic `/status` route is retained for maintainers, but runtime internals are not linked from the public user interface.

## Repository Structure

```text
.
├── app.py                         # Flask application, retrieval, routes, chat logging, admin views
├── static/index.html              # Public toolkit UI
├── documents/                     # Corpus PDFs cleared for this project repository
├── chunks.json                    # Generated retrieval chunks
├── source_metadata.json           # Bibliographic/corpus metadata
├── source_metadata.csv            # Spreadsheet-ready source register
├── evaluation_questions.json      # Seed validation questions and metrics
├── scripts/
│   ├── build_chunks.py            # Rebuild retrieval chunks from PDFs
│   ├── build_source_metadata.py   # Rebuild source metadata records
│   ├── build_vector_index.py      # Build vector index from chunks
│   └── ensure_vector_index.py     # Build/reuse deployment vector index
├── tests/test_app.py              # Route and static-surface tests
├── render.yaml                    # Render deployment configuration
├── START.command                  # macOS local launcher
└── requirements.txt               # Python dependencies
```

## Running Locally

### 1. Create a virtual environment

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On macOS, `START.command` performs the setup and starts the app for non-technical local use.

### 2. Configure model keys

The server reads credentials from environment variables. Do not enter API keys in the browser.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

Required and optional variables:

| Variable | Required | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Required for chat answers | Calls the answer and query-expansion models. |
| `OPENAI_API_KEY` | Optional | Enables vector retrieval and vector-index generation. |
| `ANTHROPIC_MODEL` | Optional | Defaults to `claude-sonnet-4-6`. |
| `ANTHROPIC_EXPANSION_MODEL` | Optional | Defaults to `claude-haiku-4-5-20251001`. |
| `EMBEDDING_MODEL` | Optional | Defaults to `text-embedding-3-small`. |
| `PORT` | Optional | Defaults to `5000` in `app.py`; `5050` is commonly used locally on macOS. |
| `ADMIN_PASSWORD` | Optional | Enables password-protected access to admin chat-log views. |
| `ADMIN_SESSION_SECRET` | Recommended when admin is enabled | Keeps admin sessions stable across restarts. |
| `CHAT_LOG_DB` | Optional | Path to the SQLite chat-log database. |
| `CHAT_LOG_IPS` | Optional | Set to `true` to store client IPs; disabled by default. |

### 3. Start the app

```bash
PORT=5050 python app.py
```

Then open:

```text
http://localhost:5050
```

## Updating the Corpus

When source PDFs change, rebuild the generated artifacts before deployment.

```bash
python scripts/build_chunks.py
python scripts/build_source_metadata.py
python scripts/build_vector_index.py
```

Commit the updated corpus artifacts that are intended for publication:

- `documents/`
- `chunks.json`
- `source_metadata.json`
- `source_metadata.csv`

The vector index is treated as a generated retrieval artifact and is excluded from git by default. On Render, `scripts/ensure_vector_index.py` can rebuild `/var/data/vector_index.json.gz` on startup when `OPENAI_API_KEY` is configured.

## Deployment

The included `render.yaml` deploys the app as a Render web service:

- installs Python dependencies from `requirements.txt`;
- rebuilds source metadata when corpus files are present;
- ensures a vector index exists at startup when possible;
- runs the Flask app with Gunicorn;
- stores chat logs and the vector index on a persistent disk mounted at `/var/data`.

The production start command is:

```bash
python scripts/ensure_vector_index.py && gunicorn app:app
```

## Evaluation and Quality Assurance

Run the test suite with:

```bash
PYTHONPATH=. pytest -q
```

The current tests check:

- core public routes;
- source metadata and CSV export behavior;
- evaluation endpoint structure;
- public toolkit surfaces in the homepage;
- presence of methodological, privacy, checklist, source-register, and citation-linking UI elements.

The public `/evaluation` endpoint provides a seed validation set for future systematic assessment. The intended metrics include response accuracy, citation validity, citation precision, hallucination rate, completeness, refusal accuracy, and retrieval coverage.

## Privacy and Logging

Chat requests are logged to SQLite for evaluation and debugging. Logged fields can include:

- user prompt and assistant reply;
- anonymous visitor and conversation identifiers;
- answer mode and temperature;
- retrieval mode and retrieval backend;
- sources used;
- scope classification and gate score;
- errors and latency;
- user agent.

Client IP addresses are not stored by default. Set `CHAT_LOG_IPS=true` only when there is a clear operational reason and an appropriate privacy basis.

Admin access to logs is disabled unless `ADMIN_PASSWORD` is configured. If enabled, admin routes are available under `/admin/login`, `/admin/chats`, `/admin/chats.csv`, and `/admin/sessions.csv`.

## Limitations

- The bot answers from the indexed corpus only; it is not a general web-search assistant.
- Legal and policy answers are informational and should not be treated as legal advice.
- Source-grounded synthesis can still omit nuance or over-compress a contested issue.
- OCR quality, PDF formatting, and metadata completeness affect retrieval quality.
- The source register records bibliographic metadata; it is not a substitute for the original source documents.
- The corpus includes project-cleared PDFs and derived artifacts. Reuse or redistribution should be checked against the relevant document permissions.

## Citation

If you use this repository, the public toolkit, or its methodology in academic work, cite the software repository and the underlying EMERGE deliverables that support the claim being discussed. A GitHub-compatible citation file is provided in `CITATION.cff`.

Suggested repository citation:

```text
EMERGE Project and MI3, Ludwig-Maximilians-Universität München. EMERGE Ethics Toolkit: Research software for source-grounded exploration of aware and collective AI ethics. GitHub repository, 2026.
```

## Reuse Status

No repository-wide open-source license is declared in this snapshot. Before reuse, redistribution, or derivative publication, verify the rights status of the code, PDFs, generated chunks, metadata, and any hosted chat logs.

Runtime secrets and private artifacts must not be committed:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `.env` files
- `chat_logs.db`
- `vector_index.json.gz`

## Acknowledgement

EMERGE Project · WP2 Deliverable 2.6<br>
MI3 · Ludwig-Maximilians-Universität München
