# EMERGE AI Ethics Information Bot
## Setup & Usage Guide

---

## What this is
A local web app that retrieves relevant excerpts from your EMERGE corpus and passes them to Claude with the bot's full system prompt and sourcing rules. Type a free-form prompt to explore the corpus.

---

## First-time setup (do this once)

### Step 1 — Add your PDFs
Put all corpus PDFs into the **`documents/`** folder. The filename becomes the raw source name; the system prompt maps known filenames to friendly citations (e.g. `D2.4 Map of Ethical Virtues` → "EMERGE D2.4").

### Step 2 — Set your API key
Set the `ANTHROPIC_API_KEY` environment variable before starting the bot. The server reads it from the environment — there is no in-browser key form.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

For vector retrieval, also set `OPENAI_API_KEY`. If a vector index or OpenAI key is missing, the app falls back to the older TF-IDF retriever.

```bash
export OPENAI_API_KEY=sk-...
```

To enable the password-protected chat log admin page, set `ADMIN_PASSWORD`:

```bash
export ADMIN_PASSWORD='choose-a-long-password'
```

(Add it to your shell profile if you want it set automatically.)

### Step 3 — Start the bot
**On Mac:** Double-click **START.command**
- If Mac asks "are you sure?", click Open
- The first run installs dependencies into `.venv/`
- When you see `Running on http://localhost:5050`, the bot is ready

### Step 4 — Open your browser
Go to: **http://localhost:5050**

(The default is `5050` because macOS AirPlay Receiver squats on `5000`. To change it, set `PORT=5000` in your shell before launching, or pass it inline.)

---

## Everyday use
1. Double-click START.command
2. Open http://localhost:5050 in your browser
3. Ask questions

The chat has three answer modes:
- **Accurate:** tight retrieval, lower context count, and temperature `0.0` for maximum source fidelity.
- **Balanced:** default source-grounded answers with normal retrieval and temperature `0.2`.
- **Brainstorm:** wider retrieval and temperature `0.45` for source-grounded ideation.

Before retrieval, the bot also runs a lightweight triage. Greetings, very vague prompts, and broad technical implementation requests are answered with a clarifying question instead of searching the corpus.

The bot page includes a corpus status bar showing whether chunks are loaded, how many sources are indexed, which retrieval backend is active, and whether the vector index is missing or stale.

**To stop:** Press Ctrl+C in the Terminal window, or just close it.

## Admin chat logs
When `ADMIN_PASSWORD` is set, every chat exchange is stored in `chat_logs.db` for later analysis. Open:

```bash
http://localhost:5050/admin/login
```

Log in with username `admin` and your `ADMIN_PASSWORD`. You can also open `/admin/chats` directly and the app will redirect you to the login page. The admin page groups turns into full chat sessions, with each user prompt and bot answer shown chronologically. It supports search, session-level CSV export, and raw turn-level CSV export. The log includes the user message, bot reply, anonymous visitor/session IDs, answer mode, temperature, sources used, retrieval mode, scope result, gate score, errors, and latency.

If the login page says admin logs are disabled, `ADMIN_PASSWORD` is not set on the running server. Set it in your local shell or hosting environment, then restart/redeploy the app.

By default the app does not store client IP addresses. To include them, set:

```bash
export CHAT_LOG_IPS=true
```

---

## Adding new documents
1. Stop the bot (Ctrl+C)
2. Add the new PDF to the `documents/` folder
3. Rebuild the corpus index:
```bash
python scripts/build_chunks.py
```
4. Rebuild the vector index:
```bash
python scripts/build_vector_index.py
```
5. Rebuild the source metadata:
```bash
python scripts/build_source_metadata.py
```
6. For a private deployment, make sure the updated corpus artifacts are available to the server, then redeploy/restart the bot. For a public academic repository, do not commit `documents/`, `chunks.json`, or `vector_index.json.gz` unless redistribution rights are explicit.

---

## Troubleshooting

**"Server not reachable" in the corpus bar**
→ The Python server isn't running. Start it with START.command first.

**"No documents loaded"**
→ The `documents/` folder is empty. Add your PDFs.

**A PDF is uploaded but the bot ignores it**
→ The bot answers from `chunks.json`, not live PDF reads. Run:
```bash
python scripts/build_chunks.py
python scripts/build_vector_index.py
```
Then check `/status`; `missing_pdf_sources` should be empty or explain what still needs attention.

If a PDF still appears in `missing_pdf_sources` after rebuilding, it may be scanned or image-only. Convert it to a text-readable/OCR PDF, then rerun both build commands.

**"Invalid API key"** or **"API key not configured on server"**
→ The server reads the key from the `ANTHROPIC_API_KEY` environment variable. Set it before starting:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
Then restart the bot.

**Bot gives vague answers without named sources**
→ The retrieved chunks may not contain enough source context. 
   Try renaming your PDFs to match their citation names exactly.

---

## Cost estimate
At ~100 questions/month: roughly $2–5/month in API costs.
Claude Sonnet is used for all responses. If vector retrieval is enabled, OpenAI embeddings are also used once when building the index and once per retrieval query at runtime.

---

## For publication / methodology description
This system uses:
- **PDF extraction:** pdfplumber (offline preprocessing into `chunks.json`)
- **Chunking:** sliding window, 400 words per chunk, 80-word overlap
- **Retrieval:** embedding cosine over `vector_index.json.gz` when available, with TF-IDF fallback; three-pass retrieval (original query + LLM-generated paraphrase + LLM-generated counter-query), source diversity, and broader context windows to improve recall and surface contrasting positions
- **Scope gating:** mean cosine of top-10 chunks; below threshold, the system refuses and points to an external resource category
- **Source metadata:** `source_metadata.json` contains bibliography/source records for the corpus without document text, chunks, embeddings, or chat logs. The `/sources` route exposes the same records at runtime.
- **Models:** OpenAI `text-embedding-3-small` for vector retrieval; Claude Haiku 4.5 for query expansion; Claude Sonnet 4 for user-facing answers
- **System prompt:** sourcing rules, two-sidedness on contested questions, scope boundaries, friendly-name conventions

For local development the app defaults to port `5050`. Hosted deployments may set a different `PORT` value through the hosting environment.

For production deployment the Render configuration runs the Flask app through Gunicorn (`gunicorn app:app`). The corpus is loaded when the app module is imported, so both local `python app.py` and Gunicorn deployments use the same indexed chunks.

For a public academic release, keep secrets and corpus-derived artifacts out of git:
- Never commit `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `.env`, or Render secret values.
- Do not commit `chat_logs.db`; logs can contain user prompts and failure details.
- Do not commit `documents/` or `vector_index.json.gz` unless the PDFs and derived data are cleared for redistribution.
- If the public Render service deploys directly from this repository, `chunks.json` must be present or supplied by another private build step. Without it, the app will refuse chat requests because it has no retrievable corpus.
- It is acceptable to publish `source_metadata.json` when bibliography-level source labels and document metadata are cleared. Review it before release, especially DOI/URL/author fields.
- Publish the code, setup instructions, methodology, source metadata, and rebuild scripts instead. Use a private deployment source or institutional storage for the actual corpus artifacts.

Render can still use the secret `OPENAI_API_KEY` to build `vector_index.json.gz` during deployment when `chunks.json` is present in the private deployment source.

---

EMERGE Project · WP2 Deliverable 2.6
Ludwig-Maximilians-Universität München
