# EMERGE AI Ethics Information Bot
## Setup & Usage Guide

---

## What this is
A local web app that retrieves relevant excerpts from your EMERGE corpus and passes them to Claude with the bot's full system prompt and sourcing rules. Two ways to interact: type a free-form prompt, or pick a predefined vignette.

---

## First-time setup (do this once)

### Step 1 — Add your PDFs
Put all corpus PDFs into the **`documents/`** folder. The filename becomes the raw source name; the system prompt maps known filenames to friendly citations (e.g. `D2.4 Map of Ethical Virtues` → "EMERGE D2.4").

### Step 2 — Set your API key
Set the `ANTHROPIC_API_KEY` environment variable before starting the bot. The server reads it from the environment — there is no in-browser key form.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

(Add it to your shell profile if you want it set automatically.)

### Step 3 — Start the bot
**On Mac:** Double-click **START.command**
- If Mac asks "are you sure?", click Open
- The first run installs dependencies into `.venv/`
- When you see `Running on http://localhost:5000`, the bot is ready

### Step 4 — Open your browser
Go to: **http://localhost:5050**

(The default is `5050` because macOS AirPlay Receiver squats on `5000`. To change it, set `PORT=5000` in your shell before launching, or pass it inline.)

---

## Everyday use
1. Double-click START.command
2. Open http://localhost:5000 in your browser
3. Ask questions

**To stop:** Press Ctrl+C in the Terminal window, or just close it.

---

## Adding new documents
1. Stop the bot (Ctrl+C)
2. Add the new PDF to the `documents/` folder
3. Rebuild the corpus index:
```bash
python scripts/build_chunks.py
```
4. Commit the updated `documents/` and `chunks.json`, then redeploy/restart the bot.

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
```
Then check `/status`; `missing_pdf_sources` should be empty or explain what still needs attention.

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
Claude Sonnet is used for all responses.

---

## For publication / methodology description
This system uses:
- **PDF extraction:** pdfplumber (offline preprocessing into `chunks.json`)
- **Chunking:** sliding window, 400 words per chunk, 80-word overlap
- **Retrieval:** TF-IDF cosine over the chunk index, with three-pass retrieval (original query + LLM-generated paraphrase + LLM-generated counter-query), source diversity, and broader context windows to improve recall and surface contrasting positions
- **Scope gating:** mean cosine of top-10 chunks; below threshold, the system refuses and points to an external resource category
- **Models:** Claude Haiku 4.5 for query expansion; Claude Sonnet 4 for user-facing answers
- **System prompt:** sourcing rules, two-sidedness on contested questions, scope boundaries, friendly-name conventions

---

EMERGE Project · WP2 Deliverable 2.6 · EU Horizon Europe Grant 101070918
Ludwig-Maximilians-Universität München
