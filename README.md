# EMERGE AI Ethics Information Bot
## Setup & Usage Guide

---

## What this is
A local web app that reads your PDF corpus, retrieves relevant excerpts for each question, and passes them to Claude with your full system prompt and sourcing rules.

---

## First-time setup (do this once)

### Step 1 — Add your PDFs
Put all your corpus PDFs into the **`documents/`** folder inside this project folder.
- You can add as many PDFs as you like
- Rename them clearly if needed (the filename becomes the source name in responses)
  - e.g. `D2_4_Map_of_Ethical_Virtues.pdf` → cited as "D2_4_Map_of_Ethical_Virtues"
  - Tip: rename to match how you want them cited, e.g. `EMERGE_D2_4.pdf`

### Step 2 — Start the bot
**On Mac:** Double-click **START.command**
- If Mac asks "are you sure?", click Open
- A Terminal window opens and installs dependencies automatically (first time only)
- When you see `Running on http://localhost:5000`, the bot is ready

### Step 3 — Open your browser
Go to: **http://localhost:5000**

### Step 4 — Enter your API key
Paste your Anthropic API key (from console.anthropic.com) in the yellow bar at the top.
It saves in your browser — you only need to do this once per browser.

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
3. Delete the file called `chunks_cache.json` (this forces a re-read of all PDFs)
4. Start the bot again

---

## Troubleshooting

**"Server not reachable" in the corpus bar**
→ The Python server isn't running. Start it with START.command first.

**"No documents loaded"**
→ The `documents/` folder is empty. Add your PDFs.

**"Invalid API key"**
→ Check you copied the full key from console.anthropic.com (starts with sk-ant-)

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
- **PDF extraction:** pdfplumber
- **Chunking:** sliding window, 400 words per chunk, 80-word overlap
- **Retrieval:** keyword overlap scoring (TF-style), top-6 chunks retrieved per query
- **Model:** Claude claude-sonnet-4-20250514 via Anthropic API
- **System prompt:** embedded sourcing rules, priority sources by topic, scope boundaries

---

EMERGE Project · WP2 Deliverable 2.6 · EU Horizon Europe Grant 101070918
Ludwig-Maximilians-Universität München
