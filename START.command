#!/bin/bash
# EMERGE Ethics Bot - Start Script (Mac)
# Double-click this file to start the bot, OR run it in Terminal.

echo "================================================"
echo "  EMERGE AI Ethics Information Bot"
echo "================================================"
echo ""

# Go to the folder this script is in
cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found."
    echo "Please install from https://python.org/downloads"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Python found: $(python3 --version)"
echo ""

# Use a local venv to avoid PEP 668 / Homebrew clashes
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install -q -r requirements.txt

# Default to port 5050 — macOS AirPlay Receiver squats on 5000
export PORT="${PORT:-5050}"

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "WARNING: ANTHROPIC_API_KEY is not set. The /chat endpoint will return an error."
    echo "Set it in your shell before running, e.g.:"
    echo "    export ANTHROPIC_API_KEY=sk-ant-..."
    echo ""
fi

echo ""
echo "Starting server..."
echo "When you see 'Running on http://localhost:$PORT', open your browser"
echo "and go to:  http://localhost:$PORT"
echo ""
echo "To stop the bot: press Ctrl+C"
echo "================================================"
echo ""

.venv/bin/python app.py
