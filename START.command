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

# Install dependencies if needed
echo "Checking dependencies..."
python3 -m pip install -r requirements.txt -q --break-system-packages 2>/dev/null || python3 -m pip install -r requirements.txt -q

echo ""
echo "Starting server..."
echo "When you see 'Running on http://localhost:5000', open your browser"
echo "and go to:  http://localhost:5000"
echo ""
echo "To stop the bot: press Ctrl+C"
echo "================================================"
echo ""

python3 app.py
