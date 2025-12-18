#!/bin/bash
# MYND Brain - Startup Script
# Run this to start your local ML server

echo "🧠 Starting MYND Brain..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "📦 Installing dependencies (first run)..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/.installed
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   🧠  MYND BRAIN - Local ML Server                           ║"
echo "║                                                               ║"
echo "║   Running on: http://localhost:8420                          ║"
echo "║   Health check: http://localhost:8420/health                 ║"
echo "║                                                               ║"
echo "║   Press Ctrl+C to stop                                       ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Run the server
python -m uvicorn server:app --host 127.0.0.1 --port 8420 --reload
