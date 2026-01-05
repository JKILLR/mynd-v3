#!/bin/bash
# MYND Brain - Runpod Startup Script
# ====================================
# This script configures and starts the MYND Brain server on Runpod GPU instances.
#
# Usage:
#   chmod +x start-runpod.sh
#   ./start-runpod.sh
#
# Environment Variables (optional):
#   WHISPER_MODEL - Whisper model size (tiny/base/small/medium/large)
#   MYND_BRAIN_PORT - Server port (default: 8420)

set -e

echo "🧠 MYND Brain - Runpod Startup"
echo "==============================="

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    export TORCH_DEVICE=cuda
else
    echo "⚠️ No NVIDIA GPU detected, using CPU"
    export TORCH_DEVICE=cpu
fi

# Set defaults for Runpod (larger models since we have GPU)
export WHISPER_MODEL="${WHISPER_MODEL:-medium}"
export MYND_BRAIN_PORT="${MYND_BRAIN_PORT:-8420}"
export MYND_BRAIN_HOST="0.0.0.0"

echo ""
echo "📋 Configuration:"
echo "   Port: $MYND_BRAIN_PORT"
echo "   Device: $TORCH_DEVICE"
echo "   Whisper Model: $WHISPER_MODEL"
echo ""

# Check if Claude CLI is authenticated
echo "🔑 Checking Claude CLI authentication..."
if ! command -v claude &> /dev/null; then
    echo "⚠️ Claude CLI not found. Installing..."
    npm install -g @anthropic-ai/claude-code
fi

# Attempt to check CLI status - this will prompt if not authenticated
if claude auth status &> /dev/null; then
    echo "✅ Claude CLI authenticated"
else
    echo "⚠️ Claude CLI not authenticated. Chat will use fallback mode."
    echo "   To authenticate, run: claude login"
fi

# Create data directories if needed
mkdir -p /app/data /app/models_cache 2>/dev/null || true

echo ""
echo "🚀 Starting MYND Brain server..."
echo "   Access at: http://0.0.0.0:$MYND_BRAIN_PORT"
echo ""

# Start the server
cd "$(dirname "$0")"
python3 -m uvicorn server:app --host 0.0.0.0 --port $MYND_BRAIN_PORT
