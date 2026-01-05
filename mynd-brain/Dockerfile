# MYND Brain - Runpod GPU Dockerfile
# =====================================
# Runs the MYND Brain ML server with NVIDIA GPU acceleration
# Includes Claude CLI for chat functionality

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22 (required for Claude CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user "axel" for Claude CLI execution
# (Claude CLI refuses to run as root)
RUN useradd -m -s /bin/bash axel \
    && mkdir -p /home/axel/.config \
    && chown -R axel:axel /home/axel

# Set up working directory
WORKDIR /app

# Install Python dependencies first (for layer caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Claude CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data /app/models_cache \
    && chown -R axel:axel /app/data /app/models_cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/models_cache
ENV TORCH_HOME=/app/models_cache
ENV XDG_CACHE_HOME=/app/models_cache

# Port for the server
EXPOSE 8420

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8420/health || exit 1

# Start server
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8420"]
