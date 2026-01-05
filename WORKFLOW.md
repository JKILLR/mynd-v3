# MYND Development Workflow

## Git Workflow

1. **Claude pushes** commits to feature branch (`claude/...`)
2. **You review and merge** to `main` on GitHub
3. **Pull to local**: `cd /Users/jellingson/Dev/mynd-server && git pull origin main`
4. **Pull to Runpod**: `cd /workspace/mynd-v3 && git pull origin main`

## Runpod First-Time Setup

### 1. Install Node.js (required for Claude CLI)
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs
```

### 2. Install Claude CLI
```bash
npm install -g @anthropic-ai/claude-code
```

### 3. Authenticate Claude CLI
```bash
claude login
```
Follow prompts to authenticate with Max subscription.

### 4. Create "axel" user (required for CLI to run as non-root)
```bash
useradd -m -s /bin/bash axel
chown -R axel:axel /home/axel
```
The brain server runs Claude CLI as this user because `--dangerously-skip-permissions` requires non-root.

### 5. Install unzip (for data uploads)
```bash
apt-get update && apt-get install -y unzip
```

### 6. Clone repo (if not already done)
```bash
cd /workspace
git clone https://github.com/JKILLR/mynd-v3.git
```

### 7. Install Python dependencies
```bash
cd /workspace/mynd-v3/mynd-brain
pip install -r requirements.txt
```

## Starting the System

### Local Frontend (on Mac)
```bash
cd /Users/jellingson/Dev/mynd-server
python3 -m http.server 8080
```
Access: `http://localhost:8080/self-dev.html`

### Runpod Brain Server
```bash
cd /workspace/mynd-v3/mynd-brain
python server.py
```
Server runs on port 8420.

### Cloudflare Tunnel (required - Runpod proxy doesn't work well)

Runpod's built-in proxy (`[POD-ID]-8420.proxy.runpod.net`) has issues. Use Cloudflare tunnel instead:

```bash
cd /workspace
./cloudflared tunnel --url http://localhost:8420
```

If cloudflared not installed:
```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:8420
```

This outputs a URL like: `https://something-random.trycloudflare.com`

**Note:** The Cloudflare URL changes each time you restart the tunnel.

### Connecting Local Frontend to Runpod
```
http://localhost:8080/self-dev.html?brain=https://[CLOUDFLARE-URL]
```

Example:
```
http://localhost:8080/self-dev.html?brain=https://fair-nationally-ddr-practical.trycloudflare.com
```

### Local-only Setup (no Runpod)
Terminal 1 - Frontend:
```bash
cd /Users/jellingson/Dev/mynd-server
python3 -m http.server 8080
```

Terminal 2 - Brain server:
```bash
cd /Users/jellingson/Dev/mynd-server/mynd-brain
python3 server.py
```

Access: `http://localhost:8080/self-dev.html` (auto-connects to localhost:8420)

## Syncing Data to Runpod

Axel's brain data (conversations, GT weights, etc.) must be manually uploaded:

### On Mac - Create zip:
```bash
cd /Users/jellingson/Dev/mynd-server/mynd-brain
zip -r data.zip data/
```

### Upload to Runpod:
1. Open Jupyter Lab: `https://[POD-ID]-8888.proxy.runpod.net/lab`
2. Upload `data.zip` via file browser (drag & drop)
3. In terminal:
```bash
cd /workspace/mynd-v3/mynd-brain
unzip /workspace/data.zip -o
```
(The `-o` flag overwrites existing files without prompting)

## Frontend Versions

- `index.html` - Consumer version
- `self-dev.html` - Development version (primary)

## Ports

| Service | Port | URL Pattern |
|---------|------|-------------|
| Local Frontend | 8080 | `http://localhost:8080/` |
| Local Brain | 8420 | `http://localhost:8420/` |
| Runpod Jupyter | 8888 | `https://[POD-ID]-8888.proxy.runpod.net/` |
| Runpod Brain | 8420 | Via Cloudflare tunnel |

## Quick Start Checklist (Runpod)

1. [ ] Start Runpod pod from dashboard
2. [ ] Open Jupyter Lab terminal
3. [ ] Start brain server: `cd /workspace/mynd-v3/mynd-brain && python server.py`
4. [ ] In another terminal, start tunnel: `cd /workspace && ./cloudflared tunnel --url http://localhost:8420`
5. [ ] Copy the Cloudflare URL
6. [ ] On Mac: `cd /Users/jellingson/Dev/mynd-server && python3 -m http.server 8080`
7. [ ] Open: `http://localhost:8080/self-dev.html?brain=https://[CLOUDFLARE-URL]`

## Troubleshooting

### "claude: command not found"
Install Node.js and Claude CLI (see First-Time Setup above)

### CORS errors in browser
Use Cloudflare tunnel instead of Runpod proxy

### "Cannot read properties of undefined (reading 'includes')"
Usually means LocalBrain request failed - check Cloudflare tunnel is running and URL is correct

### Server not loading data
Make sure you uploaded and unzipped the data.zip file in the correct location
