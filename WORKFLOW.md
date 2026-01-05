# MYND Development Workflow

## Git Workflow

1. **Claude pushes** commits to feature branch (`claude/...`)
2. **You review and merge** to `main` on GitHub
3. **Pull to local**: `cd /Users/jellingson/Dev/mynd-server && git pull origin main`
4. **Pull to Runpod**: `cd /workspace/mynd-v3 && git pull origin main`

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

### Connecting Local Frontend to Runpod
```
http://localhost:8080/self-dev.html?brain=https://[POD-ID]-8420.proxy.runpod.net
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
2. Upload `data.zip` via file browser
3. In terminal:
```bash
cd /workspace/mynd-v3/mynd-brain
apt-get update && apt-get install -y unzip  # if needed
unzip /workspace/data.zip
```

## Frontend Versions

- `index.html` - Consumer version
- `self-dev.html` - Development version (primary)

## Claude CLI Authentication

On Runpod, Claude CLI must be authenticated for chat to work:
```bash
claude login
```
Follow prompts to authenticate with Max subscription.

## Ports

| Service | Port | URL Pattern |
|---------|------|-------------|
| Local Frontend | 8080 | `http://localhost:8080/` |
| Local Brain | 8420 | `http://localhost:8420/` |
| Runpod Jupyter | 8888 | `https://[POD-ID]-8888.proxy.runpod.net/` |
| Runpod Brain | 8420 | `https://[POD-ID]-8420.proxy.runpod.net/` |
