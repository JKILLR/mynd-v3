# MYND Project State

> Last updated: 2025-01-05

## Current Setup

### Local Machine (Mac)
- **Path**: `/Users/jellingson/Dev/mynd-server`
- **User**: jellingson

### Runpod Pod
- **Pod ID**: `ztbjpe1pv628kq`
- **Jupyter**: `https://ztbjpe1pv628kq-8888.proxy.runpod.net/lab`
- **Brain Server**: `https://ztbjpe1pv628kq-8420.proxy.runpod.net`
- **SSH**: `ssh root@69.30.85.79 -p 22028`
- **Workspace**: `/workspace/mynd-v3`

### Vercel (frontend hosting - currently not used)
- URL: `https://mynd-v3.vercel.app`
- Has CORS issues with Runpod - using local frontend instead

## What's Working
- [ ] Local frontend server
- [ ] Runpod brain server
- [ ] LocalBrain connection (frontend -> Runpod)
- [ ] Claude CLI authentication on Runpod
- [ ] Chat with Axel via CLI
- [ ] Data sync to Runpod

## Current Issues
- CORS errors when using Vercel frontend with Runpod backend
- Need to verify Claude CLI is authenticated on Runpod

## Data Files (Axel's Brain)

Located in `mynd-brain/data/`:

| File/Folder | Purpose |
|-------------|---------|
| `gt_weights.pt` | Graph Transformer trained weights |
| `conversations/*.json` | Archived conversations (~400+) |
| `graph/graph.json` | Mind map structure |
| `learning/meta_learner.json` | Meta learner state |
| `learning/prediction_tracker.json` | Prediction accuracy tracking |
| `learning/knowledge_distiller.json` | Extracted knowledge patterns |

**Important**: These files are NOT in git (personal data). Must be manually synced to Runpod.

## Recent Changes
- 2025-01-05: Added Runpod GPU deployment support (Dockerfile, env vars, ?brain= URL param)
- 2025-01-05: Migrated from Anthropic API to Claude CLI (uses Max subscription)

## Architecture

```
[Browser: self-dev.html]
        |
        | HTTP (localhost:8080)
        v
[Local File Server: python -m http.server]
        |
        | ?brain= parameter
        v
[Runpod Brain Server: port 8420]
        |
        | subprocess
        v
[Claude CLI: Max subscription]
```

## Quick Start Checklist

1. [ ] Start Runpod pod
2. [ ] On Runpod: `cd /workspace/mynd-v3/mynd-brain && python server.py`
3. [ ] Verify Claude CLI auth: `claude auth status`
4. [ ] On Mac: `cd /Users/jellingson/Dev/mynd-server && python3 -m http.server 8080`
5. [ ] Open: `http://localhost:8080/self-dev.html?brain=https://ztbjpe1pv628kq-8420.proxy.runpod.net`
