# Claude API to CLI Migration Guide

## Overview

This document describes the migration from Anthropic API (per-token billing) to Claude Code CLI (included in Max subscription).

## Migration Status

### Completed Migrations

| Component | File | Status |
|-----------|------|--------|
| `/brain/chat` endpoint | `server.py` | ✅ Migrated to CLI |
| Evolution Daemon | `evolution_daemon.py` | ✅ Migrated to CLI |
| Knowledge Extractor | `models/knowledge_extractor.py` | ✅ Migrated to CLI |

### Pending Decision: Supabase Edge Function

**File:** `supabase/functions/claude-api/index.ts`

The Supabase edge function is a special case because:
1. It runs on Supabase's Deno infrastructure
2. Cannot run the Claude CLI (which requires local installation)
3. Has complex tool definitions (codebase, GitHub, self-query tools)
4. Frontend calls this via Supabase client

#### Options

**Option A: Keep API for Edge Function (Recommended for MVP)**
- Pros: No frontend changes, works immediately
- Cons: Still pays per-token for interactive chat
- Best for: Getting core brain functions on CLI first

**Option B: Move to Local Python Server**
- Create new `/api/chat` endpoint in `server.py` that replaces edge function
- Frontend calls local server instead of Supabase
- Pros: Full CLI usage, zero API costs
- Cons: Requires frontend changes, local server must be running
- Implementation:
  1. Copy tool definitions to Python
  2. Create new endpoint with tool handling
  3. Update frontend to call localhost:8420 instead of Supabase

**Option C: Hybrid Approach**
- Use CLI for background tasks (evolution, extraction)
- Keep API for interactive chat
- Pros: Balance between cost and complexity
- Cons: Still has some API costs

## CLI Executor Module

Created `utils/cli_executor.py` with two main functions:

```python
# Simple single prompt
response = await call_claude_cli(
    prompt="Hello",
    system_prompt="You are helpful.",
    timeout=120.0
)

# With conversation history
response = await call_claude_cli_with_conversation(
    messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ],
    system_prompt="You are helpful.",
    timeout=120.0
)
```

## Prerequisites

```bash
# Install Claude CLI
npm install -g @anthropic-ai/claude-code

# Login (uses Max subscription)
claude login

# Verify it works
claude -p "Hello"

# Unset API key to force CLI auth
unset ANTHROPIC_API_KEY
```

## Testing

```bash
# Test CLI executor standalone
cd mynd-brain
python -c "import asyncio; from utils.cli_executor import call_claude_cli; print(asyncio.run(call_claude_cli('Hello')))"

# Test /brain/chat endpoint
curl -X POST http://localhost:8420/brain/chat \
  -H "Content-Type: application/json" \
  -d '{"user_message": "Hello", "conversation_history": []}'
```

## Cost Savings

With Max subscription:
- **Before:** ~$0.015-$0.075 per 1K tokens (depending on model)
- **After:** $0 per token (included in subscription)

For typical usage (1M tokens/month), this saves $15-75/month.
