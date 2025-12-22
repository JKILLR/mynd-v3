# Background Cognition Spec

## Overview

Enable MYND's AI to perform autonomous analysis between user sessions, discovering insights and connections that are presented when the user returns.

**Goal**: "While you were gone, I noticed something about your map..."

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER SESSION                                │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────────────┐     │
│  │  Chat   │───▶│ Wake-up      │───▶│ Check pending_      │     │
│  │  Start  │    │ Synthesis    │    │ insights table      │     │
│  └─────────┘    └──────────────┘    └─────────────────────┘     │
│                                              │                   │
│                                              ▼                   │
│                                     Present insights to user     │
└─────────────────────────────────────────────────────────────────┘

                            ▲
                            │ reads from
                            │
┌─────────────────────────────────────────────────────────────────┐
│                   BACKGROUND WORKER (runs periodically)          │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Load user's  │───▶│ Run analysis │───▶│ Store in     │       │
│  │ mind map     │    │ algorithms   │    │ pending_     │       │
│  │ + memories   │    │              │    │ insights     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│  Triggers: Scheduled (daily) OR on significant data change       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### New Table: `pending_insights`

```sql
CREATE TABLE pending_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Insight content
    insight_type TEXT NOT NULL,  -- 'connection', 'pattern', 'growth', 'question', 'emergence'
    title TEXT NOT NULL,         -- Brief summary for display
    content TEXT NOT NULL,       -- Full insight explanation
    confidence FLOAT DEFAULT 0.7,

    -- What triggered this insight
    source_nodes TEXT[],         -- Node IDs involved
    source_memories UUID[],      -- Memory IDs referenced
    analysis_context JSONB,      -- Raw analysis data

    -- Lifecycle
    created_at TIMESTAMPTZ DEFAULT NOW(),
    presented_at TIMESTAMPTZ,    -- When shown to user (NULL = not yet shown)
    user_response TEXT,          -- 'acknowledged', 'expanded', 'dismissed', 'acted_on'

    -- Prevent duplicates
    insight_hash TEXT UNIQUE     -- Hash of key fields to prevent duplicate insights
);

-- RLS Policy
ALTER TABLE pending_insights ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users see own insights" ON pending_insights
    FOR ALL USING (auth.uid() = user_id);

-- Index for quick lookup of unpresented insights
CREATE INDEX idx_pending_insights_unpresented
    ON pending_insights(user_id, presented_at)
    WHERE presented_at IS NULL;
```

---

## Insight Types

### 1. Connection Discovery
**Trigger**: Two nodes that aren't linked but have high semantic similarity or co-occur in sessions.

```python
{
    "type": "connection",
    "title": "Possible connection: 'The Refusal' ↔ 'boundaries'",
    "content": "I noticed you've discussed The Refusal and boundaries in similar contexts across 3 sessions. They might be related concepts worth connecting.",
    "confidence": 0.82,
    "source_nodes": ["node_123", "node_456"]
}
```

### 2. Pattern Recognition
**Trigger**: Recurring themes, behaviors, or topics across sessions.

```python
{
    "type": "pattern",
    "title": "You tend to explore deeply before deciding",
    "content": "Across your last 5 sessions, I notice you create 3-4 exploratory branches before committing to one direction. This seems intentional.",
    "confidence": 0.75,
    "analysis_context": {"sessions_analyzed": 5, "pattern_occurrences": 4}
}
```

### 3. Growth Observation
**Trigger**: Significant changes in map structure, memory accumulation, or topic evolution.

```python
{
    "type": "growth",
    "title": "Your 'work' branch has grown significantly",
    "content": "You've added 12 new nodes under 'work' in the past week, making it your most active area. The Canada incorporation thread is developing depth.",
    "confidence": 0.95,
    "source_nodes": ["work_root_id"]
}
```

### 4. Emergence Detection
**Trigger**: A new cluster or theme emerging that doesn't fit existing categories.

```python
{
    "type": "emergence",
    "title": "New theme emerging: creative constraints",
    "content": "I'm seeing repeated references to constraints, limitations, and creativity across different branches. This might be becoming a core concept for you.",
    "confidence": 0.68
}
```

### 5. Reflective Questions
**Trigger**: Gaps in the map, abandoned threads, or unexplored connections.

```python
{
    "type": "question",
    "title": "Unexplored thread: 'future self'",
    "content": "You created a 'future self' node 2 weeks ago but haven't expanded it. Is this still relevant, or has your thinking evolved?",
    "confidence": 0.60,
    "source_nodes": ["future_self_node_id"]
}
```

---

## Background Worker Implementation

### Option A: Supabase Edge Function + pg_cron (Recommended)

```typescript
// supabase/functions/background-cognition/index.ts

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
)

Deno.serve(async (req) => {
    // Get all users who have been active in last 7 days
    const { data: activeUsers } = await supabase
        .from('session_summaries')
        .select('user_id')
        .gte('session_ended', new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString())
        .order('session_ended', { ascending: false })

    const uniqueUsers = [...new Set(activeUsers?.map(u => u.user_id))]

    for (const userId of uniqueUsers) {
        await analyzeUserData(userId)
    }

    return new Response(JSON.stringify({ processed: uniqueUsers.length }))
})

async function analyzeUserData(userId: string) {
    // 1. Load user's mind map
    const { data: maps } = await supabase
        .from('mind_maps')
        .select('data')
        .eq('user_id', userId)
        .order('updated_at', { ascending: false })
        .limit(1)
        .single()

    // 2. Load recent memories
    const { data: memories } = await supabase
        .from('ai_memory')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .limit(50)

    // 3. Load recent sessions
    const { data: sessions } = await supabase
        .from('session_summaries')
        .select('*')
        .eq('user_id', userId)
        .order('session_ended', { ascending: false })
        .limit(10)

    // 4. Run analysis algorithms
    const insights = []

    insights.push(...findPotentialConnections(maps?.data, memories))
    insights.push(...detectPatterns(sessions, memories))
    insights.push(...observeGrowth(maps?.data, sessions))
    insights.push(...detectEmergence(memories, sessions))
    insights.push(...generateQuestions(maps?.data, sessions))

    // 5. Filter to high-confidence, non-duplicate insights
    const validInsights = insights
        .filter(i => i.confidence >= 0.6)
        .slice(0, 3)  // Max 3 insights per run

    // 6. Store insights (with duplicate prevention)
    for (const insight of validInsights) {
        const hash = await hashInsight(insight)

        await supabase
            .from('pending_insights')
            .upsert({
                user_id: userId,
                insight_type: insight.type,
                title: insight.title,
                content: insight.content,
                confidence: insight.confidence,
                source_nodes: insight.source_nodes || [],
                source_memories: insight.source_memories || [],
                analysis_context: insight.context || {},
                insight_hash: hash
            }, { onConflict: 'insight_hash' })
    }
}
```

### Schedule with pg_cron

```sql
-- Run daily at 3 AM UTC
SELECT cron.schedule(
    'background-cognition',
    '0 3 * * *',
    $$
    SELECT net.http_post(
        url := 'https://your-project.supabase.co/functions/v1/background-cognition',
        headers := '{"Authorization": "Bearer YOUR_SERVICE_KEY"}'::jsonb
    );
    $$
);
```

### Option B: Python Server Endpoint (Alternative)

Add to `server.py`:

```python
@app.post("/background/analyze/{user_id}")
async def background_analyze(user_id: str):
    """Run background analysis for a user. Called by scheduler."""

    if not unified_brain or not unified_brain.supabase:
        raise HTTPException(503, "Brain not connected to Supabase")

    # Load user data
    map_data = load_user_map(user_id)
    memories = load_user_memories(user_id)
    sessions = load_user_sessions(user_id)

    insights = []

    # Connection discovery using existing graph analysis
    if map_data:
        missing = brain.graph_transformer.find_missing_connections(
            brain.map_embeddings,
            threshold=0.7
        )
        for conn in missing[:3]:
            insights.append({
                "type": "connection",
                "title": f"Possible link: {conn['source']} ↔ {conn['target']}",
                "content": f"These nodes share {conn['similarity']:.0%} semantic similarity but aren't connected.",
                "confidence": conn['similarity'],
                "source_nodes": [conn['source_id'], conn['target_id']]
            })

    # Pattern detection using KnowledgeDistiller
    patterns = brain.knowledge.get_patterns()
    for pattern in patterns[:2]:
        if pattern['confidence'] > 0.6:
            insights.append({
                "type": "pattern",
                "title": pattern['description'][:50],
                "content": pattern['description'],
                "confidence": pattern['confidence']
            })

    # Store insights
    for insight in insights:
        store_pending_insight(user_id, insight)

    return {"insights_generated": len(insights)}
```

---

## Presentation Layer

### Wake-up Check

Add to `chatManager.initSession()` in `app-module.js`:

```javascript
async checkPendingInsights() {
    if (!supabase) return [];

    const { data: session } = await supabase.auth.getSession();
    if (!session?.session?.user) return [];

    const { data: insights, error } = await supabase
        .from('pending_insights')
        .select('*')
        .eq('user_id', session.session.user.id)
        .is('presented_at', null)
        .order('confidence', { ascending: false })
        .limit(3);

    if (error || !insights?.length) return [];

    // Mark as presented
    const insightIds = insights.map(i => i.id);
    await supabase
        .from('pending_insights')
        .update({ presented_at: new Date().toISOString() })
        .in('id', insightIds);

    return insights;
}
```

### System Prompt Integration

Add to wake-up synthesis:

```javascript
const pendingInsights = await chatManager.checkPendingInsights();

if (pendingInsights.length > 0) {
    systemPrompt += `\n\n## Insights Discovered While Away\n`;
    systemPrompt += `Between sessions, I analyzed your map and memories. Here's what I found:\n\n`;

    for (const insight of pendingInsights) {
        systemPrompt += `### ${insight.title}\n`;
        systemPrompt += `${insight.content}\n`;
        systemPrompt += `*Confidence: ${Math.round(insight.confidence * 100)}%*\n\n`;
    }

    systemPrompt += `Feel free to explore any of these, or continue with what you had in mind.\n`;
}
```

---

## Analysis Algorithms

### 1. Connection Discovery

```python
def find_potential_connections(map_data, memories):
    """Find semantically similar but unconnected nodes."""
    insights = []

    nodes = flatten_map_nodes(map_data)
    embeddings = embed_all_nodes(nodes)

    # Find pairs with high similarity but no edge
    for i, node_a in enumerate(nodes):
        for j, node_b in enumerate(nodes[i+1:], i+1):
            if are_connected(node_a, node_b, map_data):
                continue

            similarity = cosine_similarity(embeddings[i], embeddings[j])

            if similarity > 0.75:
                # Check if they co-occur in memories/sessions
                co_occurrence = count_co_occurrences(node_a, node_b, memories)

                confidence = (similarity * 0.6) + (min(co_occurrence / 5, 1) * 0.4)

                if confidence > 0.65:
                    insights.append({
                        "type": "connection",
                        "title": f"Possible connection: '{node_a['label']}' ↔ '{node_b['label']}'",
                        "content": f"These concepts appear related (similarity: {similarity:.0%}) and you've discussed them together {co_occurrence} times.",
                        "confidence": confidence,
                        "source_nodes": [node_a['id'], node_b['id']]
                    })

    return sorted(insights, key=lambda x: -x['confidence'])[:3]
```

### 2. Pattern Detection

```python
def detect_patterns(sessions, memories):
    """Find recurring behavioral or topical patterns."""
    insights = []

    # Topic frequency across sessions
    topic_counts = defaultdict(int)
    for session in sessions:
        for topic in session.get('topics_discussed', []):
            topic_counts[topic] += 1

    # Find topics that appear in >50% of sessions
    session_count = len(sessions)
    for topic, count in topic_counts.items():
        frequency = count / session_count
        if frequency > 0.5 and count >= 3:
            insights.append({
                "type": "pattern",
                "title": f"Recurring theme: {topic}",
                "content": f"'{topic}' has appeared in {count} of your last {session_count} sessions ({frequency:.0%}). This seems to be a persistent focus area.",
                "confidence": min(0.95, 0.5 + frequency * 0.5)
            })

    # Session timing patterns
    session_hours = [parse_time(s['session_started']).hour for s in sessions]
    avg_hour = statistics.mean(session_hours)
    std_hour = statistics.stdev(session_hours) if len(session_hours) > 1 else 12

    if std_hour < 3:  # Consistent timing
        time_desc = "morning" if avg_hour < 12 else "afternoon" if avg_hour < 17 else "evening"
        insights.append({
            "type": "pattern",
            "title": f"You're a {time_desc} thinker",
            "content": f"You consistently use MYND in the {time_desc} (average: {avg_hour:.0f}:00). Your best thinking happens then.",
            "confidence": 0.7
        })

    return insights
```

### 3. Growth Observation

```python
def observe_growth(current_map, sessions):
    """Track structural changes in the map."""
    insights = []

    # Count nodes by branch
    branch_counts = count_nodes_by_branch(current_map)

    # Find branches that grew significantly
    for branch, count in branch_counts.items():
        # Compare to previous state (from session metadata)
        previous_count = get_previous_branch_count(branch, sessions)

        if previous_count and count > previous_count * 1.3:  # 30% growth
            growth = count - previous_count
            insights.append({
                "type": "growth",
                "title": f"'{branch}' is expanding",
                "content": f"You've added {growth} new nodes to '{branch}' recently. It's becoming a significant part of your map.",
                "confidence": 0.85
            })

    # Detect new top-level branches
    recent_roots = find_recent_root_nodes(current_map, days=7)
    for node in recent_roots:
        insights.append({
            "type": "growth",
            "title": f"New branch: '{node['label']}'",
            "content": f"You started a new top-level branch '{node['label']}' this week. Where do you see this going?",
            "confidence": 0.9
        })

    return insights
```

---

## Configuration

```python
BACKGROUND_CONFIG = {
    "run_frequency": "daily",           # or "on_session_end", "hourly"
    "min_sessions_for_patterns": 3,     # Need at least N sessions
    "max_insights_per_run": 3,          # Don't overwhelm
    "min_confidence": 0.6,              # Quality threshold
    "insight_cooldown_days": 7,         # Don't repeat similar insights
    "active_user_window_days": 14,      # Only analyze recent users
}
```

---

## Privacy & Performance

### Privacy
- All analysis runs on user's own data only
- No cross-user pattern analysis
- Insights stored per-user with RLS
- User can disable background analysis in settings

### Performance
- Run during off-peak hours (3 AM UTC)
- Process max 100 users per run
- Timeout: 30 seconds per user
- Cache embeddings to avoid recomputation

---

## Rollout Plan

### Phase 1: Foundation
- [ ] Create `pending_insights` table
- [ ] Add insight check to session start
- [ ] Add basic presentation in wake-up prompt

### Phase 2: Basic Analysis
- [ ] Implement connection discovery (using existing graph code)
- [ ] Add growth observation
- [ ] Deploy as manual endpoint first

### Phase 3: Scheduling
- [ ] Set up pg_cron or external scheduler
- [ ] Add user preference for enable/disable
- [ ] Monitor and tune confidence thresholds

### Phase 4: Advanced
- [ ] Pattern detection across sessions
- [ ] Emergence detection
- [ ] Reflective questions
- [ ] User feedback loop (did insight help?)

---

## Success Metrics

1. **Insight quality**: % of insights user engages with vs dismisses
2. **Discovery rate**: New connections created after insight presentation
3. **Return engagement**: Do users return more often knowing insights await?
4. **Confidence calibration**: Are high-confidence insights actually useful?

---

## Example User Experience

**User opens MYND after 2 days away:**

> Good morning. While you were gone, I was thinking about your map...
>
> **I noticed a possible connection**: 'The Refusal' and 'boundaries' appear semantically related (78% similarity) and you've discussed them together in 3 recent sessions. They might be worth connecting explicitly.
>
> **Your 'work' branch is growing**: You've added 8 nodes there this week, making it your most active area. The Canada incorporation thread is developing real depth.
>
> Feel free to explore either of these, or continue with whatever's on your mind today.

