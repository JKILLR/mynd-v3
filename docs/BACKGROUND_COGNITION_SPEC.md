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

Two primary data sources: **Conversations** and **Map Structure**

---

### Data Source 1: Conversations

Imported conversations contain rich context about the user's thinking, interests, and evolution over time.

#### 1A. Conversation-to-Map Bridging

```python
def find_conversation_map_connections(conversations, map_nodes):
    """Find concepts discussed in conversations that relate to map nodes."""
    insights = []

    # Embed all map node labels
    node_embeddings = {
        node['id']: embed(node['label'])
        for node in map_nodes
    }

    for conv in conversations:
        # Extract key topics/entities from conversation
        conv_topics = extract_topics(conv['content'])
        conv_embedding = embed(conv['content'][:2000])

        # Find map nodes semantically similar to conversation content
        for node_id, node_emb in node_embeddings.items():
            similarity = cosine_similarity(conv_embedding, node_emb)

            if similarity > 0.7:
                node_label = get_node_label(node_id, map_nodes)
                insights.append({
                    "type": "conversation_link",
                    "title": f"'{node_label}' discussed in past conversation",
                    "content": f"Your conversation from {conv['date']} touched on themes related to '{node_label}'. The context might enrich this node.",
                    "confidence": similarity,
                    "source_nodes": [node_id],
                    "source_conversation": conv['id']
                })

    return sorted(insights, key=lambda x: -x['confidence'])[:3]
```

#### 1B. Conversation Thread Detection

```python
def detect_conversation_threads(conversations):
    """Find recurring themes across multiple conversations."""
    insights = []

    # Cluster conversations by semantic similarity
    embeddings = [embed(c['content'][:2000]) for c in conversations]
    clusters = cluster_embeddings(embeddings, threshold=0.6)

    for cluster in clusters:
        if len(cluster) >= 3:  # Theme appears in 3+ conversations
            # Find common topics in cluster
            cluster_convs = [conversations[i] for i in cluster]
            common_themes = extract_common_themes(cluster_convs)

            insights.append({
                "type": "recurring_thread",
                "title": f"Recurring theme: {common_themes[0]}",
                "content": f"Across {len(cluster)} conversations, you keep returning to: {', '.join(common_themes[:3])}. This might be worth a dedicated map branch.",
                "confidence": 0.75,
                "source_conversations": [c['id'] for c in cluster_convs]
            })

    return insights
```

#### 1C. Conversation Evolution

```python
def track_thinking_evolution(conversations):
    """Detect how user's thinking on a topic has evolved."""
    insights = []

    # Sort by date
    sorted_convs = sorted(conversations, key=lambda c: c['date'])

    # For each major topic, track stance/framing changes
    topics = extract_major_topics(sorted_convs)

    for topic in topics:
        relevant_convs = [c for c in sorted_convs if topic_in_conv(topic, c)]

        if len(relevant_convs) >= 2:
            early_stance = summarize_stance(relevant_convs[0], topic)
            recent_stance = summarize_stance(relevant_convs[-1], topic)

            if stances_differ(early_stance, recent_stance):
                insights.append({
                    "type": "evolution",
                    "title": f"Your thinking on '{topic}' has evolved",
                    "content": f"Earlier: {early_stance[:100]}... Now: {recent_stance[:100]}... Interesting shift.",
                    "confidence": 0.7
                })

    return insights
```

---

### Data Source 2: Map Structure

The mind map's nodes, hierarchy, and connections reveal cognitive organization.

#### 2A. Missing Connections (Semantic)

```python
def find_missing_connections(map_data):
    """Find nodes that should probably be connected but aren't."""
    insights = []

    nodes = flatten_nodes(map_data)
    embeddings = {n['id']: embed(n['label']) for n in nodes}
    existing_edges = get_all_edges(map_data)

    # Check all pairs
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i+1:]:
            if (node_a['id'], node_b['id']) in existing_edges:
                continue

            similarity = cosine_similarity(
                embeddings[node_a['id']],
                embeddings[node_b['id']]
            )

            if similarity > 0.75:
                insights.append({
                    "type": "missing_connection",
                    "title": f"Connect '{node_a['label']}' ↔ '{node_b['label']}'?",
                    "content": f"These nodes are {similarity:.0%} semantically similar but not linked. They might benefit from an explicit connection.",
                    "confidence": similarity,
                    "source_nodes": [node_a['id'], node_b['id']]
                })

    return sorted(insights, key=lambda x: -x['confidence'])[:3]
```

#### 2B. Structural Imbalance

```python
def detect_structural_patterns(map_data):
    """Analyze map structure for imbalances or opportunities."""
    insights = []

    root = map_data
    branches = root.get('children', [])

    # Branch depth analysis
    branch_stats = []
    for branch in branches:
        stats = {
            'label': branch['label'],
            'depth': get_max_depth(branch),
            'node_count': count_nodes(branch),
            'leaf_count': count_leaves(branch)
        }
        branch_stats.append(stats)

    # Find imbalanced branches
    avg_depth = statistics.mean([b['depth'] for b in branch_stats])
    avg_nodes = statistics.mean([b['node_count'] for b in branch_stats])

    for branch in branch_stats:
        # Very deep but narrow (might need restructuring)
        if branch['depth'] > avg_depth * 2 and branch['node_count'] < avg_nodes * 0.5:
            insights.append({
                "type": "structure",
                "title": f"'{branch['label']}' is deep but narrow",
                "content": f"This branch goes {branch['depth']} levels deep but only has {branch['node_count']} nodes. Consider whether some could be siblings instead of nested.",
                "confidence": 0.65
            })

        # Very wide but shallow (might need hierarchy)
        if branch['leaf_count'] > 10 and branch['depth'] < 2:
            insights.append({
                "type": "structure",
                "title": f"'{branch['label']}' might benefit from sub-categories",
                "content": f"You have {branch['leaf_count']} items directly under '{branch['label']}'. Grouping some might help organization.",
                "confidence": 0.6
            })

    return insights
```

#### 2C. Orphan and Stale Detection

```python
def find_orphans_and_stale(map_data, conversations):
    """Find nodes that are isolated or haven't been touched."""
    insights = []

    nodes = flatten_nodes(map_data)

    # Find orphan nodes (no children, no connections, not referenced)
    for node in nodes:
        children_count = len(node.get('children', []))
        connection_count = count_connections(node['id'], map_data)
        conv_mentions = count_mentions_in_conversations(node['label'], conversations)

        if children_count == 0 and connection_count == 0 and conv_mentions == 0:
            # Check age (if metadata available)
            age_days = get_node_age_days(node)

            if age_days and age_days > 14:
                insights.append({
                    "type": "orphan",
                    "title": f"'{node['label']}' seems isolated",
                    "content": f"This node has no children or connections, and hasn't come up in conversations. Still relevant, or ready to archive?",
                    "confidence": 0.55,
                    "source_nodes": [node['id']]
                })

    return insights
```

#### 2D. Cross-Branch Connections

```python
def suggest_cross_branch_links(map_data):
    """Find potential connections between different branches."""
    insights = []

    root = map_data
    branches = root.get('children', [])

    # Get all nodes per branch
    branch_nodes = {}
    for branch in branches:
        branch_nodes[branch['label']] = flatten_nodes(branch)

    # Compare nodes across branches
    for branch_a, nodes_a in branch_nodes.items():
        for branch_b, nodes_b in branch_nodes.items():
            if branch_a >= branch_b:  # Avoid duplicates
                continue

            # Find similar nodes across branches
            for node_a in nodes_a:
                for node_b in nodes_b:
                    similarity = cosine_similarity(
                        embed(node_a['label']),
                        embed(node_b['label'])
                    )

                    if similarity > 0.7:
                        insights.append({
                            "type": "cross_branch",
                            "title": f"Bridge: '{node_a['label']}' ({branch_a}) ↔ '{node_b['label']}' ({branch_b})",
                            "content": f"These nodes in different branches are highly related. A cross-link might reveal how {branch_a} connects to {branch_b}.",
                            "confidence": similarity,
                            "source_nodes": [node_a['id'], node_b['id']]
                        })

    return sorted(insights, key=lambda x: -x['confidence'])[:2]
```

---

### Combined Analysis

```python
def run_background_analysis(user_id):
    """Main analysis function combining all sources."""

    # Load data
    map_data = load_user_map(user_id)
    conversations = load_user_conversations(user_id)
    map_nodes = flatten_nodes(map_data)

    insights = []

    # Conversation analysis
    if conversations:
        insights += find_conversation_map_connections(conversations, map_nodes)
        insights += detect_conversation_threads(conversations)
        insights += track_thinking_evolution(conversations)

    # Map structure analysis
    if map_data:
        insights += find_missing_connections(map_data)
        insights += detect_structural_patterns(map_data)
        insights += find_orphans_and_stale(map_data, conversations)
        insights += suggest_cross_branch_links(map_data)

    # Filter and rank
    insights = [i for i in insights if i['confidence'] >= 0.6]
    insights = sorted(insights, key=lambda x: -x['confidence'])

    # Deduplicate similar insights
    insights = deduplicate_insights(insights)

    return insights[:3]  # Top 3 only
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

