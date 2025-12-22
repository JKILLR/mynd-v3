-- ═══════════════════════════════════════════════════════════════════
-- PENDING INSIGHTS TABLE - Background Cognition Results
-- Stores AI-discovered insights from between-session analysis
-- ═══════════════════════════════════════════════════════════════════

-- Create the pending_insights table
CREATE TABLE IF NOT EXISTS pending_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,

    -- Insight content
    insight_type TEXT NOT NULL CHECK (insight_type IN (
        'connection',           -- Missing link between nodes
        'pattern',              -- Recurring behavior/theme
        'growth',               -- Significant map area development
        'emergence',            -- New theme forming
        'question',             -- Reflective question about gaps/stale areas
        'conversation_link',    -- Connection to past conversation
        'recurring_thread',     -- Theme across multiple conversations
        'evolution',            -- Thinking has evolved on a topic
        'memory_cluster',       -- Related memories forming insight
        'memory_enrichment',    -- Memory could enhance a node
        'cross_branch',         -- Connection between different branches
        'important_gap'         -- High-importance memory not in map
    )),
    title TEXT NOT NULL,         -- Brief summary for display
    content TEXT NOT NULL,       -- Full insight explanation
    confidence FLOAT DEFAULT 0.7 CHECK (confidence >= 0 AND confidence <= 1),

    -- What triggered this insight
    source_nodes TEXT[],         -- Node IDs involved
    source_memories UUID[],      -- Memory IDs referenced
    source_conversations UUID[], -- Conversation IDs referenced
    analysis_context JSONB,      -- Raw analysis data

    -- Lifecycle
    created_at TIMESTAMPTZ DEFAULT NOW(),
    presented_at TIMESTAMPTZ,    -- When shown to user (NULL = not yet shown)
    user_response TEXT CHECK (user_response IS NULL OR user_response IN (
        'acknowledged',  -- User saw it
        'expanded',      -- User asked for more
        'dismissed',     -- User didn't find it useful
        'acted_on'       -- User made changes based on it
    )),

    -- Prevent duplicates
    insight_hash TEXT UNIQUE     -- Hash of key fields to prevent duplicate insights
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_pending_insights_user ON pending_insights(user_id);
CREATE INDEX IF NOT EXISTS idx_pending_insights_type ON pending_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_pending_insights_confidence ON pending_insights(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_pending_insights_created ON pending_insights(created_at DESC);

-- Index for quick lookup of unpresented insights
CREATE INDEX IF NOT EXISTS idx_pending_insights_unpresented
    ON pending_insights(user_id, presented_at)
    WHERE presented_at IS NULL;

-- Row Level Security
ALTER TABLE pending_insights ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only access their own insights
CREATE POLICY "Users can read own insights" ON pending_insights
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own insights" ON pending_insights
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own insights" ON pending_insights
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own insights" ON pending_insights
    FOR DELETE USING (auth.uid() = user_id);

-- Function to mark insights as presented
CREATE OR REPLACE FUNCTION mark_insights_presented(insight_ids UUID[])
RETURNS void AS $$
BEGIN
    UPDATE pending_insights
    SET presented_at = NOW()
    WHERE id = ANY(insight_ids)
      AND user_id = auth.uid()
      AND presented_at IS NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to record user response to an insight
CREATE OR REPLACE FUNCTION respond_to_insight(
    p_insight_id UUID,
    p_response TEXT
)
RETURNS void AS $$
BEGIN
    UPDATE pending_insights
    SET user_response = p_response
    WHERE id = p_insight_id
      AND user_id = auth.uid();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get unpresented insights for wake-up
CREATE OR REPLACE FUNCTION get_pending_insights(
    p_user_id UUID DEFAULT auth.uid(),
    p_limit INT DEFAULT 3
)
RETURNS TABLE (
    id UUID,
    insight_type TEXT,
    title TEXT,
    content TEXT,
    confidence FLOAT,
    source_nodes TEXT[],
    source_memories UUID[],
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        pi.id,
        pi.insight_type,
        pi.title,
        pi.content,
        pi.confidence,
        pi.source_nodes,
        pi.source_memories,
        pi.created_at
    FROM pending_insights pi
    WHERE
        pi.user_id = p_user_id
        AND pi.presented_at IS NULL
    ORDER BY pi.confidence DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Comments for documentation
COMMENT ON TABLE pending_insights IS 'Background cognition results - AI-discovered insights between sessions';
COMMENT ON COLUMN pending_insights.insight_type IS 'Type of insight: connection, pattern, growth, emergence, question, etc.';
COMMENT ON COLUMN pending_insights.confidence IS '0-1 confidence score from analysis algorithm';
COMMENT ON COLUMN pending_insights.presented_at IS 'When shown to user. NULL means not yet presented';
COMMENT ON COLUMN pending_insights.user_response IS 'How user reacted: acknowledged, expanded, dismissed, acted_on';
COMMENT ON COLUMN pending_insights.insight_hash IS 'Hash to prevent duplicate insights on same topic';
COMMENT ON FUNCTION get_pending_insights(UUID, INT) IS 'Get unpresented insights for wake-up synthesis';
COMMENT ON FUNCTION mark_insights_presented(UUID[]) IS 'Mark insights as shown to user';
COMMENT ON FUNCTION respond_to_insight(UUID, TEXT) IS 'Record user reaction to an insight';
