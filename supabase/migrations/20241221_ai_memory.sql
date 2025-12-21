-- ═══════════════════════════════════════════════════════════════════
-- AI MEMORY TABLE - Claude's Persistent Memory System
-- Enables Claude to maintain its own synthesized understanding
-- ═══════════════════════════════════════════════════════════════════

-- Enable the vector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the ai_memory table
CREATE TABLE IF NOT EXISTS ai_memory (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,

  -- Memory classification
  memory_type TEXT NOT NULL CHECK (memory_type IN ('synthesis', 'realization', 'goal_tracking', 'pattern', 'relationship')),

  -- Core content
  content TEXT NOT NULL,

  -- Importance for retrieval prioritization (0-1)
  importance FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),

  -- Evergreen flag: if true, memory never decays (foundational knowledge)
  -- Use for: synthesis, realization, pattern types
  -- Don't use for: goal_tracking (situational, should decay)
  evergreen BOOLEAN DEFAULT FALSE,

  -- Connections to map nodes
  related_nodes TEXT[] DEFAULT '{}',

  -- Links to other memories (for memory graphs)
  related_memories UUID[] DEFAULT '{}',

  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  last_accessed TIMESTAMPTZ DEFAULT NOW(),

  -- Access tracking for importance decay
  access_count INT DEFAULT 0,

  -- Vector embedding for semantic retrieval (1536 = OpenAI ada-002 / Claude embeddings)
  embedding VECTOR(1536)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_ai_memory_user ON ai_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_memory_type ON ai_memory(memory_type);
CREATE INDEX IF NOT EXISTS idx_ai_memory_importance ON ai_memory(importance DESC);
CREATE INDEX IF NOT EXISTS idx_ai_memory_last_accessed ON ai_memory(last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_ai_memory_created ON ai_memory(created_at DESC);

-- Composite index for queries that sort by both importance and last_accessed
CREATE INDEX IF NOT EXISTS idx_ai_memory_importance_recency ON ai_memory(user_id, importance DESC, last_accessed DESC);

-- Vector similarity search index using HNSW (better for incremental inserts than IVFFlat)
CREATE INDEX IF NOT EXISTS idx_ai_memory_embedding ON ai_memory
  USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Row Level Security
ALTER TABLE ai_memory ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only access their own memories
CREATE POLICY "Users can read own memories" ON ai_memory
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own memories" ON ai_memory
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own memories" ON ai_memory
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own memories" ON ai_memory
  FOR DELETE USING (auth.uid() = user_id);

-- Function to update last_accessed and access_count when memory is retrieved
-- Includes ownership validation to prevent unauthorized access
CREATE OR REPLACE FUNCTION touch_memory(memory_id UUID)
RETURNS void AS $$
BEGIN
  UPDATE ai_memory
  SET
    last_accessed = NOW(),
    access_count = access_count + 1
  WHERE id = memory_id
    AND user_id = auth.uid();  -- Ownership validation
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function for semantic search (requires embedding)
-- Uses CTE to compute similarity once for efficiency
-- Returns evergreen flag for recency calculation in application code
CREATE OR REPLACE FUNCTION search_memories(
  query_embedding VECTOR(1536),
  match_threshold FLOAT DEFAULT 0.7,
  match_count INT DEFAULT 10,
  p_user_id UUID DEFAULT auth.uid()
)
RETURNS TABLE (
  id UUID,
  memory_type TEXT,
  content TEXT,
  importance FLOAT,
  evergreen BOOLEAN,
  related_nodes TEXT[],
  similarity FLOAT,
  created_at TIMESTAMPTZ,
  last_accessed TIMESTAMPTZ
) AS $$
BEGIN
  RETURN QUERY
  WITH scored AS (
    SELECT
      m.id,
      m.memory_type,
      m.content,
      m.importance,
      m.evergreen,
      m.related_nodes,
      1 - (m.embedding <=> query_embedding) AS similarity,
      m.created_at,
      m.last_accessed
    FROM ai_memory m
    WHERE
      m.user_id = p_user_id
      AND m.embedding IS NOT NULL
  )
  SELECT *
  FROM scored
  WHERE scored.similarity > match_threshold
  ORDER BY scored.similarity DESC
  LIMIT match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get top memories by importance and recency (no embedding required)
-- Evergreen memories always have recency_factor = 1.0
CREATE OR REPLACE FUNCTION get_top_memories(
  p_user_id UUID DEFAULT auth.uid(),
  p_limit INT DEFAULT 20,
  p_memory_type TEXT DEFAULT NULL
)
RETURNS TABLE (
  id UUID,
  memory_type TEXT,
  content TEXT,
  importance FLOAT,
  evergreen BOOLEAN,
  related_nodes TEXT[],
  created_at TIMESTAMPTZ,
  last_accessed TIMESTAMPTZ,
  access_count INT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    m.id,
    m.memory_type,
    m.content,
    m.importance,
    m.evergreen,
    m.related_nodes,
    m.created_at,
    m.last_accessed,
    m.access_count
  FROM ai_memory m
  WHERE
    m.user_id = p_user_id
    AND (p_memory_type IS NULL OR m.memory_type = p_memory_type)
  ORDER BY
    -- Score = importance * recency_factor
    -- Evergreen memories: recency_factor = 1.0 (never decays)
    -- Regular memories: recency decays over 30 days
    m.importance * (
      CASE WHEN m.evergreen THEN 1.0
      ELSE (1.0 - LEAST(EXTRACT(EPOCH FROM (NOW() - m.last_accessed)) / (30 * 24 * 60 * 60), 0.5))
      END
    ) DESC
  LIMIT p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function for memory decay (scheduled job only - not for end-user exposure)
-- WARNING: This function operates across ALL users and should only be called
-- by a scheduled database job (e.g., pg_cron), never from client code.
-- Reduces importance by 10% for memories not accessed in 30+ days.
-- RESPECTS evergreen flag - evergreen memories never decay.
CREATE OR REPLACE FUNCTION decay_stale_memories()
RETURNS INT AS $$
DECLARE
  affected_count INT;
BEGIN
  UPDATE ai_memory
  SET
    importance = GREATEST(importance * 0.9, 0.1),  -- Don't go below 0.1
    updated_at = NOW()
  WHERE
    last_accessed < NOW() - INTERVAL '30 days'
    AND importance > 0.1
    AND evergreen = FALSE;  -- Never decay evergreen memories

  GET DIAGNOSTICS affected_count = ROW_COUNT;
  RETURN affected_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to atomically reinforce a memory (avoids race conditions)
-- Increases importance by 10% (capped at 1.0) and updates access time
CREATE OR REPLACE FUNCTION reinforce_memory(p_memory_id UUID)
RETURNS FLOAT AS $$
DECLARE
  new_importance FLOAT;
BEGIN
  UPDATE ai_memory
  SET
    importance = LEAST(importance * 1.1, 1.0),
    last_accessed = NOW(),
    access_count = access_count + 1,
    updated_at = NOW()
  WHERE id = p_memory_id
    AND user_id = auth.uid()
  RETURNING importance INTO new_importance;

  RETURN new_importance;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Comments for documentation
COMMENT ON TABLE ai_memory IS 'Claude persistent memory - synthesized understanding that persists across sessions';
COMMENT ON COLUMN ai_memory.memory_type IS 'synthesis: unified understanding | realization: aha moments | goal_tracking: active goals | pattern: behavioral patterns | relationship: concept connections';
COMMENT ON COLUMN ai_memory.importance IS '0-1 priority score for retrieval. Decays over time if not accessed (unless evergreen).';
COMMENT ON COLUMN ai_memory.evergreen IS 'If true, memory never decays. Use for foundational knowledge (synthesis, realization, pattern). Situational memories (goal_tracking) should be false.';
COMMENT ON COLUMN ai_memory.related_nodes IS 'Array of map node IDs this memory connects to';
COMMENT ON COLUMN ai_memory.related_memories IS 'Links to other memories for building memory graphs';
COMMENT ON FUNCTION decay_stale_memories() IS 'Scheduled job function - decays importance of stale memories. NOT for client use.';
COMMENT ON FUNCTION reinforce_memory(UUID) IS 'Atomically reinforces a memory, increasing importance by 10% (max 1.0).';
