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

-- Vector similarity search index (for semantic retrieval)
CREATE INDEX IF NOT EXISTS idx_ai_memory_embedding ON ai_memory
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

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
CREATE OR REPLACE FUNCTION touch_memory(memory_id UUID)
RETURNS void AS $$
BEGIN
  UPDATE ai_memory
  SET
    last_accessed = NOW(),
    access_count = access_count + 1
  WHERE id = memory_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function for semantic search (requires embedding)
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
  related_nodes TEXT[],
  similarity FLOAT,
  created_at TIMESTAMPTZ,
  last_accessed TIMESTAMPTZ
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    m.id,
    m.memory_type,
    m.content,
    m.importance,
    m.related_nodes,
    1 - (m.embedding <=> query_embedding) AS similarity,
    m.created_at,
    m.last_accessed
  FROM ai_memory m
  WHERE
    m.user_id = p_user_id
    AND m.embedding IS NOT NULL
    AND 1 - (m.embedding <=> query_embedding) > match_threshold
  ORDER BY m.embedding <=> query_embedding
  LIMIT match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get top memories by importance and recency (no embedding required)
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
    -- Recency factor: 1.0 for today, decays over 30 days
    m.importance * (1.0 - LEAST(EXTRACT(EPOCH FROM (NOW() - m.last_accessed)) / (30 * 24 * 60 * 60), 0.5)) DESC
  LIMIT p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function for memory decay (to be called by a scheduled job)
CREATE OR REPLACE FUNCTION decay_stale_memories()
RETURNS INT AS $$
DECLARE
  affected_count INT;
BEGIN
  -- Reduce importance by 10% for memories not accessed in 30+ days
  UPDATE ai_memory
  SET
    importance = GREATEST(importance * 0.9, 0.1),  -- Don't go below 0.1
    updated_at = NOW()
  WHERE
    last_accessed < NOW() - INTERVAL '30 days'
    AND importance > 0.1;

  GET DIAGNOSTICS affected_count = ROW_COUNT;
  RETURN affected_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Comments for documentation
COMMENT ON TABLE ai_memory IS 'Claude persistent memory - synthesized understanding that persists across sessions';
COMMENT ON COLUMN ai_memory.memory_type IS 'synthesis: unified understanding | realization: aha moments | goal_tracking: active goals | pattern: behavioral patterns | relationship: concept connections';
COMMENT ON COLUMN ai_memory.importance IS '0-1 priority score for retrieval. Decays over time if not accessed.';
COMMENT ON COLUMN ai_memory.related_nodes IS 'Array of map node IDs this memory connects to';
COMMENT ON COLUMN ai_memory.related_memories IS 'Links to other memories for building memory graphs';
