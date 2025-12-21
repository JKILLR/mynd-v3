-- Wake-Up Synthesis Cache
-- Stores deep context expansions generated during sessions
-- Cross-references session summaries with memories and nodes for richer context

CREATE TABLE IF NOT EXISTS wakeup_synthesis (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,

  -- Session identifier (matches the session_id used in app)
  session_id TEXT NOT NULL,

  -- The synthesized deep context
  synthesis_content TEXT NOT NULL,

  -- What topics/phrases were expanded
  topics_expanded TEXT[] DEFAULT '{}',

  -- What sessions were cross-referenced
  sessions_referenced UUID[] DEFAULT '{}',

  -- The user message that triggered synthesis (for relevance)
  trigger_context TEXT,

  -- Metadata
  generated_at TIMESTAMPTZ DEFAULT NOW(),

  -- Index for fast lookup by session
  UNIQUE(user_id, session_id)
);

-- Enable RLS
ALTER TABLE wakeup_synthesis ENABLE ROW LEVEL SECURITY;

-- Users can only access their own synthesis cache
CREATE POLICY "Users can view own synthesis"
  ON wakeup_synthesis FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own synthesis"
  ON wakeup_synthesis FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own synthesis"
  ON wakeup_synthesis FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own synthesis"
  ON wakeup_synthesis FOR DELETE
  USING (auth.uid() = user_id);

-- Index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_wakeup_synthesis_session
  ON wakeup_synthesis(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_wakeup_synthesis_generated
  ON wakeup_synthesis(generated_at DESC);

-- Comment
COMMENT ON TABLE wakeup_synthesis IS 'Caches deep context synthesis that cross-references session summaries with memories and nodes';
