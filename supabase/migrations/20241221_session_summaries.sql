-- ═══════════════════════════════════════════════════════════════════
-- SESSION SUMMARIES TABLE - Experiential Continuity System
-- Enables Claude to maintain narrative continuity across sessions
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS session_summaries (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,

  -- Session timing
  session_started TIMESTAMPTZ NOT NULL,
  session_ended TIMESTAMPTZ DEFAULT NOW(),

  -- Threading - links sessions into a chain
  previous_session_id UUID REFERENCES session_summaries(id),

  -- What we discussed
  topics_discussed TEXT[] DEFAULT '{}',
  nodes_touched TEXT[] DEFAULT '{}',

  -- What we decided/realized (key outcomes)
  key_outcomes TEXT,

  -- What's unfinished (open threads to pick up)
  open_threads TEXT,

  -- Relational context
  session_type TEXT CHECK (session_type IN (
    'building',      -- Creating/expanding the map
    'troubleshooting', -- Fixing issues, debugging
    'vision',        -- Big picture thinking, goals
    'exploration',   -- Discovering, questioning
    'reflection',    -- Processing, synthesizing
    'planning',      -- Organizing, strategizing
    'casual'         -- Light conversation, check-ins
  )),

  -- Full narrative summary (Claude's synthesis)
  summary TEXT NOT NULL,

  -- Emotional/relational tone
  tone TEXT,

  -- Metadata
  message_count INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_session_summaries_user ON session_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_session_summaries_time ON session_summaries(user_id, session_ended DESC);
CREATE INDEX IF NOT EXISTS idx_session_summaries_chain ON session_summaries(previous_session_id);

-- Row Level Security
ALTER TABLE session_summaries ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own session summaries" ON session_summaries
  FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own session summaries" ON session_summaries
  FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own session summaries" ON session_summaries
  FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own session summaries" ON session_summaries
  FOR DELETE USING (auth.uid() = user_id);

-- Function to get recent session summaries for wake-up synthesis
CREATE OR REPLACE FUNCTION get_recent_sessions(
  p_user_id UUID DEFAULT auth.uid(),
  p_limit INT DEFAULT 5
)
RETURNS TABLE (
  id UUID,
  session_started TIMESTAMPTZ,
  session_ended TIMESTAMPTZ,
  topics_discussed TEXT[],
  nodes_touched TEXT[],
  key_outcomes TEXT,
  open_threads TEXT,
  session_type TEXT,
  summary TEXT,
  tone TEXT,
  message_count INT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    s.id,
    s.session_started,
    s.session_ended,
    s.topics_discussed,
    s.nodes_touched,
    s.key_outcomes,
    s.open_threads,
    s.session_type,
    s.summary,
    s.tone,
    s.message_count
  FROM session_summaries s
  WHERE s.user_id = p_user_id
  ORDER BY s.session_ended DESC
  LIMIT p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get the last session (for threading)
CREATE OR REPLACE FUNCTION get_last_session(p_user_id UUID DEFAULT auth.uid())
RETURNS UUID AS $$
DECLARE
  last_id UUID;
BEGIN
  SELECT id INTO last_id
  FROM session_summaries
  WHERE user_id = p_user_id
  ORDER BY session_ended DESC
  LIMIT 1;

  RETURN last_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
