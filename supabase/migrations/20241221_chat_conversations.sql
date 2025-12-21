-- ═══════════════════════════════════════════════════════════════════
-- CHAT CONVERSATIONS TABLE - Axel Conversation History
-- Stores all chat messages server-side for persistence across devices
-- ═══════════════════════════════════════════════════════════════════

-- Create the chat_conversations table
CREATE TABLE IF NOT EXISTS chat_conversations (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,

  -- Message content
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'bapi', 'system')),
  content TEXT NOT NULL,

  -- Attached images (stored as data URLs or references)
  images TEXT[] DEFAULT '{}',

  -- Action results from assistant responses
  actions JSONB DEFAULT '[]',

  -- Suggestions offered with this message
  suggestions TEXT[] DEFAULT '{}',

  -- Message timestamp (user-provided, for ordering)
  message_timestamp BIGINT NOT NULL,

  -- Server timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_chat_conversations_user ON chat_conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_timestamp ON chat_conversations(user_id, message_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_created ON chat_conversations(created_at DESC);

-- Row Level Security
ALTER TABLE chat_conversations ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only access their own conversations
CREATE POLICY "Users can read own conversations" ON chat_conversations
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own conversations" ON chat_conversations
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own conversations" ON chat_conversations
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own conversations" ON chat_conversations
  FOR DELETE USING (auth.uid() = user_id);

-- Function to get recent conversation history
CREATE OR REPLACE FUNCTION get_chat_history(
  p_user_id UUID DEFAULT auth.uid(),
  p_limit INT DEFAULT 100
)
RETURNS TABLE (
  id UUID,
  role TEXT,
  content TEXT,
  images TEXT[],
  actions JSONB,
  suggestions TEXT[],
  message_timestamp BIGINT,
  created_at TIMESTAMPTZ
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    c.id,
    c.role,
    c.content,
    c.images,
    c.actions,
    c.suggestions,
    c.message_timestamp,
    c.created_at
  FROM chat_conversations c
  WHERE c.user_id = p_user_id
  ORDER BY c.message_timestamp DESC
  LIMIT p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to sync a batch of messages (for initial upload from localStorage)
CREATE OR REPLACE FUNCTION sync_chat_messages(
  p_messages JSONB
)
RETURNS INT AS $$
DECLARE
  msg JSONB;
  inserted_count INT := 0;
BEGIN
  FOR msg IN SELECT * FROM jsonb_array_elements(p_messages)
  LOOP
    INSERT INTO chat_conversations (
      user_id,
      role,
      content,
      images,
      actions,
      suggestions,
      message_timestamp
    ) VALUES (
      auth.uid(),
      msg->>'role',
      msg->>'content',
      COALESCE((SELECT array_agg(elem::text) FROM jsonb_array_elements_text(msg->'images') AS elem), '{}'),
      COALESCE(msg->'actions', '[]'),
      COALESCE((SELECT array_agg(elem::text) FROM jsonb_array_elements_text(msg->'suggestions') AS elem), '{}'),
      (msg->>'timestamp')::BIGINT
    )
    ON CONFLICT DO NOTHING;

    inserted_count := inserted_count + 1;
  END LOOP;

  RETURN inserted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to clear conversation history (with confirmation)
CREATE OR REPLACE FUNCTION clear_chat_history()
RETURNS INT AS $$
DECLARE
  deleted_count INT;
BEGIN
  DELETE FROM chat_conversations
  WHERE user_id = auth.uid();

  GET DIAGNOSTICS deleted_count = ROW_COUNT;
  RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Comments for documentation
COMMENT ON TABLE chat_conversations IS 'Axel conversation history - synced from all devices';
COMMENT ON COLUMN chat_conversations.role IS 'user: human message | assistant: Axel response | bapi: BAPI insight | system: system message';
COMMENT ON COLUMN chat_conversations.actions IS 'JSON array of action results from assistant responses';
COMMENT ON COLUMN chat_conversations.message_timestamp IS 'Client timestamp in milliseconds for ordering';
COMMENT ON FUNCTION sync_chat_messages(JSONB) IS 'Batch sync messages from localStorage to server';
