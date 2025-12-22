-- Node Access Heat Tracking
-- Tracks AI and user access patterns for each node to enable metabolic architecture

CREATE TABLE IF NOT EXISTS node_access_heat (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  node_id TEXT NOT NULL,  -- References node.id in the mind map

  -- Heat metrics
  access_count INT DEFAULT 0,                    -- Total accesses
  ai_context_inclusions INT DEFAULT 0,           -- Times included in Claude's context
  ai_modifications INT DEFAULT 0,                -- Times Claude modified/referenced this node
  ai_creations INT DEFAULT 0,                    -- Times Claude created children from this
  user_selections INT DEFAULT 0,                 -- Times user selected/focused on node
  user_expansions INT DEFAULT 0,                 -- Times user expanded this node's children

  -- Temporal data
  last_accessed TIMESTAMPTZ DEFAULT NOW(),       -- Last any access
  last_ai_accessed TIMESTAMPTZ,                  -- Last AI context inclusion
  last_user_accessed TIMESTAMPTZ,                -- Last user selection
  first_accessed TIMESTAMPTZ DEFAULT NOW(),      -- When tracking began

  -- Session metrics
  current_session_accesses INT DEFAULT 0,        -- Accesses this session
  total_sessions_touched INT DEFAULT 0,          -- Unique sessions that touched this

  -- Computed heat score (updated by daemon)
  heat_score FLOAT DEFAULT 0.5 CHECK (heat_score >= 0 AND heat_score <= 1),
  heat_tier TEXT DEFAULT 'warm' CHECK (heat_tier IN ('hot', 'warm', 'cool', 'cold', 'dormant')),

  -- Connection metrics (for circulation)
  connection_count INT DEFAULT 0,                -- Number of child nodes
  orphan_risk BOOLEAN DEFAULT FALSE,             -- True if isolated and cooling

  -- Metadata
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Unique constraint: one heat record per user per node
CREATE UNIQUE INDEX IF NOT EXISTS idx_node_heat_unique ON node_access_heat(user_id, node_id);

-- Efficient queries for heat-based operations
CREATE INDEX IF NOT EXISTS idx_node_heat_user_access ON node_access_heat(user_id, last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_node_heat_score ON node_access_heat(user_id, heat_score DESC);
CREATE INDEX IF NOT EXISTS idx_node_heat_tier ON node_access_heat(user_id, heat_tier);
CREATE INDEX IF NOT EXISTS idx_node_heat_cold ON node_access_heat(user_id, heat_tier, last_accessed)
  WHERE heat_tier IN ('cold', 'dormant');

-- Enable RLS
ALTER TABLE node_access_heat ENABLE ROW LEVEL SECURITY;

-- Policy: users can only access their own heat data
CREATE POLICY "Users can view their own heat data"
  ON node_access_heat FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own heat data"
  ON node_access_heat FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own heat data"
  ON node_access_heat FOR UPDATE
  USING (auth.uid() = user_id);

-- Function to update heat score based on activity
CREATE OR REPLACE FUNCTION calculate_heat_score(
  p_access_count INT,
  p_ai_context_inclusions INT,
  p_ai_modifications INT,
  p_last_accessed TIMESTAMPTZ
) RETURNS FLOAT AS $$
DECLARE
  base_score FLOAT;
  recency_factor FLOAT;
  days_since_access FLOAT;
BEGIN
  -- Base score from activity (0-0.7)
  base_score := LEAST(0.7, (
    p_access_count * 0.01 +
    p_ai_context_inclusions * 0.02 +
    p_ai_modifications * 0.05
  ));

  -- Recency factor (0-0.3) with 30-day half-life decay
  days_since_access := EXTRACT(EPOCH FROM (NOW() - COALESCE(p_last_accessed, NOW()))) / 86400.0;
  recency_factor := 0.3 * POWER(0.5, days_since_access / 30.0);

  RETURN LEAST(1.0, base_score + recency_factor);
END;
$$ LANGUAGE plpgsql;

-- Function to determine heat tier from score
CREATE OR REPLACE FUNCTION get_heat_tier(score FLOAT) RETURNS TEXT AS $$
BEGIN
  IF score >= 0.8 THEN RETURN 'hot';
  ELSIF score >= 0.5 THEN RETURN 'warm';
  ELSIF score >= 0.3 THEN RETURN 'cool';
  ELSIF score >= 0.1 THEN RETURN 'cold';
  ELSE RETURN 'dormant';
  END IF;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update heat score and tier on any change
CREATE OR REPLACE FUNCTION update_node_heat_trigger() RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at := NOW();
  NEW.heat_score := calculate_heat_score(
    NEW.access_count,
    NEW.ai_context_inclusions,
    NEW.ai_modifications,
    NEW.last_accessed
  );
  NEW.heat_tier := get_heat_tier(NEW.heat_score);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER node_heat_update_trigger
  BEFORE UPDATE ON node_access_heat
  FOR EACH ROW
  EXECUTE FUNCTION update_node_heat_trigger();

-- View for aggregated heat statistics per user
CREATE OR REPLACE VIEW user_heat_summary AS
SELECT
  user_id,
  COUNT(*) as total_tracked_nodes,
  COUNT(*) FILTER (WHERE heat_tier = 'hot') as hot_count,
  COUNT(*) FILTER (WHERE heat_tier = 'warm') as warm_count,
  COUNT(*) FILTER (WHERE heat_tier = 'cool') as cool_count,
  COUNT(*) FILTER (WHERE heat_tier = 'cold') as cold_count,
  COUNT(*) FILTER (WHERE heat_tier = 'dormant') as dormant_count,
  AVG(heat_score) as avg_heat_score,
  MAX(last_accessed) as last_activity,
  SUM(access_count) as total_accesses
FROM node_access_heat
GROUP BY user_id;

-- Grant access to view
GRANT SELECT ON user_heat_summary TO authenticated;

COMMENT ON TABLE node_access_heat IS 'Tracks AI and user access patterns for metabolic architecture - nodes gain heat through interaction and cool over time';
COMMENT ON COLUMN node_access_heat.heat_score IS 'Computed score 0-1 based on activity and recency with 30-day half-life decay';
COMMENT ON COLUMN node_access_heat.heat_tier IS 'Categorical tier: hot (>0.8), warm (0.5-0.8), cool (0.3-0.5), cold (0.1-0.3), dormant (<0.1)';
