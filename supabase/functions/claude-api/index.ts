// Supabase Edge Function for MYND Chat
// Simple proxy to Claude API - client handles tool execution
// Deploy with: supabase functions deploy claude-api

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'

const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'
const CLAUDE_MODEL = 'claude-opus-4-5-20251101'

// CORS headers
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// ═══════════════════════════════════════════════════════════════════
// TOOL DEFINITIONS
// ═══════════════════════════════════════════════════════════════════

const CODEBASE_TOOLS = [
  {
    name: "read_file",
    description: "Read the contents of a file from the codebase.",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path relative to project root" }
      },
      required: ["path"]
    }
  },
  {
    name: "search_code",
    description: "Search for patterns across all code files.",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search pattern" },
        filePattern: { type: "string", description: "Optional glob pattern" }
      },
      required: ["query"]
    }
  },
  {
    name: "list_files",
    description: "List files matching a pattern.",
    input_schema: {
      type: "object",
      properties: {
        pattern: { type: "string", description: "Glob pattern or directory" }
      },
      required: ["pattern"]
    }
  },
  {
    name: "get_codebase_overview",
    description: "Get high-level codebase architecture summary.",
    input_schema: { type: "object", properties: {}, required: [] }
  },
  {
    name: "get_function_definition",
    description: "Find a specific function or class definition.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Function or class name" }
      },
      required: ["name"]
    }
  }
]

const GITHUB_TOOLS = [
  {
    name: "github_create_branch",
    description: "Create a new branch for changes.",
    input_schema: {
      type: "object",
      properties: {
        branch_name: { type: "string", description: "Name for new branch" }
      },
      required: ["branch_name"]
    }
  },
  {
    name: "github_get_file",
    description: "Read file contents from GitHub.",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path" },
        branch: { type: "string", description: "Branch name" }
      },
      required: ["path"]
    }
  },
  {
    name: "github_write_file",
    description: "Create or update a file (commits the change).",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path" },
        content: { type: "string", description: "File content" },
        message: { type: "string", description: "Commit message" },
        branch: { type: "string", description: "Branch to commit to" }
      },
      required: ["path", "content", "message", "branch"]
    }
  },
  {
    name: "github_create_pr",
    description: "Create a pull request.",
    input_schema: {
      type: "object",
      properties: {
        branch: { type: "string", description: "Branch with changes" },
        title: { type: "string", description: "PR title" },
        body: { type: "string", description: "PR description" }
      },
      required: ["branch", "title", "body"]
    }
  },
  {
    name: "github_list_branches",
    description: "List all branches in the repository.",
    input_schema: {
      type: "object",
      properties: {
        per_page: { type: "number", description: "Number to return" }
      },
      required: []
    }
  },
  {
    name: "github_list_commits",
    description: "List recent commits.",
    input_schema: {
      type: "object",
      properties: {
        branch: { type: "string", description: "Branch name" },
        path: { type: "string", description: "File path filter" },
        author: { type: "string", description: "Author filter" },
        per_page: { type: "number", description: "Number to return" }
      },
      required: []
    }
  },
  {
    name: "github_get_commit",
    description: "Get detailed commit information including diff.",
    input_schema: {
      type: "object",
      properties: {
        sha: { type: "string", description: "Commit SHA" }
      },
      required: ["sha"]
    }
  },
  {
    name: "github_compare",
    description: "Compare two branches or commits.",
    input_schema: {
      type: "object",
      properties: {
        base: { type: "string", description: "Base branch/commit" },
        head: { type: "string", description: "Head branch/commit" }
      },
      required: ["base", "head"]
    }
  }
]

// ═══════════════════════════════════════════════════════════════════
// SELF-QUERY TOOLS (Inner Dialogue)
// These enable Claude to query its own knowledge systems mid-reasoning
// ═══════════════════════════════════════════════════════════════════

const SELF_QUERY_TOOLS = [
  {
    name: "think",
    description: "Pause to think and continue reasoning. Use this when you want to reflect on information gathered, consider multiple angles, or think through a complex response before continuing. Your thought will be logged but not shown to user. After thinking, you can respond again or use more tools.",
    input_schema: {
      type: "object",
      properties: {
        thought: { type: "string", description: "Your internal thought or reflection" }
      },
      required: ["thought"]
    }
  },
  {
    name: "query_focus",
    description: "Get current session context - recently viewed/edited nodes, active branch, what user is working on right now. Use to understand immediate context.",
    input_schema: {
      type: "object",
      properties: {},
      required: []
    }
  },
  {
    name: "query_similar",
    description: "Find semantically similar nodes using embeddings. Use when looking for related concepts, potential duplicates, or connection opportunities.",
    input_schema: {
      type: "object",
      properties: {
        concept: { type: "string", description: "The concept or text to find similar nodes for" },
        threshold: { type: "number", description: "Similarity threshold 0-1. Higher = more similar. Default: 0.6" },
        limit: { type: "number", description: "Maximum results to return. Default: 5" }
      },
      required: ["concept"]
    }
  },
  {
    name: "query_insights",
    description: "Retrieve taught neural insights - connections you've learned, patterns discovered, user goals. Use when you need your accumulated understanding.",
    input_schema: {
      type: "object",
      properties: {
        insight_type: { type: "string", description: "Type: 'connection_insight', 'user_goal', 'neural_insight', or 'all'. Default: all" },
        related_to: { type: "string", description: "Optional: filter insights related to a specific concept" }
      },
      required: []
    }
  },
  {
    name: "query_memory",
    description: "Search conversation history and past interactions. Use when you need to recall what the user has said about something before.",
    input_schema: {
      type: "object",
      properties: {
        topic: { type: "string", description: "The topic, concept, or keyword to search for in memory" },
        limit: { type: "number", description: "Maximum results to return. Default: 5" }
      },
      required: ["topic"]
    }
  },
  {
    name: "query_patterns",
    description: "Retrieve learned user preferences and behavioral patterns. Use when you need to understand how the user typically works or what they prefer.",
    input_schema: {
      type: "object",
      properties: {
        domain: { type: "string", description: "Domain: 'naming', 'colors', 'structure', 'categories', 'all'. Default: all" }
      },
      required: []
    }
  },
  {
    name: "query_connections",
    description: "Find what connects to a concept or node - parents, children, semantic relationships. Use when understanding context around an idea.",
    input_schema: {
      type: "object",
      properties: {
        concept: { type: "string", description: "The concept or node label to find connections for" },
        depth: { type: "number", description: "How many levels of connections to traverse. Default: 1" }
      },
      required: ["concept"]
    }
  }
]

// ═══════════════════════════════════════════════════════════════════
// MAIN HANDLER - Simple proxy, client handles tool execution
// ═══════════════════════════════════════════════════════════════════

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Get API key from environment
    const ANTHROPIC_API_KEY = Deno.env.get('ANTHROPIC_API_KEY')
    if (!ANTHROPIC_API_KEY) {
      throw new Error('ANTHROPIC_API_KEY not configured')
    }

    // Parse request
    const {
      messages,
      maxTokens = 4096,
      webSearch = false,
      systemPrompt,
      enableCodebaseTools = true,
      enableGithubTools = false,
      enableSelfQueryTools = true  // Inner dialogue tools - enabled by default
    } = await req.json()

    // Build tools array
    const tools: any[] = []

    // Web search tool
    if (webSearch) {
      tools.push({
        type: 'web_search_20250305',
        name: 'web_search',
        max_uses: 3
      })
    }

    // Codebase tools
    if (enableCodebaseTools) {
      tools.push(...CODEBASE_TOOLS)
    }

    // GitHub tools
    if (enableGithubTools) {
      tools.push(...GITHUB_TOOLS)
    }

    // Self-query tools (inner dialogue)
    if (enableSelfQueryTools) {
      tools.push(...SELF_QUERY_TOOLS)
    }

    // Build request body
    const requestBody: any = {
      model: CLAUDE_MODEL,
      max_tokens: maxTokens,
      messages: messages
    }

    // Handle system prompt - can be string or array with cache_control
    // Array format enables prompt caching for 60-70% token savings
    if (systemPrompt) {
      requestBody.system = systemPrompt
    }

    if (tools.length > 0) {
      requestBody.tools = tools
    }

    // Build headers - include prompt caching beta if system is array format
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'x-api-key': ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01'
    }

    // Enable prompt caching if using array format with cache_control
    if (Array.isArray(systemPrompt)) {
      headers['anthropic-beta'] = 'prompt-caching-2024-07-31'
      console.log('🔄 Prompt caching enabled (system prompt array format)')
    }

    // Single call to Claude API - return full response
    const response = await fetch(ANTHROPIC_API_URL, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody)
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`Anthropic API error: ${error}`)
    }

    const data = await response.json()

    // Log cache statistics if available
    if (data.usage) {
      const { input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens } = data.usage
      if (cache_read_input_tokens || cache_creation_input_tokens) {
        console.log(`📊 Prompt caching stats:
  - Cache read: ${cache_read_input_tokens || 0} tokens (saved)
  - Cache creation: ${cache_creation_input_tokens || 0} tokens (first time)
  - Non-cached input: ${input_tokens - (cache_read_input_tokens || 0) - (cache_creation_input_tokens || 0)} tokens
  - Output: ${output_tokens} tokens`)
      }
    }

    // Check for tool use - return full response for client to handle
    const toolUseBlocks = data.content?.filter((b: any) => b.type === 'tool_use') || []
    const textBlocks = data.content?.filter((b: any) => b.type === 'text') || []

    // Include cache stats in response for client visibility
    const cacheStats = data.usage ? {
      cache_read_input_tokens: data.usage.cache_read_input_tokens || 0,
      cache_creation_input_tokens: data.usage.cache_creation_input_tokens || 0,
      input_tokens: data.usage.input_tokens,
      output_tokens: data.usage.output_tokens
    } : null

    if (toolUseBlocks.length > 0) {
      // Return full response so client can execute tools and continue
      return new Response(
        JSON.stringify({
          needsToolExecution: true,
          content: data.content,
          stop_reason: data.stop_reason,
          toolCalls: toolUseBlocks.map((t: any) => ({
            id: t.id,
            name: t.name,
            input: t.input
          })),
          textSoFar: textBlocks.map((b: any) => b.text).join('\n'),
          cacheStats
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 200,
        }
      )
    }

    // No tools - return text response
    const responseText = textBlocks.map((b: any) => b.text).join('\n')

    return new Response(
      JSON.stringify({ response: responseText, cacheStats }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )

  } catch (error) {
    console.error('Error:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 400,
      }
    )
  }
})
