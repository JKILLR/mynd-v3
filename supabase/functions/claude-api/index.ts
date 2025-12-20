// Supabase Edge Function for MYND Chat with full tool support
// Deploy with: supabase functions deploy claude-api

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'
const CLAUDE_MODEL = 'claude-sonnet-4-20250514'

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
    description: "Read the contents of a file from the codebase. Use this to examine specific code.",
    input_schema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "The file path relative to project root (e.g., 'js/app-module.js')"
        }
      },
      required: ["path"]
    }
  },
  {
    name: "search_code",
    description: "Search for patterns or text across all code files. Returns matching lines with context.",
    input_schema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The search pattern or text to find"
        },
        filePattern: {
          type: "string",
          description: "Optional glob pattern to filter files (e.g., '*.js', '*.html')"
        }
      },
      required: ["query"]
    }
  },
  {
    name: "list_files",
    description: "List files in a directory or matching a pattern.",
    input_schema: {
      type: "object",
      properties: {
        pattern: {
          type: "string",
          description: "Glob pattern or directory path (e.g., 'js/*.js', 'components/')"
        }
      },
      required: ["pattern"]
    }
  },
  {
    name: "get_codebase_overview",
    description: "Get a high-level overview of the codebase architecture and structure.",
    input_schema: {
      type: "object",
      properties: {},
      required: []
    }
  },
  {
    name: "get_function_definition",
    description: "Find and return the definition of a specific function or class.",
    input_schema: {
      type: "object",
      properties: {
        name: {
          type: "string",
          description: "The function or class name to find"
        }
      },
      required: ["name"]
    }
  }
]

const GITHUB_TOOLS = [
  {
    name: "github_create_branch",
    description: "Create a new branch from the base branch for making changes.",
    input_schema: {
      type: "object",
      properties: {
        branch_name: {
          type: "string",
          description: "Name for the new branch (e.g., 'fix-login-bug', 'add-dark-mode')"
        }
      },
      required: ["branch_name"]
    }
  },
  {
    name: "github_get_file",
    description: "Read the current contents of a file from GitHub.",
    input_schema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "File path relative to repo root"
        },
        branch: {
          type: "string",
          description: "Branch to read from (defaults to base branch)"
        }
      },
      required: ["path"]
    }
  },
  {
    name: "github_write_file",
    description: "Create or update a file in the repository. This commits the change.",
    input_schema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "File path relative to repo root"
        },
        content: {
          type: "string",
          description: "The full file content to write"
        },
        message: {
          type: "string",
          description: "Commit message describing the change"
        },
        branch: {
          type: "string",
          description: "Branch to commit to (must use your created branch, not main)"
        }
      },
      required: ["path", "content", "message", "branch"]
    }
  },
  {
    name: "github_create_pr",
    description: "Create a pull request for your changes.",
    input_schema: {
      type: "object",
      properties: {
        branch: {
          type: "string",
          description: "The branch with your changes"
        },
        title: {
          type: "string",
          description: "PR title"
        },
        body: {
          type: "string",
          description: "PR description"
        }
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
        per_page: {
          type: "number",
          description: "Number of branches to return (default 30)"
        }
      },
      required: []
    }
  },
  {
    name: "github_list_commits",
    description: "List recent commits. Can filter by branch, path, or author.",
    input_schema: {
      type: "object",
      properties: {
        branch: {
          type: "string",
          description: "Branch name to list commits from"
        },
        path: {
          type: "string",
          description: "Only commits affecting this file path"
        },
        author: {
          type: "string",
          description: "GitHub username to filter by"
        },
        per_page: {
          type: "number",
          description: "Number of commits to return (default 10)"
        }
      },
      required: []
    }
  },
  {
    name: "github_get_commit",
    description: "Get detailed information about a specific commit including diff.",
    input_schema: {
      type: "object",
      properties: {
        sha: {
          type: "string",
          description: "The commit SHA"
        }
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
        base: {
          type: "string",
          description: "Base branch or commit SHA"
        },
        head: {
          type: "string",
          description: "Head branch or commit SHA"
        }
      },
      required: ["base", "head"]
    }
  }
]

// ═══════════════════════════════════════════════════════════════════
// GITHUB API HELPERS
// ═══════════════════════════════════════════════════════════════════

async function githubRequest(
  endpoint: string,
  token: string,
  owner: string,
  repo: string,
  options: RequestInit = {}
): Promise<any> {
  const url = `https://api.github.com/repos/${owner}/${repo}${endpoint}`
  const response = await fetch(url, {
    ...options,
    headers: {
      'Authorization': `Bearer ${token}`,
      'Accept': 'application/vnd.github.v3+json',
      'User-Agent': 'MYND-App',
      ...options.headers,
    },
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`GitHub API error ${response.status}: ${error}`)
  }

  return response.json()
}

// ═══════════════════════════════════════════════════════════════════
// TOOL EXECUTION
// ═══════════════════════════════════════════════════════════════════

async function executeTool(
  toolName: string,
  toolInput: any,
  context: { githubToken?: string; githubOwner?: string; githubRepo?: string; baseBranch?: string }
): Promise<any> {
  const { githubToken, githubOwner, githubRepo, baseBranch = 'main' } = context

  // GitHub tools require configuration
  if (toolName.startsWith('github_')) {
    if (!githubToken || !githubOwner || !githubRepo) {
      return { error: 'GitHub integration not configured. Please set up GitHub in the Neural panel.' }
    }
  }

  try {
    switch (toolName) {
      // ─────────────────────────────────────────────────────────────
      // CODEBASE TOOLS (return placeholder - client handles these)
      // ─────────────────────────────────────────────────────────────
      case 'read_file':
      case 'search_code':
      case 'list_files':
      case 'get_codebase_overview':
      case 'get_function_definition':
        return {
          note: 'Codebase tools are executed client-side where the code is available.',
          tool: toolName,
          input: toolInput
        }

      // ─────────────────────────────────────────────────────────────
      // GITHUB TOOLS
      // ─────────────────────────────────────────────────────────────
      case 'github_create_branch': {
        const { branch_name } = toolInput
        // Get the SHA of the base branch
        const baseRef = await githubRequest(`/git/refs/heads/${baseBranch}`, githubToken!, githubOwner!, githubRepo!)
        const baseSha = baseRef.object.sha

        // Create new branch
        await githubRequest('/git/refs', githubToken!, githubOwner!, githubRepo!, {
          method: 'POST',
          body: JSON.stringify({
            ref: `refs/heads/${branch_name}`,
            sha: baseSha
          })
        })

        return {
          success: true,
          branch: branch_name,
          basedOn: baseBranch,
          message: `Branch '${branch_name}' created from '${baseBranch}'`
        }
      }

      case 'github_get_file': {
        const { path, branch } = toolInput
        const targetBranch = branch || baseBranch
        const file = await githubRequest(`/contents/${path}?ref=${targetBranch}`, githubToken!, githubOwner!, githubRepo!)

        if (file.type !== 'file') {
          return { error: `Path '${path}' is not a file` }
        }

        const content = atob(file.content)
        return {
          success: true,
          path: file.path,
          content: content,
          sha: file.sha,
          size: file.size
        }
      }

      case 'github_write_file': {
        const { path, content, message, branch } = toolInput

        // Safety: prevent writes to protected branches
        const protectedBranches = ['main', 'master', baseBranch]
        if (protectedBranches.includes(branch)) {
          return { error: `Cannot write directly to protected branch '${branch}'. Create a feature branch first.` }
        }

        // Check if file exists to get SHA
        let existingSha: string | undefined
        try {
          const existing = await githubRequest(`/contents/${path}?ref=${branch}`, githubToken!, githubOwner!, githubRepo!)
          existingSha = existing.sha
        } catch {
          // File doesn't exist yet
        }

        const result = await githubRequest(`/contents/${path}`, githubToken!, githubOwner!, githubRepo!, {
          method: 'PUT',
          body: JSON.stringify({
            message,
            content: btoa(content),
            branch,
            ...(existingSha && { sha: existingSha })
          })
        })

        return {
          success: true,
          path: result.content.path,
          sha: result.content.sha,
          commit: result.commit.sha.substring(0, 7),
          message: `File ${existingSha ? 'updated' : 'created'}: ${path}`
        }
      }

      case 'github_create_pr': {
        const { branch, title, body } = toolInput
        const result = await githubRequest('/pulls', githubToken!, githubOwner!, githubRepo!, {
          method: 'POST',
          body: JSON.stringify({
            title,
            body,
            head: branch,
            base: baseBranch
          })
        })

        return {
          success: true,
          number: result.number,
          title: result.title,
          url: result.html_url,
          message: `Pull request #${result.number} created: ${result.html_url}`
        }
      }

      case 'github_list_branches': {
        const { per_page = 30 } = toolInput
        const branches = await githubRequest(`/branches?per_page=${Math.min(per_page, 100)}`, githubToken!, githubOwner!, githubRepo!)

        return {
          success: true,
          count: branches.length,
          branches: branches.map((b: any) => ({
            name: b.name,
            sha: b.commit.sha.substring(0, 7),
            protected: b.protected
          }))
        }
      }

      case 'github_list_commits': {
        const { branch, path, author, per_page = 10 } = toolInput
        const params = new URLSearchParams()
        params.set('per_page', Math.min(per_page, 100).toString())
        if (branch) params.set('sha', branch)
        if (path) params.set('path', path)
        if (author) params.set('author', author)

        const commits = await githubRequest(`/commits?${params.toString()}`, githubToken!, githubOwner!, githubRepo!)

        return {
          success: true,
          count: commits.length,
          commits: commits.map((c: any) => ({
            sha: c.sha.substring(0, 7),
            full_sha: c.sha,
            message: c.commit.message.split('\n')[0],
            author: c.commit.author.name,
            date: c.commit.author.date
          }))
        }
      }

      case 'github_get_commit': {
        const { sha } = toolInput
        const commit = await githubRequest(`/commits/${sha}`, githubToken!, githubOwner!, githubRepo!)

        return {
          success: true,
          sha: commit.sha,
          short_sha: commit.sha.substring(0, 7),
          message: commit.commit.message,
          author: {
            name: commit.commit.author.name,
            date: commit.commit.author.date
          },
          stats: commit.stats,
          files_changed: commit.files.length,
          files: commit.files.slice(0, 20).map((f: any) => ({
            filename: f.filename,
            status: f.status,
            additions: f.additions,
            deletions: f.deletions,
            patch: f.patch?.substring(0, 1000)
          }))
        }
      }

      case 'github_compare': {
        const { base, head } = toolInput
        const comparison = await githubRequest(`/compare/${base}...${head}`, githubToken!, githubOwner!, githubRepo!)

        return {
          success: true,
          status: comparison.status,
          ahead_by: comparison.ahead_by,
          behind_by: comparison.behind_by,
          total_commits: comparison.total_commits,
          commits: comparison.commits.slice(0, 10).map((c: any) => ({
            sha: c.sha.substring(0, 7),
            message: c.commit.message.split('\n')[0]
          })),
          files_changed: comparison.files.length
        }
      }

      default:
        return { error: `Unknown tool: ${toolName}` }
    }
  } catch (error) {
    return { error: error.message }
  }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN HANDLER
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
      type,
      messages,
      maxTokens = 4096,
      webSearch = false,
      systemPrompt,
      // GitHub configuration (optional)
      githubToken,
      githubOwner,
      githubRepo,
      githubBaseBranch = 'main',
      // Tool configuration
      enableCodebaseTools = true,
      enableGithubTools = true
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

    // Codebase tools (always available - executed client-side)
    if (enableCodebaseTools) {
      tools.push(...CODEBASE_TOOLS)
    }

    // GitHub tools (only if configured)
    if (enableGithubTools && githubToken && githubOwner && githubRepo) {
      tools.push(...GITHUB_TOOLS)
    }

    // Build request body
    const requestBody: any = {
      model: CLAUDE_MODEL,
      max_tokens: maxTokens,
      messages: messages
    }

    if (systemPrompt) {
      requestBody.system = systemPrompt
    }

    if (tools.length > 0) {
      requestBody.tools = tools
    }

    // Codebase tools that need client-side execution
    const CODEBASE_TOOL_NAMES = ['read_file', 'search_code', 'list_files', 'get_codebase_overview', 'get_function_definition']

    // Agentic loop for tool execution
    let iterations = 0
    const maxIterations = 10
    let currentMessages = [...messages]
    let finalResponse = ''

    while (iterations < maxIterations) {
      iterations++

      const response = await fetch(ANTHROPIC_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': ANTHROPIC_API_KEY,
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
          ...requestBody,
          messages: currentMessages
        })
      })

      if (!response.ok) {
        const error = await response.text()
        throw new Error(`Anthropic API error: ${error}`)
      }

      const data = await response.json()

      // Check for tool use
      const toolUseBlocks = data.content?.filter((b: any) => b.type === 'tool_use') || []
      const textBlocks = data.content?.filter((b: any) => b.type === 'text') || []

      // If no tool calls or end_turn, we're done
      if (toolUseBlocks.length === 0 || data.stop_reason === 'end_turn') {
        finalResponse = textBlocks.map((b: any) => b.text).join('\n')
        break
      }

      // Check for codebase tools that need client-side execution
      const codebaseTools = toolUseBlocks.filter((t: any) => CODEBASE_TOOL_NAMES.includes(t.name))
      const serverTools = toolUseBlocks.filter((t: any) => !CODEBASE_TOOL_NAMES.includes(t.name) && t.name !== 'web_search')

      // If there are codebase tools, return them to the client for execution
      if (codebaseTools.length > 0) {
        // Add assistant message to current messages
        currentMessages.push({ role: 'assistant', content: data.content })

        return new Response(
          JSON.stringify({
            pendingTools: codebaseTools.map((t: any) => ({
              id: t.id,
              name: t.name,
              input: t.input
            })),
            currentMessages: currentMessages,
            textSoFar: textBlocks.map((b: any) => b.text).join('\n')
          }),
          {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
            status: 200,
          }
        )
      }

      // Execute server-side tools (GitHub tools)
      currentMessages.push({ role: 'assistant', content: data.content })

      const toolResults = []
      for (const toolUse of serverTools) {
        const result = await executeTool(toolUse.name, toolUse.input, {
          githubToken,
          githubOwner,
          githubRepo,
          baseBranch: githubBaseBranch
        })

        toolResults.push({
          type: 'tool_result',
          tool_use_id: toolUse.id,
          content: JSON.stringify(result, null, 2)
        })
      }

      if (toolResults.length > 0) {
        currentMessages.push({ role: 'user', content: toolResults })
      } else {
        // web_search only
        finalResponse = textBlocks.map((b: any) => b.text).join('\n')
        break
      }
    }

    return new Response(
      JSON.stringify({ response: finalResponse }),
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
