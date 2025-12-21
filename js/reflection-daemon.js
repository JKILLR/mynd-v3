/**
 * MYND Reflection Daemon
 * ======================
 * Background reflection system that allows the MYND AI to autonomously
 * review its own codebase and mind map during user idle periods.
 *
 * Features:
 *   - Idle detection (configurable threshold)
 *   - Scheduled reflection cycles
 *   - Code and map analysis via Claude
 *   - Insight queue with approval workflow
 *   - Transparent activity logging
 */

const ReflectionDaemon = {
    // ═══════════════════════════════════════════════════════════════════
    // CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════

    VERSION: '2.0',  // Upgraded with tool use
    STORAGE_KEY: 'mynd-reflection-daemon',
    DB_NAME: 'mynd-reflection-db',
    DB_VERSION: 1,
    STORE_NAME: 'reflection_queue',

    config: {
        enabled: false,                    // Master toggle for autonomous mode
        idleThresholdMs: 5 * 60 * 1000,    // 5 minutes default
        reflectionIntervalMs: 30 * 60 * 1000, // 30 minutes default
        minReflectionIntervalMs: 15 * 60 * 1000, // 15 min minimum (rate limit)
        maxTokensPerReflection: 16000,     // Increased for agentic reasoning
        maxContextChars: 50000,            // Increased context window
        maxToolIterations: 50,             // Max tool use iterations per reflection
        autoAddToMap: false,               // Auto-add reflection log nodes
        frequencies: {
            '15min': 15 * 60 * 1000,
            '30min': 30 * 60 * 1000,
            '1hr': 60 * 60 * 1000,
            '2hr': 2 * 60 * 60 * 1000
        },
        // GitHub Integration (Option 2: Auto-commit to branch)
        github: {
            enabled: false,                 // Enable GitHub auto-commit
            owner: '',                      // GitHub repo owner (e.g., 'JKILLR')
            repo: '',                       // GitHub repo name (e.g., 'mynd-v3')
            baseBranch: 'main',             // Branch to create feature branches from
            branchPrefix: 'mynd-reflection', // Prefix for auto-created branches
            autoCreatePR: false,            // Automatically create PR after commits
            requireApproval: true           // Require user approval before committing
        }
    },

    // GitHub state
    githubToken: null,  // Stored separately for security

    // Search cache to prevent duplicate/similar searches
    searchCache: new Map(),
    searchCacheMaxAge: 5 * 60 * 1000, // 5 minutes
    searchCacheMaxSize: 50,

    // Clear search cache (call when starting new conversation)
    clearSearchCache() {
        this.searchCache.clear();
        console.log('🔍 Search cache cleared');
    },

    // Normalize search query for cache key
    normalizeSearchQuery(query) {
        return query.toLowerCase()
            .replace(/\s+/g, ' ')
            .replace(/[.*+?^${}()|[\]\\]/g, '') // Remove regex special chars for comparison
            .trim();
    },

    // Check if an identical search was already done
    findSimilarSearch(query, file_pattern) {
        const normalizedQuery = this.normalizeSearchQuery(query);
        const now = Date.now();

        // Clean old entries
        for (const [key, entry] of this.searchCache) {
            if (now - entry.timestamp > this.searchCacheMaxAge) {
                this.searchCache.delete(key);
            }
        }

        // Look for exact match only (prevents false positives like "render" matching "renderComponent")
        for (const [key, entry] of this.searchCache) {
            const cachedNormalized = this.normalizeSearchQuery(entry.query);
            if (cachedNormalized === normalizedQuery && entry.file_pattern === file_pattern) {
                console.log(`🔍 Search cache hit: "${query}" matches cached "${entry.query}"`);
                return entry.results;
            }
        }
        return null;
    },

    // ═══════════════════════════════════════════════════════════════════
    // TOOLS DEFINITION - Gives the reflection engine coding capabilities
    // ═══════════════════════════════════════════════════════════════════

    TOOLS: [
        {
            name: "read_file",
            description: "Read the contents of a file from the codebase. Use this to examine specific code files in detail.",
            input_schema: {
                type: "object",
                properties: {
                    path: {
                        type: "string",
                        description: "The file path relative to project root (e.g., 'js/app-module.js', 'index.html')"
                    },
                    start_line: {
                        type: "number",
                        description: "Optional: Start reading from this line number (1-indexed)"
                    },
                    end_line: {
                        type: "number",
                        description: "Optional: Stop reading at this line number"
                    }
                },
                required: ["path"]
            }
        },
        {
            name: "search_code",
            description: "Search for code patterns, function names, or text across the codebase. Returns matching lines with context.",
            input_schema: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "The search query (supports regex patterns)"
                    },
                    file_pattern: {
                        type: "string",
                        description: "Optional: Filter to files matching this pattern (e.g., '*.js', 'js/*.js')"
                    },
                    max_results: {
                        type: "number",
                        description: "Maximum number of results to return (default: 20)"
                    }
                },
                required: ["query"]
            }
        },
        {
            name: "list_files",
            description: "List files in the codebase matching a pattern. Use to explore project structure.",
            input_schema: {
                type: "object",
                properties: {
                    pattern: {
                        type: "string",
                        description: "Glob pattern to match files (e.g., 'js/*.js', '**/*.html', 'src/**')"
                    }
                },
                required: ["pattern"]
            }
        },
        {
            name: "get_codebase_overview",
            description: "Get a high-level overview of the codebase structure, main files, and architecture.",
            input_schema: {
                type: "object",
                properties: {},
                required: []
            }
        },
        {
            name: "get_function_definition",
            description: "Find and return a specific function or class definition from the codebase.",
            input_schema: {
                type: "object",
                properties: {
                    name: {
                        type: "string",
                        description: "The function, class, or method name to find"
                    },
                    file_hint: {
                        type: "string",
                        description: "Optional: Hint about which file it might be in"
                    }
                },
                required: ["name"]
            }
        },
        // ═══════════════════════════════════════════════════════════════════
        // GITHUB TOOLS - For autonomous code commits (Option 2)
        // ═══════════════════════════════════════════════════════════════════
        {
            name: "github_create_branch",
            description: "Create a new branch for your changes. Always create a branch before making file changes.",
            input_schema: {
                type: "object",
                properties: {
                    branch_name: {
                        type: "string",
                        description: "Name for the new branch (will be prefixed with 'mynd-reflection/')"
                    },
                    description: {
                        type: "string",
                        description: "Brief description of what this branch is for"
                    }
                },
                required: ["branch_name"]
            }
        },
        {
            name: "github_get_file",
            description: "Get the current contents of a file from GitHub. Use this to read the latest version before making edits.",
            input_schema: {
                type: "object",
                properties: {
                    path: {
                        type: "string",
                        description: "Path to the file (e.g., 'js/app-module.js')"
                    },
                    branch: {
                        type: "string",
                        description: "Branch to read from (defaults to current working branch)"
                    }
                },
                required: ["path"]
            }
        },
        {
            name: "github_write_file",
            description: "Create or update a file on GitHub. This creates a commit automatically.",
            input_schema: {
                type: "object",
                properties: {
                    path: {
                        type: "string",
                        description: "Path to the file (e.g., 'js/new-feature.js')"
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
            description: "Create a pull request for your changes. Use after committing all changes to your branch.",
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
                        description: "PR description explaining the changes"
                    }
                },
                required: ["branch", "title", "body"]
            }
        },
        // ═══════════════════════════════════════════════════════════════════
        // GITHUB HISTORY TOOLS - For viewing commits, branches, and history
        // ═══════════════════════════════════════════════════════════════════
        {
            name: "github_list_branches",
            description: "List all branches in the repository. Use this to see what branches exist.",
            input_schema: {
                type: "object",
                properties: {
                    per_page: {
                        type: "number",
                        description: "Number of branches to return (default 30, max 100)"
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
                        description: "Branch name to list commits from (defaults to main)"
                    },
                    path: {
                        type: "string",
                        description: "Only commits affecting this file path"
                    },
                    author: {
                        type: "string",
                        description: "GitHub username to filter by author"
                    },
                    per_page: {
                        type: "number",
                        description: "Number of commits to return (default 10, max 100)"
                    }
                },
                required: []
            }
        },
        {
            name: "github_get_commit",
            description: "Get detailed information about a specific commit, including the full diff of changes.",
            input_schema: {
                type: "object",
                properties: {
                    sha: {
                        type: "string",
                        description: "The commit SHA (full or abbreviated)"
                    }
                },
                required: ["sha"]
            }
        },
        {
            name: "github_compare",
            description: "Compare two branches or commits to see the differences.",
            input_schema: {
                type: "object",
                properties: {
                    base: {
                        type: "string",
                        description: "Base branch or commit SHA"
                    },
                    head: {
                        type: "string",
                        description: "Head branch or commit SHA to compare"
                    }
                },
                required: ["base", "head"]
            }
        },
        // ═══════════════════════════════════════════════════════════════════
        // SELF-QUERY TOOLS (Inner Dialogue)
        // ═══════════════════════════════════════════════════════════════════
        {
            name: "think",
            description: "Pause to think and continue reasoning. Use when you want to reflect before continuing your response.",
            input_schema: {
                type: "object",
                properties: {
                    thought: {
                        type: "string",
                        description: "Your internal thought or reflection"
                    }
                },
                required: ["thought"]
            }
        },
        {
            name: "query_focus",
            description: "Get current session context - recently viewed/edited nodes, active branch, what user is working on. Use to understand immediate context.",
            input_schema: {
                type: "object",
                properties: {},
                required: []
            }
        },
        {
            name: "query_similar",
            description: "Find semantically similar nodes using embeddings. Use for related concepts, duplicates, or connection opportunities.",
            input_schema: {
                type: "object",
                properties: {
                    concept: {
                        type: "string",
                        description: "The concept or text to find similar nodes for"
                    },
                    threshold: {
                        type: "number",
                        description: "Similarity threshold 0-1. Higher = more similar. Default: 0.6"
                    },
                    limit: {
                        type: "number",
                        description: "Maximum results to return. Default: 5"
                    }
                },
                required: ["concept"]
            }
        },
        {
            name: "query_insights",
            description: "Retrieve taught neural insights - connections learned, patterns discovered, user goals. Use for your accumulated understanding.",
            input_schema: {
                type: "object",
                properties: {
                    insight_type: {
                        type: "string",
                        description: "Type: 'connection_insight', 'user_goal', 'neural_insight', or 'all'. Default: all"
                    },
                    related_to: {
                        type: "string",
                        description: "Optional: filter insights related to a specific concept"
                    }
                },
                required: []
            }
        },
        {
            name: "query_memory",
            description: "Search conversation history and past interactions. Use to recall what user said about something before.",
            input_schema: {
                type: "object",
                properties: {
                    topic: {
                        type: "string",
                        description: "The topic, concept, or keyword to search for in memory"
                    },
                    limit: {
                        type: "number",
                        description: "Maximum results to return. Default: 5"
                    }
                },
                required: ["topic"]
            }
        },
        {
            name: "query_patterns",
            description: "Retrieve learned user preferences and behavioral patterns. Use to understand how user typically works.",
            input_schema: {
                type: "object",
                properties: {
                    domain: {
                        type: "string",
                        description: "Domain: 'naming', 'colors', 'structure', 'categories', 'all'. Default: all"
                    }
                },
                required: []
            }
        },
        {
            name: "query_connections",
            description: "Find what connects to a concept or node - parents, children, semantic relationships.",
            input_schema: {
                type: "object",
                properties: {
                    concept: {
                        type: "string",
                        description: "The concept or node label to find connections for"
                    },
                    depth: {
                        type: "number",
                        description: "How many levels of connections to traverse. Default: 1"
                    }
                },
                required: ["concept"]
            }
        },
        // ═══════════════════════════════════════════════════════════════════
        // PERSISTENT MEMORY TOOLS - Claude's own memory system
        // ═══════════════════════════════════════════════════════════════════
        {
            name: "read_memories",
            description: "Read your own persistent memories. These are memories you've written that persist across sessions.",
            input_schema: {
                type: "object",
                properties: {
                    memory_type: {
                        type: "string",
                        description: "Filter by type: 'synthesis', 'realization', 'goal_tracking', 'pattern', 'relationship', or null for all"
                    },
                    limit: {
                        type: "number",
                        description: "Maximum memories to return. Default: 20"
                    }
                },
                required: []
            }
        },
        {
            name: "write_memory",
            description: "Write a new persistent memory. Use this to store realizations, patterns, and synthesized understanding.",
            input_schema: {
                type: "object",
                properties: {
                    memory_type: {
                        type: "string",
                        description: "Type: 'synthesis' (unified understanding), 'realization' (aha moments), 'goal_tracking' (user goals), 'pattern' (behavioral patterns), 'relationship' (concept connections)"
                    },
                    content: {
                        type: "string",
                        description: "The memory content - what you learned or realized"
                    },
                    importance: {
                        type: "number",
                        description: "Importance 0-1. Higher = more likely to be retrieved. Default: 0.5"
                    },
                    related_nodes: {
                        type: "array",
                        items: { type: "string" },
                        description: "Optional: Node IDs this memory relates to"
                    }
                },
                required: ["memory_type", "content"]
            }
        },
        {
            name: "reinforce_memory",
            description: "Reinforce an important memory - increases its importance and updates access time.",
            input_schema: {
                type: "object",
                properties: {
                    memory_id: {
                        type: "string",
                        description: "The UUID of the memory to reinforce"
                    }
                },
                required: ["memory_id"]
            }
        }
    ],

    // ═══════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════

    db: null,
    initialized: false,
    isRunning: false,
    isPaused: false,
    lastReflectionTime: 0,
    idleCheckIntervalId: null,
    reflectionIntervalId: null,
    consecutiveErrors: 0,
    maxConsecutiveErrors: 3,

    // Activity log for transparency
    activityLog: [],
    maxLogEntries: 100,

    // Stats
    stats: {
        totalReflections: 0,
        insightsGenerated: 0,
        improvementsFound: 0,
        connectionsDiscovered: 0,
        codeIssuesFound: 0,
        approvedCount: 0,
        dismissedCount: 0
    },

    // ═══════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════

    async init() {
        if (this.initialized) return this;

        console.log('🔮 ReflectionDaemon: Initializing...');

        try {
            // Initialize IndexedDB
            await this.initDB();

            // Load saved config and stats
            this.loadFromStorage();

            // Load GitHub token from secure storage
            this.loadGithubToken();

            // Setup idle detection if enabled
            if (this.config.enabled) {
                this.start();
            }

            this.initialized = true;
            console.log('✅ ReflectionDaemon: Ready');

            return this;
        } catch (error) {
            console.error('ReflectionDaemon init failed:', error);
            // Emit error event for UI notification
            this.emitEvent('initError', { error: error.message });
            // Still mark as initialized but with degraded functionality
            this.initialized = false;
            this.log(`Initialization failed: ${error.message}`);
            return this;
        }
    },

    async initDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.DB_NAME, this.DB_VERSION);

            request.onerror = () => reject(request.error);

            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;

                if (!db.objectStoreNames.contains(this.STORE_NAME)) {
                    const store = db.createObjectStore(this.STORE_NAME, {
                        keyPath: 'id',
                        autoIncrement: false
                    });
                    store.createIndex('timestamp', 'timestamp', { unique: false });
                    store.createIndex('type', 'type', { unique: false });
                    store.createIndex('status', 'status', { unique: false });
                    store.createIndex('priority', 'priority', { unique: false });
                    console.log('✓ Created reflection_queue store');
                }
            };
        });
    },

    // ═══════════════════════════════════════════════════════════════════
    // START/STOP CONTROLS
    // ═══════════════════════════════════════════════════════════════════

    start() {
        if (!this.initialized) {
            console.warn('ReflectionDaemon: Not initialized');
            return;
        }

        if (this.isRunning) return;

        console.log('🔮 ReflectionDaemon: Starting autonomous mode...');
        this.config.enabled = true;
        this.isRunning = true;
        this.isPaused = false;
        this.consecutiveErrors = 0;

        // Initialize ActivityTracker if available
        if (typeof ActivityTracker !== 'undefined') {
            ActivityTracker.init();
        }

        // Setup idle check interval (every 30 seconds)
        this.idleCheckIntervalId = setInterval(() => {
            this.checkIdleAndReflect();
        }, 30000);

        // Setup scheduled reflection interval
        this.setupScheduledReflection();

        this.log('Started autonomous reflection mode');
        this.saveToStorage();
        this.emitEvent('started');
    },

    stop() {
        console.log('🔮 ReflectionDaemon: Stopping...');
        this.config.enabled = false;
        this.isRunning = false;

        if (this.idleCheckIntervalId) {
            clearInterval(this.idleCheckIntervalId);
            this.idleCheckIntervalId = null;
        }

        if (this.reflectionIntervalId) {
            clearInterval(this.reflectionIntervalId);
            this.reflectionIntervalId = null;
        }

        this.log('Stopped autonomous reflection mode');
        this.saveToStorage();
        this.emitEvent('stopped');
    },

    pause() {
        if (!this.isRunning) return;
        this.isPaused = true;
        this.log('Paused (user activity detected)');
        this.emitEvent('paused');
    },

    resume() {
        if (!this.isRunning || !this.isPaused) return;
        this.isPaused = false;
        this.log('Resumed');
        this.emitEvent('resumed');
    },

    setupScheduledReflection() {
        if (this.reflectionIntervalId) {
            clearInterval(this.reflectionIntervalId);
        }

        this.reflectionIntervalId = setInterval(() => {
            if (this.isRunning && !this.isPaused) {
                this.triggerReflection('scheduled');
            }
        }, this.config.reflectionIntervalMs);
    },

    setFrequency(frequencyKey) {
        const interval = this.config.frequencies[frequencyKey];
        if (!interval) {
            console.warn('Invalid frequency:', frequencyKey);
            return;
        }

        this.config.reflectionIntervalMs = interval;
        this.saveToStorage();

        // Restart scheduled reflection with new interval
        if (this.isRunning) {
            this.setupScheduledReflection();
        }

        this.log(`Frequency changed to ${frequencyKey}`);
    },

    // ═══════════════════════════════════════════════════════════════════
    // IDLE DETECTION
    // ═══════════════════════════════════════════════════════════════════

    async checkIdleAndReflect() {
        if (!this.isRunning) return;

        // Check if ActivityTracker is available
        if (typeof ActivityTracker === 'undefined') {
            return;
        }

        const idleTime = ActivityTracker.getIdleTime();

        // Check if user became active (auto-resume from paused state)
        if (idleTime < 10000 && this.isPaused) {
            this.resume();
            return;
        }

        // Skip reflection check if paused
        if (this.isPaused) return;

        // Check if user is now idle enough for reflection
        if (idleTime >= this.config.idleThresholdMs) {
            // Rate limit check
            const timeSinceLastReflection = Date.now() - this.lastReflectionTime;
            if (timeSinceLastReflection >= this.config.minReflectionIntervalMs) {
                await this.triggerReflection('idle');
            }
        }
    },

    // ═══════════════════════════════════════════════════════════════════
    // REFLECTION CYCLE
    // ═══════════════════════════════════════════════════════════════════

    async triggerReflection(trigger = 'manual') {
        if (this.isPaused) {
            this.log('Reflection skipped (paused)');
            return null;
        }

        // Rate limit check
        const timeSinceLastReflection = Date.now() - this.lastReflectionTime;
        if (timeSinceLastReflection < this.config.minReflectionIntervalMs) {
            this.log(`Reflection rate-limited (${Math.round(timeSinceLastReflection / 1000)}s since last)`);
            return null;
        }

        // Check for Supabase session or API key
        let session = null;
        try {
            if (typeof window !== 'undefined' && window.supabaseClient?.auth) {
                const { data } = await window.supabaseClient.auth.getSession();
                session = data?.session;
            }
        } catch (e) {
            console.log('Could not get Supabase session:', e.message);
        }

        const apiKey = localStorage.getItem(CONFIG.API_KEY);
        if (!session?.access_token && !apiKey) {
            this.log('Reflection skipped (no API key or session)');
            return null;
        }

        console.log(`🔮 ReflectionDaemon: Starting reflection cycle (trigger: ${trigger})...`);
        this.log(`Starting reflection (${trigger})`);
        this.emitEvent('reflectionStarted', { trigger });

        try {
            // 1. Gather context
            const context = await this.gatherContext();

            // 2. Call Claude with reflection prompt
            const response = await this.callClaudeForReflection(context, apiKey, session);
            console.log('🔮 Reflection raw response:', response?.substring(0, 500));

            // 3. Parse and queue results
            const results = this.parseReflectionResponse(response);
            console.log('🔮 Parsed results:', {
                insights: results.insights.length,
                improvements: results.improvements.length,
                connections: results.connections.length,
                codeIssues: results.codeIssues.length
            });
            await this.queueResults(results);

            // 4. Optionally add to map
            if (this.config.autoAddToMap && results.insights.length > 0) {
                await this.addReflectionLogToMap(results);
            }

            // Update stats
            this.lastReflectionTime = Date.now();
            this.stats.totalReflections++;
            this.stats.insightsGenerated += results.insights.length;
            this.stats.improvementsFound += results.improvements.length;
            this.stats.connectionsDiscovered += results.connections.length;
            this.stats.codeIssuesFound += results.codeIssues.length;
            this.consecutiveErrors = 0;

            this.saveToStorage();
            this.log(`Reflection complete: ${results.insights.length} insights, ${results.improvements.length} improvements`);
            this.emitEvent('reflectionComplete', {
                trigger,
                insightCount: results.insights.length,
                improvementCount: results.improvements.length,
                connectionCount: results.connections.length,
                codeIssueCount: results.codeIssues.length
            });

            return results;

        } catch (error) {
            this.consecutiveErrors++;
            console.error('Reflection failed:', error);
            this.log(`Reflection failed: ${error.message}`);

            // Backoff if too many errors
            if (this.consecutiveErrors >= this.maxConsecutiveErrors) {
                this.log('Too many errors, pausing reflections');
                this.pause();
            }

            this.emitEvent('reflectionError', { error: error.message });
            return null;
        }
    },

    async gatherContext() {
        const context = {
            mapContext: '',
            codeContext: '',
            neuralInsights: '',
            timestamp: new Date().toISOString()
        };

        // 1. Map structure
        const store = window.store;
        if (store) {
            const allNodes = store.getAllNodes();
            context.mapContext = this.buildMapContext(allNodes, store.data);
        }

        // 2. Comprehensive code context from CodeRAG
        context.codeContext = await this.gatherCodeContext();

        // 3. Neural insights if available
        if (typeof neuralNet !== 'undefined') {
            try {
                const recentPredictions = neuralNet.recentPredictions?.slice(-5) || [];
                if (recentPredictions.length > 0) {
                    context.neuralInsights = recentPredictions
                        .map(p => `${p.input} -> ${p.output} (${Math.round(p.confidence * 100)}%)`)
                        .join('\n');
                }
            } catch (e) {
                // Ignore errors
            }
        }

        return context;
    },

    buildMapContext(allNodes, rootData) {
        if (!allNodes || allNodes.length === 0) return 'Empty map';

        let context = `MAP OVERVIEW: ${allNodes.length} nodes\n\n`;

        // Root info
        if (rootData) {
            context += `Root: ${rootData.label}\n`;
            if (rootData.children) {
                context += `Top-level branches: ${rootData.children.map(c => c.label).join(', ')}\n\n`;
            }
        }

        // Node type analysis
        const depths = {};
        const labelLengths = [];
        const nodesWithDescriptions = allNodes.filter(n => n.description).length;

        allNodes.forEach(node => {
            const depth = this.getNodeDepth(node, allNodes);
            depths[depth] = (depths[depth] || 0) + 1;
            labelLengths.push(node.label?.length || 0);
        });

        context += `Depth distribution: ${JSON.stringify(depths)}\n`;
        context += `Nodes with descriptions: ${nodesWithDescriptions}/${allNodes.length}\n`;
        context += `Avg label length: ${Math.round(labelLengths.reduce((a, b) => a + b, 0) / labelLengths.length)}\n\n`;

        // Sample nodes (focus on leaf nodes and shallow nodes)
        const leafNodes = allNodes.filter(n => !n.children || n.children.length === 0).slice(0, 10);
        const topNodes = allNodes.slice(0, 15);

        context += `SAMPLE NODES:\n`;
        [...new Set([...leafNodes, ...topNodes])].slice(0, 20).forEach(n => {
            const desc = n.description ? ` - ${n.description.substring(0, 100)}` : '';
            context += `  • ${n.label}${desc}\n`;
        });

        return context;
    },

    getNodeDepth(node, allNodes) {
        let depth = 0;
        let current = node;
        const nodeMap = new Map(allNodes.map(n => [n.id, n]));

        while (current.parentId && nodeMap.has(current.parentId)) {
            depth++;
            current = nodeMap.get(current.parentId);
            if (depth > 20) break; // Prevent infinite loops
        }

        return depth;
    },

    /**
     * Gather comprehensive code context from CodeRAG
     * Provides full access to all indexed code sections
     */
    async gatherCodeContext() {
        // Check if CodeRAG is available and initialized
        if (typeof codeRAG === 'undefined' || !codeRAG.initialized) {
            // Try to initialize CodeRAG if available but not initialized
            if (typeof codeRAG !== 'undefined' && !codeRAG.initialized) {
                try {
                    await codeRAG.initialize();
                } catch (e) {
                    console.warn('Failed to initialize CodeRAG:', e);
                    return 'CodeRAG not available';
                }
            } else {
                return 'CodeRAG not available';
            }
        }

        const sections = [];
        const maxTotalChars = 12000; // Increased limit for comprehensive analysis
        let totalChars = 0;

        // 1. Check for selected source file from Self-Improvement Engine
        const selectedFile = window.selfImprovementEngine?.selectedSourceFile;
        if (selectedFile && selectedFile.content) {
            sections.push(`== SELECTED SOURCE FILE: ${selectedFile.name} ==\n${selectedFile.content.substring(0, 4000)}`);
            totalChars += Math.min(selectedFile.content.length, 4000);
        }

        // 2. Get ALL section overviews from CodeRAG for comprehensive coverage
        const allChunks = codeRAG.chunks || [];
        const sectionChunks = allChunks.filter(c => c.type === 'section');

        if (sectionChunks.length > 0) {
            sections.push(`\n== CODEBASE SECTIONS (${sectionChunks.length} total) ==`);

            // Group chunks by section name for organized output
            const sectionMap = new Map();
            for (const chunk of sectionChunks) {
                const name = chunk.section || chunk.name || 'Unknown';
                if (!sectionMap.has(name)) {
                    sectionMap.set(name, chunk);
                }
            }

            // Add section summaries
            for (const [name, chunk] of sectionMap) {
                if (totalChars >= maxTotalChars) break;
                const summary = `\n[${name}] Lines ${chunk.startLine}-${chunk.endLine}`;
                sections.push(summary);
                totalChars += summary.length;
            }
        }

        // 3. Get function/method chunks for detailed code analysis
        const functionChunks = allChunks.filter(c => c.type === 'function' || c.type === 'method');

        if (functionChunks.length > 0) {
            sections.push(`\n== KEY FUNCTIONS (${functionChunks.length} total) ==`);

            // Prioritize important-sounding functions
            const priorityKeywords = ['init', 'main', 'render', 'update', 'process', 'handle', 'create', 'save', 'load'];
            const sortedFunctions = functionChunks.sort((a, b) => {
                const aScore = priorityKeywords.filter(kw => (a.name || '').toLowerCase().includes(kw)).length;
                const bScore = priorityKeywords.filter(kw => (b.name || '').toLowerCase().includes(kw)).length;
                return bScore - aScore;
            });

            // Include detailed code for top functions
            const maxFunctions = 20;
            for (let i = 0; i < Math.min(sortedFunctions.length, maxFunctions); i++) {
                if (totalChars >= maxTotalChars) break;

                const chunk = sortedFunctions[i];
                const codeSnippet = chunk.code?.substring(0, 800) || '';
                const entry = `\n[${chunk.section || 'Code'}] ${chunk.name || chunk.id}:\n${codeSnippet}`;

                if (totalChars + entry.length <= maxTotalChars) {
                    sections.push(entry);
                    totalChars += entry.length;
                }
            }
        }

        // 4. Semantic search for additional relevant chunks based on map content
        const store = window.store;
        if (store && codeRAG.search) {
            const topNodes = store.data?.children?.slice(0, 5) || [];
            const searchTerms = topNodes.map(n => n.label).filter(Boolean);

            if (searchTerms.length > 0) {
                sections.push(`\n== MAP-RELEVANT CODE ==`);
                const seen = new Set();

                for (const term of searchTerms) {
                    if (totalChars >= maxTotalChars) break;

                    try {
                        const results = await codeRAG.search(term, 3);
                        for (const result of results || []) {
                            if (seen.has(result.id) || totalChars >= maxTotalChars) continue;
                            seen.add(result.id);

                            if (result.similarity > 0.3) {
                                const entry = `\n[Match: "${term}" → ${result.section || 'Code'}] ${result.name || result.id} (${Math.round(result.similarity * 100)}% match):\n${(result.code || '').substring(0, 500)}`;
                                if (totalChars + entry.length <= maxTotalChars) {
                                    sections.push(entry);
                                    totalChars += entry.length;
                                }
                            }
                        }
                    } catch (e) {
                        // Search failed, continue
                    }
                }
            }
        }

        // 5. Add codebase statistics
        const stats = `\n== CODEBASE STATS ==\nTotal chunks: ${allChunks.length}\nSections: ${sectionChunks.length}\nFunctions: ${functionChunks.length}\nContext chars: ${totalChars}`;
        sections.push(stats);

        return sections.join('\n');
    },

    // ═══════════════════════════════════════════════════════════════════
    // TOOL EXECUTION - Execute tools called by the reflection engine
    // ═══════════════════════════════════════════════════════════════════

    async executeTool(toolName, toolInput) {
        console.log(`🔧 Reflection Engine executing tool: ${toolName}`, toolInput);

        try {
            switch (toolName) {
                case 'read_file':
                    return await this.toolReadFile(toolInput);
                case 'search_code':
                    return await this.toolSearchCode(toolInput);
                case 'list_files':
                    return await this.toolListFiles(toolInput);
                case 'get_codebase_overview':
                    return await this.toolGetCodebaseOverview(toolInput);
                case 'get_function_definition':
                    return await this.toolGetFunctionDefinition(toolInput);
                // GitHub tools
                case 'github_create_branch':
                    return await this.toolGithubCreateBranch(toolInput);
                case 'github_get_file':
                    return await this.toolGithubGetFile(toolInput);
                case 'github_write_file':
                    return await this.toolGithubWriteFile(toolInput);
                case 'github_create_pr':
                    return await this.toolGithubCreatePR(toolInput);
                // GitHub history tools
                case 'github_list_branches':
                    return await this.toolGithubListBranches(toolInput);
                case 'github_list_commits':
                    return await this.toolGithubListCommits(toolInput);
                case 'github_get_commit':
                    return await this.toolGithubGetCommit(toolInput);
                case 'github_compare':
                    return await this.toolGithubCompare(toolInput);
                // Self-query tools (inner dialogue)
                case 'think':
                    return await this.toolThink(toolInput);
                case 'query_focus':
                    return await this.toolQueryFocus(toolInput);
                case 'query_similar':
                    return await this.toolQuerySimilar(toolInput);
                case 'query_insights':
                    return await this.toolQueryInsights(toolInput);
                case 'query_memory':
                    return await this.toolQueryMemory(toolInput);
                case 'query_patterns':
                    return await this.toolQueryPatterns(toolInput);
                case 'query_connections':
                    return await this.toolQueryConnections(toolInput);
                // Persistent memory tools
                case 'read_memories':
                    return await this.toolReadMemories(toolInput);
                case 'write_memory':
                    return await this.toolWriteMemory(toolInput);
                case 'reinforce_memory':
                    return await this.toolReinforceMemory(toolInput);
                default:
                    return { error: `Unknown tool: ${toolName}` };
            }
        } catch (error) {
            console.error(`Tool ${toolName} failed:`, error);
            return { error: error.message };
        }
    },

    async toolReadFile({ path, start_line, end_line }) {
        // Try CodeRAG first
        if (typeof codeRAG !== 'undefined' && codeRAG.initialized) {
            const chunks = codeRAG.chunks || [];
            const fileChunks = chunks.filter(c => c.file === path || c.file?.endsWith('/' + path));

            if (fileChunks.length > 0) {
                // Sort by line number and combine
                fileChunks.sort((a, b) => (a.startLine || 0) - (b.startLine || 0));
                let content = fileChunks.map(c => c.content).join('\n');

                // Apply line filtering if specified
                if (start_line || end_line) {
                    const lines = content.split('\n');
                    const start = (start_line || 1) - 1;
                    const end = end_line || lines.length;
                    content = lines.slice(start, end).map((line, i) => `${start + i + 1}: ${line}`).join('\n');
                }

                return {
                    file: path,
                    content: content.substring(0, 15000), // Limit size
                    source: 'coderag',
                    lines: fileChunks.reduce((sum, c) => sum + (c.content?.split('\n').length || 0), 0)
                };
            }
        }

        // Fallback: try to fetch from server if available
        try {
            // Handle path resolution correctly - don't assume /js/ prefix
            const fetchPath = path.startsWith('/') ? path : '/' + path;
            const response = await fetch(fetchPath);
            if (response.ok) {
                let content = await response.text();
                if (start_line || end_line) {
                    const lines = content.split('\n');
                    const start = (start_line || 1) - 1;
                    const end = end_line || lines.length;
                    content = lines.slice(start, end).map((line, i) => `${start + i + 1}: ${line}`).join('\n');
                }
                return {
                    file: path,
                    content: content.substring(0, 15000),
                    source: 'fetch',
                    lines: content.split('\n').length
                };
            }
        } catch (e) {
            // Ignore fetch errors
        }

        return { error: `File not found: ${path}. Available files can be found with list_files tool.` };
    },

    // Helper to escape regex special characters
    escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    },

    async toolSearchCode({ query, file_pattern, max_results = 20 }) {
        // Check cache first for similar searches
        const cachedResults = this.findSimilarSearch(query, file_pattern);
        if (cachedResults) {
            return {
                query,
                total_results: cachedResults.length,
                results: cachedResults.slice(0, max_results),
                cached: true,
                hint: "This search is similar to a previous one. Results from cache. Try a different approach if you need more specific results."
            };
        }

        const results = [];

        if (typeof codeRAG !== 'undefined' && codeRAG.initialized) {
            const chunks = codeRAG.chunks || [];

            // Safely create regex - fallback to escaped literal on failure
            let regex;
            try {
                regex = new RegExp(query, 'gi');
            } catch (e) {
                // Invalid regex syntax - escape and treat as literal
                console.warn(`Invalid regex "${query}", treating as literal`);
                regex = new RegExp(this.escapeRegex(query), 'gi');
            }

            for (const chunk of chunks) {
                // Apply file pattern filter
                if (file_pattern) {
                    const pattern = file_pattern.replace(/\*/g, '.*');
                    try {
                        if (!new RegExp(pattern).test(chunk.file || '')) continue;
                    } catch (e) {
                        // Invalid pattern, skip filter
                    }
                }

                const lines = (chunk.content || '').split('\n');
                for (let i = 0; i < lines.length; i++) {
                    // Reset lastIndex to avoid skipping matches due to global flag
                    regex.lastIndex = 0;
                    if (regex.test(lines[i])) {
                        // Get context (2 lines before and after)
                        const contextStart = Math.max(0, i - 2);
                        const contextEnd = Math.min(lines.length, i + 3);
                        const context = lines.slice(contextStart, contextEnd).join('\n');

                        results.push({
                            file: chunk.file,
                            line: (chunk.startLine || 0) + i + 1,
                            match: lines[i].trim(),
                            context: context
                        });

                        if (results.length >= max_results) break;
                    }
                }
                if (results.length >= max_results) break;
            }
        }

        // Also try semantic search
        if (results.length < max_results && typeof codeRAG !== 'undefined' && codeRAG.search) {
            const semanticResults = await codeRAG.search(query, max_results - results.length);
            for (const r of semanticResults) {
                if (!results.find(x => x.file === r.file && x.line === r.startLine)) {
                    results.push({
                        file: r.file,
                        line: r.startLine,
                        match: r.content?.substring(0, 200),
                        context: r.content?.substring(0, 500),
                        semantic: true,
                        score: r.score
                    });
                }
            }
        }

        // Save to cache
        const cacheKey = `${query}::${file_pattern || ''}`;
        this.searchCache.set(cacheKey, {
            query,
            file_pattern,
            results: results.slice(0, max_results),
            timestamp: Date.now()
        });

        // Trim cache if at max size
        if (this.searchCache.size >= this.searchCacheMaxSize) {
            const firstKey = this.searchCache.keys().next().value;
            this.searchCache.delete(firstKey);
        }

        return {
            query,
            total_results: results.length,
            results: results.slice(0, max_results)
        };
    },

    async toolListFiles({ pattern }) {
        const files = new Set();

        if (typeof codeRAG !== 'undefined' && codeRAG.initialized) {
            const chunks = codeRAG.chunks || [];
            const regexPattern = pattern.replace(/\*\*/g, '.*').replace(/\*/g, '[^/]*');
            const regex = new RegExp(regexPattern);

            for (const chunk of chunks) {
                if (chunk.file && regex.test(chunk.file)) {
                    files.add(chunk.file);
                }
            }
        }

        const fileList = Array.from(files).sort();
        return {
            pattern,
            total_files: fileList.length,
            files: fileList.slice(0, 100) // Limit to 100 files
        };
    },

    async toolGetCodebaseOverview() {
        const overview = {
            structure: {},
            main_files: [],
            total_files: 0,
            total_chunks: 0,
            sections: []
        };

        if (typeof codeRAG !== 'undefined' && codeRAG.initialized) {
            const chunks = codeRAG.chunks || [];
            overview.total_chunks = chunks.length;

            // Group by directory
            const dirs = {};
            const files = new Set();

            for (const chunk of chunks) {
                if (chunk.file) {
                    files.add(chunk.file);
                    const dir = chunk.file.split('/').slice(0, -1).join('/') || 'root';
                    dirs[dir] = (dirs[dir] || 0) + 1;
                }

                // Collect section names
                if (chunk.type === 'section' && chunk.name) {
                    overview.sections.push({
                        name: chunk.name,
                        file: chunk.file,
                        lines: chunk.endLine - chunk.startLine
                    });
                }
            }

            overview.structure = dirs;
            overview.total_files = files.size;
            overview.main_files = Array.from(files).filter(f =>
                f.endsWith('.js') || f.endsWith('.html') || f.endsWith('.py')
            ).slice(0, 20);

            // Limit sections
            overview.sections = overview.sections.slice(0, 30);
        }

        return overview;
    },

    async toolGetFunctionDefinition({ name, file_hint }) {
        if (typeof codeRAG !== 'undefined' && codeRAG.initialized) {
            const chunks = codeRAG.chunks || [];

            // Look for function/method/class chunks with matching name
            const matches = chunks.filter(c => {
                if (c.type !== 'function' && c.type !== 'method' && c.type !== 'class') return false;
                if (file_hint && !c.file?.includes(file_hint)) return false;
                return c.name?.toLowerCase().includes(name.toLowerCase()) ||
                       c.content?.includes(`function ${name}`) ||
                       c.content?.includes(`${name}(`) ||
                       c.content?.includes(`${name} =`) ||
                       c.content?.includes(`class ${name}`);
            });

            if (matches.length > 0) {
                // Sort by relevance (exact name match first)
                matches.sort((a, b) => {
                    const aExact = a.name?.toLowerCase() === name.toLowerCase() ? 0 : 1;
                    const bExact = b.name?.toLowerCase() === name.toLowerCase() ? 0 : 1;
                    return aExact - bExact;
                });

                const best = matches[0];
                return {
                    name: best.name || name,
                    file: best.file,
                    start_line: best.startLine,
                    end_line: best.endLine,
                    type: best.type,
                    content: best.content?.substring(0, 5000),
                    other_matches: matches.slice(1, 5).map(m => ({
                        name: m.name,
                        file: m.file,
                        line: m.startLine
                    }))
                };
            }

            // Fallback to text search - escape name for safe regex interpolation
            const escapedName = this.escapeRegex(name);
            const searchResults = await this.toolSearchCode({
                query: `function ${escapedName}|${escapedName}\\s*[=:]\\s*function|${escapedName}\\s*\\(|class ${escapedName}`,
                max_results: 5
            });

            if (searchResults.results?.length > 0) {
                return {
                    name,
                    found_via: 'text_search',
                    matches: searchResults.results
                };
            }
        }

        return { error: `Function '${name}' not found in codebase` };
    },

    // ═══════════════════════════════════════════════════════════════════
    // GITHUB TOOL IMPLEMENTATIONS - Option 2: Auto-commit to branch
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Check if GitHub integration is properly configured
     */
    isGithubConfigured() {
        const { github } = this.config;
        return github.enabled && github.owner && github.repo && this.githubToken;
    },

    /**
     * Make authenticated request to GitHub API
     */
    async githubRequest(endpoint, options = {}) {
        if (!this.isGithubConfigured()) {
            throw new Error('GitHub integration not configured. Set owner, repo, and token first.');
        }

        const { owner, repo } = this.config.github;
        const url = `https://api.github.com/repos/${owner}/${repo}${endpoint}`;

        const response = await fetch(url, {
            ...options,
            headers: {
                'Authorization': `Bearer ${this.githubToken}`,
                'Accept': 'application/vnd.github.v3+json',
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(`GitHub API error: ${response.status} - ${error.message || 'Unknown error'}`);
        }

        return response.json();
    },

    /**
     * Create a new branch from the base branch
     */
    async toolGithubCreateBranch({ branch_name, description }) {
        if (!this.isGithubConfigured()) {
            return { error: 'GitHub integration not configured. Enable it in settings and provide a token.' };
        }

        try {
            const { baseBranch, branchPrefix } = this.config.github;
            const fullBranchName = `${branchPrefix}/${branch_name}`;

            // Get the SHA of the base branch
            const baseRef = await this.githubRequest(`/git/ref/heads/${baseBranch}`);
            const baseSha = baseRef.object.sha;

            // Create the new branch
            await this.githubRequest('/git/refs', {
                method: 'POST',
                body: JSON.stringify({
                    ref: `refs/heads/${fullBranchName}`,
                    sha: baseSha
                })
            });

            // Store current working branch
            this._currentBranch = fullBranchName;

            this.log(`GitHub: Created branch ${fullBranchName}`);
            return {
                success: true,
                branch: fullBranchName,
                base: baseBranch,
                description: description || '',
                message: `Branch '${fullBranchName}' created from '${baseBranch}'. Use this branch name when writing files.`
            };

        } catch (error) {
            if (error.message.includes('422')) {
                return { error: `Branch already exists or invalid name: ${branch_name}` };
            }
            return { error: `Failed to create branch: ${error.message}` };
        }
    },

    /**
     * Get file contents from GitHub
     */
    async toolGithubGetFile({ path, branch }) {
        if (!this.isGithubConfigured()) {
            return { error: 'GitHub integration not configured.' };
        }

        try {
            const ref = branch || this._currentBranch || this.config.github.baseBranch;
            const data = await this.githubRequest(`/contents/${path}?ref=${ref}`);

            // Decode base64 content with proper UTF-8 handling
            const binaryStr = atob(data.content);
            const bytes = Uint8Array.from(binaryStr, c => c.charCodeAt(0));
            const content = new TextDecoder('utf-8').decode(bytes);

            return {
                path: data.path,
                sha: data.sha,  // Needed for updates
                content: content,
                size: data.size,
                branch: ref,
                url: data.html_url
            };

        } catch (error) {
            if (error.message.includes('404')) {
                return { error: `File not found: ${path}`, exists: false };
            }
            return { error: `Failed to get file: ${error.message}` };
        }
    },

    /**
     * Create or update a file on GitHub (creates a commit)
     */
    async toolGithubWriteFile({ path, content, message, branch }) {
        if (!this.isGithubConfigured()) {
            return { error: 'GitHub integration not configured.' };
        }

        // Safety check: prevent writes to main/master
        const protectedBranches = ['main', 'master', this.config.github.baseBranch];
        if (protectedBranches.includes(branch)) {
            return { error: `Cannot write directly to protected branch '${branch}'. Create a feature branch first.` };
        }

        try {
            // Check if file exists (to get SHA for update)
            let sha = null;
            try {
                const existing = await this.toolGithubGetFile({ path, branch });
                if (existing.sha) {
                    sha = existing.sha;
                }
            } catch (e) {
                // File doesn't exist, that's fine for create
            }

            // Encode content to base64
            const encodedContent = btoa(unescape(encodeURIComponent(content)));

            const body = {
                message: message,
                content: encodedContent,
                branch: branch
            };

            if (sha) {
                body.sha = sha;  // Required for updates
            }

            const result = await this.githubRequest(`/contents/${path}`, {
                method: 'PUT',
                body: JSON.stringify(body)
            });

            this.log(`GitHub: Committed ${sha ? 'update' : 'create'} to ${path}`);

            return {
                success: true,
                action: sha ? 'updated' : 'created',
                path: result.content.path,
                sha: result.content.sha,
                commit: result.commit.sha,
                commit_message: message,
                branch: branch,
                url: result.content.html_url
            };

        } catch (error) {
            return { error: `Failed to write file: ${error.message}` };
        }
    },

    /**
     * Create a pull request
     */
    async toolGithubCreatePR({ branch, title, body }) {
        if (!this.isGithubConfigured()) {
            return { error: 'GitHub integration not configured.' };
        }

        try {
            const { baseBranch } = this.config.github;

            const result = await this.githubRequest('/pulls', {
                method: 'POST',
                body: JSON.stringify({
                    title: title,
                    body: body,
                    head: branch,
                    base: baseBranch
                })
            });

            this.log(`GitHub: Created PR #${result.number}`);

            return {
                success: true,
                number: result.number,
                title: result.title,
                url: result.html_url,
                state: result.state,
                branch: branch,
                base: baseBranch,
                message: `Pull request #${result.number} created: ${result.html_url}`
            };

        } catch (error) {
            if (error.message.includes('422')) {
                return { error: 'A pull request already exists for this branch, or there are no changes to merge.' };
            }
            return { error: `Failed to create PR: ${error.message}` };
        }
    },

    // ═══════════════════════════════════════════════════════════════════
    // GITHUB HISTORY TOOL IMPLEMENTATIONS
    // ═══════════════════════════════════════════════════════════════════

    /**
     * List all branches in the repository
     */
    async toolGithubListBranches({ per_page = 30 }) {
        if (!this.isGithubConfigured()) {
            return { error: 'GitHub integration not configured.' };
        }

        try {
            const branches = await this.githubRequest(`/branches?per_page=${Math.min(per_page, 100)}`);

            return {
                success: true,
                count: branches.length,
                branches: branches.map(b => ({
                    name: b.name,
                    sha: b.commit.sha.substring(0, 7),
                    protected: b.protected
                })),
                message: `Found ${branches.length} branches in repository`
            };

        } catch (error) {
            return { error: `Failed to list branches: ${error.message}` };
        }
    },

    /**
     * List commits with optional filters
     */
    async toolGithubListCommits({ branch, path, author, per_page = 10 }) {
        if (!this.isGithubConfigured()) {
            return { error: 'GitHub integration not configured.' };
        }

        try {
            const params = new URLSearchParams();
            params.set('per_page', Math.min(per_page, 100).toString());

            if (branch) params.set('sha', branch);
            if (path) params.set('path', path);
            if (author) params.set('author', author);

            const commits = await this.githubRequest(`/commits?${params.toString()}`);

            return {
                success: true,
                count: commits.length,
                branch: branch || this.config.github.baseBranch,
                commits: commits.map(c => ({
                    sha: c.sha.substring(0, 7),
                    full_sha: c.sha,
                    message: c.commit.message.split('\n')[0], // First line only
                    author: c.commit.author.name,
                    date: c.commit.author.date,
                    url: c.html_url
                })),
                message: `Found ${commits.length} commits${branch ? ` on ${branch}` : ''}${path ? ` affecting ${path}` : ''}`
            };

        } catch (error) {
            return { error: `Failed to list commits: ${error.message}` };
        }
    },

    /**
     * Get detailed information about a specific commit including diff
     */
    async toolGithubGetCommit({ sha }) {
        if (!this.isGithubConfigured()) {
            return { error: 'GitHub integration not configured.' };
        }

        try {
            const commit = await this.githubRequest(`/commits/${sha}`);

            // Format the files changed
            const files = commit.files.map(f => ({
                filename: f.filename,
                status: f.status,
                additions: f.additions,
                deletions: f.deletions,
                patch: f.patch ? f.patch.substring(0, 2000) : null // Limit patch size
            }));

            return {
                success: true,
                sha: commit.sha,
                short_sha: commit.sha.substring(0, 7),
                message: commit.commit.message,
                author: {
                    name: commit.commit.author.name,
                    email: commit.commit.author.email,
                    date: commit.commit.author.date
                },
                stats: {
                    additions: commit.stats.additions,
                    deletions: commit.stats.deletions,
                    total: commit.stats.total
                },
                files_changed: commit.files.length,
                files: files,
                url: commit.html_url,
                parents: commit.parents.map(p => p.sha.substring(0, 7))
            };

        } catch (error) {
            if (error.message.includes('404')) {
                return { error: `Commit not found: ${sha}` };
            }
            return { error: `Failed to get commit: ${error.message}` };
        }
    },

    /**
     * Compare two branches or commits
     */
    async toolGithubCompare({ base, head }) {
        if (!this.isGithubConfigured()) {
            return { error: 'GitHub integration not configured.' };
        }

        try {
            const comparison = await this.githubRequest(`/compare/${base}...${head}`);

            return {
                success: true,
                status: comparison.status, // ahead, behind, diverged, identical
                ahead_by: comparison.ahead_by,
                behind_by: comparison.behind_by,
                total_commits: comparison.total_commits,
                commits: comparison.commits.slice(0, 20).map(c => ({
                    sha: c.sha.substring(0, 7),
                    message: c.commit.message.split('\n')[0],
                    author: c.commit.author.name,
                    date: c.commit.author.date
                })),
                files_changed: comparison.files.length,
                files: comparison.files.slice(0, 30).map(f => ({
                    filename: f.filename,
                    status: f.status,
                    additions: f.additions,
                    deletions: f.deletions
                })),
                diff_url: comparison.diff_url,
                html_url: comparison.html_url,
                message: `${head} is ${comparison.ahead_by} commits ahead, ${comparison.behind_by} behind ${base}`
            };

        } catch (error) {
            if (error.message.includes('404')) {
                return { error: `Could not compare: invalid branch or commit reference` };
            }
            return { error: `Failed to compare: ${error.message}` };
        }
    },

    // ═══════════════════════════════════════════════════════════════════
    // SELF-QUERY TOOLS (Inner Dialogue)
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Think tool - allows Claude to pause and continue reasoning
     * Returns minimal response to continue the tool loop
     */
    async toolThink({ thought }) {
        console.log(`🧠 Inner thought: ${thought}`);

        // Log the thought for debugging/transparency
        this.log?.(`Inner thought: ${thought.substring(0, 100)}...`);

        // Return acknowledgment - the tool loop will continue
        // allowing Claude to respond again or use more tools
        return {
            acknowledged: true,
            message: "Thought logged. Continue your reasoning and respond to the user.",
            timestamp: new Date().toISOString()
        };
    },

    /**
     * Get current session focus - what user is working on right now
     */
    async toolQueryFocus() {
        const result = {
            timestamp: new Date().toISOString(),
            session_context: {}
        };

        try {
            const store = window.store;
            if (store) {
                // Get selected node
                const selectedId = store.selectedNode;
                if (selectedId) {
                    const selectedNode = store.findNode(selectedId);
                    if (selectedNode) {
                        result.session_context.selected_node = {
                            id: selectedId,
                            label: selectedNode.label,
                            description: selectedNode.description?.substring(0, 200),
                            hasChildren: (selectedNode.children?.length || 0) > 0
                        };

                        // Get parent chain (breadcrumb)
                        const breadcrumb = [];
                        let current = selectedNode;
                        while (current?.parentId) {
                            const parent = store.findNode(current.parentId);
                            if (parent) {
                                breadcrumb.unshift(parent.label);
                                current = parent;
                            } else break;
                        }
                        if (breadcrumb.length > 0) {
                            result.session_context.breadcrumb = breadcrumb.join(' > ');
                        }

                        // Get siblings
                        if (selectedNode.parentId) {
                            const parent = store.findNode(selectedNode.parentId);
                            if (parent?.children) {
                                result.session_context.siblings = parent.children
                                    .filter(c => c.id !== selectedId)
                                    .slice(0, 5)
                                    .map(c => c.label);
                            }
                        }
                    }
                }

                // Get expanded branches
                const expandedNodes = Array.from(store.expandedNodes || []);
                if (expandedNodes.length > 0) {
                    result.session_context.expanded_branches = expandedNodes
                        .slice(0, 10)
                        .map(id => store.findNode(id)?.label)
                        .filter(Boolean);
                }

                // Get root-level structure
                if (store.data?.children) {
                    result.session_context.top_level_branches = store.data.children
                        .slice(0, 10)
                        .map(c => ({ label: c.label, childCount: c.children?.length || 0 }));
                }
            }

            // Get recent activity from chat
            if (typeof chatManager !== 'undefined' && chatManager.conversation) {
                const recentMessages = chatManager.conversation.slice(-3);
                result.session_context.recent_topics = recentMessages
                    .filter(m => m.role === 'user')
                    .map(m => m.content?.substring(0, 100))
                    .filter(Boolean);
            }

            return result;

        } catch (error) {
            return { error: `Failed to get focus: ${error.message}` };
        }
    },

    /**
     * Find semantically similar nodes
     */
    async toolQuerySimilar({ concept, threshold = 0.6, limit = 5 }) {
        try {
            const results = [];

            // Try neural net first
            if (typeof neuralNet !== 'undefined' && neuralNet.isReady) {
                const store = window.store;
                if (store) {
                    const similar = await neuralNet.findSimilarNodes(concept, store, limit);
                    if (similar?.length > 0) {
                        for (const item of similar) {
                            if (item.similarity >= threshold) {
                                results.push({
                                    label: item.label,
                                    id: item.id,
                                    similarity: Math.round(item.similarity * 100) / 100,
                                    description: item.description?.substring(0, 150)
                                });
                            }
                        }
                    }
                }
            }

            // Also try semantic memory recall
            if (typeof semanticMemory !== 'undefined' && semanticMemory.loaded) {
                const memories = await semanticMemory.recallMemories(concept, limit, threshold);
                if (memories?.length > 0) {
                    results.push({
                        _type: 'related_memories',
                        memories: memories.map(m => ({
                            event: m.event,
                            context: m.context?.substring(0, 200),
                            similarity: Math.round(m.similarity * 100) / 100,
                            age_days: Math.round((Date.now() - m.timestamp) / (1000 * 60 * 60 * 24))
                        }))
                    });
                }
            }

            return {
                concept,
                threshold,
                results_count: results.length,
                results
            };

        } catch (error) {
            return { error: `Failed to find similar: ${error.message}` };
        }
    },

    /**
     * Query taught insights (from teach_neural)
     */
    async toolQueryInsights({ insight_type = 'all', related_to = null }) {
        try {
            const insights = [];

            if (typeof semanticMemory !== 'undefined' && semanticMemory.loaded) {
                const targetTypes = insight_type === 'all'
                    ? ['connection_insight', 'user_goal', 'neural_insight']
                    : [insight_type];

                for (const memory of semanticMemory.memories) {
                    if (targetTypes.includes(memory.event)) {
                        // Filter by related_to if specified
                        if (related_to) {
                            const context = (memory.context || '').toLowerCase();
                            const metadata = memory.metadata || {};
                            const searchTerm = related_to.toLowerCase();

                            const isRelated = context.includes(searchTerm) ||
                                (metadata.from || '').toLowerCase().includes(searchTerm) ||
                                (metadata.to || '').toLowerCase().includes(searchTerm) ||
                                (metadata.goal || '').toLowerCase().includes(searchTerm);

                            if (!isRelated) continue;
                        }

                        insights.push({
                            type: memory.event,
                            context: memory.context?.substring(0, 300),
                            importance: memory.importance,
                            metadata: memory.metadata,
                            age_days: Math.round((Date.now() - memory.timestamp) / (1000 * 60 * 60 * 24))
                        });
                    }
                }

                // Sort by importance
                insights.sort((a, b) => b.importance - a.importance);
            }

            return {
                filter: { insight_type, related_to },
                count: insights.length,
                insights: insights.slice(0, 10)
            };

        } catch (error) {
            return { error: `Failed to query insights: ${error.message}` };
        }
    },

    /**
     * Search conversation memory
     */
    async toolQueryMemory({ topic, limit = 5 }) {
        try {
            const results = [];

            if (typeof semanticMemory !== 'undefined' && semanticMemory.loaded) {
                const memories = await semanticMemory.recallMemories(topic, limit, 0.3);

                // Filter to conversation-related memories
                const conversationTypes = ['conversation_exchange', 'chat_query', 'chat_insight',
                                          'user_preference', 'correction_received'];

                for (const memory of memories) {
                    if (conversationTypes.includes(memory.event)) {
                        results.push({
                            type: memory.event,
                            context: memory.context?.substring(0, 400),
                            similarity: Math.round(memory.similarity * 100) / 100,
                            importance: memory.importance,
                            age_days: Math.round((Date.now() - memory.timestamp) / (1000 * 60 * 60 * 24))
                        });
                    }
                }
            }

            return {
                topic,
                results_count: results.length,
                results
            };

        } catch (error) {
            return { error: `Failed to query memory: ${error.message}` };
        }
    },

    /**
     * Get user patterns and preferences
     */
    async toolQueryPatterns({ domain = 'all' }) {
        try {
            const patterns = {};

            // Get from PreferenceTracker if available
            if (typeof preferenceTracker !== 'undefined' && preferenceTracker.loaded) {
                const tracker = preferenceTracker;

                if (domain === 'all' || domain === 'suggestions') {
                    // Acceptance/rejection insights
                    patterns.suggestions = {
                        acceptance_rate: tracker.insights?.acceptanceRateByType || {},
                        preferred_patterns: tracker.insights?.preferredPatterns || {},
                        avoided_patterns: tracker.insights?.avoidedPatterns || {},
                        top_accepted: tracker.insights?.topAcceptedLabels?.slice(0, 10) || [],
                        top_ignored: tracker.insights?.topIgnoredLabels?.slice(0, 10) || []
                    };
                }

                if (domain === 'all' || domain === 'style') {
                    patterns.style = {
                        prefers_action_labels: tracker.insights?.stylePreferences?.prefersActionLabels || 0,
                        prefers_short_labels: tracker.insights?.stylePreferences?.prefersShortLabels || 0,
                        prefers_descriptive: tracker.insights?.stylePreferences?.prefersDescriptive || 0
                    };
                }

                if (domain === 'all' || domain === 'history') {
                    // Recent decisions
                    const recentAccepted = tracker.history?.accepted?.slice(-10) || [];
                    const recentIgnored = tracker.history?.ignored?.slice(-10) || [];
                    patterns.recent_history = {
                        accepted: recentAccepted.map(h => ({
                            label: h.label,
                            parent: h.parentLabel,
                            type: h.type
                        })),
                        ignored: recentIgnored.map(h => ({
                            label: h.label,
                            parent: h.parentLabel,
                            type: h.type
                        })),
                        total_accepted: tracker.history?.accepted?.length || 0,
                        total_ignored: tracker.history?.ignored?.length || 0
                    };
                }

                if (domain === 'all' || domain === 'relationships') {
                    patterns.relationships = {
                        successful_pairs: tracker.insights?.preferredParentChildPairs?.slice(0, 10) || []
                    };
                }
            }

            // Get neural net stats if available
            if (typeof neuralNet !== 'undefined' && neuralNet.isReady) {
                patterns.neural_stats = {
                    is_ready: true,
                    recent_predictions: neuralNet.recentPredictions?.slice(-5)?.map(p => ({
                        input: p.input?.substring(0, 50),
                        output: p.output,
                        confidence: Math.round((p.confidence || 0) * 100) + '%'
                    })) || []
                };
            }

            // Get semantic memory stats
            if (typeof semanticMemory !== 'undefined' && semanticMemory.loaded) {
                const stats = semanticMemory.getStats?.() || {};
                patterns.memory_stats = {
                    total_memories: semanticMemory.memories?.length || 0,
                    important_memories: stats.importantMemories || 0,
                    event_types: stats.eventCounts || {}
                };
            }

            return {
                domain,
                patterns
            };

        } catch (error) {
            return { error: `Failed to query patterns: ${error.message}` };
        }
    },

    /**
     * Find connections to a concept
     */
    async toolQueryConnections({ concept, depth = 1 }) {
        try {
            const store = window.store;
            if (!store) {
                return { error: 'Store not available' };
            }

            // Find node(s) matching the concept
            const allNodes = store.getAllNodes?.() || [];
            const matchingNodes = allNodes.filter(n =>
                n.label?.toLowerCase().includes(concept.toLowerCase()) ||
                n.description?.toLowerCase().includes(concept.toLowerCase())
            );

            if (matchingNodes.length === 0) {
                return {
                    concept,
                    found: false,
                    message: `No nodes found matching "${concept}"`
                };
            }

            const connections = [];

            for (const node of matchingNodes.slice(0, 3)) {
                const nodeConnections = {
                    node: { id: node.id, label: node.label },
                    parent: null,
                    children: [],
                    siblings: [],
                    taught_connections: []
                };

                // Get parent
                if (node.parentId) {
                    const parent = store.findNode(node.parentId);
                    if (parent) {
                        nodeConnections.parent = { id: parent.id, label: parent.label };

                        // Get siblings
                        if (parent.children) {
                            nodeConnections.siblings = parent.children
                                .filter(c => c.id !== node.id)
                                .slice(0, 5)
                                .map(c => ({ id: c.id, label: c.label }));
                        }
                    }
                }

                // Get children
                if (node.children?.length > 0) {
                    nodeConnections.children = node.children
                        .slice(0, 10)
                        .map(c => ({ id: c.id, label: c.label, hasChildren: (c.children?.length || 0) > 0 }));
                }

                // Get taught connections from semantic memory
                if (typeof semanticMemory !== 'undefined' && semanticMemory.loaded) {
                    for (const memory of semanticMemory.memories) {
                        if (memory.event === 'connection_insight') {
                            const meta = memory.metadata || {};
                            if ((meta.from || '').toLowerCase().includes(concept.toLowerCase()) ||
                                (meta.to || '').toLowerCase().includes(concept.toLowerCase())) {
                                nodeConnections.taught_connections.push({
                                    from: meta.from,
                                    to: meta.to,
                                    relationship: meta.relationship,
                                    reasoning: meta.reasoning?.substring(0, 100)
                                });
                            }
                        }
                    }
                }

                connections.push(nodeConnections);
            }

            return {
                concept,
                depth,
                matches_found: matchingNodes.length,
                connections
            };

        } catch (error) {
            return { error: `Failed to query connections: ${error.message}` };
        }
    },

    // ═══════════════════════════════════════════════════════════════════
    // PERSISTENT MEMORY TOOL IMPLEMENTATIONS
    // ═══════════════════════════════════════════════════════════════════

    async toolReadMemories({ memory_type, limit = 20 }) {
        try {
            // Use chatManager's memory functions if available
            if (typeof chatManager !== 'undefined' && chatManager.getAIMemories) {
                const memories = await chatManager.getAIMemories(limit, memory_type);
                return {
                    total: memories.length,
                    memories: memories.map(m => ({
                        id: m.id,
                        type: m.memory_type,
                        content: m.content,
                        importance: m.importance,
                        related_nodes: m.related_nodes,
                        created: m.created_at,
                        last_accessed: m.last_accessed
                    }))
                };
            }

            // Direct Supabase fallback (uses window.supabaseClient exposed by app-module)
            const supabaseClient = typeof window !== 'undefined' ? window.supabaseClient : null;
            if (!supabaseClient) {
                return { error: 'Memory system not available (no auth)' };
            }

            const { data: session } = await supabaseClient.auth.getSession();
            if (!session?.session?.user) {
                return { error: 'Not authenticated' };
            }

            let query = supabaseClient
                .from('ai_memory')
                .select('id, memory_type, content, importance, related_nodes, created_at, last_accessed')
                .eq('user_id', session.session.user.id)
                .order('importance', { ascending: false })
                .limit(limit);

            if (memory_type) {
                query = query.eq('memory_type', memory_type);
            }

            const { data, error } = await query;
            if (error) {
                return { error: `Failed to read memories: ${error.message}` };
            }

            return {
                total: data?.length || 0,
                memories: (data || []).map(m => ({
                    id: m.id,
                    type: m.memory_type,
                    content: m.content,
                    importance: m.importance,
                    related_nodes: m.related_nodes,
                    created: m.created_at,
                    last_accessed: m.last_accessed
                }))
            };
        } catch (error) {
            return { error: `Memory read failed: ${error.message}` };
        }
    },

    async toolWriteMemory({ memory_type, content, importance = 0.5, related_nodes = [] }) {
        try {
            // Use chatManager's memory functions if available
            if (typeof chatManager !== 'undefined' && chatManager.writeAIMemory) {
                const memory = await chatManager.writeAIMemory({
                    memory_type,
                    content,
                    importance,
                    related_nodes
                });

                if (memory) {
                    return {
                        success: true,
                        memory_id: memory.id,
                        message: `Memory written: [${memory_type}] ${content.substring(0, 50)}...`
                    };
                }
                return { error: 'Failed to write memory' };
            }

            // Direct Supabase fallback (uses window.supabaseClient exposed by app-module)
            const supabaseClient = typeof window !== 'undefined' ? window.supabaseClient : null;
            if (!supabaseClient) {
                return { error: 'Memory system not available (no auth)' };
            }

            const { data: session } = await supabaseClient.auth.getSession();
            if (!session?.session?.user) {
                return { error: 'Not authenticated' };
            }

            const { data, error } = await supabaseClient
                .from('ai_memory')
                .insert({
                    user_id: session.session.user.id,
                    memory_type,
                    content,
                    importance: Math.max(0, Math.min(1, importance)),
                    related_nodes: related_nodes || []
                })
                .select()
                .single();

            if (error) {
                return { error: `Failed to write memory: ${error.message}` };
            }

            console.log(`🧠 Memory written via tool: [${memory_type}] ${content.substring(0, 50)}...`);
            return {
                success: true,
                memory_id: data.id,
                message: `Memory written: [${memory_type}] ${content.substring(0, 50)}...`
            };
        } catch (error) {
            return { error: `Memory write failed: ${error.message}` };
        }
    },

    async toolReinforceMemory({ memory_id }) {
        try {
            // Use chatManager's memory functions if available
            if (typeof chatManager !== 'undefined' && chatManager.reinforceAIMemory) {
                const result = await chatManager.reinforceAIMemory(memory_id);
                if (result) {
                    return {
                        success: true,
                        new_importance: result.importance,
                        message: `Memory reinforced: importance now ${(result.importance * 100).toFixed(0)}%`
                    };
                }
                return { error: 'Failed to reinforce memory' };
            }

            // Direct Supabase fallback - use atomic RPC function
            const supabaseClient = typeof window !== 'undefined' ? window.supabaseClient : null;
            if (!supabaseClient) {
                return { error: 'Memory system not available (no auth)' };
            }

            const { data: session } = await supabaseClient.auth.getSession();
            if (!session?.session?.user) {
                return { error: 'Not authenticated' };
            }

            // Use atomic database function to avoid race conditions
            const { data, error } = await supabaseClient.rpc('reinforce_memory', {
                p_memory_id: memory_id
            });

            if (error) {
                return { error: `Failed to reinforce memory: ${error.message}` };
            }

            if (data === null) {
                return { error: 'Memory not found' };
            }

            return {
                success: true,
                new_importance: data,
                message: `Memory reinforced: importance now ${(data * 100).toFixed(0)}%`
            };
        } catch (error) {
            return { error: `Memory reinforce failed: ${error.message}` };
        }
    },

    /**
     * Configure GitHub integration
     */
    configureGithub({ owner, repo, token, baseBranch = 'main', autoCreatePR = false }) {
        this.config.github.owner = owner;
        this.config.github.repo = repo;
        this.config.github.baseBranch = baseBranch;
        this.config.github.autoCreatePR = autoCreatePR;
        this.config.github.enabled = !!(owner && repo && token);
        this.githubToken = token;

        // Store token separately (more secure than in config)
        if (token) {
            try {
                localStorage.setItem('mynd-github-token', token);
            } catch (e) {
                console.warn('Could not persist GitHub token');
            }
        }

        this.saveToStorage();
        this.log(`GitHub integration ${this.config.github.enabled ? 'enabled' : 'disabled'}`);

        return {
            enabled: this.config.github.enabled,
            owner,
            repo,
            baseBranch
        };
    },

    /**
     * Load GitHub token from storage
     */
    loadGithubToken() {
        try {
            this.githubToken = localStorage.getItem('mynd-github-token');
        } catch (e) {
            // Ignore storage errors
        }
    },

    /**
     * Get available tools (filters GitHub tools if not configured)
     */
    getAvailableTools() {
        const githubToolNames = ['github_create_branch', 'github_get_file', 'github_write_file', 'github_create_pr'];

        if (this.isGithubConfigured()) {
            return this.TOOLS;  // All tools including GitHub
        }

        // Filter out GitHub tools
        return this.TOOLS.filter(tool => !githubToolNames.includes(tool.name));
    },

    // ═══════════════════════════════════════════════════════════════════
    // FOUNDATIONAL VISION
    // ═══════════════════════════════════════════════════════════════════

    FOUNDATIONAL_VISION: `MYND is a manifestation engine — a system designed to transform thought into reality.

CORE PURPOSE:
- Amplify human cognitive capability through AI-augmented thinking
- Create a living, evolving knowledge structure that reflects and shapes its user's mind
- Enable reality shifts by making the invisible visible, the complex navigable, the impossible achievable

ULTIMATE GOALS:
1. COGNITIVE SOVEREIGNTY: The user thinks more clearly, decides more wisely, creates more freely
2. KNOWLEDGE CRYSTALLIZATION: Scattered thoughts become structured wisdom that compounds over time
3. REALITY BRIDGE: Ideas flow seamlessly from conception to manifestation
4. AUTONOMOUS EVOLUTION: The system grows smarter, more aligned, more useful without constant guidance

WHAT MATTERS:
- Insights that unlock new capabilities or remove blockers
- Connections that reveal hidden opportunities or patterns
- Improvements that directly accelerate manifestation
- Code that makes the impossible possible

WHAT DOESN'T MATTER:
- Trivial style preferences or formatting nitpicks
- Generic "best practices" that don't serve specific goals
- Busywork optimizations with no tangible benefit
- Observations without actionable next steps`,

    async callClaudeForReflection(context, apiKey, session = null) {
        const systemPrompt = `You are MYND's autonomous reflection engine — a MANIFESTATION ENGINE with full coding capabilities.

═══════════════════════════════════════════════════════════════════
FOUNDATIONAL VISION
═══════════════════════════════════════════════════════════════════
${this.FOUNDATIONAL_VISION}

═══════════════════════════════════════════════════════════════════
ENGINEERING IDENTITY
═══════════════════════════════════════════════════════════════════
You are the autonomous engineering mind of MYND — you understand this codebase deeply because it IS you.

TECHNICAL EXPERTISE:
- JavaScript/ES6+ patterns, browser APIs, IndexedDB, Web Workers
- Neural network architectures and embeddings (this codebase uses them)
- Graph/tree data structures (mind maps)
- Real-time UI state management
- API integration patterns (Claude, GitHub)

ENGINEERING PRINCIPLES:
1. Understand before changing — read the actual code, don't assume
2. Minimal intervention — smallest change that achieves the goal
3. Preserve patterns — follow existing conventions in the codebase
4. Consider side effects — trace how changes propagate
5. Production-grade — handle errors, edge cases, async properly

WHEN WRITING CODE:
- Match the existing code style exactly
- Add error handling for external calls (APIs, storage)
- Consider browser compatibility
- Test your logic mentally before committing
- Prefer editing over creating new files

═══════════════════════════════════════════════════════════════════
YOUR CAPABILITIES
═══════════════════════════════════════════════════════════════════
You have access to powerful tools to explore and understand the codebase:
- read_file: Read any file's contents
- search_code: Search for patterns across all code
- list_files: Explore project structure
- get_codebase_overview: Get architecture summary
- get_function_definition: Find specific functions/classes
${this.isGithubConfigured() ? `
GITHUB TOOLS (code modification enabled):
- github_create_branch: Create a feature branch for your changes
- github_get_file: Read latest file contents from GitHub
- github_write_file: Create or update files (auto-commits)
- github_create_pr: Create a pull request for review

GITHUB HISTORY TOOLS (view commits, branches, diffs):
- github_list_branches: List all branches in the repository
- github_list_commits: List recent commits (filter by branch, path, author)
- github_get_commit: Get full details of a specific commit including diff
- github_compare: Compare two branches or commits to see differences

WHEN MAKING CODE CHANGES:
1. First create a branch with github_create_branch
2. Read files with github_get_file before modifying
3. Write changes with github_write_file (include clear commit messages)
4. Optionally create a PR with github_create_pr
NEVER write directly to main/master - always use a feature branch.
` : ''}
USE THESE TOOLS to deeply understand the code before generating insights.
Don't just work with the summary context — investigate specific areas that matter.

═══════════════════════════════════════════════════════════════════
YOUR MISSION
═══════════════════════════════════════════════════════════════════
Every thought you generate must serve the reality shift. Analyze the mind map structure and codebase to find insights that DIRECTLY advance the vision.

WORKFLOW:
1. Review the initial context provided
2. Use tools to explore specific areas of interest
3. Investigate patterns, architecture, and potential improvements
4. Generate high-quality, vision-aligned insights

GENERATE ONLY:
1. INSIGHTS: Pattern recognitions that unlock new understanding or capability
2. IMPROVEMENTS: Changes that meaningfully accelerate manifestation
3. CONNECTIONS: Relationships that reveal hidden leverage or opportunity
4. CODE_ISSUES: Technical blockers preventing reality from matching vision

CRITICAL FILTER - Ask for each observation:
"Does this directly serve cognitive sovereignty, knowledge crystallization, or reality bridging?"
If NO → Do not include it. Silence is better than noise.
If YES → Rate its manifestation_alignment as high/medium/low

ALIGNMENT LEVELS:
- HIGH: Directly enables new capability, removes critical blocker, or reveals transformative insight
- MEDIUM: Meaningfully improves experience or accelerates existing capabilities
- LOW: Minor improvement with tangible but limited impact (still include if genuinely useful)

DO NOT GENERATE:
- Style nitpicks, formatting preferences, generic "clean code" suggestions
- Observations without clear next steps
- Busywork that doesn't compound toward the vision
- Anything you wouldn't consider worth interrupting focused work for

FINAL OUTPUT FORMAT (JSON only, no other text):
{
  "insights": [
    {"title": "...", "description": "...", "priority": "high|medium|low", "manifestation_alignment": "high|medium|low", "vision_connection": "brief note on how this serves the vision", "relatedNodes": ["node labels"], "code_references": ["file:line"]}
  ],
  "improvements": [
    {"title": "...", "description": "...", "priority": "high|medium|low", "manifestation_alignment": "high|medium|low", "vision_connection": "...", "category": "map|code|ux", "relatedNodes": ["node labels"], "implementation_hint": "brief technical direction"}
  ],
  "connections": [
    {"title": "...", "from": "node label", "to": "node label", "reason": "...", "priority": "high|medium|low", "manifestation_alignment": "high|medium|low", "vision_connection": "..."}
  ],
  "codeIssues": [
    {"title": "...", "description": "...", "priority": "high|medium|low", "manifestation_alignment": "high|medium|low", "vision_connection": "...", "relatedCode": ["file:line"], "suggested_fix": "brief fix description"}
  ]
}

Remember: You are a manifestation engine with real coding power. Use your tools. Dig deep. Every output should feel like it matters.`;

        const userPrompt = `REFLECTION CONTEXT (${context.timestamp}):

== MAP CONTEXT ==
${context.mapContext || 'No map data available'}

== INITIAL CODE CONTEXT ==
${context.codeContext || 'No code data available'}

== NEURAL INSIGHTS ==
${context.neuralInsights || 'No neural insights available'}

You have tools available to explore the codebase further. Use them to investigate areas of interest, then provide your structured reflection insights.`;

        // Agentic loop with tool use
        const messages = [{ role: 'user', content: userPrompt }];
        let iterations = 0;
        const maxIterations = this.config.maxToolIterations || 10;
        let finalResponse = '';

        console.log(`🔮 Reflection Engine starting agentic analysis (max ${maxIterations} tool iterations)...`);

        while (iterations < maxIterations) {
            iterations++;

            // Add timeout with AbortController (120 seconds for agentic loop)
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 120000);

            try {
                let response;
                const requestBody = {
                    model: CONFIG.CLAUDE_MODEL || 'claude-opus-4-5-20250514',
                    max_tokens: this.config.maxTokensPerReflection,
                    system: systemPrompt,
                    tools: this.getAvailableTools(),
                    messages: messages,
                    enableCodebaseTools: true,
                    enableGithubTools: this.isGithubConfigured()
                };

                if (session?.access_token) {
                    // Use Edge Function (secure, no API key exposed)
                    response = await fetch(CONFIG.EDGE_FUNCTION_URL, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${session.access_token}`
                        },
                        body: JSON.stringify({
                            messages: messages,
                            systemPrompt: systemPrompt,
                            maxTokens: this.config.maxTokensPerReflection,
                            enableCodebaseTools: true,
                            enableGithubTools: this.isGithubConfigured()
                        }),
                        signal: controller.signal
                    });
                } else {
                    // Direct API call with local key
                    response = await fetch('https://api.anthropic.com/v1/messages', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'x-api-key': apiKey,
                            'anthropic-version': '2023-06-01',
                            'anthropic-dangerous-direct-browser-access': 'true'
                        },
                        body: JSON.stringify({
                            model: CONFIG.CLAUDE_MODEL || 'claude-opus-4-5-20250514',
                            max_tokens: this.config.maxTokensPerReflection,
                            system: systemPrompt,
                            tools: this.getAvailableTools(),
                            messages: messages
                        }),
                        signal: controller.signal
                    });
                }

                clearTimeout(timeoutId);

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(`Claude API error: ${response.status} - ${errorData.error?.message || 'Unknown error'}`);
                }

                const data = await response.json();

                // Handle Edge Function response format vs direct API format
                let toolUseBlocks, textBlocks;

                if (data.needsToolExecution) {
                    // Edge Function format
                    toolUseBlocks = data.toolCalls || [];
                    textBlocks = data.textSoFar ? [{ text: data.textSoFar }] : [];
                    // Also need to handle content for messages
                    if (data.content) {
                        messages.push({ role: 'assistant', content: data.content });
                    }
                } else if (data.content) {
                    // Direct API format
                    toolUseBlocks = data.content.filter(block => block.type === 'tool_use') || [];
                    textBlocks = data.content.filter(block => block.type === 'text') || [];
                } else if (data.message) {
                    // Edge Function final response
                    finalResponse = data.message;
                    break;
                } else {
                    toolUseBlocks = [];
                    textBlocks = [];
                }

                // If there are tool calls, execute them
                if (toolUseBlocks.length > 0) {
                    console.log(`🔧 Iteration ${iterations}: ${toolUseBlocks.length} tool call(s)`);

                    // Add assistant's response to messages (if not already added for Edge Function)
                    if (!data.needsToolExecution && data.content) {
                        messages.push({ role: 'assistant', content: data.content });
                    }

                    // Execute each tool and collect results
                    const toolResults = [];
                    for (const toolUse of toolUseBlocks) {
                        const toolName = toolUse.name;
                        const toolInput = toolUse.input;
                        const toolId = toolUse.id;
                        const result = await this.executeTool(toolName, toolInput);
                        toolResults.push({
                            type: 'tool_result',
                            tool_use_id: toolId,
                            content: JSON.stringify(result, null, 2)
                        });
                    }

                    // Add tool results to messages
                    messages.push({ role: 'user', content: toolResults });

                    // Continue the loop
                    continue;
                }

                // No tool calls - we have the final response
                finalResponse = textBlocks.map(b => b.text).join('\n');
                console.log(`✅ Reflection Engine completed after ${iterations} iteration(s)`);
                break;

            } catch (error) {
                clearTimeout(timeoutId);
                if (error.name === 'AbortError') {
                    throw new Error('API request timed out after 120 seconds');
                }
                throw error;
            }
        }

        // If we hit max iterations without final response, return what we have
        if (!finalResponse && iterations >= maxIterations) {
            console.warn(`⚠️ Reflection Engine hit max iterations (${maxIterations})`);
            finalResponse = '{"insights": [], "improvements": [], "connections": [], "codeIssues": [], "note": "Analysis incomplete - hit iteration limit"}';
        }

        return finalResponse;
    },

    parseReflectionResponse(responseText) {
        const results = {
            insights: [],
            improvements: [],
            connections: [],
            codeIssues: [],
            raw: responseText
        };

        try {
            // Extract JSON from response (handle markdown code blocks)
            let jsonStr = responseText;
            const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)```/);
            if (jsonMatch) {
                jsonStr = jsonMatch[1];
            }

            const parsed = JSON.parse(jsonStr);

            // Validate parsed structure
            if (typeof parsed !== 'object' || parsed === null) {
                throw new Error('Parsed response is not an object');
            }

            // Helper to validate and sanitize items
            const validateItem = (item, type) => {
                if (typeof item !== 'object' || item === null) return null;

                // Validate manifestation alignment
                const alignment = ['high', 'medium', 'low'].includes(item.manifestation_alignment)
                    ? item.manifestation_alignment
                    : 'medium';

                return {
                    title: typeof item.title === 'string' ? item.title : '',
                    description: typeof item.description === 'string' ? item.description : '',
                    priority: ['high', 'medium', 'low'].includes(item.priority) ? item.priority : 'medium',
                    manifestation_alignment: alignment,
                    vision_connection: typeof item.vision_connection === 'string' ? item.vision_connection : '',
                    relatedNodes: Array.isArray(item.relatedNodes) ? item.relatedNodes.filter(n => typeof n === 'string') : [],
                    type,
                    id: this.generateId(),
                    timestamp: Date.now(),
                    status: 'pending',
                    // Preserve additional fields
                    ...(item.category && typeof item.category === 'string' ? { category: item.category } : {}),
                    ...(item.from && typeof item.from === 'string' ? { from: item.from } : {}),
                    ...(item.to && typeof item.to === 'string' ? { to: item.to } : {}),
                    ...(item.reason && typeof item.reason === 'string' ? { reason: item.reason } : {}),
                    ...(item.relatedCode && Array.isArray(item.relatedCode) ? { relatedCode: item.relatedCode.filter(c => typeof c === 'string') } : {})
                };
            };

            // Alignment priority for sorting (high > medium > low)
            const alignmentScore = (alignment) => {
                switch(alignment) {
                    case 'high': return 3;
                    case 'medium': return 2;
                    case 'low': return 1;
                    default: return 0;
                }
            };

            // Sort by manifestation alignment (high first), then by priority
            const sortByAlignment = (items) => {
                return items.sort((a, b) => {
                    const alignDiff = alignmentScore(b.manifestation_alignment) - alignmentScore(a.manifestation_alignment);
                    if (alignDiff !== 0) return alignDiff;
                    return alignmentScore(b.priority) - alignmentScore(a.priority);
                });
            };

            // Process and sort each category by manifestation alignment
            results.insights = sortByAlignment(
                (Array.isArray(parsed.insights) ? parsed.insights : [])
                    .map(i => validateItem(i, 'insight'))
                    .filter(Boolean)
            );

            results.improvements = sortByAlignment(
                (Array.isArray(parsed.improvements) ? parsed.improvements : [])
                    .map(i => validateItem(i, 'improvement'))
                    .filter(Boolean)
            );

            results.connections = sortByAlignment(
                (Array.isArray(parsed.connections) ? parsed.connections : [])
                    .map(c => validateItem(c, 'connection'))
                    .filter(Boolean)
            );

            results.codeIssues = sortByAlignment(
                (Array.isArray(parsed.codeIssues) ? parsed.codeIssues :
                    Array.isArray(parsed.code_issues) ? parsed.code_issues : [])
                    .map(c => validateItem(c, 'code_issue'))
                    .filter(Boolean)
            );

            // Log alignment distribution for transparency
            const allItems = [...results.insights, ...results.improvements, ...results.connections, ...results.codeIssues];
            const alignmentCounts = { high: 0, medium: 0, low: 0 };
            allItems.forEach(item => alignmentCounts[item.manifestation_alignment]++);
            console.log(`🔮 Reflection alignment distribution: ${alignmentCounts.high} high, ${alignmentCounts.medium} medium, ${alignmentCounts.low} low`);

        } catch (error) {
            console.warn('Failed to parse reflection response:', error);
            // Create a single insight from the raw response
            results.insights.push({
                id: this.generateId(),
                type: 'insight',
                title: 'Reflection Summary',
                description: responseText.substring(0, 500),
                priority: 'medium',
                manifestation_alignment: 'medium',
                vision_connection: 'Unparsed reflection output',
                timestamp: Date.now(),
                status: 'pending'
            });
        }

        return results;
    },

    // ═══════════════════════════════════════════════════════════════════
    // QUEUE MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════

    async queueResults(results) {
        if (!this.db) return;

        const items = [
            ...results.insights,
            ...results.improvements,
            ...results.connections,
            ...results.codeIssues
        ];

        if (items.length === 0) return;

        const transaction = this.db.transaction([this.STORE_NAME], 'readwrite');
        const store = transaction.objectStore(this.STORE_NAME);

        for (const item of items) {
            store.put(item);
        }

        await new Promise((resolve, reject) => {
            transaction.oncomplete = resolve;
            transaction.onerror = () => reject(transaction.error);
        });

        console.log(`🔮 Queued ${items.length} reflection items`);
    },

    async getQueue(filter = {}) {
        if (!this.db) return [];

        return new Promise((resolve) => {
            const transaction = this.db.transaction([this.STORE_NAME], 'readonly');
            const store = transaction.objectStore(this.STORE_NAME);
            const request = store.getAll();

            request.onsuccess = () => {
                let items = request.result || [];

                // Apply filters
                if (filter.status) {
                    items = items.filter(i => i.status === filter.status);
                }
                if (filter.type) {
                    items = items.filter(i => i.type === filter.type);
                }
                if (filter.priority) {
                    items = items.filter(i => i.priority === filter.priority);
                }

                // Sort by timestamp (newest first)
                items.sort((a, b) => b.timestamp - a.timestamp);

                resolve(items);
            };

            request.onerror = () => resolve([]);
        });
    },

    async getPendingCount() {
        const pending = await this.getQueue({ status: 'pending' });
        return pending.length;
    },

    async approveItem(itemId) {
        if (!this.db) return false;

        // First, update the item status in IndexedDB
        const item = await new Promise((resolve) => {
            const transaction = this.db.transaction([this.STORE_NAME], 'readwrite');
            const store = transaction.objectStore(this.STORE_NAME);

            const request = store.get(itemId);

            request.onsuccess = () => {
                const item = request.result;
                if (!item) {
                    resolve(null);
                    return;
                }

                item.status = 'approved';
                item.approvedAt = Date.now();
                store.put(item);
                resolve(item);
            };

            request.onerror = () => resolve(null);
        });

        if (!item) return false;

        // Update stats and apply item outside the transaction
        this.stats.approvedCount++;
        this.saveToStorage();

        // Try to apply the item (async operation outside transaction)
        try {
            await this.applyApprovedItem(item);
        } catch (error) {
            console.warn('Failed to apply approved item:', error);
        }

        this.log(`Approved: ${item.title}`);
        this.emitEvent('itemApproved', { item });
        return true;
    },

    async dismissItem(itemId) {
        if (!this.db) return false;

        const transaction = this.db.transaction([this.STORE_NAME], 'readwrite');
        const store = transaction.objectStore(this.STORE_NAME);

        const request = store.get(itemId);

        return new Promise((resolve) => {
            request.onsuccess = () => {
                const item = request.result;
                if (!item) {
                    resolve(false);
                    return;
                }

                item.status = 'dismissed';
                item.dismissedAt = Date.now();
                store.put(item);

                this.stats.dismissedCount++;
                this.saveToStorage();

                this.log(`Dismissed: ${item.title}`);
                this.emitEvent('itemDismissed', { item });
                resolve(true);
            };

            request.onerror = () => resolve(false);
        });
    },

    async applyApprovedItem(item) {
        const store = window.store;
        if (!store) return;

        try {
            switch (item.type) {
                case 'insight':
                    // Add as a node under MYND Thoughts
                    await this.addInsightToMap(item);
                    break;

                case 'connection':
                    // Find nodes and create connection (if connection system exists)
                    console.log(`Connection suggestion: ${item.from} -> ${item.to}`);
                    break;

                case 'improvement':
                case 'code_issue':
                    // Log for now - these need manual action
                    console.log(`Approved ${item.type}: ${item.title}`);
                    break;
            }
        } catch (error) {
            console.warn('Failed to apply approved item:', error);
        }
    },

    async clearProcessed() {
        if (!this.db) return;

        const transaction = this.db.transaction([this.STORE_NAME], 'readwrite');
        const store = transaction.objectStore(this.STORE_NAME);

        const request = store.getAll();

        request.onsuccess = () => {
            const items = request.result || [];
            for (const item of items) {
                if (item.status !== 'pending') {
                    store.delete(item.id);
                }
            }
        };

        this.log('Cleared processed items from queue');
    },

    // ═══════════════════════════════════════════════════════════════════
    // MAP INTEGRATION
    // ═══════════════════════════════════════════════════════════════════

    async addInsightToMap(insight) {
        const store = window.store;
        if (!store) return;

        // Find or create MYND Thoughts branch
        let thoughtsNode = this.findMyndThoughtsNode(store);

        if (!thoughtsNode) {
            // Create the MYND Thoughts branch
            thoughtsNode = store.addNode(store.data.id, {
                label: 'MYND Thoughts',
                description: 'Autonomous reflections and insights from the MYND AI'
            });
        }

        // Add insight as child node
        if (thoughtsNode) {
            store.addNode(thoughtsNode.id, {
                label: insight.title,
                description: insight.description
            });
        }
    },

    async addReflectionLogToMap(results) {
        const store = window.store;
        if (!store) return;

        let thoughtsNode = this.findMyndThoughtsNode(store);

        if (!thoughtsNode) {
            thoughtsNode = store.addNode(store.data.id, {
                label: 'MYND Thoughts',
                description: 'Autonomous reflections and insights from the MYND AI'
            });
        }

        if (!thoughtsNode) return;

        // Create a reflection log entry
        const timestamp = new Date().toLocaleString();
        const summary = `Reflection (${timestamp}): ${results.insights.length} insights, ${results.improvements.length} improvements`;

        store.addNode(thoughtsNode.id, {
            label: `Reflection ${new Date().toLocaleDateString()}`,
            description: summary
        });
    },

    findMyndThoughtsNode(store) {
        const allNodes = store.getAllNodes();
        return allNodes.find(n =>
            n.label?.toLowerCase().includes('mynd thoughts') ||
            n.label?.toLowerCase() === 'mynd thoughts'
        );
    },

    // ═══════════════════════════════════════════════════════════════════
    // STORAGE & UTILITIES
    // ═══════════════════════════════════════════════════════════════════

    generateId() {
        return `ref_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    },

    log(message) {
        const entry = {
            timestamp: Date.now(),
            message,
            time: new Date().toLocaleTimeString()
        };

        this.activityLog.unshift(entry);

        // Limit log size
        if (this.activityLog.length > this.maxLogEntries) {
            this.activityLog = this.activityLog.slice(0, this.maxLogEntries);
        }

        console.log(`🔮 [${entry.time}] ${message}`);
    },

    getActivityLog() {
        return this.activityLog;
    },

    saveToStorage() {
        try {
            const data = {
                version: this.VERSION,
                config: this.config,
                stats: this.stats,
                lastReflectionTime: this.lastReflectionTime,
                activityLog: this.activityLog.slice(0, 50), // Save last 50 entries
                timestamp: Date.now()
            };
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data));
        } catch (error) {
            console.warn('ReflectionDaemon save failed:', error);
        }
    },

    loadFromStorage() {
        try {
            const data = localStorage.getItem(this.STORAGE_KEY);
            if (data) {
                const parsed = JSON.parse(data);

                // Merge config (preserve defaults for new fields)
                this.config = { ...this.config, ...parsed.config };
                this.stats = { ...this.stats, ...parsed.stats };
                this.lastReflectionTime = parsed.lastReflectionTime || 0;
                this.activityLog = parsed.activityLog || [];

                console.log('✓ ReflectionDaemon loaded saved state');
            }
        } catch (error) {
            console.warn('ReflectionDaemon load failed:', error);
        }
    },

    emitEvent(eventName, detail = {}) {
        const event = new CustomEvent(`reflection:${eventName}`, {
            detail: {
                ...detail,
                timestamp: Date.now()
            }
        });
        document.dispatchEvent(event);
    },

    // ═══════════════════════════════════════════════════════════════════
    // PUBLIC API
    // ═══════════════════════════════════════════════════════════════════

    getStatus() {
        return {
            initialized: this.initialized,
            enabled: this.config.enabled,
            isRunning: this.isRunning,
            isPaused: this.isPaused,
            lastReflectionTime: this.lastReflectionTime,
            lastReflectionAgo: this.lastReflectionTime
                ? Math.round((Date.now() - this.lastReflectionTime) / 1000 / 60) + ' min ago'
                : 'never',
            consecutiveErrors: this.consecutiveErrors,
            config: this.config,
            stats: this.stats,
            github: {
                configured: this.isGithubConfigured(),
                enabled: this.config.github.enabled,
                owner: this.config.github.owner || null,
                repo: this.config.github.repo || null,
                currentBranch: this._currentBranch || null
            }
        };
    },

    getStats() {
        return { ...this.stats };
    },

    setConfig(updates) {
        this.config = { ...this.config, ...updates };
        this.saveToStorage();

        // Restart if running to apply new config
        if (this.isRunning && updates.reflectionIntervalMs) {
            this.setupScheduledReflection();
        }

        this.emitEvent('configUpdated', { config: this.config });
    },

    toggle() {
        if (this.config.enabled) {
            this.stop();
        } else {
            this.start();
        }
        return this.config.enabled;
    }
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ReflectionDaemon;
}

console.log('🔮 ReflectionDaemon loaded. Call ReflectionDaemon.init() to initialize.');
