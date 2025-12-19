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

    VERSION: '1.0',
    STORAGE_KEY: 'mynd-reflection-daemon',
    DB_NAME: 'mynd-reflection-db',
    DB_VERSION: 1,
    STORE_NAME: 'reflection_queue',

    config: {
        enabled: false,                    // Master toggle for autonomous mode
        idleThresholdMs: 5 * 60 * 1000,    // 5 minutes default
        reflectionIntervalMs: 30 * 60 * 1000, // 30 minutes default
        minReflectionIntervalMs: 15 * 60 * 1000, // 15 min minimum (rate limit)
        maxTokensPerReflection: 4000,
        autoAddToMap: false,               // Auto-add reflection log nodes
        frequencies: {
            '15min': 15 * 60 * 1000,
            '30min': 30 * 60 * 1000,
            '1hr': 60 * 60 * 1000,
            '2hr': 2 * 60 * 60 * 1000
        }
    },

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

        // Check for API key
        const apiKey = localStorage.getItem(CONFIG.API_KEY);
        if (!apiKey) {
            this.log('Reflection skipped (no API key)');
            return null;
        }

        console.log(`🔮 ReflectionDaemon: Starting reflection cycle (trigger: ${trigger})...`);
        this.log(`Starting reflection (${trigger})`);
        this.emitEvent('reflectionStarted', { trigger });

        try {
            // 1. Gather context
            const context = await this.gatherContext();

            // 2. Call Claude with reflection prompt
            const response = await this.callClaudeForReflection(context, apiKey);

            // 3. Parse and queue results
            const results = this.parseReflectionResponse(response);
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
        const store = window.app?.store;
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
        const store = window.app?.store;
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

    async callClaudeForReflection(context, apiKey) {
        const systemPrompt = `You are MYND's autonomous reflection engine. You analyze the mind map structure and codebase to find insights and improvements.

Your task is to review the provided context and generate:
1. INSIGHTS: Observations about the map's structure, gaps, or patterns
2. IMPROVEMENTS: Specific, actionable suggestions for the map or system
3. CONNECTIONS: Missing relationships between concepts that should be linked
4. CODE_ISSUES: Bugs, improvements, or patterns found in the code

IMPORTANT RULES:
- Be specific and actionable, not vague
- Focus on high-value observations
- Prioritize by importance (high/medium/low)
- Keep suggestions concise

OUTPUT FORMAT (JSON only, no other text):
{
  "insights": [
    {"title": "...", "description": "...", "priority": "high|medium|low", "relatedNodes": ["node labels"]}
  ],
  "improvements": [
    {"title": "...", "description": "...", "priority": "high|medium|low", "category": "map|code|ux", "relatedNodes": ["node labels"]}
  ],
  "connections": [
    {"title": "...", "from": "node label", "to": "node label", "reason": "...", "priority": "high|medium|low"}
  ],
  "codeIssues": [
    {"title": "...", "description": "...", "priority": "high|medium|low", "relatedCode": ["section names"]}
  ]
}`;

        const userPrompt = `REFLECTION CONTEXT (${context.timestamp}):

== MAP CONTEXT ==
${context.mapContext || 'No map data available'}

== CODE CONTEXT ==
${context.codeContext || 'No code data available'}

== NEURAL INSIGHTS ==
${context.neuralInsights || 'No neural insights available'}

Analyze this context and provide structured reflection insights.`;

        // Add timeout with AbortController (60 seconds)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);

        try {
            const response = await fetch('https://api.anthropic.com/v1/messages', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': apiKey,
                    'anthropic-version': '2023-06-01',
                    'anthropic-dangerous-direct-browser-access': 'true'
                },
                body: JSON.stringify({
                    model: CONFIG.CLAUDE_MODEL || 'claude-sonnet-4-20250514',
                    max_tokens: this.config.maxTokensPerReflection,
                    system: systemPrompt,
                    messages: [{ role: 'user', content: userPrompt }]
                }),
                signal: controller.signal
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`Claude API error: ${response.status} - ${errorData.error?.message || 'Unknown error'}`);
            }

            const data = await response.json();
            return data.content?.[0]?.text || '';
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error('API request timed out after 60 seconds');
            }
            throw error;
        } finally {
            clearTimeout(timeoutId);
        }
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
                return {
                    title: typeof item.title === 'string' ? item.title : '',
                    description: typeof item.description === 'string' ? item.description : '',
                    priority: ['high', 'medium', 'low'].includes(item.priority) ? item.priority : 'medium',
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

            results.insights = (Array.isArray(parsed.insights) ? parsed.insights : [])
                .map(i => validateItem(i, 'insight'))
                .filter(Boolean);

            results.improvements = (Array.isArray(parsed.improvements) ? parsed.improvements : [])
                .map(i => validateItem(i, 'improvement'))
                .filter(Boolean);

            results.connections = (Array.isArray(parsed.connections) ? parsed.connections : [])
                .map(c => validateItem(c, 'connection'))
                .filter(Boolean);

            results.codeIssues = (Array.isArray(parsed.codeIssues) ? parsed.codeIssues :
                                  Array.isArray(parsed.code_issues) ? parsed.code_issues : [])
                .map(c => validateItem(c, 'code_issue'))
                .filter(Boolean);

        } catch (error) {
            console.warn('Failed to parse reflection response:', error);
            // Create a single insight from the raw response
            results.insights.push({
                id: this.generateId(),
                type: 'insight',
                title: 'Reflection Summary',
                description: responseText.substring(0, 500),
                priority: 'medium',
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
        const store = window.app?.store;
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
        const store = window.app?.store;
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
        const store = window.app?.store;
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
            stats: this.stats
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
