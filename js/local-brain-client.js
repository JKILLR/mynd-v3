/**
 * MYND Local Brain Client
 * =======================
 * Connects the browser app to the local Python ML server.
 * Falls back to browser TensorFlow.js if server unavailable.
 *
 * Features:
 *   - Text embeddings (sentence-transformers)
 *   - Graph Transformer connection predictions
 *   - Voice transcription (Whisper) - speak to create nodes
 *   - Image understanding (CLIP) - drop images to describe them
 *
 * Usage:
 *   1. Include this file in your HTML
 *   2. LocalBrain.init() on startup
 *   3. Replace neuralNet calls with LocalBrain calls (or use LocalBrain as wrapper)
 */

const LocalBrain = {
    // Server configuration
    serverUrl: 'http://localhost:8420',
    isAvailable: false,
    lastCheck: 0,
    checkInterval: 30000, // Re-check every 30 seconds

    // Status
    status: {
        connected: false,
        device: 'unknown',
        lastLatency: 0,
        fallbackMode: true,
        voiceAvailable: false,
        visionAvailable: false
    },

    /**
     * Initialize and check server availability
     */
    async init() {
        console.log('🧠 LocalBrain: Initializing...');
        await this.checkAvailability();

        // Periodic health check
        setInterval(() => this.checkAvailability(), this.checkInterval);

        if (this.isAvailable) {
            console.log(`✅ LocalBrain: Connected to local server (${this.status.device})`);
        } else {
            console.log('⚠️ LocalBrain: Server not available, using browser ML fallback');
        }

        return this.isAvailable;
    },

    /**
     * Check if local server is running
     */
    async checkAvailability() {
        try {
            const start = performance.now();
            const res = await fetch(`${this.serverUrl}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(2000) // 2 second timeout
            });

            if (res.ok) {
                const health = await res.json();
                this.isAvailable = true;
                this.status.connected = true;
                this.status.device = health.device;
                this.status.lastLatency = performance.now() - start;
                this.status.fallbackMode = false;
                this.status.voiceAvailable = !!health.voice_model;
                this.status.visionAvailable = !!health.vision_model;
                this.lastCheck = Date.now();
                return true;
            }
        } catch (e) {
            // Server not running - this is fine, we'll use fallback
        }

        this.isAvailable = false;
        this.status.connected = false;
        this.status.fallbackMode = true;
        this.status.voiceAvailable = false;
        this.status.visionAvailable = false;
        this.lastCheck = Date.now();
        return false;
    },

    /**
     * Generate embedding for text
     * Falls back to browser neuralNet if server unavailable
     */
    async embed(text) {
        if (this.isAvailable) {
            try {
                const start = performance.now();
                const res = await fetch(`${this.serverUrl}/embed`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });

                if (res.ok) {
                    const data = await res.json();
                    this.status.lastLatency = performance.now() - start;
                    console.log(`🧠 LocalBrain.embed: ${data.time_ms.toFixed(1)}ms (server)`);
                    return data.embedding;
                }
            } catch (e) {
                console.warn('LocalBrain.embed failed, falling back:', e);
                this.isAvailable = false;
            }
        }

        // Fallback to browser
        if (typeof neuralNet !== 'undefined' && neuralNet.embed) {
            console.log('🧠 LocalBrain.embed: using browser fallback');
            return await neuralNet.embed(text);
        }

        throw new Error('No embedding method available');
    },

    /**
     * Generate embeddings for multiple texts
     */
    async embedBatch(texts) {
        if (this.isAvailable) {
            try {
                const start = performance.now();
                const res = await fetch(`${this.serverUrl}/embed/batch`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ texts })
                });

                if (res.ok) {
                    const data = await res.json();
                    this.status.lastLatency = performance.now() - start;
                    console.log(`🧠 LocalBrain.embedBatch: ${texts.length} texts in ${data.time_ms.toFixed(1)}ms`);
                    return data.embeddings;
                }
            } catch (e) {
                console.warn('LocalBrain.embedBatch failed, falling back:', e);
            }
        }

        // Fallback: embed one by one
        const embeddings = [];
        for (const text of texts) {
            embeddings.push(await this.embed(text));
        }
        return embeddings;
    },

    /**
     * Predict connections for a node using Graph Transformer
     * This is the key capability - attention across entire map
     */
    async predictConnections(nodeId, mapData, topK = 5) {
        if (this.isAvailable) {
            try {
                // Format map data for server
                const formattedMap = this._formatMapForServer(mapData);

                const start = performance.now();
                const res = await fetch(`${this.serverUrl}/predict/connections`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        node_id: nodeId,
                        map_data: formattedMap,
                        top_k: topK
                    })
                });

                if (res.ok) {
                    const data = await res.json();
                    this.status.lastLatency = performance.now() - start;
                    console.log(`🧠 LocalBrain.predictConnections: ${data.connections.length} predictions in ${data.time_ms.toFixed(1)}ms`);
                    return {
                        connections: data.connections,
                        attentionWeights: data.attention_weights,
                        source: 'local_server'
                    };
                }
            } catch (e) {
                console.warn('LocalBrain.predictConnections failed, falling back:', e);
            }
        }

        // Fallback to browser neuralNet
        if (typeof neuralNet !== 'undefined' && neuralNet.predictConnections) {
            console.log('🧠 LocalBrain.predictConnections: using browser fallback');
            const predictions = await neuralNet.predictConnections(nodeId);
            return {
                connections: predictions,
                attentionWeights: null,
                source: 'browser_fallback'
            };
        }

        return { connections: [], attentionWeights: null, source: 'none' };
    },

    /**
     * Record feedback for learning
     */
    async recordFeedback(nodeId, action, context) {
        if (this.isAvailable) {
            try {
                await fetch(`${this.serverUrl}/train/feedback`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        node_id: nodeId,
                        action: action,
                        context: context
                    })
                });
                console.log(`🧠 LocalBrain.recordFeedback: ${action}`);
            } catch (e) {
                console.warn('LocalBrain.recordFeedback failed:', e);
            }
        }

        // Also record in browser (for redundancy)
        if (typeof preferenceTracker !== 'undefined') {
            if (action === 'accepted') {
                preferenceTracker.recordAccept?.(nodeId);
            }
        }
    },

    /**
     * Format map data for the server
     */
    _formatMapForServer(mapData) {
        const nodes = [];

        const traverse = (node, parentId = null) => {
            nodes.push({
                id: node.id,
                label: node.label,
                parentId: parentId,
                children: node.children?.map(c => c.id) || []
            });

            if (node.children) {
                node.children.forEach(child => traverse(child, node.id));
            }
        };

        // Handle both store.data format and array format
        if (mapData.id) {
            traverse(mapData);
        } else if (Array.isArray(mapData)) {
            mapData.forEach(node => nodes.push({
                id: node.id,
                label: node.label,
                parentId: node.parentId || null,
                children: node.children?.map(c => c.id) || []
            }));
        }

        return { nodes };
    },

    // ═══════════════════════════════════════════════════════════════════
    // BAPI - Full Map Awareness
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Sync the full map to BAPI's context window.
     * Call this on map load and after significant changes.
     * @param {Object} mapData - The full map data (store.data or array format)
     * @returns {Promise<{synced: number, time_ms: number}>}
     */
    async syncMap(mapData) {
        if (!this.isAvailable) {
            console.log('🧠 LocalBrain.syncMap: Server not available');
            return { synced: 0, time_ms: 0 };
        }

        try {
            const formattedMap = this._formatMapForServer(mapData);
            const start = performance.now();

            const res = await fetch(`${this.serverUrl}/map/sync`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formattedMap)
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🧠 BAPI synced: ${result.synced} nodes in ${result.time_ms.toFixed(0)}ms`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.syncMap failed:', e);
        }

        return { synced: 0, time_ms: 0 };
    },

    /**
     * Get BAPI's analysis of the current map.
     * Returns observations about missing connections, important nodes, etc.
     * @returns {Promise<Object>} Analysis results
     */
    async analyze() {
        if (!this.isAvailable) {
            console.log('🧠 LocalBrain.analyze: Server not available');
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/map/analyze`);

            if (res.ok) {
                const result = await res.json();
                console.log(`🧠 BAPI analysis: ${result.observations?.length || 0} observations`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.analyze failed:', e);
        }

        return { error: 'Analysis failed' };
    },

    /**
     * Get current map sync status
     */
    async getMapStatus() {
        if (!this.isAvailable) {
            return { synced: false, node_count: 0 };
        }

        try {
            const res = await fetch(`${this.serverUrl}/map/status`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getMapStatus failed:', e);
        }

        return { synced: false, node_count: 0 };
    },

    // ═══════════════════════════════════════════════════════════════════
    // UNIFIED BRAIN - Complete Self-Aware Context
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Get complete context for Claude from the Unified Brain.
     * This is THE method to use for all Claude API calls.
     * One call = complete self-awareness.
     *
     * @param {Object} options
     * @param {string} options.requestType - 'chat', 'action', 'code_review', 'self_improve'
     * @param {string} options.userMessage - The user's message
     * @param {string} options.selectedNodeId - Currently selected node ID
     * @param {Object} options.mapData - Map data (store.data format)
     * @param {Object} options.include - What to include in context
     * @returns {Promise<{contextDocument: string, tokenCount: number, brainState: Object}>}
     */
    async getBrainContext(options = {}) {
        if (!this.isAvailable) {
            console.log('🧠 LocalBrain.getBrainContext: Server not available');
            return { contextDocument: null, error: 'Server not available' };
        }

        try {
            const start = performance.now();

            // Format map data if provided
            let mapData = null;
            if (options.mapData) {
                mapData = this._formatMapForServer(options.mapData);
            }

            const res = await fetch(`${this.serverUrl}/brain/context`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    request_type: options.requestType || 'chat',
                    user_message: options.userMessage || '',
                    selected_node_id: options.selectedNodeId || null,
                    map_data: mapData,
                    include: options.include || {
                        self_awareness: true,
                        map_context: true,
                        memories: true,
                        user_profile: true,
                        neural_insights: true
                    }
                })
            });

            if (res.ok) {
                const result = await res.json();
                const latency = performance.now() - start;
                console.log(`🧠 Brain context: ${result.token_count} tokens in ${latency.toFixed(0)}ms`);

                return {
                    contextDocument: result.context_document,
                    tokenCount: result.token_count,
                    breakdown: result.breakdown,
                    brainState: result.brain_state,
                    timeMs: result.time_ms
                };
            }
        } catch (e) {
            console.warn('LocalBrain.getBrainContext failed:', e);
        }

        return { contextDocument: null, error: 'Failed to get brain context' };
    },

    /**
     * Get current brain state for debugging/display
     */
    async getBrainState() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/state`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getBrainState failed:', e);
        }

        return { error: 'Failed to get brain state' };
    },

    /**
     * Record feedback for brain learning
     * Call this when user accepts, rejects, or corrects something
     */
    async recordBrainFeedback(nodeId, action, context = {}) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    node_id: nodeId,
                    action: action,
                    context: context
                })
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🧠 Brain feedback recorded: ${action}`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.recordBrainFeedback failed:', e);
        }

        return { error: 'Failed to record feedback' };
    },

    // ═══════════════════════════════════════════════════════════════════
    // SELF-LEARNING - Brain learns from its own predictions
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Record predictions made by the Graph Transformer.
     * Call this when predictions are generated/shown to user.
     * This enables the brain to learn from prediction outcomes.
     *
     * @param {string} sourceId - The node predictions were made for
     * @param {Array} predictions - Array of {target_id, target_label, score}
     */
    async recordPredictions(sourceId, predictions) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/predictions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source_id: sourceId,
                    predictions: predictions
                })
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🧠 Predictions recorded: ${predictions.length} for node ${sourceId}`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.recordPredictions failed:', e);
        }

        return { error: 'Failed to record predictions' };
    },

    /**
     * Tell the brain about a new connection being created.
     * The brain checks if it predicted this and learns accordingly.
     *
     * This is the KEY self-learning method:
     * - If brain predicted this: Reinforces the pattern
     * - If brain missed this: Learns the new pattern
     *
     * @param {string} sourceId - Source node of connection
     * @param {string} targetId - Target node of connection
     * @param {string} connectionType - 'manual', 'suggested', 'ai'
     * @returns {Promise<{was_predicted, learning_signal, accuracy}>}
     */
    async learnFromConnection(sourceId, targetId, connectionType = 'manual') {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/learn-connection`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source_id: sourceId,
                    target_id: targetId,
                    connection_type: connectionType
                })
            });

            if (res.ok) {
                const result = await res.json();
                if (result.was_predicted) {
                    console.log(`🧠 Self-learning: Brain correctly predicted this connection! (accuracy: ${(result.accuracy * 100).toFixed(1)}%)`);
                } else {
                    console.log(`🧠 Self-learning: Brain learned new pattern from this connection`);
                }
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.learnFromConnection failed:', e);
        }

        return { error: 'Failed to learn from connection' };
    },

    /**
     * Get the brain's learning statistics
     * Shows prediction accuracy and learning history
     */
    async getLearningStats() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/learning`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getLearningStats failed:', e);
        }

        return { error: 'Failed to get learning stats' };
    },

    // ═══════════════════════════════════════════════════════════════════
    // CLAUDE ↔ BRAIN - Bidirectional Learning & Knowledge Distillation
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Send Claude's response to the brain for knowledge extraction.
     * This is how Claude TEACHES the brain.
     *
     * @param {Object} claudeResponse - Structured response from Claude
     * @param {string} claudeResponse.response - The text response
     * @param {Array} claudeResponse.insights - Key facts with confidence
     * @param {Array} claudeResponse.patterns - Patterns identified
     * @param {Array} claudeResponse.corrections - Things corrected
     * @param {Object} claudeResponse.explanations - Concept explanations
     */
    async sendToBrain(claudeResponse) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/receive-from-claude`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(claudeResponse)
            });

            if (res.ok) {
                const result = await res.json();
                const stats = result.knowledge_stats || {};
                console.log(`🧠 Brain learned from Claude: ${stats.distilled_facts || 0} facts, ${stats.patterns_learned || 0} patterns`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.sendToBrain failed:', e);
        }

        return { error: 'Failed to send to brain' };
    },

    /**
     * Get a teaching prompt for Claude.
     * Use this to have Claude teach the brain about a topic.
     *
     * @param {string} topic - What to learn about
     * @returns {Promise<{teaching_request, instructions_for_claude}>}
     */
    async askClaudeToTeach(topic) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/ask-to-teach`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic })
            });

            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.askClaudeToTeach failed:', e);
        }

        return { error: 'Failed to generate teaching request' };
    },

    /**
     * Get all knowledge the brain has learned from Claude.
     */
    async getBrainKnowledge() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/knowledge`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getBrainKnowledge failed:', e);
        }

        return { error: 'Failed to get brain knowledge' };
    },

    /**
     * Get the teaching prompt to include in Claude's system prompt.
     * This enables structured knowledge transfer.
     */
    async getTeachingPrompt() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/teaching-prompt`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getTeachingPrompt failed:', e);
        }

        return { error: 'Failed to get teaching prompt' };
    },

    /**
     * Get comprehensive brain statistics.
     */
    async getFullBrainStats() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/full-stats`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getFullBrainStats failed:', e);
        }

        return { error: 'Failed to get brain stats' };
    },

    // ═══════════════════════════════════════════════════════════════════
    // CODE SELF-AWARENESS - Deep Code Understanding for Claude (Legacy)
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Get the Code Self-Awareness Document for Claude.
     * This document gives Claude true understanding of the MYND codebase.
     * Include this in ALL Claude API calls (~500-1000 tokens).
     * @param {boolean} forceRegenerate - Force regeneration even if cached
     * @returns {Promise<{document: string, cached: boolean, token_estimate: number}>}
     */
    async getCodeSelfAwareness(forceRegenerate = false) {
        if (!this.isAvailable) {
            console.log('🧠 LocalBrain.getCodeSelfAwareness: Server not available');
            return { document: null, error: 'Server not available' };
        }

        try {
            const url = `${this.serverUrl}/code/self-awareness${forceRegenerate ? '?regenerate=true' : ''}`;
            const res = await fetch(url);

            if (res.ok) {
                const result = await res.json();
                console.log(`🧠 Code Self-Awareness: ${result.cached ? 'cached' : 'fresh'} (${result.token_estimate || Math.round(result.document?.length / 4)} tokens)`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.getCodeSelfAwareness failed:', e);
        }

        return { document: null, error: 'Failed to fetch' };
    },

    // ═══════════════════════════════════════════════════════════════════
    // CODE EMBEDDING - Parse codebase into map
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Parse the MYND codebase into map-ready nodes.
     * Use this for MYND to analyze its own architecture.
     * @returns {Promise<Object>} Parsed code structure
     */
    async parseCodebase() {
        if (!this.isAvailable) {
            console.log('🧠 LocalBrain.parseCodebase: Server not available');
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/code/parse`);

            if (res.ok) {
                const result = await res.json();
                console.log(`🧠 Codebase parsed: ${result.stats.total_nodes} nodes (${result.stats.js_files} JS, ${result.stats.py_files} Python)`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.parseCodebase failed:', e);
        }

        return { error: 'Parse failed' };
    },

    /**
     * Import parsed codebase as a new branch in the map.
     * @param {Function} addChildFn - Function to add child nodes (e.g., from store)
     * @param {string} parentId - ID of parent node to attach code tree to
     */
    async importCodebaseToMap(addChildFn, parentId) {
        const result = await this.parseCodebase();
        if (result.error) {
            console.error('Failed to parse codebase:', result.error);
            return { success: false, error: result.error };
        }

        // Convert parsed nodes to map format
        const codeRoot = result.nodes.find(n => n.type === 'root');
        if (!codeRoot) {
            return { success: false, error: 'No root node in parsed code' };
        }

        // Recursive function to add nodes
        let addedCount = 0;
        const addNodes = async (nodeData, targetParentId) => {
            const newNode = await addChildFn(targetParentId, {
                label: nodeData.label,
                description: nodeData.type,
                color: this._getCodeNodeColor(nodeData.type)
            });

            addedCount++;

            // Add children
            if (nodeData.children && nodeData.children.length > 0) {
                for (const childId of nodeData.children) {
                    const childNode = result.nodes.find(n => n.id === childId);
                    if (childNode) {
                        await addNodes(childNode, newNode.id);
                    }
                }
            }

            return newNode;
        };

        // Start adding from code root
        await addNodes(codeRoot, parentId);

        console.log(`🧠 Imported ${addedCount} code nodes into map`);
        return { success: true, nodesAdded: addedCount };
    },

    // Get color based on code node type
    _getCodeNodeColor(type) {
        const colors = {
            'root': '#a855f7',      // Purple - code root
            'directory': '#3b82f6', // Blue - directories
            'file': '#22c55e',      // Green - files
            'class': '#f59e0b',     // Orange - classes
            'object': '#ec4899',    // Pink - objects
            'function': '#06b6d4'   // Cyan - functions
        };
        return colors[type] || null;
    },

    /**
     * Refresh existing code nodes in the map with current source code.
     * Matches nodes by label and updates their descriptions.
     * @param {Object} store - The store object with findNode and data
     * @returns {Promise<Object>} Stats on what was updated
     */
    async refreshCodebase(store) {
        if (!this.isAvailable) {
            return { success: false, error: 'Server not available' };
        }

        const result = await this.parseCodebase();
        if (result.error) {
            return { success: false, error: result.error };
        }

        // Build lookup of fresh code by label
        const freshCodeByLabel = {};
        for (const node of result.nodes) {
            freshCodeByLabel[node.label] = node;
        }

        // Find and update existing code nodes in the map
        let updated = 0;
        let notFound = 0;

        const updateNode = (mapNode) => {
            // Check if this is a code node (has description with code block)
            if (mapNode.description &&
                (mapNode.description.includes('```javascript') ||
                 mapNode.description.includes('```python') ||
                 mapNode.description.includes('=== FILE START ==='))) {

                // Try to find fresh code for this node
                const freshNode = freshCodeByLabel[mapNode.label];
                if (freshNode && freshNode.description) {
                    mapNode.description = freshNode.description;
                    if (freshNode.stats) {
                        mapNode.stats = freshNode.stats;
                    }
                    updated++;
                } else {
                    notFound++;
                }
            }

            // Recursively update children
            if (mapNode.children) {
                for (const child of mapNode.children) {
                    updateNode(child);
                }
            }
        };

        // Start from root
        updateNode(store.data);

        // Save the updated map
        if (store.save) {
            store.save();
        }

        console.log(`🧠 Codebase refreshed: ${updated} nodes updated, ${notFound} not found in current code`);

        return {
            success: true,
            updated,
            notFound,
            freshNodeCount: result.nodes.length
        };
    },

    /**
     * Find the code root node in the map (if previously imported)
     * @param {Object} store - The store object
     * @returns {Object|null} The code root node or null
     */
    findCodeRoot(store) {
        const search = (node) => {
            if (node.label === 'MYND Codebase') {
                return node;
            }
            if (node.children) {
                for (const child of node.children) {
                    const found = search(child);
                    if (found) return found;
                }
            }
            return null;
        };
        return search(store.data);
    },

    // ═══════════════════════════════════════════════════════════════════
    // VOICE (Whisper)
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Transcribe audio to text using local Whisper model
     * @param {Blob|ArrayBuffer} audioData - Audio data to transcribe
     * @param {Object} options - { language: 'en', task: 'transcribe'|'translate' }
     * @returns {Promise<{success: boolean, text?: string, error?: string}>}
     */
    async transcribe(audioData, options = {}) {
        if (!this.isAvailable || !this.status.voiceAvailable) {
            return { success: false, error: 'Voice model not available' };
        }

        try {
            // Convert to base64
            let base64;
            if (audioData instanceof Blob) {
                const buffer = await audioData.arrayBuffer();
                base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
            } else if (audioData instanceof ArrayBuffer) {
                base64 = btoa(String.fromCharCode(...new Uint8Array(audioData)));
            } else {
                return { success: false, error: 'Invalid audio data format' };
            }

            const start = performance.now();
            const res = await fetch(`${this.serverUrl}/voice/transcribe`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    audio_base64: base64,
                    language: options.language || null,
                    task: options.task || 'transcribe'
                })
            });

            if (res.ok) {
                const data = await res.json();
                this.status.lastLatency = performance.now() - start;

                if (data.success) {
                    console.log(`🎤 LocalBrain.transcribe: "${data.text}" (${data.time_ms?.toFixed(0)}ms)`);
                }

                return data;
            }

            return { success: false, error: 'Server error' };

        } catch (e) {
            console.warn('LocalBrain.transcribe failed:', e);
            return { success: false, error: e.message };
        }
    },

    /**
     * Check if voice input is available
     */
    isVoiceAvailable() {
        return this.isAvailable && this.status.voiceAvailable;
    },

    // ═══════════════════════════════════════════════════════════════════
    // VISION (CLIP)
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Describe an image using local CLIP model
     * @param {Blob|ArrayBuffer} imageData - Image data to describe
     * @param {Object} options - { candidateLabels: [...], topK: 5 }
     * @returns {Promise<{success: boolean, description?: string, confidence?: number, error?: string}>}
     */
    async describeImage(imageData, options = {}) {
        if (!this.isAvailable || !this.status.visionAvailable) {
            return { success: false, error: 'Vision model not available' };
        }

        try {
            // Convert to base64
            let base64;
            if (imageData instanceof Blob) {
                const buffer = await imageData.arrayBuffer();
                base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
            } else if (imageData instanceof ArrayBuffer) {
                base64 = btoa(String.fromCharCode(...new Uint8Array(imageData)));
            } else if (typeof imageData === 'string' && imageData.startsWith('data:')) {
                // Already a data URL, extract base64 part
                base64 = imageData.split(',')[1];
            } else {
                return { success: false, error: 'Invalid image data format' };
            }

            const start = performance.now();
            const res = await fetch(`${this.serverUrl}/image/describe`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_base64: base64,
                    candidate_labels: options.candidateLabels || null,
                    top_k: options.topK || 5
                })
            });

            if (res.ok) {
                const data = await res.json();
                this.status.lastLatency = performance.now() - start;

                if (data.success) {
                    console.log(`🖼️ LocalBrain.describeImage: "${data.description}" (${Math.round(data.confidence * 100)}% confidence, ${data.time_ms?.toFixed(0)}ms)`);
                }

                return data;
            }

            return { success: false, error: 'Server error' };

        } catch (e) {
            console.warn('LocalBrain.describeImage failed:', e);
            return { success: false, error: e.message };
        }
    },

    /**
     * Generate embedding for an image (for similarity search)
     * @param {Blob|ArrayBuffer} imageData - Image data
     * @returns {Promise<number[]|null>}
     */
    async embedImage(imageData) {
        if (!this.isAvailable || !this.status.visionAvailable) {
            return null;
        }

        try {
            let base64;
            if (imageData instanceof Blob) {
                const buffer = await imageData.arrayBuffer();
                base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
            } else if (imageData instanceof ArrayBuffer) {
                base64 = btoa(String.fromCharCode(...new Uint8Array(imageData)));
            } else {
                return null;
            }

            const res = await fetch(`${this.serverUrl}/image/embed`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_base64: base64 })
            });

            if (res.ok) {
                const data = await res.json();
                console.log(`🖼️ LocalBrain.embedImage: ${data.dim} dimensions (${data.time_ms?.toFixed(0)}ms)`);
                return data.embedding;
            }

            return null;

        } catch (e) {
            console.warn('LocalBrain.embedImage failed:', e);
            return null;
        }
    },

    /**
     * Check if vision/image input is available
     */
    isVisionAvailable() {
        return this.isAvailable && this.status.visionAvailable;
    },

    /**
     * Get status for debugging/display
     */
    getStatus() {
        return {
            ...this.status,
            serverUrl: this.serverUrl,
            lastCheck: this.lastCheck ? new Date(this.lastCheck).toISOString() : null
        };
    }
};

// Auto-initialize when script loads (optional)
// Uncomment if you want automatic initialization:
// document.addEventListener('DOMContentLoaded', () => LocalBrain.init());

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LocalBrain;
}

console.log('🧠 LocalBrain client loaded. Call LocalBrain.init() to connect.');
