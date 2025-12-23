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

    // Track if initial map sync has been done
    _initialSyncDone: false,

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
                const wasConnected = this.isAvailable;
                this.isAvailable = true;
                this.status.connected = true;
                this.status.device = health.device;
                this.status.lastLatency = performance.now() - start;
                this.status.fallbackMode = false;
                this.status.voiceAvailable = !!health.voice_model;
                this.status.visionAvailable = !!health.vision_model;
                this.lastCheck = Date.now();

                // Initial map sync when first connected
                if (!wasConnected && !this._initialSyncDone) {
                    this._doInitialMapSync().then(() => {
                        this._initialSyncDone = true;
                    }).catch(() => {
                        // Will retry on next checkAvailability
                    });
                }
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
     * Predict category for text (replaces browser TensorFlow.js)
     * @param {string} text - Text to categorize
     * @param {Object} mapData - Map data (store.data format)
     * @param {number} topK - Number of predictions to return
     * @returns {Promise<Array<{category: string, node_id: string, confidence: number}>>}
     */
    async predictCategory(text, mapData, topK = 5) {
        if (!this.isAvailable) {
            return null; // Fall back to browser
        }

        try {
            const start = performance.now();
            const formattedMap = this._formatMapForServerWithDepth(mapData);

            const res = await fetch(`${this.serverUrl}/predict/category`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: AbortSignal.timeout(10000), // 10 second timeout
                body: JSON.stringify({
                    text: text,
                    map_data: formattedMap,
                    top_k: topK
                })
            });

            if (res.ok) {
                const result = await res.json();
                const elapsed = performance.now() - start;
                console.log(`🧠 LocalBrain.predictCategory: ${result.predictions.length} predictions in ${elapsed.toFixed(1)}ms`);

                // Convert to browser format: [{category, confidence}]
                return result.predictions.map(p => ({
                    category: p.category,
                    confidence: p.confidence,
                    nodeId: p.node_id
                }));
            }
        } catch (e) {
            console.warn('LocalBrain.predictCategory failed:', e);
        }

        return null; // Fall back to browser
    },

    /**
     * Format map data with depth info for category prediction
     */
    _formatMapForServerWithDepth(mapData) {
        const nodes = [];

        const traverse = (node, parentId = null, depth = 0) => {
            nodes.push({
                id: node.id,
                label: node.label,
                description: node.description || '',
                parentId: parentId,
                depth: depth,
                children: node.children?.map(c => c.id) || []
            });

            if (node.children) {
                node.children.forEach(child => traverse(child, node.id, depth + 1));
            }
        };

        if (mapData.id) {
            traverse(mapData);
        }

        return { nodes };
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
        // ALWAYS try to get brain context - don't bail early
        // This ensures ASA learns from every conversation

        // Try to reconnect if we think we're not available
        if (!this.isAvailable) {
            console.log('🧠 LocalBrain.getBrainContext: Attempting to reconnect...');
            await this.checkAvailability();
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
                signal: AbortSignal.timeout(30000), // 30 second timeout
                body: JSON.stringify({
                    request_type: options.requestType || 'chat',
                    user_message: options.userMessage || '',
                    selected_node_id: options.selectedNodeId || null,
                    map_data: mapData,
                    user_id: options.userId || null,  // For Supabase AI memory queries
                    goals: options.goals || null,     // For goal-aware context synthesis
                    include: options.include || {
                        self_awareness: true,
                        map_context: true,
                        memories: true,
                        user_profile: true,
                        neural_insights: true,
                        synthesized_context: true
                    }
                })
            });

            if (res.ok) {
                const result = await res.json();
                const latency = performance.now() - start;
                console.log(`🧠 Brain context: ${result.token_count} tokens in ${latency.toFixed(0)}ms`);

                // Mark as available since call succeeded
                this.isAvailable = true;
                this.status.connected = true;

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
            // Mark as unavailable so we retry next time
            this.isAvailable = false;
            this.status.connected = false;
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
    // META-LEARNING - Learning how to learn
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Get detailed meta-learning statistics.
     * Shows how the brain is learning to learn:
     * - Source effectiveness (which knowledge sources work best)
     * - Confidence calibration (is the brain over/under confident)
     * - Learning rates per domain
     * - Best learning strategies
     * - Improvement trend over time
     */
    async getMetaStats() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/meta`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getMetaStats failed:', e);
        }

        return { error: 'Failed to get meta-learning stats' };
    },

    /**
     * Get a human-readable summary of meta-learning state.
     * Useful for debugging and displaying brain behavior.
     */
    async getMetaSummary() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/meta/summary`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getMetaSummary failed:', e);
        }

        return { error: 'Failed to get meta-learning summary' };
    },

    /**
     * Check if the brain's confidence scores are calibrated.
     * Shows whether it's over-confident, under-confident, or well-calibrated.
     *
     * Good calibration means:
     * - When brain says 80% confident, it's right ~80% of the time
     */
    async getCalibrationReport() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/meta/calibration`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getCalibrationReport failed:', e);
        }

        return { error: 'Failed to get calibration report' };
    },

    /**
     * Check if the brain is improving over time.
     * Shows learning velocity and effectiveness trends.
     */
    async getImprovementTrend() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/meta/improvement`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getImprovementTrend failed:', e);
        }

        return { error: 'Failed to get improvement trend' };
    },

    /**
     * Get recommendations on which knowledge sources to prioritize.
     * The meta-learner tracks which sources are most effective
     * and adjusts attention weights accordingly.
     *
     * @param {string} context - Optional context for recommendations
     */
    async getSourceRecommendations(context = '') {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const url = context
                ? `${this.serverUrl}/brain/meta/recommendations?context=${encodeURIComponent(context)}`
                : `${this.serverUrl}/brain/meta/recommendations`;
            const res = await fetch(url);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getSourceRecommendations failed:', e);
        }

        return { error: 'Failed to get source recommendations' };
    },

    /**
     * Record feedback on a knowledge source's effectiveness.
     * Call this when you know a source helped or didn't help.
     *
     * This updates the meta-learner's attention weights -
     * effective sources get prioritized in future context building.
     *
     * @param {string} source - 'predictions', 'distilled_knowledge', 'patterns', 'corrections', 'memories'
     * @param {boolean} success - Whether the source helped
     * @param {Object} context - Optional context about the usage
     */
    async recordSourceFeedback(source, success, context = null) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/meta/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source,
                    success,
                    context
                })
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🧠 Meta-learning: Recorded ${source} ${success ? 'success' : 'failure'} (new weight: ${result.new_weight?.toFixed(2)})`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.recordSourceFeedback failed:', e);
        }

        return { error: 'Failed to record source feedback' };
    },

    /**
     * Adjust the learning rate for a domain.
     * Positive delta = learn faster, negative = learn slower.
     *
     * @param {string} domain - 'connections', 'patterns', 'corrections', 'insights'
     * @param {number} delta - Adjustment amount (e.g., 0.05 or -0.02)
     */
    async adjustLearningRate(domain, delta) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/meta/learning-rate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ domain, delta })
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🧠 Meta-learning: ${domain} learning rate ${result.old_rate.toFixed(3)} → ${result.new_rate.toFixed(3)}`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.adjustLearningRate failed:', e);
        }

        return { error: 'Failed to adjust learning rate' };
    },

    /**
     * Manually save a meta-learning epoch.
     * Epochs capture the brain's learning state at a point in time.
     * Useful after significant learning events.
     */
    async saveMetaEpoch() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/meta/save-epoch`, {
                method: 'POST'
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🧠 Meta-learning: Saved epoch ${result.epoch?.epoch}`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.saveMetaEpoch failed:', e);
        }

        return { error: 'Failed to save meta epoch' };
    },

    // ═══════════════════════════════════════════════════════════════════
    // SELF-IMPROVEMENT - Analyze weaknesses and suggest improvements
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Run a complete self-analysis of the brain.
     * Generates improvement suggestions based on performance metrics.
     * Uses the vision statement to prioritize suggestions.
     */
    async runSelfAnalysis() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/analyze`, {
                method: 'POST'
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🔍 Self-analysis: ${result.suggestion_count} suggestions generated`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.runSelfAnalysis failed:', e);
        }

        return { error: 'Failed to run self-analysis' };
    },

    /**
     * Get current improvement suggestions.
     * @param {string} category - Optional: architecture, training, integration, data_flow, user_experience, performance, accuracy
     * @param {string} priority - Optional: high, medium, low
     */
    async getImprovementSuggestions(category = null, priority = null) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            let url = `${this.serverUrl}/brain/suggestions`;
            const params = [];
            if (category) params.push(`category=${encodeURIComponent(category)}`);
            if (priority) params.push(`priority=${encodeURIComponent(priority)}`);
            if (params.length) url += '?' + params.join('&');

            const res = await fetch(url);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getImprovementSuggestions failed:', e);
        }

        return { error: 'Failed to get suggestions' };
    },

    /**
     * Get top improvement suggestions by priority.
     * @param {number} limit - Max number of suggestions to return
     */
    async getTopSuggestions(limit = 5) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/suggestions/top?limit=${limit}`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getTopSuggestions failed:', e);
        }

        return { error: 'Failed to get top suggestions' };
    },

    /**
     * Get a human-readable summary of improvement suggestions.
     * Returns markdown formatted by priority.
     */
    async getImprovementSummary() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/suggestions/summary`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getImprovementSummary failed:', e);
        }

        return { error: 'Failed to get improvement summary' };
    },

    /**
     * Mark a suggestion's status.
     * @param {string} suggestionId - The suggestion ID
     * @param {string} status - 'accepted', 'rejected', or 'implemented'
     * @param {string} notes - Optional notes about the decision
     */
    async markSuggestionStatus(suggestionId, status, notes = '') {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/suggestions/status`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    suggestion_id: suggestionId,
                    status,
                    notes
                })
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🔍 Suggestion ${suggestionId}: ${status}`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.markSuggestionStatus failed:', e);
        }

        return { error: 'Failed to mark suggestion status' };
    },

    /**
     * Get self-improvement statistics.
     */
    async getImprovementStats() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/improvement-stats`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getImprovementStats failed:', e);
        }

        return { error: 'Failed to get improvement stats' };
    },

    // ═══════════════════════════════════════════════════════════════════
    // VISION - User-editable goals and priorities
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Get the brain's vision statement, goals, and priorities.
     * This guides what improvements the brain suggests.
     */
    async getVision() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/vision`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getVision failed:', e);
        }

        return { error: 'Failed to get vision' };
    },

    /**
     * Update the vision statement, goals, or priorities.
     * @param {Object} updates
     * @param {string} updates.statement - The vision statement text
     * @param {string[]} updates.goals - Array of goals
     * @param {string[]} updates.priorities - Array of priorities in order
     */
    async setVision(updates) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/vision`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updates)
            });

            if (res.ok) {
                const result = await res.json();
                console.log('🎯 Vision updated');
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.setVision failed:', e);
        }

        return { error: 'Failed to update vision' };
    },

    /**
     * Add a goal to the vision.
     * @param {string} goal - The goal to add
     */
    async addVisionGoal(goal) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/vision/goals`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ goal })
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🎯 Goal added: ${goal}`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.addVisionGoal failed:', e);
        }

        return { error: 'Failed to add goal' };
    },

    /**
     * Remove a goal from the vision.
     * @param {string} goal - The goal to remove
     */
    async removeVisionGoal(goal) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/brain/vision/goals?goal=${encodeURIComponent(goal)}`, {
                method: 'DELETE'
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🎯 Goal removed: ${goal}`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.removeVisionGoal failed:', e);
        }

        return { error: 'Failed to remove goal' };
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
    },

    // ═══════════════════════════════════════════════════════════════════
    // CONVERSATION STORAGE - Import and search AI conversations
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Import a conversation from any AI chat (Claude, ChatGPT, Grok, etc.)
     * Full text is stored and embedded for semantic search.
     *
     * @param {string} text - Full conversation text
     * @param {string} source - Source AI: 'claude', 'chatgpt', 'grok', etc.
     * @param {string} title - Optional title for the conversation
     * @returns {Promise<{status, id, title, chars}>}
     */
    async importConversation(text, source = 'unknown', title = null) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const start = performance.now();
            const res = await fetch(`${this.serverUrl}/conversations/import`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, source, title })
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`💬 Conversation imported: "${result.title}" (${result.chars} chars)`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.importConversation failed:', e);
        }

        return { error: 'Failed to import conversation' };
    },

    /**
     * List all stored conversations.
     * @param {string} source - Optional filter by source
     * @returns {Promise<{conversations: Array, stats: Object}>}
     */
    async listConversations(source = null) {
        if (!this.isAvailable) {
            return { conversations: [], stats: {}, error: 'Server not available' };
        }

        try {
            const url = source
                ? `${this.serverUrl}/conversations?source=${encodeURIComponent(source)}`
                : `${this.serverUrl}/conversations`;
            const res = await fetch(url);

            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.listConversations failed:', e);
        }

        return { conversations: [], stats: {}, error: 'Failed to list conversations' };
    },

    /**
     * Get a specific conversation by ID.
     * @param {string} convId - Conversation ID
     * @returns {Promise<Object>} Full conversation data
     */
    async getConversation(convId) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/conversations/${convId}`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getConversation failed:', e);
        }

        return { error: 'Failed to get conversation' };
    },

    /**
     * Search conversations by semantic similarity.
     * @param {string} query - Search query
     * @param {number} topK - Max results to return
     * @param {string} source - Optional source filter
     * @returns {Promise<{results: Array, query: string}>}
     */
    async searchConversations(query, topK = 5, source = null) {
        if (!this.isAvailable) {
            return { results: [], error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/conversations/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k: topK, source_filter: source })
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`💬 Found ${result.results.length} relevant conversations for: "${query}"`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.searchConversations failed:', e);
        }

        return { results: [], error: 'Search failed' };
    },

    /**
     * Get relevant context from past conversations for injection into Claude.
     * This is THE key method for unified context.
     *
     * @param {string} query - The user's current query/message
     * @param {number} maxTokens - Approximate max tokens for context
     * @param {boolean} includeFullText - Include full conversations or just summaries
     * @returns {Promise<{context: string, chars: number}>}
     */
    async getConversationContext(query, maxTokens = 4000, includeFullText = false) {
        if (!this.isAvailable) {
            return { context: '', error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/conversations/context`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    max_tokens: maxTokens,
                    include_full_text: includeFullText
                })
            });

            if (res.ok) {
                const result = await res.json();
                if (result.chars > 0) {
                    console.log(`💬 Retrieved ${result.chars} chars of conversation context`);
                }
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.getConversationContext failed:', e);
        }

        return { context: '', error: 'Failed to get context' };
    },

    /**
     * Get conversation storage statistics.
     * @returns {Promise<{total_conversations, total_chars, total_mb, sources}>}
     */
    async getConversationStats() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/conversations/stats`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getConversationStats failed:', e);
        }

        return { error: 'Failed to get stats' };
    },

    /**
     * Internal: Perform initial map sync when server connects.
     * Uses window.store if available (set by app-module.js).
     */
    async _doInitialMapSync() {
        // Wait a bit for store to be ready
        await new Promise(resolve => setTimeout(resolve, 500));

        if (typeof window !== 'undefined' && window.store && window.store.data) {
            try {
                console.log('🔄 LocalBrain: Performing initial map sync...');

                // Run both syncs in parallel for better performance
                const [unifiedResult, bapiResult] = await Promise.all([
                    this.syncMapToServer(window.store.data),
                    this.syncMap(window.store.data)
                ]);

                if (unifiedResult.status === 'synced') {
                    console.log(`✅ LocalBrain: Initial sync complete - ${unifiedResult.nodes} nodes`);
                }
                if (bapiResult.synced > 0) {
                    console.log(`🧠 BAPI: Map synced - ${bapiResult.synced} nodes in ${bapiResult.time_ms?.toFixed(0) || 0}ms`);
                }
            } catch (e) {
                console.warn('LocalBrain: Initial map sync failed:', e.message);
            }
        } else {
            console.log('⚠️ LocalBrain: No store available for initial sync');
        }
    },

    // ═══════════════════════════════════════════════════════════════════
    // UNIFIED SYSTEM - Map as Vector Database
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Sync the browser's map to the server's unified graph.
     * This makes the map the source of truth for the vector database.
     * @param {Object} mapData - The map data (recursive node structure)
     * @param {boolean} reEmbedAll - Whether to regenerate all embeddings
     * @returns {Promise<{status, nodes, embedded, time_ms}>}
     */
    async syncMapToServer(mapData, reEmbedAll = false) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/unified/map/sync`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    map_data: mapData,
                    re_embed_all: reEmbedAll
                })
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🗺️ Map synced to server: ${result.nodes} nodes, ${result.embedded} embedded`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.syncMapToServer failed:', e);
        }

        return { error: 'Failed to sync map' };
    },

    /**
     * Export the server's unified graph to browser map format.
     * Use this to initialize the browser from server state.
     * @returns {Promise<{map_data, stats}>}
     */
    async exportMapFromServer() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/unified/map/export`);
            if (res.ok) {
                const result = await res.json();
                if (result.map_data) {
                    console.log(`🗺️ Map exported from server: ${result.stats?.total_nodes} nodes`);
                }
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.exportMapFromServer failed:', e);
        }

        return { error: 'Failed to export map' };
    },

    /**
     * Semantic search across the unified map.
     * @param {string} query - Search query
     * @param {number} topK - Maximum results
     * @param {number} threshold - Minimum similarity threshold
     * @param {string[]} nodeTypes - Filter by node types
     * @returns {Promise<{results, query, time_ms}>}
     */
    async searchUnifiedMap(query, topK = 10, threshold = 0.35, nodeTypes = null) {
        if (!this.isAvailable) {
            return { error: 'Server not available', results: [] };
        }

        try {
            const res = await fetch(`${this.serverUrl}/unified/map/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    top_k: topK,
                    threshold,
                    node_types: nodeTypes
                })
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`🔍 Map search: ${result.results?.length} results for "${query}"`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.searchUnifiedMap failed:', e);
        }

        return { error: 'Failed to search map', results: [] };
    },

    /**
     * Get unified context for RAG from the map.
     * This is THE method to call before every Claude request.
     * @param {string} query - Query to find relevant context for
     * @param {number} maxTokens - Maximum tokens to return
     * @param {boolean} includeSources - Whether to include source excerpts
     * @returns {Promise<{context, nodes_used, chars, time_ms}>}
     */
    async getUnifiedContext(query, maxTokens = 8000, includeSources = false) {
        if (!this.isAvailable) {
            return { context: '', error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/unified/context`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    max_tokens: maxTokens,
                    include_sources: includeSources
                })
            });

            if (res.ok) {
                const result = await res.json();
                if (result.chars > 0) {
                    console.log(`🧠 Unified context: ${result.chars} chars from ${result.nodes_used} nodes`);
                }
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.getUnifiedContext failed:', e);
        }

        return { context: '', error: 'Failed to get context' };
    },

    /**
     * Get unified map statistics.
     * @returns {Promise<{total_nodes, embedded_nodes, type_counts}>}
     */
    async getUnifiedMapStats() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/unified/map/stats`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getUnifiedMapStats failed:', e);
        }

        return { error: 'Failed to get stats' };
    },

    // ═══════════════════════════════════════════════════════════════════
    // CONVERSATION ARCHIVE + KNOWLEDGE EXTRACTION
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Import a conversation and extract knowledge into the map.
     * @param {string} text - Full conversation text
     * @param {string} source - Source AI (claude, chatgpt, grok)
     * @param {string} title - Optional title
     * @param {boolean} processImmediately - Extract knowledge now
     * @returns {Promise<{conversation_id, nodes_created, nodes_enriched}>}
     */
    async importConversationUnified(text, source = 'unknown', title = null, processImmediately = true) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/unified/conversations/import`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    source,
                    title,
                    process_immediately: processImmediately
                })
            });

            if (res.ok) {
                const result = await res.json();
                if (result.processed) {
                    console.log(`📚 Conversation imported: ${result.concepts_extracted} concepts → ${result.nodes_created} new, ${result.nodes_enriched} enriched`);
                } else {
                    console.log(`📚 Conversation archived: ${result.title}`);
                }
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.importConversationUnified failed:', e);
        }

        return { error: 'Failed to import conversation' };
    },

    /**
     * List archived conversations.
     * @param {string} source - Filter by source
     * @param {boolean} processed - Filter by processed status
     * @returns {Promise<{conversations, stats}>}
     */
    async listArchivedConversations(source = null, processed = null) {
        if (!this.isAvailable) {
            return { error: 'Server not available', conversations: [] };
        }

        try {
            const params = new URLSearchParams();
            if (source) params.set('source', source);
            if (processed !== null) params.set('processed', processed);

            const res = await fetch(`${this.serverUrl}/unified/conversations?${params}`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.listArchivedConversations failed:', e);
        }

        return { error: 'Failed to list conversations', conversations: [] };
    },

    /**
     * Process pending (unprocessed) conversations.
     * Extracts knowledge and integrates into the map.
     * @param {number} limit - Maximum conversations to process
     * @returns {Promise<{processed, results}>}
     */
    async processPendingConversations(limit = 5) {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/unified/conversations/process-pending?limit=${limit}`, {
                method: 'POST'
            });

            if (res.ok) {
                const result = await res.json();
                console.log(`⚙️ Processed ${result.processed} pending conversations`);
                return result;
            }
        } catch (e) {
            console.warn('LocalBrain.processPendingConversations failed:', e);
        }

        return { error: 'Failed to process conversations' };
    },

    /**
     * Get conversation archive statistics.
     * @returns {Promise<{total_conversations, processed, unprocessed, sources}>}
     */
    async getArchiveStats() {
        if (!this.isAvailable) {
            return { error: 'Server not available' };
        }

        try {
            const res = await fetch(`${this.serverUrl}/unified/conversations/stats`);
            if (res.ok) {
                return await res.json();
            }
        } catch (e) {
            console.warn('LocalBrain.getArchiveStats failed:', e);
        }

        return { error: 'Failed to get stats' };
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
