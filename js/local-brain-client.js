/**
 * MYND Local Brain Client
 * =======================
 * Connects the browser app to the local Python ML server.
 * Falls back to browser TensorFlow.js if server unavailable.
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
        fallbackMode: true
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
                this.lastCheck = Date.now();
                return true;
            }
        } catch (e) {
            // Server not running - this is fine, we'll use fallback
        }

        this.isAvailable = false;
        this.status.connected = false;
        this.status.fallbackMode = true;
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
