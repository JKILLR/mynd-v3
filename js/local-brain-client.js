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
