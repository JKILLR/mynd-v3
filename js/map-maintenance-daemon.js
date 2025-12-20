/**
 * MYND Map Maintenance Daemon
 * ===========================
 * Autonomous background system that proactively monitors and maintains
 * the map structure - like having a personal assistant constantly tidying
 * and optimizing your cognitive space.
 *
 * Features:
 *   - Duplicate detection using semantic embeddings
 *   - Structural analysis (imbalance, orphans, misplacement)
 *   - Auto-reorganization with user approval workflow
 *   - Smart consolidation of redundant nodes
 *   - Gap detection for missing connections
 *   - Periodic maintenance cycles
 */

const MapMaintenanceDaemon = {
    // ═══════════════════════════════════════════════════════════════════
    // CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════

    VERSION: '1.0',
    STORAGE_KEY: 'mynd-map-maintenance',

    config: {
        enabled: false,                     // Master toggle
        autoApply: false,                   // Auto-apply fixes vs queue for review
        checkIntervalMs: 10 * 60 * 1000,    // Check every 10 minutes
        minIdleTimeMs: 2 * 60 * 1000,       // Wait 2 min of idle before running

        // Feature toggles
        features: {
            duplicateDetection: true,
            structuralAnalysis: true,
            gapDetection: true,
            autoConsolidation: false,       // Requires explicit enable (destructive)
            autoReorganization: false       // Requires explicit enable
        },

        // Thresholds
        thresholds: {
            similarityForDuplicate: 0.85,   // Embedding similarity threshold
            nameSimilarityMin: 0.7,         // Levenshtein similarity for names
            maxChildrenImbalance: 5,        // Max difference between sibling branch sizes
            maxDepth: 10,                   // Warn if tree gets too deep
            minBranchSize: 2,               // Branches with fewer nodes may need attention
            orphanAgeMs: 7 * 24 * 60 * 60 * 1000  // Orphan detection age (7 days)
        },

        // Core concept protection - nodes matching these patterns are protected from suggestions
        protectedPatterns: [
            /vision/i, /mission/i, /goal/i, /purpose/i, /core/i,
            /foundation/i, /principle/i, /value/i, /manifest/i,
            /mynd/i, /meta/i, /root/i
        ]
    },

    // Foundational Vision Context - used to align suggestions with the app's purpose
    VISION_CONTEXT: `MYND is a manifestation engine — a system designed to transform thought into reality.
The map structure represents the user's cognitive space and should be optimized for:
1. Cognitive Sovereignty - Clear thinking and wise decisions
2. Knowledge Crystallization - Structured wisdom that compounds over time
3. Reality Bridge - Ideas flowing from conception to manifestation

When analyzing the map, prioritize:
- Protecting foundational/vision nodes (never suggest removing core concepts)
- Ensuring important nodes are well-connected
- Maintaining clarity of hierarchy for key concepts
- Identifying gaps that block manifestation`,

    // ═══════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════

    initialized: false,
    isRunning: false,
    lastMaintenanceTime: 0,
    checkIntervalId: null,
    encoder: null,  // Shared from neuralNet

    // Maintenance queue (items awaiting user review)
    maintenanceQueue: [],
    maxQueueSize: 50,

    // Stats
    stats: {
        totalScans: 0,
        duplicatesFound: 0,
        issuesDetected: 0,
        autoFixed: 0,
        userApproved: 0,
        userDismissed: 0
    },

    // Activity log
    activityLog: [],
    maxLogEntries: 50,

    // ═══════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════

    async init() {
        if (this.initialized) return this;

        console.log('🔧 MapMaintenanceDaemon: Initializing...');

        try {
            // Load saved config and stats
            this.loadFromStorage();

            // Try to get encoder from neuralNet
            if (typeof neuralNet !== 'undefined' && neuralNet.encoder) {
                this.encoder = neuralNet.encoder;
                console.log('✓ MapMaintenanceDaemon: Encoder linked from neuralNet');
            }

            this.initialized = true;

            // Start if enabled
            if (this.config.enabled) {
                this.start();
            }

            console.log('✅ MapMaintenanceDaemon: Ready');
            return this;

        } catch (error) {
            console.error('MapMaintenanceDaemon init failed:', error);
            this.initialized = false;
            return this;
        }
    },

    // Set encoder (called from neuralNet after it loads)
    setEncoder(encoder) {
        this.encoder = encoder;
        console.log('✓ MapMaintenanceDaemon: Encoder set');
    },

    // ═══════════════════════════════════════════════════════════════════
    // START/STOP CONTROLS
    // ═══════════════════════════════════════════════════════════════════

    start() {
        if (!this.initialized) {
            console.warn('MapMaintenanceDaemon: Not initialized');
            return;
        }

        if (this.isRunning) return;

        console.log('🔧 MapMaintenanceDaemon: Starting autonomous maintenance...');
        this.config.enabled = true;
        this.isRunning = true;

        // Setup periodic check
        this.checkIntervalId = setInterval(() => {
            this.checkAndMaintain();
        }, this.config.checkIntervalMs);

        this.log('Started autonomous map maintenance');
        this.saveToStorage();
        this.emitEvent('started');
    },

    stop() {
        console.log('🔧 MapMaintenanceDaemon: Stopping...');
        this.config.enabled = false;
        this.isRunning = false;

        if (this.checkIntervalId) {
            clearInterval(this.checkIntervalId);
            this.checkIntervalId = null;
        }

        this.log('Stopped autonomous map maintenance');
        this.saveToStorage();
        this.emitEvent('stopped');
    },

    toggle() {
        if (this.config.enabled) {
            this.stop();
        } else {
            this.start();
        }
        return this.config.enabled;
    },

    // ═══════════════════════════════════════════════════════════════════
    // MAINTENANCE CYCLE
    // ═══════════════════════════════════════════════════════════════════

    async checkAndMaintain() {
        if (!this.isRunning) return;

        // Check idle time if ActivityTracker available
        if (typeof ActivityTracker !== 'undefined') {
            const idleTime = ActivityTracker.getIdleTime();
            if (idleTime < this.config.minIdleTimeMs) {
                return; // User is active, skip maintenance
            }
        }

        // Rate limit
        const timeSinceLast = Date.now() - this.lastMaintenanceTime;
        if (timeSinceLast < this.config.checkIntervalMs * 0.9) {
            return;
        }

        await this.runMaintenance();
    },

    async runMaintenance(options = {}) {
        const store = window.app?.store;
        if (!store) {
            this.log('No store available');
            return null;
        }

        console.log('🔧 MapMaintenanceDaemon: Running maintenance scan...');
        this.log('Starting maintenance scan');
        this.emitEvent('scanStarted');

        const report = {
            timestamp: Date.now(),
            duplicates: [],
            structuralIssues: [],
            gaps: [],
            suggestions: [],
            summary: {}
        };

        try {
            const allNodes = store.getAllNodes();

            if (allNodes.length < 3) {
                this.log('Map too small for maintenance');
                return report;
            }

            // 1. Duplicate Detection
            if (this.config.features.duplicateDetection) {
                report.duplicates = await this.detectDuplicates(allNodes);
            }

            // 2. Structural Analysis
            if (this.config.features.structuralAnalysis) {
                report.structuralIssues = this.analyzeStructure(allNodes, store.data);
            }

            // 3. Gap Detection
            if (this.config.features.gapDetection) {
                report.gaps = await this.detectGaps(allNodes);
            }

            // Generate suggestions from findings
            report.suggestions = this.generateSuggestions(report);

            // Build summary
            report.summary = {
                totalNodes: allNodes.length,
                duplicatesFound: report.duplicates.length,
                structuralIssues: report.structuralIssues.length,
                gapsFound: report.gaps.length,
                suggestionsGenerated: report.suggestions.length
            };

            // Update stats
            this.stats.totalScans++;
            this.stats.duplicatesFound += report.duplicates.length;
            this.stats.issuesDetected += report.structuralIssues.length + report.gaps.length;
            this.lastMaintenanceTime = Date.now();

            // Queue items for review or auto-apply
            await this.processFindings(report);

            this.log(`Scan complete: ${report.duplicates.length} duplicates, ${report.structuralIssues.length} issues, ${report.suggestions.length} suggestions`);
            this.saveToStorage();
            this.emitEvent('scanComplete', { report });

            return report;

        } catch (error) {
            console.error('Maintenance scan failed:', error);
            this.log(`Scan failed: ${error.message}`);
            this.emitEvent('scanError', { error: error.message });
            return report;
        }
    },

    // ═══════════════════════════════════════════════════════════════════
    // NODE IMPORTANCE & PROTECTION
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Check if a node is protected (core concept, vision-related, etc.)
     * Protected nodes are never suggested for removal or significant changes
     */
    isProtectedNode(node) {
        if (!node || !node.label) return false;

        const label = node.label.toLowerCase();

        // Check against protected patterns
        for (const pattern of this.config.protectedPatterns) {
            if (pattern.test(label)) {
                return true;
            }
        }

        // Check if node has high importance metadata
        if (node.importance && node.importance >= 0.8) {
            return true;
        }

        // Check if it's a top-level branch (direct child of root)
        if (!node.parentId || node.depth === 1) {
            return true;
        }

        return false;
    },

    /**
     * Calculate importance score for a node based on various factors
     */
    calculateNodeImportance(node, allNodes) {
        let score = 0.5; // Base score

        // Boost for vision-related content
        const visionKeywords = ['vision', 'mission', 'goal', 'purpose', 'core', 'value', 'principle'];
        const label = (node.label || '').toLowerCase();
        const desc = (node.description || '').toLowerCase();

        for (const keyword of visionKeywords) {
            if (label.includes(keyword) || desc.includes(keyword)) {
                score += 0.15;
            }
        }

        // Boost for nodes with many children (structural importance)
        const childCount = allNodes.filter(n => n.parentId === node.id).length;
        if (childCount > 3) score += 0.1;
        if (childCount > 7) score += 0.1;

        // Boost for nodes with descriptions (more developed)
        if (node.description && node.description.length > 50) {
            score += 0.1;
        }

        // Boost for shallow depth (closer to root = more structural importance)
        const depth = this.getNodeDepth(node, allNodes);
        if (depth <= 2) score += 0.15;
        else if (depth <= 4) score += 0.05;

        // Cap at 1.0
        return Math.min(1.0, score);
    },

    /**
     * Get the depth of a node in the tree
     */
    getNodeDepth(node, allNodes) {
        let depth = 0;
        let current = node;
        const nodeMap = new Map(allNodes.map(n => [n.id, n]));

        while (current && current.parentId && nodeMap.has(current.parentId)) {
            depth++;
            current = nodeMap.get(current.parentId);
            if (depth > 20) break; // Prevent infinite loops
        }

        return depth;
    },

    // ═══════════════════════════════════════════════════════════════════
    // DUPLICATE DETECTION
    // ═══════════════════════════════════════════════════════════════════

    async detectDuplicates(nodes) {
        const duplicates = [];
        const checked = new Set();

        // Group nodes by parent for same-parent duplicate detection
        const nodesByParent = new Map();
        for (const node of nodes) {
            const parentId = node.parentId || 'root';
            if (!nodesByParent.has(parentId)) {
                nodesByParent.set(parentId, []);
            }
            nodesByParent.get(parentId).push(node);
        }

        // Check for duplicates within same parent (most likely issues)
        for (const [parentId, siblings] of nodesByParent) {
            if (siblings.length < 2) continue;

            for (let i = 0; i < siblings.length; i++) {
                for (let j = i + 1; j < siblings.length; j++) {
                    const nodeA = siblings[i];
                    const nodeB = siblings[j];

                    // Skip protected nodes - never suggest merging core concepts
                    if (this.isProtectedNode(nodeA) || this.isProtectedNode(nodeB)) {
                        continue;
                    }

                    const pairKey = [nodeA.id, nodeB.id].sort().join('::');

                    if (checked.has(pairKey)) continue;
                    checked.add(pairKey);

                    const similarity = await this.calculateNodeSimilarity(nodeA, nodeB);

                    if (similarity.combined >= this.config.thresholds.similarityForDuplicate) {
                        // Calculate importance to adjust severity
                        const importanceA = this.calculateNodeImportance(nodeA, nodes);
                        const importanceB = this.calculateNodeImportance(nodeB, nodes);
                        const avgImportance = (importanceA + importanceB) / 2;

                        // Lower severity for more important nodes
                        let severity = similarity.combined > 0.95 ? 'high' : 'medium';
                        if (avgImportance > 0.7) severity = 'low'; // Important nodes get lower severity

                        duplicates.push({
                            type: 'duplicate',
                            severity,
                            nodeA: { id: nodeA.id, label: nodeA.label, importance: importanceA },
                            nodeB: { id: nodeB.id, label: nodeB.label, importance: importanceB },
                            parentId,
                            similarity: similarity.combined,
                            nameSimilarity: similarity.name,
                            semanticSimilarity: similarity.semantic,
                            sameParent: true,
                            suggestion: `Consider merging "${nodeA.label}" and "${nodeB.label}" - they appear to be ${Math.round(similarity.combined * 100)}% similar`
                        });
                    }
                }
            }
        }

        // Also check for semantic duplicates across entire map (lower threshold)
        if (this.encoder && nodes.length <= 500) { // Limit for performance
            const crossMapDuplicates = await this.detectCrossMapDuplicates(nodes, checked);
            duplicates.push(...crossMapDuplicates);
        }

        // Sort by severity and similarity
        duplicates.sort((a, b) => {
            if (a.severity !== b.severity) {
                return a.severity === 'high' ? -1 : 1;
            }
            return b.similarity - a.similarity;
        });

        return duplicates.slice(0, 20); // Limit to top 20
    },

    async calculateNodeSimilarity(nodeA, nodeB) {
        const result = {
            name: 0,
            semantic: 0,
            combined: 0
        };

        // Name similarity using Levenshtein distance
        result.name = this.calculateNameSimilarity(nodeA.label, nodeB.label);

        // Semantic similarity using embeddings
        if (this.encoder) {
            try {
                const textA = `${nodeA.label} ${nodeA.description || ''}`.trim();
                const textB = `${nodeB.label} ${nodeB.description || ''}`.trim();

                const embeddings = await this.encoder.embed([textA, textB]);
                const embArray = await embeddings.array();
                result.semantic = this.cosineSimilarity(embArray[0], embArray[1]);
                embeddings.dispose();
            } catch (e) {
                // Fall back to name only
                result.semantic = result.name;
            }
        } else {
            result.semantic = result.name;
        }

        // Combined score (weighted)
        result.combined = result.name * 0.4 + result.semantic * 0.6;

        return result;
    },

    calculateNameSimilarity(strA, strB) {
        if (!strA || !strB) return 0;

        const a = strA.toLowerCase().trim();
        const b = strB.toLowerCase().trim();

        if (a === b) return 1.0;

        // Levenshtein distance
        const matrix = [];
        for (let i = 0; i <= a.length; i++) {
            matrix[i] = [i];
        }
        for (let j = 0; j <= b.length; j++) {
            matrix[0][j] = j;
        }
        for (let i = 1; i <= a.length; i++) {
            for (let j = 1; j <= b.length; j++) {
                const cost = a[i - 1] === b[j - 1] ? 0 : 1;
                matrix[i][j] = Math.min(
                    matrix[i - 1][j] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j - 1] + cost
                );
            }
        }

        const distance = matrix[a.length][b.length];
        const maxLen = Math.max(a.length, b.length);
        return 1 - (distance / maxLen);
    },

    async detectCrossMapDuplicates(nodes, alreadyChecked) {
        const duplicates = [];

        // Get embeddings for all nodes
        const nodeTexts = nodes.map(n => `${n.label} ${n.description || ''}`.trim());

        try {
            const embeddings = await this.encoder.embed(nodeTexts);
            const embArray = await embeddings.array();
            embeddings.dispose();

            // Compare all pairs (O(n^2) but limited to 500 nodes)
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const nodeA = nodes[i];
                    const nodeB = nodes[j];

                    // Skip same parent (already checked)
                    if (nodeA.parentId === nodeB.parentId) continue;

                    const pairKey = [nodeA.id, nodeB.id].sort().join('::');
                    if (alreadyChecked.has(pairKey)) continue;

                    const semantic = this.cosineSimilarity(embArray[i], embArray[j]);

                    // Higher threshold for cross-map (0.9)
                    if (semantic >= 0.9) {
                        duplicates.push({
                            type: 'cross-map-duplicate',
                            severity: semantic > 0.95 ? 'high' : 'medium',
                            nodeA: { id: nodeA.id, label: nodeA.label },
                            nodeB: { id: nodeB.id, label: nodeB.label },
                            similarity: semantic,
                            semanticSimilarity: semantic,
                            sameParent: false,
                            suggestion: `"${nodeA.label}" and "${nodeB.label}" are semantically similar but in different locations - consider linking or consolidating`
                        });
                    }
                }
            }
        } catch (e) {
            console.warn('Cross-map duplicate detection failed:', e);
        }

        return duplicates;
    },

    cosineSimilarity(vecA, vecB) {
        if (!vecA || !vecB || vecA.length !== vecB.length) return 0;

        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }

        const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
        return magnitude === 0 ? 0 : dotProduct / magnitude;
    },

    // ═══════════════════════════════════════════════════════════════════
    // STRUCTURAL ANALYSIS
    // ═══════════════════════════════════════════════════════════════════

    analyzeStructure(nodes, rootData) {
        const issues = [];
        const nodeMap = new Map(nodes.map(n => [n.id, n]));

        // Build tree statistics
        const stats = this.buildTreeStats(nodes, rootData);

        // 1. Check for imbalanced branches
        if (rootData.children && rootData.children.length > 1) {
            const childSizes = rootData.children.map(child =>
                this.countDescendants(child, nodeMap)
            );

            const maxSize = Math.max(...childSizes);
            const minSize = Math.min(...childSizes);

            if (maxSize - minSize > this.config.thresholds.maxChildrenImbalance) {
                const largestIdx = childSizes.indexOf(maxSize);
                const smallestIdx = childSizes.indexOf(minSize);

                issues.push({
                    type: 'imbalanced-branches',
                    severity: maxSize / Math.max(minSize, 1) > 10 ? 'high' : 'medium',
                    description: `Branch sizes are imbalanced`,
                    details: {
                        largest: {
                            label: rootData.children[largestIdx].label,
                            size: maxSize
                        },
                        smallest: {
                            label: rootData.children[smallestIdx].label,
                            size: minSize
                        }
                    },
                    suggestion: `Consider breaking down "${rootData.children[largestIdx].label}" (${maxSize} nodes) or expanding "${rootData.children[smallestIdx].label}" (${minSize} nodes)`
                });
            }
        }

        // 2. Check for overly deep branches
        if (stats.maxDepth > this.config.thresholds.maxDepth) {
            issues.push({
                type: 'excessive-depth',
                severity: 'medium',
                description: `Tree has excessive depth (${stats.maxDepth} levels)`,
                details: { maxDepth: stats.maxDepth, deepestPath: stats.deepestPath },
                suggestion: 'Consider flattening deeply nested structures for better navigation'
            });
        }

        // 3. Check for orphaned nodes (nodes with missing parents)
        for (const node of nodes) {
            if (node.parentId && !nodeMap.has(node.parentId) && node.parentId !== rootData.id) {
                issues.push({
                    type: 'orphaned-node',
                    severity: 'high',
                    nodeId: node.id,
                    label: node.label,
                    description: `Node "${node.label}" has a missing parent`,
                    suggestion: 'Move this node to an appropriate location or delete if no longer needed'
                });
            }
        }

        // 4. Check for empty branches (nodes with no content and no children)
        for (const node of nodes) {
            // Skip protected/important nodes - they may be placeholders for future content
            if (this.isProtectedNode(node)) continue;

            const hasChildren = nodes.some(n => n.parentId === node.id);
            const hasContent = node.description && node.description.trim().length > 0;

            if (!hasChildren && !hasContent && node.id !== rootData.id) {
                // Check if it's old (not recently created)
                const createdAt = node.createdAt || node.timestamp || 0;
                const age = Date.now() - createdAt;

                if (age > this.config.thresholds.orphanAgeMs) {
                    // Calculate importance - never suggest removing important empty nodes
                    const importance = this.calculateNodeImportance(node, nodes);
                    if (importance > 0.6) continue; // Skip important nodes even if empty

                    issues.push({
                        type: 'empty-leaf',
                        severity: 'low',
                        nodeId: node.id,
                        label: node.label,
                        importance,
                        description: `"${node.label}" has no content or children`,
                        suggestion: 'Add content to this node or consider removing it'
                    });
                }
            }
        }

        // 5. Check for overly long labels
        for (const node of nodes) {
            if (node.label && node.label.length > 60) {
                issues.push({
                    type: 'long-label',
                    severity: 'low',
                    nodeId: node.id,
                    label: node.label,
                    description: `Label is very long (${node.label.length} chars)`,
                    suggestion: 'Consider shortening the label and moving details to description'
                });
            }
        }

        // 6. Check for flat structure (too many children under one node)
        for (const node of nodes) {
            const directChildren = nodes.filter(n => n.parentId === node.id);
            if (directChildren.length > 15) {
                issues.push({
                    type: 'flat-structure',
                    severity: 'medium',
                    nodeId: node.id,
                    label: node.label,
                    childCount: directChildren.length,
                    description: `"${node.label}" has ${directChildren.length} direct children`,
                    suggestion: 'Consider grouping some children into subcategories'
                });
            }
        }

        // Sort by severity
        const severityOrder = { high: 0, medium: 1, low: 2 };
        issues.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);

        return issues.slice(0, 20); // Limit to top 20
    },

    buildTreeStats(nodes, rootData) {
        const stats = {
            totalNodes: nodes.length,
            maxDepth: 0,
            deepestPath: [],
            avgBranchFactor: 0,
            leafNodes: 0
        };

        const nodeMap = new Map(nodes.map(n => [n.id, n]));

        // Calculate depth for each node
        const calculateDepth = (nodeId, path = []) => {
            const node = nodeMap.get(nodeId) || (nodeId === rootData.id ? rootData : null);
            if (!node) return 0;

            const currentPath = [...path, node.label];
            let maxChildDepth = 0;

            const children = node.children || nodes.filter(n => n.parentId === nodeId);

            if (children.length === 0) {
                stats.leafNodes++;
                if (currentPath.length > stats.maxDepth) {
                    stats.maxDepth = currentPath.length;
                    stats.deepestPath = currentPath;
                }
                return 1;
            }

            for (const child of children) {
                const childId = child.id || child;
                const childDepth = calculateDepth(childId, currentPath);
                maxChildDepth = Math.max(maxChildDepth, childDepth);
            }

            return maxChildDepth + 1;
        };

        calculateDepth(rootData.id);

        // Calculate average branch factor
        const nodesWithChildren = nodes.filter(n =>
            nodes.some(c => c.parentId === n.id)
        ).length;

        if (nodesWithChildren > 0) {
            stats.avgBranchFactor = (nodes.length - 1) / nodesWithChildren;
        }

        return stats;
    },

    countDescendants(node, nodeMap) {
        let count = 1;
        const children = node.children || [];

        for (const child of children) {
            const childNode = nodeMap.get(child.id || child) || child;
            count += this.countDescendants(childNode, nodeMap);
        }

        return count;
    },

    // ═══════════════════════════════════════════════════════════════════
    // GAP DETECTION
    // ═══════════════════════════════════════════════════════════════════

    async detectGaps(nodes) {
        const gaps = [];

        // 1. Find nodes that could be connected but aren't
        if (this.encoder && nodes.length >= 5) {
            const potentialConnections = await this.findPotentialConnections(nodes);
            gaps.push(...potentialConnections);
        }

        // 2. Find incomplete patterns
        const incompletePatterns = this.findIncompletePatterns(nodes);
        gaps.push(...incompletePatterns);

        return gaps.slice(0, 15); // Limit
    },

    async findPotentialConnections(nodes) {
        const connections = [];

        try {
            // Get embeddings for all nodes
            const nodeTexts = nodes.map(n => `${n.label} ${n.description || ''}`.trim());
            const embeddings = await this.encoder.embed(nodeTexts);
            const embArray = await embeddings.array();
            embeddings.dispose();

            // Find nodes that are semantically related but not structurally connected
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const nodeA = nodes[i];
                    const nodeB = nodes[j];

                    // Skip if already connected (parent-child or siblings)
                    if (nodeA.parentId === nodeB.id || nodeB.parentId === nodeA.id) continue;
                    if (nodeA.parentId === nodeB.parentId) continue;

                    const similarity = this.cosineSimilarity(embArray[i], embArray[j]);

                    // Moderate similarity suggests potential connection
                    if (similarity >= 0.7 && similarity < 0.9) {
                        connections.push({
                            type: 'potential-connection',
                            severity: 'low',
                            nodeA: { id: nodeA.id, label: nodeA.label },
                            nodeB: { id: nodeB.id, label: nodeB.label },
                            similarity,
                            description: `"${nodeA.label}" and "${nodeB.label}" seem related`,
                            suggestion: 'Consider creating a link or cross-reference between these nodes'
                        });
                    }
                }
            }

            // Sort by similarity and limit
            connections.sort((a, b) => b.similarity - a.similarity);

        } catch (e) {
            console.warn('Potential connection detection failed:', e);
        }

        return connections.slice(0, 10);
    },

    findIncompletePatterns(nodes) {
        const patterns = [];

        // Look for common incomplete patterns
        const commonPatterns = [
            { prefix: 'pros', complement: 'cons' },
            { prefix: 'advantages', complement: 'disadvantages' },
            { prefix: 'input', complement: 'output' },
            { prefix: 'before', complement: 'after' },
            { prefix: 'problem', complement: 'solution' },
            { prefix: 'question', complement: 'answer' },
            { prefix: 'cause', complement: 'effect' },
            { prefix: 'goal', complement: 'action' }
        ];

        const nodeLabels = nodes.map(n => n.label.toLowerCase());
        const nodeMap = new Map(nodes.map(n => [n.label.toLowerCase(), n]));

        for (const pattern of commonPatterns) {
            const hasPrefix = nodeLabels.some(l => l.includes(pattern.prefix));
            const hasComplement = nodeLabels.some(l => l.includes(pattern.complement));

            if (hasPrefix && !hasComplement) {
                patterns.push({
                    type: 'incomplete-pattern',
                    severity: 'low',
                    pattern: pattern.prefix,
                    missing: pattern.complement,
                    description: `Found "${pattern.prefix}" without corresponding "${pattern.complement}"`,
                    suggestion: `Consider adding a "${pattern.complement}" node to complement your "${pattern.prefix}" node`
                });
            }
        }

        return patterns;
    },

    // ═══════════════════════════════════════════════════════════════════
    // SUGGESTION GENERATION
    // ═══════════════════════════════════════════════════════════════════

    generateSuggestions(report) {
        const suggestions = [];

        // Convert duplicates to suggestions
        for (const dup of report.duplicates) {
            suggestions.push({
                id: this.generateId(),
                type: 'merge',
                priority: dup.severity === 'high' ? 'high' : 'medium',
                title: `Merge duplicate nodes`,
                description: dup.suggestion,
                data: dup,
                action: 'consolidate',
                timestamp: Date.now()
            });
        }

        // Convert structural issues to suggestions
        for (const issue of report.structuralIssues) {
            suggestions.push({
                id: this.generateId(),
                type: 'restructure',
                priority: issue.severity === 'high' ? 'high' :
                         issue.severity === 'medium' ? 'medium' : 'low',
                title: issue.type.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                description: issue.suggestion,
                data: issue,
                action: issue.type,
                timestamp: Date.now()
            });
        }

        // Convert gaps to suggestions
        for (const gap of report.gaps) {
            suggestions.push({
                id: this.generateId(),
                type: 'connection',
                priority: 'low',
                title: gap.type === 'potential-connection' ? 'Add connection' : 'Complete pattern',
                description: gap.suggestion,
                data: gap,
                action: gap.type,
                timestamp: Date.now()
            });
        }

        return suggestions;
    },

    // ═══════════════════════════════════════════════════════════════════
    // PROCESSING & AUTO-ACTIONS
    // ═══════════════════════════════════════════════════════════════════

    async processFindings(report) {
        for (const suggestion of report.suggestions) {
            if (this.config.autoApply && this.canAutoApply(suggestion)) {
                await this.applySuggestion(suggestion);
            } else {
                this.queueSuggestion(suggestion);
            }
        }
    },

    canAutoApply(suggestion) {
        // Only auto-apply safe, non-destructive changes
        switch (suggestion.action) {
            case 'long-label':
                return false; // Needs human judgment
            case 'empty-leaf':
                return this.config.features.autoConsolidation;
            case 'consolidate':
                return this.config.features.autoConsolidation;
            default:
                return false;
        }
    },

    queueSuggestion(suggestion) {
        // Check for duplicates in queue
        const isDuplicate = this.maintenanceQueue.some(s =>
            s.type === suggestion.type &&
            JSON.stringify(s.data) === JSON.stringify(suggestion.data)
        );

        if (!isDuplicate) {
            this.maintenanceQueue.push(suggestion);

            // Trim queue if over limit
            if (this.maintenanceQueue.length > this.maxQueueSize) {
                // Remove oldest low-priority items
                this.maintenanceQueue.sort((a, b) => {
                    const priorityOrder = { high: 0, medium: 1, low: 2 };
                    return priorityOrder[a.priority] - priorityOrder[b.priority];
                });
                this.maintenanceQueue = this.maintenanceQueue.slice(0, this.maxQueueSize);
            }

            this.emitEvent('suggestionQueued', { suggestion });
        }
    },

    async applySuggestion(suggestion) {
        const store = window.app?.store;
        if (!store) return false;

        try {
            switch (suggestion.action) {
                case 'consolidate':
                    return await this.consolidateNodes(suggestion.data);

                case 'empty-leaf':
                    // Remove empty leaf node
                    if (suggestion.data.nodeId) {
                        store.removeNode(suggestion.data.nodeId);
                        this.stats.autoFixed++;
                        this.log(`Auto-removed empty node: ${suggestion.data.label}`);
                        return true;
                    }
                    break;

                default:
                    return false;
            }
        } catch (error) {
            console.error('Failed to apply suggestion:', error);
            return false;
        }
    },

    async consolidateNodes(duplicate) {
        const store = window.app?.store;
        if (!store || !duplicate.nodeA || !duplicate.nodeB) return false;

        try {
            const nodeA = store.findNode(duplicate.nodeA.id);
            const nodeB = store.findNode(duplicate.nodeB.id);

            if (!nodeA || !nodeB) return false;

            // Merge: Keep nodeA, move nodeB's children to nodeA, delete nodeB
            const childrenOfB = store.getAllNodes().filter(n => n.parentId === nodeB.id);

            for (const child of childrenOfB) {
                store.moveNode(child.id, nodeA.id);
            }

            // Merge descriptions
            if (nodeB.description && !nodeA.description?.includes(nodeB.description)) {
                const mergedDesc = [nodeA.description, nodeB.description]
                    .filter(Boolean)
                    .join('\n\n---\n\n');
                store.updateNode(nodeA.id, { description: mergedDesc });
            }

            // Remove nodeB
            store.removeNode(nodeB.id);

            this.stats.autoFixed++;
            this.log(`Consolidated "${nodeA.label}" and "${nodeB.label}"`);

            return true;

        } catch (error) {
            console.error('Consolidation failed:', error);
            return false;
        }
    },

    // ═══════════════════════════════════════════════════════════════════
    // QUEUE MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════

    getQueue(filter = {}) {
        let queue = [...this.maintenanceQueue];

        if (filter.type) {
            queue = queue.filter(s => s.type === filter.type);
        }
        if (filter.priority) {
            queue = queue.filter(s => s.priority === filter.priority);
        }

        // Sort by priority then timestamp
        const priorityOrder = { high: 0, medium: 1, low: 2 };
        queue.sort((a, b) => {
            if (a.priority !== b.priority) {
                return priorityOrder[a.priority] - priorityOrder[b.priority];
            }
            return b.timestamp - a.timestamp;
        });

        return queue;
    },

    async approveSuggestion(suggestionId) {
        const index = this.maintenanceQueue.findIndex(s => s.id === suggestionId);
        if (index === -1) return false;

        const suggestion = this.maintenanceQueue[index];
        const result = await this.applySuggestion(suggestion);

        if (result) {
            this.maintenanceQueue.splice(index, 1);
            this.stats.userApproved++;
            this.saveToStorage();
            this.emitEvent('suggestionApproved', { suggestion });
        }

        return result;
    },

    dismissSuggestion(suggestionId) {
        const index = this.maintenanceQueue.findIndex(s => s.id === suggestionId);
        if (index === -1) return false;

        const suggestion = this.maintenanceQueue.splice(index, 1)[0];
        this.stats.userDismissed++;
        this.saveToStorage();
        this.emitEvent('suggestionDismissed', { suggestion });

        return true;
    },

    clearQueue() {
        const count = this.maintenanceQueue.length;
        this.maintenanceQueue = [];
        this.saveToStorage();
        this.log(`Cleared ${count} items from queue`);
        return count;
    },

    // ═══════════════════════════════════════════════════════════════════
    // MAINTENANCE REPORT
    // ═══════════════════════════════════════════════════════════════════

    async generateReport() {
        const store = window.app?.store;
        if (!store) return null;

        const report = await this.runMaintenance({ skipQueue: true });

        return {
            ...report,
            queueSize: this.maintenanceQueue.length,
            stats: { ...this.stats },
            health: this.calculateMapHealth(report),
            recommendations: this.generateRecommendations(report)
        };
    },

    calculateMapHealth(report) {
        let score = 100;

        // Deduct for issues
        score -= report.duplicates.length * 5;
        score -= report.structuralIssues.filter(i => i.severity === 'high').length * 10;
        score -= report.structuralIssues.filter(i => i.severity === 'medium').length * 5;
        score -= report.structuralIssues.filter(i => i.severity === 'low').length * 2;

        // Ensure score is between 0 and 100
        return Math.max(0, Math.min(100, score));
    },

    generateRecommendations(report) {
        const recommendations = [];

        if (report.duplicates.length > 5) {
            recommendations.push({
                priority: 'high',
                text: 'Your map has many duplicate nodes. Consider a cleanup session to merge similar concepts.'
            });
        }

        if (report.structuralIssues.some(i => i.type === 'imbalanced-branches')) {
            recommendations.push({
                priority: 'medium',
                text: 'Some branches are much larger than others. Consider reorganizing for balance.'
            });
        }

        if (report.structuralIssues.some(i => i.type === 'excessive-depth')) {
            recommendations.push({
                priority: 'medium',
                text: 'Your map is quite deep. Consider flattening some hierarchies for easier navigation.'
            });
        }

        if (report.gaps.length > 5) {
            recommendations.push({
                priority: 'low',
                text: 'There are potential connections between nodes that could strengthen your knowledge structure.'
            });
        }

        return recommendations;
    },

    // ═══════════════════════════════════════════════════════════════════
    // STORAGE & UTILITIES
    // ═══════════════════════════════════════════════════════════════════

    generateId() {
        return `maint_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    },

    log(message) {
        const entry = {
            timestamp: Date.now(),
            message,
            time: new Date().toLocaleTimeString()
        };

        this.activityLog.unshift(entry);
        if (this.activityLog.length > this.maxLogEntries) {
            this.activityLog = this.activityLog.slice(0, this.maxLogEntries);
        }

        console.log(`🔧 [${entry.time}] ${message}`);
    },

    saveToStorage() {
        try {
            const data = {
                version: this.VERSION,
                config: this.config,
                stats: this.stats,
                maintenanceQueue: this.maintenanceQueue,
                lastMaintenanceTime: this.lastMaintenanceTime,
                activityLog: this.activityLog.slice(0, 20),
                timestamp: Date.now()
            };
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data));
        } catch (error) {
            console.warn('MapMaintenanceDaemon save failed:', error);
        }
    },

    loadFromStorage() {
        try {
            const data = localStorage.getItem(this.STORAGE_KEY);
            if (data) {
                const parsed = JSON.parse(data);
                this.config = { ...this.config, ...parsed.config };
                this.stats = { ...this.stats, ...parsed.stats };
                this.maintenanceQueue = parsed.maintenanceQueue || [];
                this.lastMaintenanceTime = parsed.lastMaintenanceTime || 0;
                this.activityLog = parsed.activityLog || [];
                console.log('✓ MapMaintenanceDaemon loaded saved state');
            }
        } catch (error) {
            console.warn('MapMaintenanceDaemon load failed:', error);
        }
    },

    emitEvent(eventName, detail = {}) {
        const event = new CustomEvent(`mapMaintenance:${eventName}`, {
            detail: { ...detail, timestamp: Date.now() }
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
            lastMaintenanceTime: this.lastMaintenanceTime,
            lastMaintenanceAgo: this.lastMaintenanceTime
                ? Math.round((Date.now() - this.lastMaintenanceTime) / 1000 / 60) + ' min ago'
                : 'never',
            queueSize: this.maintenanceQueue.length,
            config: this.config,
            stats: this.stats,
            hasEncoder: !!this.encoder
        };
    },

    setConfig(updates) {
        if (updates.features) {
            this.config.features = { ...this.config.features, ...updates.features };
            delete updates.features;
        }
        if (updates.thresholds) {
            this.config.thresholds = { ...this.config.thresholds, ...updates.thresholds };
            delete updates.thresholds;
        }
        this.config = { ...this.config, ...updates };
        this.saveToStorage();
        this.emitEvent('configUpdated', { config: this.config });
    }
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MapMaintenanceDaemon;
}

console.log('🔧 MapMaintenanceDaemon loaded. Call MapMaintenanceDaemon.init() to initialize.');
