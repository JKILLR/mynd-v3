/**
 * MYND Goal System
 * Manifestation architecture for turning desires into reality
 *
 * This module handles:
 * - Goal nodes as first-class citizens
 * - Path discovery between current and desired states
 * - Milestone generation and tracking
 * - Progress visualization
 * - Daily guidance and reflection
 */

// ═══════════════════════════════════════════════════════════════════
// GOAL REGISTRY - Central management for all goals
// ═══════════════════════════════════════════════════════════════════

const GoalRegistry = {
    goals: new Map(),
    milestones: new Map(),
    paths: new Map(),

    // Initialize from storage
    async init() {
        try {
            // NeuralDB may not be available yet (defined in main script)
            if (typeof NeuralDB === 'undefined') {
                console.log('Goal registry: NeuralDB not available yet, using empty state');
                return;
            }
            const saved = await NeuralDB.load('goal-registry');
            if (saved) {
                if (saved.goals) {
                    this.goals = new Map(saved.goals);
                }
                if (saved.milestones) {
                    this.milestones = new Map(saved.milestones);
                }
                console.log(`✓ Loaded ${this.goals.size} goals`);
            }
        } catch (error) {
            console.error('Failed to load goals:', error);
        }
    },

    // Save to storage
    async save() {
        try {
            // NeuralDB may not be available
            if (typeof NeuralDB === 'undefined') {
                console.warn('Goal registry: Cannot save, NeuralDB not available');
                return;
            }
            await NeuralDB.save('goal-registry', {
                goals: Array.from(this.goals.entries()),
                milestones: Array.from(this.milestones.entries()),
                savedAt: Date.now()
            });
        } catch (error) {
            console.error('Failed to save goals:', error);
        }
    },

    // Create a new goal
    createGoal(data) {
        const goal = {
            id: `goal_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
            type: 'goal',
            label: data.label,
            description: data.description || '',

            // Goal-specific properties
            targetDate: data.targetDate || null,
            priority: data.priority || 'medium', // high, medium, low
            status: 'active', // active, achieved, paused, abandoned

            // Manifestation properties
            desiredState: data.desiredState || '',
            currentState: data.currentState || '',
            whyItMatters: data.whyItMatters || '',

            // Progress tracking
            progress: 0,
            milestoneIds: [],

            // Metadata
            createdAt: Date.now(),
            updatedAt: Date.now(),
            lastReflection: null,
            insights: []
        };

        this.goals.set(goal.id, goal);
        this.save();

        console.log(`✨ Goal created: "${goal.label}"`);
        return goal;
    },

    // Get a goal by ID
    getGoal(goalId) {
        return this.goals.get(goalId);
    },

    // Get all active goals
    getActiveGoals() {
        return Array.from(this.goals.values())
            .filter(g => g.status === 'active')
            .sort((a, b) => {
                // Sort by priority, then by creation date
                const priorityOrder = { high: 0, medium: 1, low: 2 };
                if (priorityOrder[a.priority] !== priorityOrder[b.priority]) {
                    return priorityOrder[a.priority] - priorityOrder[b.priority];
                }
                return b.createdAt - a.createdAt;
            });
    },

    // Update goal progress
    updateProgress(goalId) {
        const goal = this.goals.get(goalId);
        if (!goal) return;

        // Calculate progress based on completed milestones
        const milestones = goal.milestoneIds
            .map(id => this.milestones.get(id))
            .filter(Boolean);

        if (milestones.length === 0) {
            goal.progress = 0;
        } else {
            const completed = milestones.filter(m => m.status === 'completed').length;
            goal.progress = completed / milestones.length;
        }

        goal.updatedAt = Date.now();
        this.save();

        return goal.progress;
    },

    // Mark goal as achieved
    achieveGoal(goalId) {
        const goal = this.goals.get(goalId);
        if (!goal) return;

        goal.status = 'achieved';
        goal.progress = 1;
        goal.achievedAt = Date.now();
        goal.updatedAt = Date.now();

        this.save();
        console.log(`🎉 Goal achieved: "${goal.label}"`);

        return goal;
    }
};

// ═══════════════════════════════════════════════════════════════════
// MILESTONE GENERATOR - Break goals into achievable steps
// ═══════════════════════════════════════════════════════════════════

const MilestoneGenerator = {
    // Generate milestones for a goal using AI
    async generateMilestones(goalId, store) {
        const goal = GoalRegistry.getGoal(goalId);
        if (!goal) return [];

        // TODO: Use Claude to generate intelligent milestones
        // For now, create placeholder milestones
        const milestones = [];

        // This will be replaced with AI-generated milestones
        const placeholders = [
            { label: 'Define success criteria', order: 1 },
            { label: 'Identify required resources', order: 2 },
            { label: 'Create action plan', order: 3 },
            { label: 'Take first step', order: 4 },
            { label: 'Review and adjust', order: 5 }
        ];

        for (const p of placeholders) {
            const milestone = {
                id: `milestone_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
                type: 'milestone',
                goalId: goalId,
                label: p.label,
                order: p.order,
                status: 'pending', // pending, in_progress, completed
                estimatedEffort: null,
                completedAt: null,
                evidence: [],
                createdAt: Date.now()
            };

            milestones.push(milestone);
            GoalRegistry.milestones.set(milestone.id, milestone);
        }

        // Update goal with milestone IDs
        goal.milestoneIds = milestones.map(m => m.id);
        GoalRegistry.save();

        return milestones;
    },

    // Complete a milestone
    completeMilestone(milestoneId, evidence = null) {
        const milestone = GoalRegistry.milestones.get(milestoneId);
        if (!milestone) return;

        milestone.status = 'completed';
        milestone.completedAt = Date.now();
        if (evidence) {
            milestone.evidence.push(evidence);
        }

        // Update parent goal progress
        GoalRegistry.updateProgress(milestone.goalId);
        GoalRegistry.save();

        console.log(`✓ Milestone completed: "${milestone.label}"`);
        return milestone;
    }
};

// ═══════════════════════════════════════════════════════════════════
// PATH FINDER - Discover connections from current to desired state
// ═══════════════════════════════════════════════════════════════════

const PathFinder = {
    // Find paths from current reality to goal
    async discoverPaths(goalId, store) {
        const goal = GoalRegistry.getGoal(goalId);
        if (!goal) return null;

        // TODO: Implement semantic path finding using:
        // - User's existing mind map nodes
        // - Neural network embeddings for similarity
        // - Claude for bridge concept generation

        const path = {
            goalId,
            nodes: [],           // Existing nodes that connect to goal
            bridgeConcepts: [],  // AI-generated intermediate steps
            connections: [],     // Edges between nodes
            discoveredAt: Date.now()
        };

        GoalRegistry.paths.set(goalId, path);
        return path;
    },

    // Find relevant existing nodes for a goal
    async findRelevantNodes(goalId, store) {
        const goal = GoalRegistry.getGoal(goalId);
        if (!goal || !neuralNet.isReady) return [];

        // Use neural network to find semantically similar nodes
        const goalText = `${goal.label}. ${goal.description}. ${goal.desiredState}`;
        const similarNodes = await neuralNet.findSimilarNodes(goalText, store, 10);

        return similarNodes;
    },

    // Generate bridge concepts (what's missing between current and goal)
    async generateBridgeConcepts(goalId, store) {
        const goal = GoalRegistry.getGoal(goalId);
        if (!goal) return [];

        // TODO: Use Claude to identify gaps and generate bridge concepts
        // "What skills/resources/connections are needed to get from A to B?"

        return [];
    }
};

// ═══════════════════════════════════════════════════════════════════
// DAILY GUIDANCE - Smart suggestions and reflections
// ═══════════════════════════════════════════════════════════════════

const DailyGuidance = {
    // Get today's suggested focus
    async getSuggestedFocus() {
        const activeGoals = GoalRegistry.getActiveGoals();
        if (activeGoals.length === 0) return null;

        // TODO: Use AI to determine optimal focus based on:
        // - Goal priority and deadline
        // - Recent activity
        // - User energy patterns (from MetaLearner)
        // - Milestone dependencies

        // For now, return highest priority goal
        const topGoal = activeGoals[0];
        const nextMilestone = this.getNextMilestone(topGoal.id);

        return {
            goal: topGoal,
            milestone: nextMilestone,
            reason: 'Highest priority active goal'
        };
    },

    // Get next incomplete milestone for a goal
    getNextMilestone(goalId) {
        const goal = GoalRegistry.getGoal(goalId);
        if (!goal) return null;

        for (const milestoneId of goal.milestoneIds) {
            const milestone = GoalRegistry.milestones.get(milestoneId);
            if (milestone && milestone.status !== 'completed') {
                return milestone;
            }
        }
        return null;
    },

    // Record a reflection
    async recordReflection(goalId, reflection) {
        const goal = GoalRegistry.getGoal(goalId);
        if (!goal) return;

        goal.insights.push({
            text: reflection,
            timestamp: Date.now()
        });
        goal.lastReflection = Date.now();
        goal.updatedAt = Date.now();

        // Keep last 50 insights
        if (goal.insights.length > 50) {
            goal.insights = goal.insights.slice(-50);
        }

        GoalRegistry.save();

        // Store in semantic memory
        if (typeof semanticMemory !== 'undefined') {
            semanticMemory.addMemory('goal_reflection', reflection, {
                goalId,
                goalLabel: goal.label
            });
        }
    },

    // Check for procrastination patterns
    async checkProcrastination(userId) {
        // TODO: Use MetaLearner to detect avoidance patterns
        // Look for goals not visited in X days
        // Identify blocked or stuck milestones

        const warnings = [];
        const now = Date.now();
        const fiveDaysMs = 5 * 24 * 60 * 60 * 1000;

        for (const goal of GoalRegistry.getActiveGoals()) {
            if (goal.updatedAt && (now - goal.updatedAt) > fiveDaysMs) {
                warnings.push({
                    type: 'neglected_goal',
                    goalId: goal.id,
                    goalLabel: goal.label,
                    daysSinceUpdate: Math.floor((now - goal.updatedAt) / (24 * 60 * 60 * 1000))
                });
            }
        }

        return warnings;
    }
};

// ═══════════════════════════════════════════════════════════════════
// PROGRESS TRACKER - Visual progress and celebrations
// ═══════════════════════════════════════════════════════════════════

const ProgressTracker = {
    // Calculate overall progress toward all goals
    getOverallProgress() {
        const activeGoals = GoalRegistry.getActiveGoals();
        if (activeGoals.length === 0) return 0;

        const totalProgress = activeGoals.reduce((sum, g) => sum + g.progress, 0);
        return totalProgress / activeGoals.length;
    },

    // Get progress summary for dashboard
    getProgressSummary() {
        const goals = Array.from(GoalRegistry.goals.values());

        return {
            total: goals.length,
            active: goals.filter(g => g.status === 'active').length,
            achieved: goals.filter(g => g.status === 'achieved').length,
            paused: goals.filter(g => g.status === 'paused').length,
            overallProgress: this.getOverallProgress(),
            topGoals: GoalRegistry.getActiveGoals().slice(0, 3).map(g => ({
                id: g.id,
                label: g.label,
                progress: g.progress,
                priority: g.priority
            }))
        };
    },

    // Predict completion date based on progress velocity
    predictCompletionDate(goalId) {
        const goal = GoalRegistry.getGoal(goalId);
        if (!goal || goal.progress === 0) return null;

        // Simple linear prediction based on progress rate
        const daysSinceCreation = (Date.now() - goal.createdAt) / (24 * 60 * 60 * 1000);
        if (daysSinceCreation < 1) return null;

        const progressPerDay = goal.progress / daysSinceCreation;
        if (progressPerDay === 0) return null;

        const remainingProgress = 1 - goal.progress;
        const daysToComplete = remainingProgress / progressPerDay;

        return new Date(Date.now() + daysToComplete * 24 * 60 * 60 * 1000);
    }
};

// ═══════════════════════════════════════════════════════════════════
// GOAL UI HELPERS - Interface utilities
// ═══════════════════════════════════════════════════════════════════

const GoalUI = {
    // Show goal creation wizard
    showCreateWizard() {
        // TODO: Implement modal wizard UI
        console.log('Goal creation wizard would open here');
    },

    // Show goal details panel
    showGoalDetails(goalId) {
        const goal = GoalRegistry.getGoal(goalId);
        if (!goal) return;

        // TODO: Implement details panel UI
        console.log('Goal details:', goal);
    },

    // Trigger celebration animation
    celebrate(type = 'milestone') {
        // TODO: Implement confetti/celebration effects
        if (type === 'goal') {
            console.log('🎉 GOAL ACHIEVED! Major celebration!');
        } else {
            console.log('✨ Milestone completed!');
        }
    }
};

// ═══════════════════════════════════════════════════════════════════
// GOAL VISUALIZATION - Distant beacon rendering
// ═══════════════════════════════════════════════════════════════════
// Goals appear as large, glowing beacons FAR from the current mind map
// They are NOT connected to the tree - they represent "New Reality"

const GoalVisualization = {
    beacons: new Map(), // goalId -> { mesh, glow, label, rings }
    baseDistance: 80, // How far goals float from center (distant beacon)
    goalSize: 2.5, // Much larger than normal nodes (normal is ~0.5)

    // Create a distant goal beacon
    createGoalBeacon(goal) {
        if (typeof THREE === 'undefined') {
            console.warn('THREE.js not available for goal visualization');
            return null;
        }

        const scene = window.scene;
        const camera = window.camera;
        const controls = window.controls;

        if (!scene) {
            console.warn('Scene not available for goal visualization');
            return null;
        }

        // Calculate position - far from center, spread out if multiple goals
        const existingGoals = this.beacons.size;
        const angle = (existingGoals * Math.PI / 3) + Math.PI / 6; // Spread goals around
        const distance = this.baseDistance + (existingGoals * 15); // Stagger distance

        const position = new THREE.Vector3(
            Math.cos(angle) * distance,
            5 + (existingGoals * 3), // Slightly elevated, staggered
            Math.sin(angle) * distance
        );

        // Priority affects size and intensity
        const priorityMultiplier = {
            high: 1.3,
            medium: 1.0,
            low: 0.8
        };
        const sizeMult = priorityMultiplier[goal.priority] || 1.0;
        const size = this.goalSize * sizeMult;

        // Create main sphere - much larger than normal nodes
        const geometry = new THREE.SphereGeometry(size, 64, 64);
        const material = new THREE.MeshStandardMaterial({
            color: 0x8B5CF6, // Purple
            emissive: 0x8B5CF6,
            emissiveIntensity: 0.8,
            roughness: 0.2,
            metalness: 0.3,
            transparent: true,
            opacity: 0.95
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(position);
        mesh.userData = {
            isGoalBeacon: true,
            goalId: goal.id,
            goal: goal
        };
        scene.add(mesh);

        // Create outer glow sphere
        const glowGeometry = new THREE.SphereGeometry(size * 1.5, 32, 32);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: 0xA78BFA,
            transparent: true,
            opacity: 0.15,
            side: THREE.BackSide
        });
        const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
        glowMesh.position.copy(position);
        scene.add(glowMesh);

        // Create pulsing rings (like a target/beacon)
        const rings = [];
        for (let i = 0; i < 3; i++) {
            const ringGeometry = new THREE.RingGeometry(
                size * (1.8 + i * 0.6),
                size * (2.0 + i * 0.6),
                64
            );
            const ringMaterial = new THREE.MeshBasicMaterial({
                color: 0x8B5CF6,
                transparent: true,
                opacity: 0.4 - (i * 0.1),
                side: THREE.DoubleSide
            });
            const ring = new THREE.Mesh(ringGeometry, ringMaterial);
            ring.position.copy(position);
            ring.userData.phaseOffset = i * (Math.PI * 2 / 3);
            scene.add(ring);
            rings.push(ring);
        }

        // Create label
        const label = this.createGoalLabel(goal.label, size);
        label.position.set(position.x, position.y - size - 1, position.z);
        scene.add(label);

        // Store beacon data
        const beacon = {
            mesh,
            glow: glowMesh,
            label,
            rings,
            goal,
            position: position.clone(),
            createdAt: Date.now()
        };
        this.beacons.set(goal.id, beacon);

        // Animate camera to show the new goal
        this.revealGoal(beacon, camera, controls);

        console.log(`🎯 Goal beacon created at distance ${distance}:`, goal.label);
        return beacon;
    },

    // Create a label sprite for the goal
    createGoalLabel(text, size) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const fontSize = 48;
        const padding = 20;

        ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`;
        const textWidth = ctx.measureText(text).width;
        const width = textWidth + padding * 2;
        const height = fontSize + padding * 2;

        canvas.width = width * 2;
        canvas.height = height * 2;
        ctx.scale(2, 2);

        // Background
        ctx.fillStyle = 'rgba(139, 92, 246, 0.9)';
        ctx.roundRect(0, 0, width, height, 12);
        ctx.fill();

        // Text
        ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`;
        ctx.fillStyle = 'white';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, width / 2, height / 2);

        // Create sprite
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        const spriteMaterial = new THREE.SpriteMaterial({
            map: texture,
            transparent: true
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(width / 80 * size, height / 80 * size, 1);

        return sprite;
    },

    // Animate camera to reveal the distant goal
    revealGoal(beacon, camera, controls) {
        if (!camera || !controls) return;

        // Store original position
        const originalTarget = controls.target.clone();
        const originalPosition = camera.position.clone();

        // Calculate where the goal is
        const goalPos = beacon.position;

        // Calculate a camera position that sees the goal
        // Position camera high and far back to see the distant beacon
        const cameraGoalView = new THREE.Vector3(
            goalPos.x * 0.4, // Between center and goal
            goalPos.y + 60,   // Very high up to see both
            goalPos.z * 0.4 + 40 // Far behind to see the distance
        );

        // Target point between center and goal
        const midTarget = new THREE.Vector3().lerpVectors(
            originalTarget,
            goalPos,
            0.5 // Look at point between map and goal
        );

        // Animate in three phases: zoom out, hold, return
        const duration = 4000;
        const startTime = Date.now();

        const animateReveal = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Smooth easing
            const ease = 1 - Math.pow(1 - progress, 3);

            if (progress < 0.4) {
                // Phase 1: Zoom out and pan to see goal (40%)
                const phaseProgress = progress / 0.4;
                const smoothPhase = 1 - Math.pow(1 - phaseProgress, 2);

                camera.position.lerpVectors(originalPosition, cameraGoalView, smoothPhase);
                controls.target.lerpVectors(originalTarget, midTarget, smoothPhase);
            } else if (progress < 0.7) {
                // Phase 2: Hold view on goal (30%)
                camera.position.copy(cameraGoalView);
                controls.target.copy(midTarget);
            } else {
                // Phase 3: Return to original view (30%)
                const phaseProgress = (progress - 0.7) / 0.3;
                const smoothPhase = phaseProgress * phaseProgress; // Ease in

                camera.position.lerpVectors(cameraGoalView, originalPosition, smoothPhase);
                controls.target.lerpVectors(midTarget, originalTarget, smoothPhase);
            }

            controls.update();

            if (progress < 1) {
                requestAnimationFrame(animateReveal);
            }
        };

        // Start after a brief delay
        setTimeout(animateReveal, 300);
    },

    // Update all beacons (called in animation loop)
    update(time) {
        this.beacons.forEach((beacon) => {
            // Pulse the glow
            const pulse = Math.sin(time * 2) * 0.5 + 0.5;
            beacon.glow.material.opacity = 0.1 + pulse * 0.1;
            beacon.glow.scale.setScalar(1 + pulse * 0.1);

            // Rotate and pulse the rings
            beacon.rings.forEach((ring, i) => {
                ring.rotation.x = Math.sin(time + ring.userData.phaseOffset) * 0.3;
                ring.rotation.y = time * 0.5 + ring.userData.phaseOffset;
                ring.material.opacity = (0.3 - i * 0.08) + Math.sin(time * 2 + ring.userData.phaseOffset) * 0.1;

                // Rings expand outward slowly
                const expandPulse = Math.sin(time * 0.5 + ring.userData.phaseOffset) * 0.1 + 1;
                ring.scale.setScalar(expandPulse);
            });

            // Subtle float for the main sphere
            beacon.mesh.position.y = beacon.position.y + Math.sin(time * 0.8) * 0.3;
            beacon.glow.position.y = beacon.mesh.position.y;
            beacon.rings.forEach(ring => {
                ring.position.y = beacon.mesh.position.y;
            });
            beacon.label.position.y = beacon.mesh.position.y - beacon.mesh.geometry.parameters.radius - 1;
        });
    },

    // Remove a goal beacon
    removeBeacon(goalId) {
        const beacon = this.beacons.get(goalId);
        if (!beacon) return;

        const scene = window.scene;
        if (scene) {
            scene.remove(beacon.mesh);
            scene.remove(beacon.glow);
            scene.remove(beacon.label);
            beacon.rings.forEach(ring => scene.remove(ring));
        }

        // Dispose geometries and materials
        beacon.mesh.geometry.dispose();
        beacon.mesh.material.dispose();
        beacon.glow.geometry.dispose();
        beacon.glow.material.dispose();
        beacon.label.material.map.dispose();
        beacon.label.material.dispose();
        beacon.rings.forEach(ring => {
            ring.geometry.dispose();
            ring.material.dispose();
        });

        this.beacons.delete(goalId);
    },

    // Restore beacons from saved goals
    restoreBeacons() {
        const activeGoals = GoalRegistry.getActiveGoals();
        activeGoals.forEach(goal => {
            if (!this.beacons.has(goal.id)) {
                this.createGoalBeacon(goal);
            }
        });
    }
};

// ═══════════════════════════════════════════════════════════════════
// GOAL WIZARD - Multi-step goal creation flow
// ═══════════════════════════════════════════════════════════════════

const GoalWizard = {
    currentStep: 1,
    totalSteps: 4,
    data: {
        desire: '',
        why: '',
        success: '',
        priority: 'medium'
    },

    // DOM elements (cached after init)
    elements: null,

    // Initialize wizard
    init() {
        this.elements = {
            wizard: document.getElementById('goal-wizard'),
            close: document.getElementById('goal-wizard-close'),
            back: document.getElementById('goal-wizard-back'),
            next: document.getElementById('goal-wizard-next'),
            steps: document.querySelectorAll('.goal-wizard-step'),
            stepContents: document.querySelectorAll('.goal-wizard-step-content'),
            priorityOptions: document.querySelectorAll('.goal-priority-option'),
            // Inputs
            desire: document.getElementById('goal-desire'),
            why: document.getElementById('goal-why'),
            success: document.getElementById('goal-success'),
            // Summary
            summaryDesire: document.getElementById('summary-desire'),
            summaryWhy: document.getElementById('summary-why'),
            summarySuccess: document.getElementById('summary-success'),
            summaryPriority: document.getElementById('summary-priority')
        };

        if (!this.elements.wizard) {
            console.warn('Goal wizard elements not found');
            return;
        }

        this.bindEvents();
        console.log('✓ Goal wizard initialized');
    },

    // Bind event listeners
    bindEvents() {
        // Close button
        this.elements.close?.addEventListener('click', () => this.close());

        // Click outside to close
        this.elements.wizard?.addEventListener('click', (e) => {
            if (e.target === this.elements.wizard) {
                this.close();
            }
        });

        // Back button
        this.elements.back?.addEventListener('click', () => this.prevStep());

        // Next button
        this.elements.next?.addEventListener('click', () => this.nextStep());

        // Priority selection
        this.elements.priorityOptions?.forEach(option => {
            option.addEventListener('click', () => {
                this.elements.priorityOptions.forEach(o => o.classList.remove('selected'));
                option.classList.add('selected');
                this.data.priority = option.dataset.priority;
            });
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (!this.elements.wizard?.classList.contains('active')) return;

            if (e.key === 'Escape') {
                this.close();
            } else if (e.key === 'Enter' && !e.shiftKey) {
                // Only trigger next on Enter if not in textarea
                if (e.target.tagName !== 'TEXTAREA') {
                    e.preventDefault();
                    this.nextStep();
                }
            }
        });
    },

    // Open wizard
    open() {
        this.reset();
        this.elements.wizard?.classList.add('active');
        document.body.style.overflow = 'hidden';

        // Focus first input after animation
        setTimeout(() => {
            this.elements.desire?.focus();
        }, 300);

        console.log('✨ Goal wizard opened');
    },

    // Close wizard
    close() {
        this.elements.wizard?.classList.remove('active');
        document.body.style.overflow = '';
        console.log('Goal wizard closed');
    },

    // Reset wizard state
    reset() {
        this.currentStep = 1;
        this.data = {
            desire: '',
            why: '',
            success: '',
            priority: 'medium'
        };

        // Clear inputs
        if (this.elements.desire) this.elements.desire.value = '';
        if (this.elements.why) this.elements.why.value = '';
        if (this.elements.success) this.elements.success.value = '';

        // Reset priority selection
        this.elements.priorityOptions?.forEach(o => {
            o.classList.remove('selected');
            if (o.dataset.priority === 'medium') {
                o.classList.add('selected');
            }
        });

        // Update UI
        this.updateStepUI();
    },

    // Go to next step
    nextStep() {
        // Validate current step
        if (!this.validateStep()) return;

        // Save current step data
        this.saveStepData();

        if (this.currentStep < this.totalSteps) {
            this.currentStep++;
            this.updateStepUI();

            // Focus appropriate input
            setTimeout(() => {
                if (this.currentStep === 2) this.elements.why?.focus();
                else if (this.currentStep === 3) this.elements.success?.focus();
            }, 100);
        } else {
            // Final step - create the goal
            this.createGoal();
        }
    },

    // Go to previous step
    prevStep() {
        if (this.currentStep > 1) {
            this.currentStep--;
            this.updateStepUI();
        }
    },

    // Validate current step
    validateStep() {
        switch (this.currentStep) {
            case 1:
                const desire = this.elements.desire?.value.trim();
                if (!desire) {
                    this.elements.desire?.focus();
                    this.shakeInput(this.elements.desire);
                    return false;
                }
                return true;
            case 2:
                // Why is optional but encouraged
                return true;
            case 3:
                // Success criteria optional
                return true;
            case 4:
                return true;
            default:
                return true;
        }
    },

    // Shake input for validation feedback
    shakeInput(element) {
        if (!element) return;
        element.style.animation = 'none';
        element.offsetHeight; // Trigger reflow
        element.style.animation = 'shake 0.5s ease-in-out';
    },

    // Save data from current step
    saveStepData() {
        switch (this.currentStep) {
            case 1:
                this.data.desire = this.elements.desire?.value.trim() || '';
                break;
            case 2:
                this.data.why = this.elements.why?.value.trim() || '';
                break;
            case 3:
                this.data.success = this.elements.success?.value.trim() || '';
                // Priority already saved via click handler
                break;
        }

        // Update summary if going to step 4
        if (this.currentStep === 3) {
            this.updateSummary();
        }
    },

    // Update summary display
    updateSummary() {
        const priorityLabels = {
            high: '🔥 High Priority',
            medium: '⭐ Medium Priority',
            low: '🌱 Someday'
        };

        if (this.elements.summaryDesire) {
            this.elements.summaryDesire.textContent = this.data.desire || '-';
        }
        if (this.elements.summaryWhy) {
            this.elements.summaryWhy.textContent = this.data.why || 'Not specified';
        }
        if (this.elements.summarySuccess) {
            this.elements.summarySuccess.textContent = this.data.success || 'Not specified';
        }
        if (this.elements.summaryPriority) {
            this.elements.summaryPriority.textContent = priorityLabels[this.data.priority] || 'Medium';
        }
    },

    // Update step UI (progress dots, visible content, buttons)
    updateStepUI() {
        // Update progress dots
        this.elements.steps?.forEach((step, i) => {
            const stepNum = i + 1;
            step.classList.remove('active', 'completed');
            if (stepNum === this.currentStep) {
                step.classList.add('active');
            } else if (stepNum < this.currentStep) {
                step.classList.add('completed');
            }
        });

        // Update visible content
        this.elements.stepContents?.forEach(content => {
            content.classList.remove('active');
            if (parseInt(content.dataset.step) === this.currentStep) {
                content.classList.add('active');
            }
        });

        // Update buttons
        if (this.elements.back) {
            this.elements.back.style.display = this.currentStep > 1 ? 'block' : 'none';
        }
        if (this.elements.next) {
            this.elements.next.textContent = this.currentStep === this.totalSteps ? 'Create Goal ✨' : 'Next';
        }
    },

    // Create the goal
    async createGoal() {
        console.log('Creating goal with data:', this.data);

        try {
            // Create goal via GoalRegistry
            const goal = GoalRegistry.createGoal({
                label: this.data.desire,
                description: this.data.success,
                desiredState: this.data.desire,
                whyItMatters: this.data.why,
                priority: this.data.priority
            });

            // Close wizard
            this.close();

            // Show success feedback
            if (typeof showToast === 'function') {
                showToast(`Goal created: "${goal.label}"`, 'success');
            }

            // Celebrate!
            GoalUI.celebrate('goal');

            // Create distant goal beacon (NOT a regular node)
            GoalVisualization.createGoalBeacon(goal);

            return goal;

        } catch (error) {
            console.error('Failed to create goal:', error);
            if (typeof showToast === 'function') {
                showToast('Failed to create goal', 'error');
            }
        }
    }
};

// Add shake animation for validation
const shakeStyle = document.createElement('style');
shakeStyle.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20%, 60% { transform: translateX(-5px); }
        40%, 80% { transform: translateX(5px); }
    }
`;
document.head.appendChild(shakeStyle);

// Initialize goal system when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Defer initialization to not block main app
    setTimeout(() => {
        GoalRegistry.init().then(() => {
            console.log('✓ Goal system initialized');
        });
        GoalWizard.init();
    }, 2000);
});

console.log('📍 Goal system module loaded');
