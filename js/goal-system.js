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

// Initialize goal system when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Defer initialization to not block main app
    setTimeout(() => {
        GoalRegistry.init().then(() => {
            console.log('✓ Goal system initialized');
        });
    }, 2000);
});

console.log('📍 Goal system module loaded');
