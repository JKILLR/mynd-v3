/**
 * MYND Demo Map Generator
 * Creates a comprehensive mind map of MYND itself with 500-1000 nodes
 * Covers: Vision, Features, Technology, AI, Business, Competition, Challenges, Roadmap
 */

const DemoMapGenerator = {
    colors: {
        vision: '#8B5CF6',      // Purple - Vision & Philosophy
        features: '#3B82F6',    // Blue - Core Features
        technology: '#10B981',  // Green - Technology
        ai: '#F59E0B',          // Amber - AI Systems
        ux: '#EC4899',          // Pink - User Experience
        business: '#EF4444',    // Red - Business
        data: '#06B6D4',        // Cyan - Data & Privacy
        competition: '#F97316', // Orange - Competition
        challenges: '#6366F1',  // Indigo - Challenges
        roadmap: '#84CC16',     // Lime - Roadmap
        investment: '#14B8A6',  // Teal - Investment
    },

    nodeCount: 0,

    // Main entry point
    async generate() {
        if (typeof store === 'undefined') {
            console.error('Store not available');
            return;
        }

        this.nodeCount = 0;
        const rootId = store.data.id;

        console.log('🚀 Starting MYND Demo Map generation...');

        // Create main MYND overview node
        const myndRoot = store.addNode(rootId, {
            label: 'MYND',
            color: '#8B5CF6',
            description: 'Cognitive Operating System - Your AI-Powered Second Brain',
            source: 'demo'
        });
        this.nodeCount++;

        if (!myndRoot) {
            console.error('Failed to create root node');
            return;
        }

        // Generate all major branches
        await this.createVisionBranch(myndRoot.id);
        await this.createFeaturesBranch(myndRoot.id);
        await this.createTechnologyBranch(myndRoot.id);
        await this.createAISystemsBranch(myndRoot.id);
        await this.createUXBranch(myndRoot.id);
        await this.createBusinessBranch(myndRoot.id);
        await this.createDataPrivacyBranch(myndRoot.id);
        await this.createCompetitionBranch(myndRoot.id);
        await this.createChallengesBranch(myndRoot.id);
        await this.createRoadmapBranch(myndRoot.id);
        await this.createInvestmentBranch(myndRoot.id);

        console.log(`✅ Demo map complete! Created ${this.nodeCount} nodes`);

        // Rebuild scene
        if (typeof buildScene === 'function') {
            buildScene();
        }

        // Expand the root
        store.expandedNodes.add(myndRoot.id);

        if (typeof showToast === 'function') {
            showToast(`Demo map created with ${this.nodeCount} nodes!`, 'success');
        }

        return this.nodeCount;
    },

    // Helper to create a node and increment counter
    addNode(parentId, data) {
        const node = store.addNode(parentId, { ...data, source: 'demo' });
        if (node) {
            this.nodeCount++;
            return node;
        }
        return null;
    },

    // Helper to create a branch from nested structure
    createBranch(parentId, structure, defaultColor) {
        if (Array.isArray(structure)) {
            structure.forEach(item => {
                if (typeof item === 'string') {
                    this.addNode(parentId, { label: item, color: defaultColor });
                } else if (typeof item === 'object') {
                    const node = this.addNode(parentId, {
                        label: item.label,
                        color: item.color || defaultColor,
                        description: item.description || ''
                    });
                    if (node && item.children) {
                        this.createBranch(node.id, item.children, item.color || defaultColor);
                    }
                }
            });
        }
    },

    // ═══════════════════════════════════════════════════════════════════
    // VISION & PHILOSOPHY BRANCH
    // ═══════════════════════════════════════════════════════════════════
    async createVisionBranch(parentId) {
        const vision = this.addNode(parentId, {
            label: 'Vision & Philosophy',
            color: this.colors.vision,
            description: 'The soul and purpose of MYND'
        });

        const structure = [
            {
                label: 'Mission Statement',
                children: [
                    'Cognitive Operating System',
                    'AI-Powered Second Brain',
                    'Antidote to Procrastination',
                    'Manifestation Tool',
                    'Life Coach in Your Pocket',
                    'Think in Constellations Not Lists',
                    {
                        label: 'Core Promise',
                        children: [
                            'Your thoughts analyzed locally',
                            'Privacy by design',
                            'You own your data',
                            'AI that understands YOU'
                        ]
                    }
                ]
            },
            {
                label: 'Target Audience',
                children: [
                    {
                        label: 'Primary Users',
                        children: [
                            'ADHD Minds',
                            'Creative Thinkers',
                            'Visual Learners',
                            'Non-Linear Processors',
                            'Overwhelmed by Linear Tools'
                        ]
                    },
                    {
                        label: 'Professional Users',
                        children: [
                            'Researchers',
                            'Writers & Authors',
                            'Strategists',
                            'Founders & Entrepreneurs',
                            'Product Managers',
                            'Consultants'
                        ]
                    },
                    {
                        label: 'Life Optimizers',
                        children: [
                            'Goal Setters',
                            'Self-Improvement Seekers',
                            'Habit Builders',
                            'Wellness Enthusiasts',
                            'Personal Development'
                        ]
                    },
                    {
                        label: 'Enterprise Teams',
                        children: [
                            'Cross-Functional Teams',
                            'Innovation Labs',
                            'Strategy Departments',
                            'R&D Groups'
                        ]
                    }
                ]
            },
            {
                label: 'Core Philosophy',
                children: [
                    {
                        label: 'Thoughts Are Spatial',
                        children: [
                            'Ideas have relationships',
                            'Context matters',
                            'Proximity = relevance',
                            '3D captures complexity'
                        ]
                    },
                    {
                        label: 'AI As Mirror',
                        children: [
                            'Reflects your patterns',
                            'Shows blind spots',
                            'Never judges',
                            'Always learning'
                        ]
                    },
                    {
                        label: 'Local-First Privacy',
                        children: [
                            'Your brain stays yours',
                            'No server surveillance',
                            'Encryption by default',
                            'You control sharing'
                        ]
                    },
                    {
                        label: 'Anticipatory Not Reactive',
                        children: [
                            'Predicts next steps',
                            'Suggests connections',
                            'Finds hidden paths',
                            'Coaches proactively'
                        ]
                    }
                ]
            },
            {
                label: 'Value Propositions',
                children: [
                    {
                        label: 'vs Traditional Tools',
                        children: [
                            'Lists trap thoughts',
                            'Folders hide connections',
                            'Linear = limited',
                            'MYND liberates thinking'
                        ]
                    },
                    {
                        label: 'Cognitive Enhancement',
                        children: [
                            'See patterns you miss',
                            'Remember what matters',
                            'Connect distant ideas',
                            'Track your evolution'
                        ]
                    },
                    {
                        label: 'Actionable Insights',
                        children: [
                            'From chaos to clarity',
                            'From dreams to plans',
                            'From stuck to moving',
                            'From overwhelmed to focused'
                        ]
                    }
                ]
            },
            {
                label: 'The MYND Difference',
                children: [
                    'Not a todo list',
                    'Not a note app',
                    'Not just mind mapping',
                    'A thinking partner',
                    'A growth accelerator',
                    'Your external consciousness'
                ]
            },
            {
                label: 'Guiding Principles',
                children: [
                    'Simplicity over complexity',
                    'Privacy over convenience',
                    'User ownership always',
                    'AI augments not replaces',
                    'Progress over perfection',
                    'Connection over isolation'
                ]
            }
        ];

        this.createBranch(vision.id, structure, this.colors.vision);
    },

    // ═══════════════════════════════════════════════════════════════════
    // CORE FEATURES BRANCH
    // ═══════════════════════════════════════════════════════════════════
    async createFeaturesBranch(parentId) {
        const features = this.addNode(parentId, {
            label: 'Core Features',
            color: this.colors.features,
            description: 'What MYND can do for you'
        });

        const structure = [
            {
                label: '3D Mind Map',
                children: [
                    {
                        label: 'Visual Elements',
                        children: [
                            'Force-Directed Graph',
                            'Node Spheres',
                            'Connection Lines',
                            'Glow Effects',
                            'Ambient Particles',
                            'Dynamic Lighting',
                            'Color Inheritance'
                        ]
                    },
                    {
                        label: 'Navigation',
                        children: [
                            'Orbit Controls',
                            'Pan & Zoom',
                            'Auto Camera',
                            'Focus Mode',
                            'Expand/Collapse',
                            'Home Position',
                            'Double-Tap Focus'
                        ]
                    },
                    {
                        label: 'Node Operations',
                        children: [
                            'Add Child Node',
                            'Edit Node',
                            'Delete Node',
                            'Move Node',
                            'Color Node',
                            'Add Description',
                            'Link External URL'
                        ]
                    },
                    {
                        label: 'Visual Feedback',
                        children: [
                            'Selection Glow',
                            'Hover Effects',
                            'Pulse Animation',
                            'Celebration Particles',
                            'Context Badges'
                        ]
                    }
                ]
            },
            {
                label: 'Goal System',
                children: [
                    {
                        label: 'Goal Wizard',
                        children: [
                            'Step 1: Define Desire',
                            'Step 2: Why It Matters',
                            'Step 3: Success Criteria',
                            'Step 4: Set Priority',
                            'Summary & Confirmation'
                        ]
                    },
                    {
                        label: 'Goal Beacons',
                        children: [
                            'Distant Visualization',
                            'Pulsing Glow Effect',
                            'Rotating Rings',
                            'Camera Reveal Animation',
                            'Priority-Based Sizing',
                            'Label Sprites'
                        ]
                    },
                    {
                        label: 'Manifestation Engine',
                        children: [
                            'Current State Mapping',
                            'Desired State Definition',
                            'Gap Analysis',
                            'Bridge Concept Generation',
                            'Path Discovery',
                            'Milestone Generation'
                        ]
                    },
                    {
                        label: 'Progress Tracking',
                        children: [
                            'Milestone Completion',
                            'Progress Percentage',
                            'Velocity Prediction',
                            'Completion Forecasting',
                            'Reflection Logging'
                        ]
                    },
                    {
                        label: 'Daily Guidance',
                        children: [
                            'Suggested Focus',
                            'Priority Ranking',
                            'Procrastination Detection',
                            'Nudge Notifications',
                            'Celebration Triggers'
                        ]
                    }
                ]
            },
            {
                label: 'AI Chat Interface',
                children: [
                    {
                        label: 'Conversation Modes',
                        children: [
                            'General Chat',
                            'Context-Aware (Node Selected)',
                            'Action Mode',
                            'Expansion Mode',
                            'Analysis Mode'
                        ]
                    },
                    {
                        label: 'AI Actions',
                        children: [
                            'Add Nodes',
                            'Edit Nodes',
                            'Move Nodes',
                            'Delete Nodes',
                            'Organize Structure',
                            'Generate Ideas',
                            'Find Connections'
                        ]
                    },
                    {
                        label: 'Quick Actions',
                        children: [
                            'Expand Ideas',
                            'Summarize Branch',
                            'Find Gaps',
                            'Suggest Next Steps',
                            'Create Timeline'
                        ]
                    }
                ]
            },
            {
                label: 'Memo System',
                children: [
                    {
                        label: 'Quick Capture',
                        children: [
                            'Floating Button',
                            'Text Input',
                            'Voice Input',
                            'Image Attachment',
                            'Link Parsing'
                        ]
                    },
                    {
                        label: 'AI Processing',
                        children: [
                            'Auto-Categorization',
                            'Suggested Placement',
                            'Entity Extraction',
                            'Sentiment Analysis',
                            'Action Item Detection'
                        ]
                    },
                    {
                        label: 'Organization',
                        children: [
                            'Inbox Queue',
                            'Quick Placement',
                            'Batch Processing',
                            'Smart Sorting'
                        ]
                    }
                ]
            },
            {
                label: 'Search & Navigation',
                children: [
                    {
                        label: 'Spotlight Search',
                        children: [
                            'Cmd+K Trigger',
                            'Fuzzy Matching',
                            'Recent Results',
                            'Quick Actions',
                            'Create from Search'
                        ]
                    },
                    {
                        label: 'Semantic Search',
                        children: [
                            'Meaning-Based Results',
                            'Similar Concepts',
                            'Cross-Branch Discovery',
                            'Embedding Similarity'
                        ]
                    },
                    {
                        label: 'Header Search Bar',
                        children: [
                            'Always Visible',
                            'Real-Time Results',
                            'Navigate to Node',
                            'Filter by Type'
                        ]
                    }
                ]
            },
            {
                label: 'Import & Export',
                children: [
                    {
                        label: 'Import Sources',
                        children: [
                            'Apple Notes',
                            'Text Files',
                            'JSON Data',
                            'Markdown',
                            'Browser Bookmarks',
                            'Screenshots',
                            'Voice Recordings'
                        ]
                    },
                    {
                        label: 'Export Formats',
                        children: [
                            'JSON (Full Data)',
                            'Markdown',
                            'Plain Text',
                            'PDF',
                            'Image (PNG)',
                            'Neural Weights'
                        ]
                    },
                    {
                        label: 'AI Import',
                        children: [
                            'Auto-Structure Notes',
                            'Duplicate Detection',
                            'Smart Merging',
                            'Category Inference'
                        ]
                    }
                ]
            },
            {
                label: 'Collaboration',
                children: [
                    {
                        label: 'Sharing Options',
                        children: [
                            'Public Link',
                            'View-Only Access',
                            'Branch Sharing',
                            'Template Export'
                        ]
                    },
                    {
                        label: 'Cloud Sync',
                        children: [
                            'Supabase Backend',
                            'Real-Time Sync',
                            'Conflict Resolution',
                            'Offline Support'
                        ]
                    },
                    {
                        label: 'User Connections',
                        children: [
                            'Opt-In Discovery',
                            'Thought Bridges',
                            'Cognitive Compatibility',
                            'Anonymous Matching'
                        ]
                    }
                ]
            },
            {
                label: 'Personalization',
                children: [
                    {
                        label: 'Themes',
                        children: [
                            'Dark Mode (Default)',
                            'Light Mode',
                            'Custom Colors',
                            'Accent Selection'
                        ]
                    },
                    {
                        label: 'Layout Options',
                        children: [
                            'Compact View',
                            'Expanded View',
                            'List Fallback',
                            'Focus Mode'
                        ]
                    },
                    {
                        label: 'Audio & Haptics',
                        children: [
                            'Sound Effects',
                            'Haptic Feedback',
                            'Volume Control',
                            'Mute Option'
                        ]
                    }
                ]
            }
        ];

        this.createBranch(features.id, structure, this.colors.features);
    },

    // ═══════════════════════════════════════════════════════════════════
    // TECHNOLOGY STACK BRANCH
    // ═══════════════════════════════════════════════════════════════════
    async createTechnologyBranch(parentId) {
        const tech = this.addNode(parentId, {
            label: 'Technology Stack',
            color: this.colors.technology,
            description: 'The engineering behind MYND'
        });

        const structure = [
            {
                label: 'Frontend',
                children: [
                    {
                        label: 'Three.js',
                        children: [
                            'WebGL Renderer',
                            'Scene Management',
                            'Camera Controls',
                            'Mesh Creation',
                            'Material System',
                            'Animation Loop',
                            'Raycasting',
                            'Sprite Labels'
                        ]
                    },
                    {
                        label: 'HTML5 Canvas',
                        children: [
                            'Label Rendering',
                            'Texture Generation',
                            'Dynamic Text',
                            '2D Overlays'
                        ]
                    },
                    {
                        label: 'CSS Architecture',
                        children: [
                            'CSS Variables',
                            'Flexbox Layout',
                            'Grid System',
                            'Animations',
                            'Transitions',
                            'Media Queries',
                            'Dark Theme'
                        ]
                    },
                    {
                        label: 'PWA Features',
                        children: [
                            'Service Worker',
                            'App Manifest',
                            'Offline Support',
                            'Install Prompt',
                            'Push Notifications',
                            'Background Sync'
                        ]
                    }
                ]
            },
            {
                label: 'AI & Machine Learning',
                children: [
                    {
                        label: 'TensorFlow.js',
                        children: [
                            'Browser Runtime',
                            'Model Loading',
                            'Inference Engine',
                            'Tensor Operations',
                            'Memory Management',
                            'WebGL Backend'
                        ]
                    },
                    {
                        label: 'Universal Sentence Encoder',
                        children: [
                            '512-Dim Embeddings',
                            'Semantic Similarity',
                            'Text Vectorization',
                            'Multilingual Support',
                            'Cached Embeddings'
                        ]
                    },
                    {
                        label: 'Custom Neural Networks',
                        children: [
                            'Category Predictor',
                            'Connection Predictor',
                            'Expansion Predictor',
                            'Online Learning',
                            'Knowledge Distillation'
                        ]
                    },
                    {
                        label: 'WebGPU Acceleration',
                        children: [
                            'GPU Compute Shaders',
                            'WGSL Programs',
                            'Parallel Similarity',
                            'Batch Processing',
                            'Fallback to CPU'
                        ]
                    }
                ]
            },
            {
                label: 'Data Layer',
                children: [
                    {
                        label: 'IndexedDB',
                        children: [
                            'NeuralDB Wrapper',
                            'Key-Value Storage',
                            'Binary Data',
                            'Transaction Support',
                            'Async Operations'
                        ]
                    },
                    {
                        label: 'LocalStorage',
                        children: [
                            'Quick Access Data',
                            'Settings Storage',
                            'Session State',
                            'Backup Fallback'
                        ]
                    },
                    {
                        label: 'Supabase',
                        children: [
                            'PostgreSQL Backend',
                            'Real-Time Subscriptions',
                            'Row-Level Security',
                            'Authentication',
                            'Edge Functions'
                        ]
                    },
                    {
                        label: 'Data Structures',
                        children: [
                            'Tree Structure',
                            'Node Objects',
                            'Edge Connections',
                            'Embedding Vectors',
                            'Neural Weights'
                        ]
                    }
                ]
            },
            {
                label: 'External APIs',
                children: [
                    {
                        label: 'Claude API',
                        children: [
                            'Chat Completions',
                            'Action Generation',
                            'Idea Expansion',
                            'Smart Organization',
                            'Knowledge Distillation'
                        ]
                    },
                    {
                        label: 'Edge Functions',
                        children: [
                            'Serverless Compute',
                            'API Proxying',
                            'Rate Limiting',
                            'Error Handling'
                        ]
                    },
                    {
                        label: 'CDN Resources',
                        children: [
                            'Three.js Library',
                            'TensorFlow Models',
                            'Font Assets',
                            'Icon Libraries'
                        ]
                    }
                ]
            },
            {
                label: 'Architecture Patterns',
                children: [
                    {
                        label: 'Current: Monolith',
                        children: [
                            'Single HTML File',
                            'Inline CSS',
                            'Inline JavaScript',
                            'Fast Prototyping',
                            'Technical Debt'
                        ]
                    },
                    {
                        label: 'Future: Modular',
                        children: [
                            'React/Vue/Svelte',
                            'Component System',
                            'State Management',
                            'Build Pipeline',
                            'Code Splitting'
                        ]
                    },
                    {
                        label: 'Design Patterns',
                        children: [
                            'Observer Pattern',
                            'Factory Pattern',
                            'Singleton Services',
                            'Command Pattern',
                            'Strategy Pattern'
                        ]
                    }
                ]
            },
            {
                label: 'Performance',
                children: [
                    {
                        label: 'Rendering',
                        children: [
                            'RequestAnimationFrame',
                            'Frustum Culling',
                            'Level of Detail',
                            'Instanced Meshes',
                            'Texture Atlases'
                        ]
                    },
                    {
                        label: 'Memory',
                        children: [
                            'Object Pooling',
                            'Geometry Disposal',
                            'Texture Management',
                            'Garbage Collection'
                        ]
                    },
                    {
                        label: 'Computation',
                        children: [
                            'Web Workers',
                            'Async Operations',
                            'Debouncing',
                            'Throttling',
                            'Lazy Loading'
                        ]
                    }
                ]
            }
        ];

        this.createBranch(tech.id, structure, this.colors.technology);
    },

    // ═══════════════════════════════════════════════════════════════════
    // AI SYSTEMS BRANCH
    // ═══════════════════════════════════════════════════════════════════
    async createAISystemsBranch(parentId) {
        const ai = this.addNode(parentId, {
            label: 'AI Systems',
            color: this.colors.ai,
            description: 'The intelligence powering MYND'
        });

        const structure = [
            {
                label: 'Neural Network Core',
                children: [
                    {
                        label: 'Architecture',
                        children: [
                            'Input Layer (512 dims)',
                            'Hidden Layer 1 (256 units)',
                            'Hidden Layer 2 (128 units)',
                            'Output Layer (variable)',
                            'ReLU Activation',
                            'Softmax Output',
                            'Dropout Regularization'
                        ]
                    },
                    {
                        label: 'Training Process',
                        children: [
                            'Batch Training',
                            'Online Learning',
                            'Incremental Updates',
                            'Cross-Entropy Loss',
                            'Adam Optimizer',
                            'Learning Rate Decay'
                        ]
                    },
                    {
                        label: 'Model Persistence',
                        children: [
                            'Weight Serialization',
                            'IndexedDB Storage',
                            'Cloud Backup',
                            'Version Control',
                            'Rollback Support'
                        ]
                    }
                ]
            },
            {
                label: 'Cognitive Graph Transformer',
                children: [
                    {
                        label: 'Graph Neural Network',
                        children: [
                            'Message Passing',
                            'Node Aggregation',
                            'Edge Features',
                            'Multi-Head Attention',
                            'Skip Connections'
                        ]
                    },
                    {
                        label: 'Structural Analysis',
                        children: [
                            'Node Role Detection',
                            'Hub Identification',
                            'Leaf Detection',
                            'Bridge Nodes',
                            'Cluster Analysis'
                        ]
                    },
                    {
                        label: 'Feature Extraction',
                        children: [
                            'Depth Features',
                            'Sibling Count',
                            'Child Count',
                            'Connection Density',
                            'Semantic Centrality'
                        ]
                    }
                ]
            },
            {
                label: 'Meta-Learner',
                children: [
                    {
                        label: 'Behavioral Metrics',
                        children: [
                            'Breadth vs Depth Ratio',
                            'Burst vs Steady Pattern',
                            'Quick vs Selective Decisions',
                            'Edit Frequency',
                            'Session Duration'
                        ]
                    },
                    {
                        label: 'Pattern Recognition',
                        children: [
                            'Time of Day Patterns',
                            'Topic Preferences',
                            'Organization Style',
                            'Expansion Habits',
                            'Deletion Patterns'
                        ]
                    },
                    {
                        label: 'Cognitive Fingerprint',
                        children: [
                            'Thinking Style Profile',
                            'Preference Vector',
                            'Habit Signatures',
                            'Growth Trajectory',
                            'Blind Spot Map'
                        ]
                    },
                    {
                        label: 'Adaptive Coaching',
                        children: [
                            'Personalized Prompts',
                            'Timing Optimization',
                            'Style Matching',
                            'Challenge Calibration'
                        ]
                    }
                ]
            },
            {
                label: 'Trajectory Predictor',
                children: [
                    {
                        label: 'Next Step Prediction',
                        children: [
                            'Action Forecasting',
                            'Node Suggestions',
                            'Connection Hints',
                            'Expansion Prompts'
                        ]
                    },
                    {
                        label: 'Pattern Analysis',
                        children: [
                            'Historical Sequences',
                            'Common Patterns',
                            'Deviation Detection',
                            'Trend Identification'
                        ]
                    },
                    {
                        label: 'Confidence Scoring',
                        children: [
                            'Prediction Certainty',
                            'Alternative Options',
                            'Risk Assessment',
                            'Fallback Suggestions'
                        ]
                    }
                ]
            },
            {
                label: 'Structural Holes Detector',
                children: [
                    {
                        label: 'Gap Analysis',
                        children: [
                            'Semantic Distance',
                            'Structural Distance',
                            'Missing Connections',
                            'Orphan Concepts'
                        ]
                    },
                    {
                        label: 'Bridge Suggestions',
                        children: [
                            'Connecting Concepts',
                            'Intermediate Steps',
                            'Hidden Relationships',
                            'Cross-Domain Links'
                        ]
                    },
                    {
                        label: 'Opportunity Detection',
                        children: [
                            'Insight Potential',
                            'Innovation Zones',
                            'Unexplored Territory',
                            'Growth Areas'
                        ]
                    }
                ]
            },
            {
                label: 'Knowledge Distillation',
                children: [
                    {
                        label: 'Teacher Model (Claude)',
                        children: [
                            'Rich Reasoning',
                            'Contextual Understanding',
                            'Creative Suggestions',
                            'Complex Analysis'
                        ]
                    },
                    {
                        label: 'Student Model (Local)',
                        children: [
                            'Compressed Knowledge',
                            'Fast Inference',
                            'Privacy Preserving',
                            'Offline Capable'
                        ]
                    },
                    {
                        label: 'Transfer Process',
                        children: [
                            'Soft Label Learning',
                            'Feature Matching',
                            'Attention Transfer',
                            'Progressive Training'
                        ]
                    }
                ]
            },
            {
                label: 'Active Learning',
                children: [
                    {
                        label: 'Uncertainty Sampling',
                        children: [
                            'Low Confidence Detection',
                            'Ambiguous Cases',
                            'Edge Cases',
                            'Novel Patterns'
                        ]
                    },
                    {
                        label: 'Human-in-Loop',
                        children: [
                            'User Corrections',
                            'Explicit Feedback',
                            'Implicit Signals',
                            'Preference Learning'
                        ]
                    },
                    {
                        label: 'Model Improvement',
                        children: [
                            'Targeted Training',
                            'Error Reduction',
                            'Coverage Expansion',
                            'Bias Correction'
                        ]
                    }
                ]
            },
            {
                label: 'Semantic Memory',
                children: [
                    {
                        label: 'Memory Types',
                        children: [
                            'Episodic Memory',
                            'Semantic Facts',
                            'Procedural Knowledge',
                            'Emotional Context'
                        ]
                    },
                    {
                        label: 'Retrieval System',
                        children: [
                            'Similarity Search',
                            'Contextual Recall',
                            'Associative Links',
                            'Temporal Ordering'
                        ]
                    },
                    {
                        label: 'Memory Consolidation',
                        children: [
                            'Importance Weighting',
                            'Decay Functions',
                            'Reinforcement',
                            'Pruning'
                        ]
                    }
                ]
            },
            {
                label: 'Feedback Loop System',
                children: [
                    {
                        label: 'Input Sources',
                        children: [
                            'Explicit Ratings',
                            'Click Behavior',
                            'Time Spent',
                            'Corrections Made',
                            'Features Used'
                        ]
                    },
                    {
                        label: 'Processing',
                        children: [
                            'Signal Aggregation',
                            'Noise Filtering',
                            'Pattern Extraction',
                            'Trend Analysis'
                        ]
                    },
                    {
                        label: 'Model Updates',
                        children: [
                            'Weight Adjustments',
                            'Threshold Tuning',
                            'Feature Importance',
                            'Architecture Adaptation'
                        ]
                    }
                ]
            }
        ];

        this.createBranch(ai.id, structure, this.colors.ai);
    },

    // ═══════════════════════════════════════════════════════════════════
    // USER EXPERIENCE BRANCH
    // ═══════════════════════════════════════════════════════════════════
    async createUXBranch(parentId) {
        const ux = this.addNode(parentId, {
            label: 'User Experience',
            color: this.colors.ux,
            description: 'How users interact with MYND'
        });

        const structure = [
            {
                label: 'Mobile Experience',
                children: [
                    {
                        label: 'Input-First Design',
                        children: [
                            'Floating Action Button',
                            'Quick Capture Sheet',
                            'Voice Input',
                            'Camera Capture',
                            'Swipe Gestures'
                        ]
                    },
                    {
                        label: 'Touch Navigation',
                        children: [
                            'Pinch to Zoom',
                            'Two-Finger Pan',
                            'Double-Tap Focus',
                            'Long Press Menu',
                            'Swipe Actions'
                        ]
                    },
                    {
                        label: 'Mobile UI',
                        children: [
                            'Bottom Sheet Panels',
                            'Collapsible Headers',
                            'Thumb-Friendly Buttons',
                            'Safe Area Handling',
                            'Keyboard Avoidance'
                        ]
                    },
                    {
                        label: 'Performance',
                        children: [
                            'Reduced Particle Count',
                            'Simplified Shadows',
                            'Touch Debouncing',
                            'Battery Optimization'
                        ]
                    }
                ]
            },
            {
                label: 'Desktop Experience',
                children: [
                    {
                        label: 'Mouse Navigation',
                        children: [
                            'Click to Select',
                            'Scroll to Zoom',
                            'Right-Click Menu',
                            'Drag to Pan',
                            'Hover Previews'
                        ]
                    },
                    {
                        label: 'Keyboard Shortcuts',
                        children: [
                            'Cmd+K Spotlight',
                            'Cmd+N New Node',
                            'Cmd+E Edit Node',
                            'Delete Key',
                            'Arrow Navigation',
                            'Tab to Expand'
                        ]
                    },
                    {
                        label: 'Desktop UI',
                        children: [
                            'Side Panels',
                            'Top Controls',
                            'Resizable Sections',
                            'Multi-Window Support'
                        ]
                    }
                ]
            },
            {
                label: 'Onboarding Flow',
                children: [
                    {
                        label: 'Welcome Screens',
                        children: [
                            'Value Proposition',
                            'Feature Highlights',
                            'Privacy Assurance',
                            'Getting Started'
                        ]
                    },
                    {
                        label: 'Questionnaire',
                        children: [
                            'Life Areas Selection',
                            'Goal Setting',
                            'Style Preferences',
                            'Import Options'
                        ]
                    },
                    {
                        label: 'Initial Map Creation',
                        children: [
                            'Template Selection',
                            'Seeded Categories',
                            'First Nodes',
                            'Tutorial Prompts'
                        ]
                    },
                    {
                        label: 'Feature Discovery',
                        children: [
                            'Contextual Tips',
                            'Progressive Disclosure',
                            'Achievement Badges',
                            'Milestone Celebrations'
                        ]
                    }
                ]
            },
            {
                label: 'Interaction Design',
                children: [
                    {
                        label: 'Feedback Systems',
                        children: [
                            'Toast Notifications',
                            'Loading States',
                            'Error Messages',
                            'Success Confirmations',
                            'Progress Indicators'
                        ]
                    },
                    {
                        label: 'Celebrations',
                        children: [
                            'Particle Bursts',
                            'Sound Effects',
                            'Haptic Feedback',
                            'Achievement Popups',
                            'Streak Recognition'
                        ]
                    },
                    {
                        label: 'Micro-Interactions',
                        children: [
                            'Button Press Effects',
                            'Smooth Transitions',
                            'Hover States',
                            'Focus Indicators',
                            'Loading Animations'
                        ]
                    }
                ]
            },
            {
                label: 'Accessibility',
                children: [
                    {
                        label: 'Visual',
                        children: [
                            'High Contrast Mode',
                            'Font Scaling',
                            'Color Blind Options',
                            'Reduced Motion'
                        ]
                    },
                    {
                        label: 'Input',
                        children: [
                            'Keyboard Navigation',
                            'Screen Reader Support',
                            'Voice Control',
                            'Switch Access'
                        ]
                    },
                    {
                        label: 'Cognitive',
                        children: [
                            'Simple Language',
                            'Clear Hierarchy',
                            'Consistent Patterns',
                            'Undo/Redo Support'
                        ]
                    }
                ]
            },
            {
                label: 'Information Architecture',
                children: [
                    {
                        label: 'Navigation Structure',
                        children: [
                            'Primary Actions',
                            'Secondary Actions',
                            'Contextual Actions',
                            'Global Actions'
                        ]
                    },
                    {
                        label: 'Content Hierarchy',
                        children: [
                            'Root Level',
                            'Category Level',
                            'Item Level',
                            'Detail Level'
                        ]
                    },
                    {
                        label: 'Mental Models',
                        children: [
                            'Spatial Memory',
                            'Color Association',
                            'Proximity Meaning',
                            'Size Importance'
                        ]
                    }
                ]
            }
        ];

        this.createBranch(ux.id, structure, this.colors.ux);
    },

    // ═══════════════════════════════════════════════════════════════════
    // BUSINESS MODEL BRANCH
    // ═══════════════════════════════════════════════════════════════════
    async createBusinessBranch(parentId) {
        const business = this.addNode(parentId, {
            label: 'Business Model',
            color: this.colors.business,
            description: 'How MYND creates and captures value'
        });

        const structure = [
            {
                label: 'Revenue Streams',
                children: [
                    {
                        label: 'Coach Subscription (B2C)',
                        children: [
                            {
                                label: 'Free Tier',
                                children: [
                                    'Local Storage Only',
                                    'Basic 3D Map',
                                    'Manual Creation',
                                    'Limited Nodes',
                                    'Community Support'
                                ]
                            },
                            {
                                label: 'Pro Tier ($20/mo)',
                                children: [
                                    'Cloud Sync',
                                    'Neural Features Active',
                                    'Meta-Learner Insights',
                                    'Trajectory Predictions',
                                    'Priority Support',
                                    'Advanced Export'
                                ]
                            },
                            {
                                label: 'Premium Add-Ons',
                                children: [
                                    'Business Lens Pack',
                                    'Wellness Lens Pack',
                                    'Creative Lens Pack',
                                    'Custom Lens Training'
                                ]
                            }
                        ]
                    },
                    {
                        label: 'Neural Templates (C2C)',
                        children: [
                            {
                                label: 'Template Types',
                                children: [
                                    'Trained Weight Sets',
                                    'Structure Templates',
                                    'Cognitive Patterns',
                                    'Domain Expertise'
                                ]
                            },
                            {
                                label: 'Marketplace Model',
                                children: [
                                    'User-Created Templates',
                                    'Verified Sellers',
                                    'Rating System',
                                    '20-30% Platform Fee'
                                ]
                            },
                            {
                                label: 'Example Templates',
                                children: [
                                    'Founder Weights',
                                    'Researcher Patterns',
                                    'Creative Writing',
                                    'Project Management'
                                ]
                            }
                        ]
                    },
                    {
                        label: 'Enterprise Hive Mind (B2B)',
                        children: [
                            {
                                label: 'Team Features',
                                children: [
                                    'Merged Embeddings',
                                    'Cross-Team Insights',
                                    'Structural Gap Detection',
                                    'Collaboration Tools'
                                ]
                            },
                            {
                                label: 'Pricing',
                                children: [
                                    'Per-Seat Licensing',
                                    'Annual Contracts',
                                    'Custom Deployments',
                                    'SLA Guarantees'
                                ]
                            },
                            {
                                label: 'Use Cases',
                                children: [
                                    'Marketing-Engineering Gaps',
                                    'Knowledge Management',
                                    'Innovation Discovery',
                                    'Strategy Alignment'
                                ]
                            }
                        ]
                    },
                    {
                        label: 'Data Licensing',
                        children: [
                            {
                                label: 'User-Controlled Access',
                                children: [
                                    'Opt-In Only',
                                    'Granular Permissions',
                                    'Revenue Sharing',
                                    'Anonymization Options'
                                ]
                            },
                            {
                                label: 'B2B Applications',
                                children: [
                                    'Market Research',
                                    'Product Testing',
                                    'Talent Assessment',
                                    'Trend Analysis'
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                label: 'Market Analysis',
                children: [
                    {
                        label: 'Market Size',
                        children: [
                            '$50B Productivity Tools',
                            '$100B Mental Health/Coaching',
                            'Trillion$ Data Economy',
                            'Growing AI Enhancement'
                        ]
                    },
                    {
                        label: 'Growth Drivers',
                        children: [
                            'Remote Work Explosion',
                            'Information Overload',
                            'ADHD Awareness',
                            'AI Acceptance',
                            'Privacy Concerns'
                        ]
                    },
                    {
                        label: 'Market Position',
                        children: [
                            'Not Productivity Tool',
                            'Cognitive Enhancement',
                            'Personal AI Coach',
                            'Data Ownership Pioneer'
                        ]
                    }
                ]
            },
            {
                label: 'Go-To-Market',
                children: [
                    {
                        label: 'Launch Strategy',
                        children: [
                            'Beta Community',
                            'Influencer Partnerships',
                            'Content Marketing',
                            'Product Hunt Launch'
                        ]
                    },
                    {
                        label: 'Growth Tactics',
                        children: [
                            'Viral Sharing',
                            'Template Marketplace',
                            'Referral Program',
                            'Community Building'
                        ]
                    },
                    {
                        label: 'Target Channels',
                        children: [
                            'ADHD Communities',
                            'Productivity Forums',
                            'Startup Ecosystems',
                            'Self-Improvement Spaces'
                        ]
                    }
                ]
            },
            {
                label: 'Unit Economics',
                children: [
                    {
                        label: 'Key Metrics',
                        children: [
                            'Customer Acquisition Cost',
                            'Lifetime Value',
                            'Monthly Recurring Revenue',
                            'Churn Rate',
                            'Net Promoter Score'
                        ]
                    },
                    {
                        label: 'Cost Structure',
                        children: [
                            'Cloud Infrastructure',
                            'AI API Costs',
                            'Development Team',
                            'Marketing Spend',
                            'Support Operations'
                        ]
                    },
                    {
                        label: 'Projections',
                        children: [
                            '1M Users Year 1',
                            '10M Users Year 3',
                            '$100M ARR Target',
                            '70% Gross Margins'
                        ]
                    }
                ]
            },
            {
                label: 'Competitive Moat',
                children: [
                    {
                        label: 'Technical Moat',
                        children: [
                            'Proprietary GNN',
                            'Meta-Learning Algorithms',
                            'Local-First Architecture',
                            'Knowledge Distillation'
                        ]
                    },
                    {
                        label: 'Data Moat',
                        children: [
                            'User Cognitive Models',
                            'Trained Neural Weights',
                            'Behavioral Patterns',
                            'Network Effects'
                        ]
                    },
                    {
                        label: 'Brand Moat',
                        children: [
                            'Trust & Privacy',
                            'Community Loyalty',
                            'Thought Leadership',
                            'First-Mover Advantage'
                        ]
                    }
                ]
            }
        ];

        this.createBranch(business.id, structure, this.colors.business);
    },

    // ═══════════════════════════════════════════════════════════════════
    // DATA & PRIVACY BRANCH
    // ═══════════════════════════════════════════════════════════════════
    async createDataPrivacyBranch(parentId) {
        const data = this.addNode(parentId, {
            label: 'Data & Privacy',
            color: this.colors.data,
            description: 'How MYND protects your mind'
        });

        const structure = [
            {
                label: 'Local-First Architecture',
                children: [
                    {
                        label: 'On-Device Processing',
                        children: [
                            'TensorFlow.js Local',
                            'WebGPU Acceleration',
                            'No Server Round-Trips',
                            'Instant Response'
                        ]
                    },
                    {
                        label: 'Local Storage',
                        children: [
                            'IndexedDB Primary',
                            'LocalStorage Backup',
                            'Browser Sandbox',
                            'No Cloud Required'
                        ]
                    },
                    {
                        label: 'Benefits',
                        children: [
                            'Privacy by Default',
                            'Offline Capable',
                            'Low Latency',
                            'No Data Leaks'
                        ]
                    }
                ]
            },
            {
                label: 'User Data Ownership',
                children: [
                    {
                        label: 'What You Own',
                        children: [
                            'Mind Map Content',
                            'Neural Weights',
                            'Cognitive Fingerprint',
                            'Behavioral Patterns',
                            'All Metadata'
                        ]
                    },
                    {
                        label: 'Export Rights',
                        children: [
                            'Full Data Export',
                            'Standard Formats',
                            'Neural Weight Export',
                            'No Lock-In'
                        ]
                    },
                    {
                        label: 'Deletion Rights',
                        children: [
                            'Instant Local Delete',
                            'Cloud Purge Request',
                            'No Residual Data',
                            'Verification Proof'
                        ]
                    }
                ]
            },
            {
                label: 'Cloud Sync (Optional)',
                children: [
                    {
                        label: 'Encryption',
                        children: [
                            'End-to-End Encryption',
                            'User-Held Keys',
                            'Zero-Knowledge Design',
                            'At-Rest Encryption'
                        ]
                    },
                    {
                        label: 'Access Control',
                        children: [
                            'Fine-Grained Permissions',
                            'Sharing Controls',
                            'Revocation Support',
                            'Audit Logging'
                        ]
                    },
                    {
                        label: 'Compliance',
                        children: [
                            'GDPR Compliant',
                            'CCPA Compliant',
                            'SOC 2 (Planned)',
                            'Data Residency Options'
                        ]
                    }
                ]
            },
            {
                label: 'Data Monetization (User-Controlled)',
                children: [
                    {
                        label: 'Opt-In Model',
                        children: [
                            'Explicit Consent',
                            'Granular Selection',
                            'Easy Opt-Out',
                            'Transparent Terms'
                        ]
                    },
                    {
                        label: 'Revenue Sharing',
                        children: [
                            'User Gets Majority',
                            'Transparent Pricing',
                            'Real-Time Dashboard',
                            'Instant Payouts'
                        ]
                    },
                    {
                        label: 'Anonymization',
                        children: [
                            'Differential Privacy',
                            'K-Anonymity',
                            'Aggregation Only',
                            'No Raw Data Sale'
                        ]
                    }
                ]
            },
            {
                label: 'Security Measures',
                children: [
                    {
                        label: 'Application Security',
                        children: [
                            'XSS Prevention',
                            'CSRF Protection',
                            'Input Validation',
                            'Content Security Policy'
                        ]
                    },
                    {
                        label: 'Infrastructure',
                        children: [
                            'HTTPS Only',
                            'Secure Headers',
                            'Rate Limiting',
                            'DDoS Protection'
                        ]
                    },
                    {
                        label: 'Authentication',
                        children: [
                            'OAuth 2.0',
                            'MFA Support',
                            'Session Management',
                            'Secure Tokens'
                        ]
                    }
                ]
            },
            {
                label: 'Trust Building',
                children: [
                    {
                        label: 'Transparency',
                        children: [
                            'Open Source (Planned)',
                            'Privacy Policy',
                            'Data Flow Diagrams',
                            'Third-Party Audits'
                        ]
                    },
                    {
                        label: 'User Control',
                        children: [
                            'Settings Dashboard',
                            'Data Viewer',
                            'Permission Manager',
                            'Activity Log'
                        ]
                    }
                ]
            }
        ];

        this.createBranch(data.id, structure, this.colors.data);
    },

    // ═══════════════════════════════════════════════════════════════════
    // COMPETITIVE LANDSCAPE BRANCH
    // ═══════════════════════════════════════════════════════════════════
    async createCompetitionBranch(parentId) {
        const competition = this.addNode(parentId, {
            label: 'Competitive Landscape',
            color: this.colors.competition,
            description: 'How MYND compares to alternatives'
        });

        const structure = [
            {
                label: 'Second Brain Tools',
                children: [
                    {
                        label: 'Notion',
                        children: [
                            'Strength: Flexibility',
                            'Strength: Team Features',
                            'Weakness: Linear Structure',
                            'Weakness: No AI Coaching',
                            'Gap: Not Anticipatory'
                        ]
                    },
                    {
                        label: 'Obsidian',
                        children: [
                            'Strength: Local-First',
                            'Strength: Graph View',
                            'Strength: Plugin Ecosystem',
                            'Weakness: Text-Centric',
                            'Gap: No Meta-Learning'
                        ]
                    },
                    {
                        label: 'Roam Research',
                        children: [
                            'Strength: Bi-Directional Links',
                            'Strength: Daily Notes',
                            'Weakness: Steep Learning Curve',
                            'Gap: No Visual 3D'
                        ]
                    },
                    {
                        label: 'Logseq',
                        children: [
                            'Strength: Open Source',
                            'Strength: Outliner',
                            'Weakness: Limited AI',
                            'Gap: No Cognitive Coaching'
                        ]
                    }
                ]
            },
            {
                label: 'AI Assistants',
                children: [
                    {
                        label: 'Personal.ai',
                        children: [
                            'Strength: Personal SLM',
                            'Strength: Digital Twin',
                            'Weakness: Chat-Only Interface',
                            'Gap: No Spatial Thinking'
                        ]
                    },
                    {
                        label: 'Saner.ai',
                        children: [
                            'Strength: ADHD Focus',
                            'Strength: Proactive Planning',
                            'Weakness: Work-Focused Only',
                            'Gap: No Life Coaching'
                        ]
                    },
                    {
                        label: 'Mem',
                        children: [
                            'Strength: AI Organization',
                            'Strength: Search',
                            'Weakness: Notes Only',
                            'Gap: No Manifestation'
                        ]
                    }
                ]
            },
            {
                label: 'Memory Tools',
                children: [
                    {
                        label: 'Rewind',
                        children: [
                            'Strength: Total Recall',
                            'Strength: Screenshot Search',
                            'Weakness: Backward-Looking',
                            'Gap: No Forward Guidance'
                        ]
                    },
                    {
                        label: 'Screenpipe',
                        children: [
                            'Strength: Open Source',
                            'Strength: Local Processing',
                            'Weakness: Passive Recording',
                            'Gap: No Active Coaching'
                        ]
                    }
                ]
            },
            {
                label: 'Mind Mapping Tools',
                children: [
                    {
                        label: 'MindMeister',
                        children: [
                            'Strength: Easy to Use',
                            'Strength: Collaboration',
                            'Weakness: 2D Only',
                            'Gap: No AI Features'
                        ]
                    },
                    {
                        label: 'Miro',
                        children: [
                            'Strength: Whiteboard',
                            'Strength: Team Features',
                            'Weakness: Not Personal',
                            'Gap: No Cognitive Model'
                        ]
                    },
                    {
                        label: 'TheBrain',
                        children: [
                            'Strength: 3D Navigation',
                            'Strength: Relationships',
                            'Weakness: Complex UI',
                            'Gap: No AI Coaching'
                        ]
                    }
                ]
            },
            {
                label: 'Data Marketplaces',
                children: [
                    {
                        label: 'Tartle',
                        children: [
                            'Strength: User Data Sales',
                            'Weakness: Raw Data Only',
                            'Gap: No Cognitive Models'
                        ]
                    },
                    {
                        label: 'Reklaim',
                        children: [
                            'Strength: Data Visibility',
                            'Weakness: Low Value Data',
                            'Gap: No Digital Twin'
                        ]
                    }
                ]
            },
            {
                label: 'MYND Differentiation',
                children: [
                    {
                        label: 'Unique Combination',
                        children: [
                            '3D Visual Second Brain',
                            'Anticipatory AI Coach',
                            'Manifestation Engine',
                            'Data Ownership Model'
                        ]
                    },
                    {
                        label: 'Technical Advantages',
                        children: [
                            'Local GNN Processing',
                            'Meta-Learning System',
                            'WebGPU Acceleration',
                            'Knowledge Distillation'
                        ]
                    },
                    {
                        label: 'Philosophy Advantages',
                        children: [
                            'User Owns Data',
                            'Privacy-First Design',
                            'Cognitive Enhancement Focus',
                            'Life Coaching Not Productivity'
                        ]
                    }
                ]
            }
        ];

        this.createBranch(competition.id, structure, this.colors.competition);
    },

    // ═══════════════════════════════════════════════════════════════════
    // TECHNICAL CHALLENGES BRANCH
    // ═══════════════════════════════════════════════════════════════════
    async createChallengesBranch(parentId) {
        const challenges = this.addNode(parentId, {
            label: 'Technical Challenges',
            color: this.colors.challenges,
            description: 'Obstacles to overcome'
        });

        const structure = [
            {
                label: 'Code Architecture',
                children: [
                    {
                        label: 'Monolith Problem',
                        children: [
                            '3500+ Lines Single File',
                            'Mixed HTML/CSS/JS',
                            'Hard to Maintain',
                            'Difficult Testing',
                            'No Code Splitting'
                        ]
                    },
                    {
                        label: 'Refactoring Needs',
                        children: [
                            'Component Framework',
                            'State Management',
                            'Build Pipeline',
                            'Type Safety',
                            'Module System'
                        ]
                    },
                    {
                        label: 'Framework Options',
                        children: [
                            'React + Vite',
                            'Vue 3 + Vite',
                            'Svelte + SvelteKit',
                            'Solid.js',
                            'Vanilla Modules'
                        ]
                    }
                ]
            },
            {
                label: 'Performance',
                children: [
                    {
                        label: 'Rendering',
                        children: [
                            'Three.js Overhead',
                            'Many Nodes Slowdown',
                            'Mobile GPU Limits',
                            'Memory Consumption',
                            'Frame Rate Drops'
                        ]
                    },
                    {
                        label: 'AI Computation',
                        children: [
                            'TensorFlow.js Heavy',
                            'Embedding Latency',
                            'GNN Message Passing',
                            'Parallel Processing',
                            'Battery Drain'
                        ]
                    },
                    {
                        label: 'Optimization Strategies',
                        children: [
                            'Web Workers',
                            'Lazy Loading',
                            'Level of Detail',
                            'Frustum Culling',
                            'Compute Offloading'
                        ]
                    }
                ]
            },
            {
                label: 'Data Integrity',
                children: [
                    {
                        label: 'Local Storage Risks',
                        children: [
                            'Browser Clears Data',
                            'Storage Limits',
                            'No Built-In Backup',
                            'Corruption Possible'
                        ]
                    },
                    {
                        label: 'Sync Challenges',
                        children: [
                            'Conflict Resolution',
                            'Offline Changes',
                            'Network Failures',
                            'Data Consistency'
                        ]
                    },
                    {
                        label: 'Solutions',
                        children: [
                            'Cloud Primary Storage',
                            'Auto-Backup System',
                            'CRDT for Sync',
                            'Version History'
                        ]
                    }
                ]
            },
            {
                label: 'UX Challenges',
                children: [
                    {
                        label: '3D Navigation',
                        children: [
                            'Disorientation Risk',
                            'Learning Curve',
                            'Motion Sickness',
                            'Touch Precision'
                        ]
                    },
                    {
                        label: 'Information Overload',
                        children: [
                            'Too Many Nodes',
                            'Visual Clutter',
                            'Finding Content',
                            'Maintaining Structure'
                        ]
                    },
                    {
                        label: 'Mitigations',
                        children: [
                            'Auto Camera Mode',
                            'Focus Mode',
                            'Collapse/Expand',
                            'Search Spotlight',
                            'Guided Tutorials'
                        ]
                    }
                ]
            },
            {
                label: 'AI Challenges',
                children: [
                    {
                        label: 'Cold Start Problem',
                        children: [
                            'New User Empty Map',
                            'No Patterns Yet',
                            'Generic Suggestions',
                            'Low Initial Value'
                        ]
                    },
                    {
                        label: 'Model Accuracy',
                        children: [
                            'Training Data Quality',
                            'Overfitting Risk',
                            'Domain Shift',
                            'Edge Cases'
                        ]
                    },
                    {
                        label: 'API Dependencies',
                        children: [
                            'Claude API Costs',
                            'Rate Limits',
                            'Latency Issues',
                            'Provider Lock-In'
                        ]
                    },
                    {
                        label: 'Solutions',
                        children: [
                            'Onboarding Questionnaire',
                            'Template Seeding',
                            'Active Learning',
                            'Local Model Fallback'
                        ]
                    }
                ]
            },
            {
                label: 'Scaling Challenges',
                children: [
                    {
                        label: 'User Growth',
                        children: [
                            'Infrastructure Scaling',
                            'Support Scaling',
                            'Cost Management',
                            'Quality Maintenance'
                        ]
                    },
                    {
                        label: 'Data Growth',
                        children: [
                            'Large Mind Maps',
                            'Embedding Storage',
                            'Search Performance',
                            'Backup Size'
                        ]
                    },
                    {
                        label: 'Feature Growth',
                        children: [
                            'Complexity Creep',
                            'UX Consistency',
                            'Technical Debt',
                            'Backward Compatibility'
                        ]
                    }
                ]
            }
        ];

        this.createBranch(challenges.id, structure, this.colors.challenges);
    },

    // ═══════════════════════════════════════════════════════════════════
    // ROADMAP BRANCH
    // ═══════════════════════════════════════════════════════════════════
    async createRoadmapBranch(parentId) {
        const roadmap = this.addNode(parentId, {
            label: 'Roadmap & Future',
            color: this.colors.roadmap,
            description: 'Where MYND is headed'
        });

        const structure = [
            {
                label: 'Phase 1: Foundation',
                children: [
                    {
                        label: 'Code Refactoring',
                        children: [
                            'Break Monolith',
                            'Setup Build Pipeline',
                            'Component Framework',
                            'State Management',
                            'Testing Framework'
                        ]
                    },
                    {
                        label: 'Performance Optimization',
                        children: [
                            'Web Workers Integration',
                            'Rendering Optimization',
                            'Mobile Performance',
                            'Memory Management',
                            'Load Time Reduction'
                        ]
                    },
                    {
                        label: 'Core Stability',
                        children: [
                            'Bug Fixes',
                            'Error Handling',
                            'Data Backup System',
                            'Crash Recovery',
                            'Offline Mode Polish'
                        ]
                    }
                ]
            },
            {
                label: 'Phase 2: AI Enhancement',
                children: [
                    {
                        label: 'Model Improvements',
                        children: [
                            'Better Predictions',
                            'Faster Training',
                            'Reduced Size',
                            'More Accuracy',
                            'New Capabilities'
                        ]
                    },
                    {
                        label: 'Edge Computing',
                        children: [
                            'Serverless GPU',
                            'Edge Functions',
                            'Hybrid Processing',
                            'Cost Optimization'
                        ]
                    },
                    {
                        label: 'New AI Features',
                        children: [
                            'Voice Interaction',
                            'Image Understanding',
                            'Document Analysis',
                            'Meeting Notes'
                        ]
                    }
                ]
            },
            {
                label: 'Phase 3: Marketplace',
                children: [
                    {
                        label: 'Template System',
                        children: [
                            'Creation Tools',
                            'Publishing Flow',
                            'Discovery System',
                            'Rating & Reviews'
                        ]
                    },
                    {
                        label: 'Commerce Features',
                        children: [
                            'Payment Processing',
                            'Revenue Sharing',
                            'Seller Dashboard',
                            'Buyer Protection'
                        ]
                    },
                    {
                        label: 'Community',
                        children: [
                            'User Profiles',
                            'Following System',
                            'Comments & Feedback',
                            'Leaderboards'
                        ]
                    }
                ]
            },
            {
                label: 'Phase 4: Enterprise',
                children: [
                    {
                        label: 'Team Features',
                        children: [
                            'Shared Workspaces',
                            'Permission System',
                            'Admin Controls',
                            'Analytics Dashboard'
                        ]
                    },
                    {
                        label: 'Integration',
                        children: [
                            'SSO Support',
                            'API Access',
                            'Webhook System',
                            'Third-Party Apps'
                        ]
                    },
                    {
                        label: 'Hive Mind',
                        children: [
                            'Collective Intelligence',
                            'Cross-Team Insights',
                            'Knowledge Gaps',
                            'Alignment Tools'
                        ]
                    }
                ]
            },
            {
                label: 'Future Vision',
                children: [
                    {
                        label: 'AR/VR Integration',
                        children: [
                            'Spatial Computing',
                            'Vision Pro Support',
                            'Quest Integration',
                            'Holographic Mind Map'
                        ]
                    },
                    {
                        label: 'Brain Interface',
                        children: [
                            'EEG Integration',
                            'Focus Detection',
                            'Thought Capture',
                            'Neurofeedback'
                        ]
                    },
                    {
                        label: 'Autonomous Agents',
                        children: [
                            'Background Processing',
                            'Proactive Research',
                            'Auto-Organization',
                            'Goal Tracking'
                        ]
                    },
                    {
                        label: 'Collective Intelligence',
                        children: [
                            'Global Knowledge Graph',
                            'Wisdom of Crowds',
                            'Emergent Insights',
                            'Humanity Mind Map'
                        ]
                    }
                ]
            }
        ];

        this.createBranch(roadmap.id, structure, this.colors.roadmap);
    },

    // ═══════════════════════════════════════════════════════════════════
    // INVESTMENT BRANCH
    // ═══════════════════════════════════════════════════════════════════
    async createInvestmentBranch(parentId) {
        const investment = this.addNode(parentId, {
            label: 'Investment Opportunity',
            color: this.colors.investment,
            description: 'Why invest in MYND'
        });

        const structure = [
            {
                label: 'Funding Ask',
                children: [
                    {
                        label: 'Round Details',
                        children: [
                            '$5-10M Seed/Series A',
                            '$50M Valuation',
                            'Equity + Board Seat',
                            '18-Month Runway'
                        ]
                    },
                    {
                        label: 'Use of Funds',
                        children: [
                            '40% Engineering',
                            '30% Product Development',
                            '20% Marketing',
                            '10% Operations/Legal'
                        ]
                    },
                    {
                        label: 'Milestones',
                        children: [
                            'MVP Launch (6 months)',
                            '100K Users (Year 1)',
                            'Marketplace Launch',
                            'Enterprise Beta'
                        ]
                    }
                ]
            },
            {
                label: 'Investment Thesis',
                children: [
                    {
                        label: 'Market Timing',
                        children: [
                            'AI Mainstream Adoption',
                            'Privacy Awareness Peak',
                            'Remote Work Standard',
                            'Mental Health Priority'
                        ]
                    },
                    {
                        label: 'Technical Moat',
                        children: [
                            'Years Ahead of Competition',
                            'GNN Innovation',
                            'Meta-Learning Patents',
                            'Local-First Architecture'
                        ]
                    },
                    {
                        label: 'Team Strength',
                        children: [
                            'Technical Founder',
                            'AI/ML Expertise',
                            'Product Vision',
                            'Execution Track Record'
                        ]
                    }
                ]
            },
            {
                label: 'Return Potential',
                children: [
                    {
                        label: 'Revenue Projections',
                        children: [
                            'Year 1: $1M ARR',
                            'Year 2: $10M ARR',
                            'Year 3: $50M ARR',
                            'Year 5: $200M ARR'
                        ]
                    },
                    {
                        label: 'Exit Scenarios',
                        children: [
                            'IPO at $1B+',
                            'Strategic Acquisition',
                            'PE Buyout',
                            'Long-Term Hold'
                        ]
                    },
                    {
                        label: 'Comparable Exits',
                        children: [
                            'Notion: $10B Valuation',
                            'Figma: $20B Acquisition',
                            'Miro: $17.5B Valuation',
                            'Canva: $40B Valuation'
                        ]
                    }
                ]
            },
            {
                label: 'Risk Mitigation',
                children: [
                    {
                        label: 'Technical Risks',
                        children: [
                            'Proven Prototype',
                            'Clear Refactor Path',
                            'Fallback Options',
                            'Incremental Approach'
                        ]
                    },
                    {
                        label: 'Market Risks',
                        children: [
                            'Clear Differentiation',
                            'Multiple Revenue Streams',
                            'Early Mover Advantage',
                            'Strong Community'
                        ]
                    },
                    {
                        label: 'Execution Risks',
                        children: [
                            'Experienced Team',
                            'Lean Operations',
                            'Agile Development',
                            'User-Driven Roadmap'
                        ]
                    }
                ]
            },
            {
                label: 'Social Impact',
                children: [
                    {
                        label: 'Individual Benefits',
                        children: [
                            'Reduced Procrastination',
                            'Better Goal Achievement',
                            'Cognitive Enhancement',
                            'Mental Wellness'
                        ]
                    },
                    {
                        label: 'Societal Benefits',
                        children: [
                            'Data Ownership Movement',
                            'Privacy-First AI',
                            'Democratized Intelligence',
                            'Human Potential'
                        ]
                    },
                    {
                        label: 'Economic Benefits',
                        children: [
                            'User Data Monetization',
                            'Creator Economy',
                            'Knowledge Workers',
                            'Innovation Acceleration'
                        ]
                    }
                ]
            }
        ];

        this.createBranch(investment.id, structure, this.colors.investment);
    }
};

// Make available globally
window.DemoMapGenerator = DemoMapGenerator;

// Add console command
console.log('📍 Demo Map Generator loaded. Run DemoMapGenerator.generate() to create MYND map.');
