/**
 * MYND Configuration
 * All constants and settings in one place
 */

const CONFIG = {
    // Storage Keys
    STORAGE_KEY: 'mynd-v6c',
    ONBOARDING_KEY: 'mynd-onboarded-v17',
    THEME_KEY: 'mynd-theme-v8',
    API_KEY: 'mynd-api-key',

    // Claude AI Model
    CLAUDE_MODEL: 'claude-opus-4-5-20251101',

    // Brain Server (Python ML backend)
    BRAIN_SERVER_URL: 'http://localhost:8000',

    // Supabase Configuration
    SUPABASE_URL: 'https://diqjasswlujwtdgsreab.supabase.co',
    SUPABASE_ANON_KEY: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRpcWphc3N3bHVqd3RkZ3NyZWFiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU0MjQyNTksImV4cCI6MjA4MTAwMDI1OX0._APhMDa1-FooORV_s5-ki6HF02TAlc_TX5CvqC7yKr4',
    EDGE_FUNCTION_URL: 'https://diqjasswlujwtdgsreab.supabase.co/functions/v1/claude-api',

    // Color Palettes for each theme
    THEME_COLORS: {
        // Sandstone - Natural earthy tones (graphite first for root)
        sandstone: [
            '#3A3A3A', '#B17457', '#8B9D83', '#C4887A', '#6B7C7A',
            '#4A4947', '#9AABA0', '#D8D2C2', '#A89078', '#7A8B70',
            '#5E6B5A', '#C0A898', '#8A9488', '#D4C4B0', '#6B6B63',
            '#A09080', '#C8B8A0', '#787870'
        ],
        // Coral - Soft pastels, flat cartoon look
        coral: [
            '#A8D5E5', '#E87D7D', '#B8A9C9', '#95D5B2', '#F5C396',
            '#89CFF0', '#F4A4A4', '#C3B1E1', '#A7E8BD', '#FFD8A8',
            '#98D8C8', '#EFA8A8', '#D4C4E0', '#BEE5B0', '#F8C8A8',
            '#A0C8D0', '#E0C0C0', '#C8C8C8'
        ],
        // Ember - Warm gradient from burnt orange to cream (rich orange first for root)
        ember: [
            '#C04818', '#E85820', '#F07028', '#F88830', '#FFA040',
            '#FFB858', '#FFD070', '#FFE088', '#D4C4A0', '#B0A080',
            '#907860', '#C89060', '#E8A848', '#F0C060', '#D8B078',
            '#A88858', '#C8A070', '#E8D098'
        ],
        // Frost - Cool soft blues and grays (white first for root)
        frost: [
            '#FFFFFF', '#5B8DEF', '#7DA8F5', '#6B9BD0', '#8FB8E8',
            '#5D7D9A', '#7A9AB8', '#4A6A8A', '#98B8D8', '#6888A8',
            '#88A8C8', '#5878A0', '#A8C8E8', '#7898B8', '#6080A0',
            '#90B0D0', '#4868A0', '#B0D0F0'
        ],
        // Obsidian - Purple accent with cool grays
        obsidian: [
            '#8B5CF6', '#A78BFA', '#BFC0C0', '#4F5D75', '#7A8899',
            '#9D6EF8', '#8D99AE', '#B07CF8', '#6B7B8C', '#7C4FE0',
            '#9EAAB8', '#6B3FD0', '#5A6A7A', '#9B7AF0', '#A8B4C0',
            '#8058C8', '#7888A0', '#C4B0F8'
        ]
    },

    // Legacy fallback (will be mapped to coral)
    COLORS: [
        '#A8D5E5', '#E87D7D', '#B8A9C9', '#95D5B2', '#F5C396',
        '#89CFF0', '#F4A4A4', '#C3B1E1', '#A7E8BD', '#FFD8A8',
        '#98D8C8', '#EFA8A8', '#D4C4E0', '#BEE5B0', '#F8C8A8',
        '#A0C8D0', '#E0C0C0', '#C8C8C8'
    ],

    COLORS_PROFESSIONAL: [
        '#A8D5E5', '#E87D7D', '#B8A9C9', '#95D5B2', '#F5C396',
        '#89CFF0', '#F4A4A4', '#C3B1E1', '#A7E8BD', '#FFD8A8',
        '#98D8C8', '#EFA8A8', '#D4C4E0', '#BEE5B0', '#F8C8A8',
        '#A0C8D0', '#E0C0C0', '#C8C8C8'
    ],

    // Spring Physics
    SPRING: {
        stiffness: 0.15,
        damping: 0.88
    },

    // Node Sizes
    NODE_SIZES: {
        root: 1.8,
        level1: 0.85,
        level2: 0.6,
        default: 0.45,
        minSize: 0.25
    },

    // Layout - dramatically increased for spiral galaxy effect
    LAYOUT: {
        level1Radius: 35.0,
        level1RadiusExpanded: 50,
        level2Radius: 28.0,
        level3Radius: 20.0,
        spreadAngle: Math.PI * 2.0,
        descendantSpacingFactor: 1.2,
        maxDescendantPush: 40,
        // Helix effect parameters - exaggerated for spiral galaxy look
        helixAmplitude: 12.0,     // Much larger vertical wave amplitude
        helixFrequency: 1.5,      // Tighter helix winds for more spirals
        verticalSpread: 3.0       // Larger vertical spacing multiplier
    },

    // Animation - consolidated timing constants
    ANIMATION: {
        springLerpBase: 0.12,
        springLerpSlow: 0.06,
        springLerpMedium: 0.08,
        scaleLerp: 0.15,
        positionThreshold: 0.001,
        scaleThreshold: 0.001,
        animatingThreshold: 0.01,
        cameraArcDuration: 600,
        cameraFollowSpeed: 0.03
    },

    // Timing (ms)
    TIMING: {
        expandStagger: 50,
        collapseStagger: 40,
        collapseHideDelay: 200,
        toastDuration: 3000,
        labelUpdateInterval: 100
    },

    // Label Decluttering
    LABELS: {
        padding: 12,
        fadeSpeed: 0.12,
        minOpacity: 0.03,
        basePriority: 100,
        selectedPriority: 1000,
        rootPriority: 500,
        depthPenalty: 10,
        declutterInterval: 3, // frames between declutter runs
        // Label sprite rendering
        fontSize: 42,
        spritePadding: 20,
        minWidth: 60,
        charWidth: 8,
        scaleFactor: 20,
        minDistance: 5
    },

    // Particles
    PARTICLES: {
        count: 200,
        spread: 100,
        baseSize: 0.15,
        opacity: 0.4,
        rotationSpeed: 0.02
    },

    // Neural Network Settings
    NEURAL_NET: {
        STORAGE_KEY: 'mynd-neural-model-v1',
        EMBEDDINGS_KEY: 'mynd-embeddings-v1',
        DB_NAME: 'mynd-neural-db',
        DB_VERSION: 1,
        STORE_NAME: 'models',
        embeddingDim: 512, // Universal Sentence Encoder output dimension
        hiddenUnits: 128,
        learningRate: 0.01,
        minTrainingNodes: 5,
        batchSize: 16,
        epochs: 50,
        // Active Learning Settings
        uncertaintyThreshold: 0.4, // Route to Claude if confidence below this
        onlineLearningRates: {
            rejected: 3.0,    // Strong correction multiplier
            modified: 2.0,    // Moderate correction multiplier
            accepted: 0.5     // Gentle reinforcement multiplier
        },
        // Feedback Loop Settings
        feedbackBatchThreshold: 5,    // Trigger batch learning after N feedback items
        feedbackMinInterval: 60000,   // Minimum ms between batch training runs
        teacherExampleWeight: 1.5     // Weight multiplier for Claude's examples
    },

    // Attachment settings
    ATTACHMENTS: {
        DB_NAME: 'mynd-attachments-db',
        DB_VERSION: 1,
        STORE_NAME: 'files',
        maxFileSize: 50 * 1024 * 1024, // 50MB
        thumbnailSize: 200, // px
        thumbnailQuality: 0.7,
        supportedImages: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
        supportedDocs: ['application/pdf'],
        maxAttachmentsPerNode: 20
    }
};

// Mobile detection and performance adjustments
const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) || window.innerWidth < 768;
const isLowPowerDevice = navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4;

if (isMobile || isLowPowerDevice) {
    // Reduce training intensity for mobile/low-power devices
    CONFIG.NEURAL_NET.epochs = isMobile ? 8 : 12;
    CONFIG.NEURAL_NET.batchSize = isMobile ? 8 : 12;
    CONFIG.NEURAL_NET.hiddenUnits = isMobile ? 64 : 96;
    console.log(`📱 Mobile/low-power mode: epochs=${CONFIG.NEURAL_NET.epochs}, batch=${CONFIG.NEURAL_NET.batchSize}`);
}

// Freeze CONFIG to prevent accidental modifications
Object.freeze(CONFIG.SPRING);
Object.freeze(CONFIG.NODE_SIZES);
Object.freeze(CONFIG.LAYOUT);
Object.freeze(CONFIG.ANIMATION);
Object.freeze(CONFIG.TIMING);
Object.freeze(CONFIG.LABELS);
Object.freeze(CONFIG.PARTICLES);
Object.freeze(CONFIG.ATTACHMENTS);
// Note: NEURAL_NET not frozen as it can be modified for mobile
