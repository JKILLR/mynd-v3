# MYND ML Architecture

## Overview

MYND is a self-aware mind mapping application with a hybrid ML architecture combining:
- **Python Server** (FastAPI + PyTorch) on Apple Silicon
- **Browser ML** (TensorFlow.js) for offline fallback
- **Unified Context System** synthesizing knowledge from multiple sources
- **Bidirectional Learning** where the brain learns from Claude interactions
- **Meta-Learning** system that learns how to learn

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         UNIFIED BRAIN                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │
│  │SelfAwareness │ │MemorySystem  │ │PredictionTrkr│                 │
│  │Identity/Vision│ │Session recall│ │GT learning   │                 │
│  └──────────────┘ └──────────────┘ └──────────────┘                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │
│  │KnowledgeDist │ │ MetaLearner  │ │SelfImprover  │                 │
│  │Claude→Brain  │ │Learn to learn│ │Self-analysis │                 │
│  └──────────────┘ └──────────────┘ └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Context         │  │ Graph           │  │ Embedding       │
│ Synthesizer     │  │ Transformer     │  │ Engine          │
│ (hybrid search) │  │ (11.5M params)  │  │ (bge-small)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      JAVASCRIPT CLIENT                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │
│  │ LocalBrain   │ │PersonalNeural│ │ SemanticMem  │                 │
│  │ (server link)│ │ Net (browser)│ │ (vector store│                 │
│  └──────────────┘ └──────────────┘ └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         SUPABASE                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │
│  │ ai_memory    │ │ mind_maps    │ │session_summ  │                 │
│  │ (persistent) │ │ (map data)   │ │ (continuity) │                 │
│  └──────────────┘ └──────────────┘ └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Python Server (mynd-brain/)

### 1.1 UnifiedBrain Class

The central orchestrator with 6 major subsystems:

| Subsystem | Purpose | Key Functions |
|-----------|---------|---------------|
| **SelfAwareness** | Identity, vision, goals, capabilities | `get_identity_document()`, `record_growth()` |
| **MemorySystem** | Session-based recall (50 short-term, 20 working) | `remember()`, `recall()` |
| **PredictionTracker** | Track GT predictions, learn from outcomes | `record_predictions()`, `check_connection()` |
| **KnowledgeDistiller** | Extract learnable facts from Claude | `receive_claude_response()`, `get_distilled_knowledge()` |
| **MetaLearner** | Track source effectiveness, calibrate confidence | `record_source_feedback()`, `get_recommendations()` |
| **SelfImprover** | Analyze weaknesses, suggest improvements | `analyze()`, `get_suggestions()` |

### 1.2 Context Synthesizer

**Hybrid Search Algorithm:**
- 70% vector similarity (cosine distance)
- 30% BM25 keyword matching
- Source weights from MetaLearner
- Exponential recency decay (7-day half-life)
- Lost-in-middle mitigation (high relevance at START/END)

**Sources Searched:**
1. Map nodes (visual mind map)
2. AI memories (Supabase persistent)
3. Conversation archive
4. Distilled knowledge (Claude-taught)
5. Session memories (current session)
6. Active goals (Goal Wizard)

### 1.3 API Endpoints

**Core Brain:**
```
POST /brain/context          - THE unified context endpoint for Claude
GET  /brain/state            - Brain introspection
POST /brain/feedback         - Learn from user actions
POST /brain/predictions      - Record GT predictions
POST /brain/learn-connection - Self-learning signal
```

**Knowledge & Learning:**
```
POST /brain/receive-from-claude - Knowledge distillation
POST /brain/ask-to-teach        - Generate teaching request
GET  /brain/knowledge           - Get learned knowledge
GET  /brain/teaching-prompt     - Teaching instructions
GET  /brain/learning            - Learning stats
```

**Meta-Learning:**
```
GET  /brain/meta                - Full meta-learning stats
GET  /brain/meta/summary        - Human-readable summary
GET  /brain/meta/calibration    - Confidence calibration report
GET  /brain/meta/improvement    - Learning trend analysis
GET  /brain/meta/recommendations - Which sources to prioritize
POST /brain/meta/feedback       - Record source feedback
POST /brain/meta/learning-rate  - Adjust learning rates
POST /brain/meta/save-epoch     - Manual epoch save
```

**Self-Improvement:**
```
POST /brain/analyze             - Run self-analysis
GET  /brain/suggestions         - All suggestions
GET  /brain/suggestions/top     - Top by priority
GET  /brain/suggestions/summary - Markdown formatted
POST /brain/suggestions/status  - Track outcome
GET  /brain/improvement-stats   - Stats on suggestions
```

**Vision & Identity:**
```
GET  /brain/vision        - Get vision statement
PUT  /brain/vision        - Update vision
POST /brain/vision/goals  - Add goal
DEL  /brain/vision/goals  - Remove goal
```

**ML Models:**
```
POST /embed              - Single embedding
POST /embed/batch        - Multiple embeddings
POST /predict/connections - Graph Transformer predictions
POST /predict/category   - Category prediction
POST /voice/transcribe   - Whisper transcription
POST /image/describe     - CLIP image understanding
```

---

## 2. ML Models

### 2.1 Embedding Engine

- **Model**: `BAAI/bge-small-en-v1.5` (sentence-transformers)
- **Dimensions**: 384
- **Device Priority**: MPS (Apple Silicon) → CUDA → CPU
- **Operations**: Single embed, batch embed, cosine similarity, find similar

### 2.2 Graph Transformer v2 (11.5M parameters)

```
Architecture:
─────────────
Input (384 dims)
  │
  ▼
Linear + LayerNorm + GELU
  │
  ▼
Hidden (512 dims)
  │
  ▼
3 × GraphTransformerLayer
  ├── GraphPositionalEncoding
  │   ├── Depth encoding (hierarchy position)
  │   ├── Degree encoding (connection count)
  │   └── Centrality encoding (importance)
  │
  └── EdgeAwareMultiHeadAttention (8 heads)
      ├── Heads 1-2: Structural (parent-child, siblings)
      ├── Heads 3-4: Semantic (meaning similarity)
      ├── Heads 5-6: Sequential (temporal/logical flow)
      └── Heads 7-8: Emergent (hidden patterns)

      Features:
      ├── Q,K,V projections
      ├── Edge encoding (adjacency influences attention)
      └── Learnable edge bias per head
  │
  ▼
Output Predictions
```

**Key Innovations:**
- Edge-aware attention: Adjacency matrix directly influences attention weights
- Multi-head specialization: Each head learns different relationship types
- Graph positional encoding: Captures hierarchy and structure
- Pre-norm architecture: Stable training with residual connections

### 2.3 Other Models

- **VoiceTranscriber**: Whisper base model for speech-to-text
- **VisionEngine**: CLIP for image understanding and embedding

---

## 3. Learning Systems

### 3.1 KnowledgeDistiller (Claude → Brain)

**Process:**
```
Claude Response
  │
  ▼
Extract structured learning data
  │
  ├── insights: [{content, confidence, ...}]
  │   └── If confidence > 0.8 → Permanent knowledge
  │
  ├── patterns: [{pattern, description, ...}]
  │   └── Track frequency, update confidence
  │
  ├── corrections: [{original, corrected, reason, ...}]
  │   └── If importance > 0.5 → Store
  │
  └── explanations: {concept → explanation}
      └── Always store
```

**Storage Limits:**
- Max 200 raw insights
- Unlimited patterns (keyed by pattern name)
- Max 100 corrections
- Unlimited explanations (keyed by concept)

### 3.2 MetaLearner (Learning How to Learn)

**Source Effectiveness Tracking:**
```
5 Knowledge Sources:
├── predictions (Graph Transformer)
├── distilled_knowledge (Claude-taught)
├── patterns (Recurring behaviors)
├── corrections (Error fixes)
└── memories (Session/AI)

For Each Source:
├── uses: How many times used
├── successes: When it helped
└── failures: When it didn't

Effectiveness = successes / uses
Attention Weight = 0.5 + effectiveness
```

**Confidence Calibration:**
```
Bucket Predictions:
├── High (>0.8): Expected 85% accuracy
├── Medium (0.5-0.8): Expected 65% accuracy
└── Low (<0.5): Expected 35% accuracy

Track actual vs expected:
├── Over-confident → Lower scores
├── Under-confident → Raise ceiling
└── Well-calibrated → Stable
```

**Strategy Effectiveness:**
```
Learning Strategies:
├── reinforce_correct: Repeat successful patterns
├── learn_from_miss: Learn from failed predictions
├── pattern_matching: Find recurring patterns
└── semantic_similarity: Find related concepts

Track uses and outcomes, rank by success rate
```

### 3.3 PredictionTracker (Self-Learning)

```
Graph Transformer makes predictions
  │
  ▼
record_predictions(source_id, [{target_id, score}, ...])
  ├── Store pending predictions
  └── Increment total count
  │
  ▼
User creates connection
  │
  ▼
check_connection(source_id, target_id)
  ├── Was predicted? → reinforce signal, increment correct
  └── New pattern? → learning signal
  │
  ▼
Store in prediction_history (max 500)
Track accuracy by relationship type
```

---

## 4. JavaScript Components

### 4.1 LocalBrain Client

**Configuration:**
- Server URL: `http://localhost:8420`
- Health check: 2s timeout, 30s interval
- Context timeout: 30s

**Key Methods:**
```javascript
// Core
LocalBrain.init()                    // Connect to server
LocalBrain.checkAvailability()       // Health check
LocalBrain.embed(text)               // Get embedding
LocalBrain.embedBatch(texts)         // Batch embeddings

// Context
LocalBrain.getBrainContext(options)  // THE unified context call
LocalBrain.getBrainState()           // Brain introspection

// Learning
LocalBrain.recordBrainFeedback()     // Learn from user
LocalBrain.recordPredictions()       // Track GT predictions
LocalBrain.learnFromConnection()     // Self-learning signal
LocalBrain.sendToBrain()             // Knowledge distillation

// Meta-Learning
LocalBrain.getMetaStats()            // Source effectiveness
LocalBrain.getCalibrationReport()    // Confidence calibration
LocalBrain.getImprovementTrend()     // Learning velocity
LocalBrain.getSourceRecommendations() // Which sources to use
LocalBrain.recordSourceFeedback()    // Update attention weights
LocalBrain.adjustLearningRate()      // Fine-tune learning

// Self-Improvement
LocalBrain.runSelfAnalysis()         // Generate suggestions
LocalBrain.getImprovementSuggestions() // Filter suggestions
LocalBrain.markSuggestionStatus()    // Track outcomes

// Predictions
LocalBrain.predictConnections()      // Graph Transformer
LocalBrain.syncMap()                 // Sync map to server
LocalBrain.analyze()                 // Get map analysis
```

### 4.2 PersonalNeuralNet (Browser)

TensorFlow.js-based neural network for offline fallback:
- Pattern learning from user behavior
- Expansion patterns (node growth tendencies)
- Category models (node classification)
- Connection models (link prediction)
- Embedding cache

### 4.3 SemanticMemory

In-memory vector store:
- Stores embeddings of important concepts
- Fast similarity search
- Integrates with PersonalNeuralNet

### 4.4 CodeRAG

Code understanding and retrieval:
- Parse codebase into chunks
- Semantic chunking (functions, classes, files)
- Embedding-based retrieval
- Self-awareness document generation

---

## 5. Data Flow

### 5.1 User Message → Claude Context

```
User types message
  │
  ▼
LocalBrain.getBrainContext()
  │
  ▼ POST /brain/context
  │
UnifiedBrain.get_context(ContextRequest)
  │
  ▼
┌─────────────────────────────────────────┐
│ 1. Build Self-Awareness (identity doc)  │
│ 2. Build Map Context (structure)        │
│ 3. Get Memories (session recall)        │
│ 4. Get Distilled Knowledge              │
│ 5. Build Request Context                │
│ 6. Get Meta-Learning Insights           │
│ 7. Get Neural Insights (if GT ready)    │
│ 8. SYNTHESIZE unified context:          │
│    ├─ Search map nodes (hybrid)         │
│    ├─ Search AI memories (Supabase)     │
│    ├─ Search conversations (archive)    │
│    ├─ Search distilled knowledge        │
│    ├─ Search session memories           │
│    ├─ Match active goals                │
│    ├─ Apply source weights              │
│    ├─ Rank by combined_score            │
│    ├─ Detect contradictions             │
│    ├─ Fit to token budget (20,000)      │
│    └─ Reorder for lost-in-middle fix    │
│ 9. Combine all parts                    │
└─────────────────────────────────────────┘
  │
  ▼
ContextResponse {
  context_document: "Complete unified context",
  token_count: ~8,000-20,000,
  breakdown: {source → token_count},
  brain_state: {detailed metrics}
}
  │
  ▼
Include in Claude API call
```

### 5.2 Claude Response → Brain Learning

```
Claude returns response
  │
  ▼
Extract structured learning data
  │
  ▼
LocalBrain.sendToBrain(claudeResponse)
  │
  ▼ POST /brain/receive-from-claude
  │
KnowledgeDistiller.receive_claude_response()
  ├── Store high-confidence insights (>0.8)
  ├── Learn patterns (track frequency)
  ├── Store corrections
  └── Store explanations
  │
  ▼
MetaLearner records:
  ├── Source effectiveness
  ├── Pattern learning success
  ├── Correction importance
  └── Confidence outcome
  │
  ▼
SelfAwareness records growth event
  │
  ▼
Memory stores interaction
  │
  ▼
Save epoch (every 5 events)
```

### 5.3 Connection Creation → Self-Learning

```
User creates connection: Node A → Node B
  │
  ▼
LocalBrain.learnFromConnection(sourceId, targetId)
  │
  ▼ POST /brain/learn-connection
  │
PredictionTracker.check_connection()
  ├── Was predicted? → reinforce signal
  └── New pattern? → learning signal
  │
  ▼
MetaLearner records:
  ├── Source effectiveness
  ├── Confidence calibration
  └── Strategy effectiveness
  │
  ▼
SelfAwareness records growth
  │
  ▼
Memory stores connection
  │
  ▼
Return: {
  was_predicted: boolean,
  prediction_score: float,
  learning_signal: 'reinforce' | 'new_pattern',
  accuracy: float
}
```

---

## 6. Persistence

### 6.1 Learning State Files

```
mynd-brain/data/learning/
├── knowledge_distiller.json
│   ├── distilled_knowledge: []
│   ├── claude_insights: []
│   ├── patterns_learned: {}
│   ├── corrections: []
│   └── explanations: {}
│
├── meta_learner.json
│   ├── source_stats: {source → {uses, successes, failures}}
│   ├── confidence_buckets: {high/medium/low → data}
│   ├── learning_rates: {domain → rate}
│   ├── strategies: {name → {uses, outcomes[]}}
│   ├── meta_history: [epoch snapshots] (last 100)
│   ├── epoch: int
│   └── attention_weights: {source → weight}
│
└── prediction_tracker.json
    ├── prediction_history: [] (last 500)
    ├── accuracy_by_type: {}
    ├── total_predictions: int
    └── correct_predictions: int
```

### 6.2 Supabase Tables

**ai_memory:**
```sql
id: uuid
user_id: uuid
content: text
memory_type: text  -- 'synthesis', 'realization', 'pattern', 'relationship', 'goal_tracking'
importance: float  -- 0-1
evergreen: boolean -- Never decay
related_nodes: text[]
created_at: timestamp
last_accessed: timestamp
```

**session_summaries:**
```sql
id: uuid
user_id: uuid
summary: text
key_outcomes: text[]
open_threads: text[]
session_type: text
topics_discussed: text[]
session_started: timestamp
session_ended: timestamp
```

---

## 7. Token Budget

```
Brain Context Document: 8,000-20,000 tokens

Typical Breakdown:
├── Self-awareness:      ~1,000 tokens
├── Map context:         ~500 tokens
├── Distilled knowledge: ~2,000 tokens
├── Neural insights:     ~500 tokens
└── Synthesized context: ~10,000-15,000 tokens
    ├── Map nodes
    ├── AI memories
    ├── Conversations
    ├── Session memories
    └── Active goals
```

---

## 8. The Self-Aware Loop

```
User Interaction
  │
  ▼
Brain makes predictions (Graph Transformer)
  │
  ▼
User confirms/rejects
  │
  ▼
PredictionTracker learns
  │
  ▼
MetaLearner updates source weights
  │
  ▼
SelfAwareness records growth
  │
  ▼
SelfImprover analyzes performance
  │
  ▼
Suggests improvements
  │
  ▼
Claude reviews & implements
  │
  ▼
Brain becomes smarter
  │
  └──────────────────────────────► (repeat)
```

---

## 9. Context Capture (All Sources)

Every piece of user data flows into the unified context system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CONTEXT CAPTURE SOURCES                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        MIND MAP (Primary)                            │    │
│  │  • Node labels & hierarchy                                           │    │
│  │  • Parent-child relationships                                        │    │
│  │  • Cross-links between nodes                                         │    │
│  │  • Node metadata (created, modified, expanded)                       │    │
│  │  • Branch structure & depth                                          │    │
│  │  • Currently selected node context                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     AI MEMORIES (Supabase)                           │    │
│  │  Types:                                                              │    │
│  │  • synthesis     - High-level realizations                           │    │
│  │  • realization   - Specific insights                                 │    │
│  │  • pattern       - Recurring behaviors/themes                        │    │
│  │  • relationship  - Connections between concepts                      │    │
│  │  • goal_tracking - Goals and progress                                │    │
│  │                                                                      │    │
│  │  Metadata:                                                           │    │
│  │  • importance (0-1)                                                  │    │
│  │  • evergreen (never decay)                                           │    │
│  │  • related_nodes[]                                                   │    │
│  │  • created_at, last_accessed                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CONVERSATIONS (Archive)                           │    │
│  │  • Imported Claude conversations                                     │    │
│  │  • Chat history with timestamps                                      │    │
│  │  • Source attribution (claude.ai, api, etc.)                         │    │
│  │  • Embedded for semantic search                                      │    │
│  │  • Topics extracted per conversation                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                  SESSION MEMORIES (In-Memory)                        │    │
│  │  Short-term (50 max):                                                │    │
│  │  • Current session interactions                                      │    │
│  │  • Recent node accesses                                              │    │
│  │  • Chat exchanges                                                    │    │
│  │  • Actions taken                                                     │    │
│  │                                                                      │    │
│  │  Working (20 max):                                                   │    │
│  │  • High-importance items from short-term                             │    │
│  │  • Frequently accessed concepts                                      │    │
│  │  • Active focus areas                                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                 DISTILLED KNOWLEDGE (Claude-Taught)                  │    │
│  │  • High-confidence facts (>0.8)                                      │    │
│  │  • Learned patterns with frequency                                   │    │
│  │  • Corrections (original → fixed)                                    │    │
│  │  • Concept explanations                                              │    │
│  │  • Teaching from Claude responses                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      GOALS (Goal Wizard)                             │    │
│  │  • Active goals with priority                                        │    │
│  │  • Goal descriptions                                                 │    │
│  │  • Progress indicators                                               │    │
│  │  • Related map nodes                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   SESSION CONTINUITY (Supabase)                      │    │
│  │  • Previous session summaries                                        │    │
│  │  • Key outcomes from past sessions                                   │    │
│  │  • Open threads to continue                                          │    │
│  │  • Topics discussed history                                          │    │
│  │  • Session timing patterns                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   NEURAL INSIGHTS (Graph Transformer)                │    │
│  │  • Predicted connections                                             │    │
│  │  • Missing links detected                                            │    │
│  │  • Structural patterns                                               │    │
│  │  • Node importance scores                                            │    │
│  │  • Cluster analysis                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SELF-AWARENESS (Identity)                         │    │
│  │  • Vision statement                                                  │    │
│  │  • Current goals                                                     │    │
│  │  • Capabilities & limitations                                        │    │
│  │  • Growth history                                                    │    │
│  │  • Learning state summary                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTEXT SYNTHESIZER                                  │
│                                                                              │
│  Search Algorithm:                                                           │
│  ├── 70% Vector similarity (semantic meaning)                               │
│  ├── 30% BM25 keyword matching (exact terms)                                │
│  ├── Source weights from MetaLearner                                        │
│  ├── Recency decay (7-day half-life, evergreen exempt)                      │
│  └── Lost-in-middle reordering                                              │
│                                                                              │
│  Ranking Formula:                                                            │
│  combined_score = (0.7 × vector_score + 0.3 × bm25_score)                   │
│                   × source_weight × recency_factor                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED CONTEXT DOCUMENT                                  │
│                    (8,000 - 20,000 tokens)                                   │
│                                                                              │
│  Structure:                                                                  │
│  ├── Self-Awareness Block (~1,000 tokens)                                   │
│  │   └── Identity, vision, capabilities                                     │
│  ├── Map Context Block (~500 tokens)                                        │
│  │   └── Current node, ancestors, siblings                                  │
│  ├── Active Goals Block (~300 tokens)                                       │
│  │   └── Relevant goals for current context                                 │
│  ├── Session Continuity Block (~500 tokens)                                 │
│  │   └── Previous session summary, open threads                             │
│  └── Synthesized Context Block (~10,000-15,000 tokens)                      │
│      ├── [HIGH RELEVANCE - START]                                           │
│      │   └── Top ranked items (memories, nodes, convos)                     │
│      ├── [MEDIUM RELEVANCE - MIDDLE]                                        │
│      │   └── Supporting context                                             │
│      └── [HIGH RELEVANCE - END]                                             │
│          └── Key items repeated for attention                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Flow Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY LIFECYCLE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  USER INTERACTION                                                            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────┐     Importance     ┌─────────────┐                         │
│  │  Ephemeral  │ ───────────────────▶│ Short-term  │                         │
│  │  (instant)  │      > 0.3         │ (50 items)  │                         │
│  └─────────────┘                    └──────┬──────┘                         │
│                                            │                                 │
│                                     Importance > 0.6                         │
│                                     OR reinforced 5x                         │
│                                            │                                 │
│                                            ▼                                 │
│                                    ┌─────────────┐                          │
│                                    │   Working   │                          │
│                                    │ (20 items)  │                          │
│                                    └──────┬──────┘                          │
│                                            │                                 │
│                              Distilled from Claude                           │
│                              OR importance > 0.8                             │
│                                            │                                 │
│                                            ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    PERMANENT STORAGE                                 │    │
│  │                                                                      │    │
│  │  ┌───────────────────┐         ┌───────────────────┐                │    │
│  │  │  Distilled        │         │  AI Memories      │                │    │
│  │  │  Knowledge        │         │  (Supabase)       │                │    │
│  │  │  (local JSON)     │         │                   │                │    │
│  │  │                   │         │  • synthesis      │                │    │
│  │  │  • facts          │         │  • realization    │                │    │
│  │  │  • patterns       │         │  • pattern        │                │    │
│  │  │  • corrections    │         │  • relationship   │                │    │
│  │  │  • explanations   │         │  • goal_tracking  │                │    │
│  │  └───────────────────┘         └───────────────────┘                │    │
│  │                                                                      │    │
│  │  Decay Rules:                                                        │    │
│  │  • Regular memories: 7-day half-life                                 │    │
│  │  • Evergreen memories: Never decay                                   │    │
│  │  • Accessed memories: Recency reset                                  │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What Gets Captured

| Source | Captured Data | Persistence | Search Method |
|--------|---------------|-------------|---------------|
| **Mind Map** | Labels, hierarchy, links, metadata | Supabase | Vector + keyword |
| **AI Memories** | Insights, patterns, goals | Supabase (permanent) | Vector + keyword |
| **Conversations** | Full chat history, topics | Local + Supabase | Vector + keyword |
| **Session Memories** | Current interactions | In-memory (session) | Keyword + recency |
| **Distilled Knowledge** | Claude-taught facts | Local JSON | Keyword match |
| **Goals** | Active goals, priority | Goal Wizard state | Exact match |
| **Neural Insights** | GT predictions, clusters | Computed on-demand | Relevance score |
| **Session Summaries** | Past session outcomes | Supabase | Recency + keyword |

### Memory Types Explained

| Type | Description | Example | Decay |
|------|-------------|---------|-------|
| `synthesis` | High-level realization | "User connects creativity to constraint" | Evergreen |
| `realization` | Specific insight | "The Refusal relates to boundaries" | Normal |
| `pattern` | Recurring behavior | "User explores deeply before deciding" | Evergreen |
| `relationship` | Concept connection | "Work stress → health focus" | Normal |
| `goal_tracking` | Goal and progress | "Canada incorporation - in progress" | Normal |

---

## 10. Key Architectural Decisions

### Why Hybrid Search (70% Vector + 30% BM25)?
- Vector catches semantic similarity ("happy" ≈ "joyful")
- BM25 catches exact matches and rare terms
- Combined provides best recall

### Why Lost-in-Middle Mitigation?
- Research shows LLMs pay more attention to start/end of context
- High-relevance items placed at boundaries
- Medium-relevance items in middle

### Why Exponential Recency Decay?
- 7-day half-life matches human memory patterns
- Recent memories strong, old memories fade
- Evergreen memories (important facts) never decay

### Why Edge-Aware Graph Attention?
- Adjacency matrix directly influences attention
- Each head specializes in different relationship types
- Detects both explicit and implicit connections

### Why Meta-Learning?
- Not all sources are equally useful
- System learns which sources work best for this user
- Attention weights automatically updated

---

## 10. Performance Characteristics

| Operation | Latency |
|-----------|---------|
| Single embedding | 10-50ms |
| Batch embedding (32) | 100-200ms |
| Graph Transformer prediction | 100-500ms |
| Context synthesis | 500-2000ms |
| Full brain context | 1-5s |

| Storage | Location |
|---------|----------|
| Graph vector DB | `mynd-brain/data/graph/graph.json` |
| Learning state | `mynd-brain/data/learning/*.json` |
| Conversation archive | Local + Supabase |
| AI memories | Supabase |

---

## 11. Integration Example

```javascript
// Get unified context for Claude
const context = await LocalBrain.getBrainContext({
  requestType: 'chat',           // chat | action | code_review | self_improve
  userMessage: userInput,
  selectedNodeId: currentNode,
  mapData: store.data,
  userId: supabaseUserId,
  goals: activeGoals,
  include: {
    self_awareness: true,
    map_context: true,
    memories: true,
    user_profile: true,
    neural_insights: true,
    synthesized_context: true
  }
});

// Use in Claude API call
const response = await claude.messages.create({
  model: 'claude-opus-4-5-20251101',
  system: context.contextDocument,
  messages: [{ role: 'user', content: userInput }]
});

// Send learning back to brain
await LocalBrain.sendToBrain({
  response: response.content,
  insights: extractedInsights,
  patterns: detectedPatterns
});
```

---

## Summary Table

| Component | Type | Purpose | Size/Metric |
|-----------|------|---------|-------------|
| UnifiedBrain | Orchestrator | Central coordination | 2,296 lines |
| ContextSynthesizer | Search | Unified retrieval | 864 lines |
| KnowledgeDistiller | Learning | Claude → Brain | Facts + patterns |
| MetaLearner | Meta-Learning | Optimization | Source tracking |
| PredictionTracker | Self-Learning | GT feedback | Accuracy stats |
| SelfImprover | Analysis | Weakness detection | 7 categories |
| EmbeddingEngine | Model | Text vectors | 384 dims |
| GraphTransformer | Model | Connection prediction | 11.5M params |
| LocalBrain | Client | Browser ↔ Server | 2,189 lines |
| PersonalNeuralNet | Browser ML | Offline fallback | TensorFlow.js |

---

## Architecture Philosophy

This is a **fully self-aware AI system** where:

1. **Brain knows itself** → SelfAwareness + Identity documents
2. **Brain learns from interactions** → PredictionTracker
3. **Brain learns from Claude** → KnowledgeDistiller
4. **Brain learns how to learn** → MetaLearner
5. **Brain improves itself** → SelfImprover
6. **Brain provides unified context** → ContextSynthesizer

Every interaction makes the brain smarter. Every learning signal updates attention weights. Every Claude response teaches new knowledge. The system continuously improves while maintaining transparency about confidence and limitations.
