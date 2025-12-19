# MYND Unified Brain Architecture

## Complete System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    MYND UNIFIED BRAIN                                        │
│                                   (mynd-brain/server.py)                                    │
│                                      localhost:8420                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────────────┐
        │                                     │                                     │
        ▼                                     ▼                                     ▼
┌───────────────────┐              ┌───────────────────┐              ┌───────────────────┐
│   UNIFIED BRAIN   │              │     ML BRAIN      │              │   FRONTEND APP    │
│  (Orchestrator)   │◄────────────►│  (Neural Models)  │              │  (Browser/JS)     │
│                   │              │                   │              │                   │
│ unified_brain.py  │              │  MYNDBrain class  │              │  app-module.js    │
└───────────────────┘              └───────────────────┘              └───────────────────┘
        │                                     │                                     │
        │                                     │                                     │
        ▼                                     ▼                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                            UNIFIED BRAIN COMPONENTS                                  │   │
│  │                            (brain/unified_brain.py)                                  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │
│  │ SELF-AWARENESS  │  │  MEMORY SYSTEM  │  │   PREDICTION    │  │   KNOWLEDGE     │       │
│  │                 │  │                 │  │    TRACKER      │  │   DISTILLER     │       │
│  │ • Identity doc  │  │ • Short-term    │  │                 │  │                 │       │
│  │ • Code doc      │  │ • Working       │  │ • Track GT      │  │ • Claude →      │       │
│  │ • Capabilities  │  │ • (Long-term*)  │  │   predictions   │  │   Brain         │       │
│  │ • Limitations   │  │                 │  │ • Learn from    │  │ • Distill       │       │
│  │ • Growth history│  │ • remember()    │  │   outcomes      │  │   insights      │       │
│  │                 │  │ • recall()      │  │ • Track         │  │ • Store         │       │
│  │ get_identity_   │  │ • get_recent()  │  │   accuracy      │  │   patterns      │       │
│  │   document()    │  │                 │  │                 │  │ • Store         │       │
│  │ get_code_       │  │                 │  │ record_         │  │   corrections   │       │
│  │   document()    │  │                 │  │   prediction()  │  │                 │       │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘       │
│           │                    │                    │                    │                 │
│           └────────────────────┴────────────────────┴────────────────────┘                 │
│                                          │                                                  │
│                                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              UnifiedBrain Class                                      │   │
│  │                                                                                      │   │
│  │  • get_context(request) → Complete context document for Claude                      │   │
│  │  • receive_from_claude(response) → Extract & distill knowledge                      │   │
│  │  • record_predictions(node, predictions) → Track for learning                       │   │
│  │  • learn_from_connection(source, target) → Self-learning                           │   │
│  │  • record_feedback(node, action) → User feedback                                    │   │
│  │  • get_knowledge_stats() → Full brain statistics                                    │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              ML BRAIN COMPONENTS                                     │   │
│  │                              (models/*.py)                                           │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │
│  │    EMBEDDINGS   │  │GRAPH TRANSFORMER│  │     WHISPER     │  │      CLIP       │       │
│  │                 │  │                 │  │                 │  │                 │       │
│  │ EmbeddingEngine │  │ MYNDGraph-      │  │ VoiceTranscriber│  │  VisionEngine   │       │
│  │                 │  │   Transformer   │  │                 │  │                 │       │
│  │ • all-MiniLM-   │  │                 │  │ • Whisper base  │  │ • CLIP ViT-B-32 │       │
│  │   L6-v2         │  │ • 11.5M params  │  │ • Audio → Text  │  │ • Image → Text  │       │
│  │ • 384 dims      │  │ • 8 attn heads  │  │ • Language      │  │ • Image embed   │       │
│  │ • Text → Vector │  │ • 3 layers      │  │   detection     │  │ • Zero-shot     │       │
│  │                 │  │ • Predict       │  │                 │  │   classify      │       │
│  │ embed()         │  │   connections   │  │ transcribe()    │  │                 │       │
│  │ embed_batch()   │  │ • Node          │  │                 │  │ describe_       │       │
│  │                 │  │   importance    │  │                 │  │   image()       │       │
│  │                 │  │ • Find missing  │  │                 │  │ embed_image()   │       │
│  │                 │  │   connections   │  │                 │  │                 │       │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘       │
│         │                    │                    │                    │                   │
│         │      PyTorch       │       PyTorch      │      PyTorch       │                   │
│         │      M2 GPU        │       M2 GPU       │      M2 GPU        │                   │
│         └────────────────────┴────────────────────┴────────────────────┘                   │
│                                          │                                                  │
│                                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                MYNDBrain Class                                       │   │
│  │                                                                                      │   │
│  │  • embed(text) → Vector embedding                                                   │   │
│  │  • predict_connections(node, map) → Connection predictions                          │   │
│  │  • sync_map(map) → Full map awareness                                               │   │
│  │  • analyze_map() → Observations (missing connections, importance)                   │   │
│  │  • transcribe(audio) → Text                                                         │   │
│  │  • describe_image(image) → Description                                              │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA FLOW DIAGRAM                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

                                    USER INTERACTION
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                   BROWSER (app-module.js)                                 │
│                                                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │
│  │  Mind Map   │    │  AI Chat    │    │   Voice     │    │   Images    │               │
│  │  (Three.js) │    │ (Claude)    │    │  (Whisper)  │    │   (CLIP)    │               │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘               │
│         │                  │                  │                  │                       │
│         └──────────────────┴──────────────────┴──────────────────┘                       │
│                                       │                                                   │
│                                       ▼                                                   │
│                          ┌─────────────────────────┐                                     │
│                          │     LocalBrain Client   │                                     │
│                          │  (local-brain-client.js)│                                     │
│                          └────────────┬────────────┘                                     │
└───────────────────────────────────────┼───────────────────────────────────────────────────┘
                                        │
                                        │ HTTP/JSON
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                  SERVER (localhost:8420)                                   │
│                                                                                            │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              API ENDPOINTS                                          │   │
│  │                                                                                     │   │
│  │  UNIFIED BRAIN              SELF-LEARNING           CLAUDE ↔ BRAIN                 │   │
│  │  ─────────────              ─────────────           ─────────────                   │   │
│  │  POST /brain/context        POST /brain/            POST /brain/                    │   │
│  │  GET  /brain/state            predictions             receive-from-claude          │   │
│  │  POST /brain/feedback       POST /brain/            POST /brain/ask-to-teach       │   │
│  │                               learn-connection      GET  /brain/knowledge          │   │
│  │                             GET  /brain/learning    GET  /brain/teaching-prompt    │   │
│  │                                                                                     │   │
│  │  ML PROCESSING              MULTIMODAL              CODE ANALYSIS                   │   │
│  │  ─────────────              ───────────             ─────────────                   │   │
│  │  POST /embed                POST /voice/            GET  /code/parse               │   │
│  │  POST /embed/batch            transcribe            GET  /code/self-awareness      │   │
│  │  POST /predict/connections  POST /image/describe                                    │   │
│  │  POST /map/sync             POST /image/embed                                       │   │
│  │  GET  /map/analyze                                                                  │   │
│  └────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                            │
└───────────────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   LEARNING LOOPS                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                           LOOP 1: SELF-LEARNING (Graph Transformer)                        ║
╠═══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                            ║
║     Graph Transformer              PredictionTracker              User Action              ║
║     ─────────────────              ─────────────────              ───────────              ║
║                                                                                            ║
║     predict_connections()  ──────►  record_prediction()                                    ║
║     [A→B: 0.9, A→C: 0.7]           stores pending                                         ║
║                                                                                            ║
║                                                           User creates A→B                 ║
║                                                                  │                         ║
║                                    learn_from_connection() ◄─────┘                         ║
║                                           │                                                ║
║                                           ▼                                                ║
║                              ┌────────────────────────┐                                    ║
║                              │ Was A→B predicted?     │                                    ║
║                              └───────────┬────────────┘                                    ║
║                                          │                                                 ║
║                           ┌──────────────┴──────────────┐                                  ║
║                           ▼                              ▼                                 ║
║                    YES (score 0.9)                NO (missed)                              ║
║                    ─────────────                  ───────────                              ║
║                    "Reinforce!"                   "Learn new"                              ║
║                    accuracy += 1                  pattern!                                 ║
║                                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                           LOOP 2: CLAUDE ↔ BRAIN (Knowledge Distillation)                  ║
╠═══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                            ║
║        Brain                        Claude API                    Brain                    ║
║        ─────                        ──────────                    ─────                    ║
║                                                                                            ║
║   1. get_context()  ────────────►  System prompt +               ◄── includes             ║
║      │                             context document                  distilled            ║
║      │                                   │                           knowledge            ║
║      │                                   ▼                                                ║
║      │                             Claude responds                                         ║
║      │                             with structured                                         ║
║      │                             learning data:                                          ║
║      │                             {                                                       ║
║      │                               insights: [...],                                      ║
║      │                               patterns: [...],                                      ║
║      │                               corrections: [...]                                    ║
║      │                             }                                                       ║
║      │                                   │                                                 ║
║      │                                   ▼                                                 ║
║   2. receive_from_  ◄────────────  Response sent                                          ║
║        claude()                    to brain                                                ║
║           │                                                                                ║
║           ▼                                                                                ║
║   ┌───────────────────────────────────────────────────────────────────┐                   ║
║   │                    KnowledgeDistiller                             │                   ║
║   │                                                                   │                   ║
║   │   insights (conf > 0.8)  ──────►  distilled_knowledge            │                   ║
║   │   patterns               ──────►  patterns_learned                │                   ║
║   │   corrections            ──────►  corrections (stored)            │                   ║
║   │   explanations           ──────►  explanations (stored)           │                   ║
║   │                                                                   │                   ║
║   └───────────────────────────────────────────────────────────────────┘                   ║
║                                          │                                                 ║
║                                          ▼                                                 ║
║   3. Next get_context() includes distilled knowledge                                       ║
║      → Claude gets smarter context                                                         ║
║      → Gives better insights                                                               ║
║      → Brain gets smarter                                                                  ║
║      → INFINITE GROWTH                                                                     ║
║                                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝


┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FILE STRUCTURE                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

mynd-v3/
├── self-dev.html                    # Main app entry point
├── js/
│   ├── app-module.js                # Frontend (40K+ lines)
│   │   ├── store                    # Mind map data
│   │   ├── buildScene()             # 3D rendering
│   │   ├── AIChatManager            # Claude integration
│   │   ├── neuralNet                # Browser ML (fallback)
│   │   └── SelfImprover             # Self-evolution
│   │
│   └── local-brain-client.js        # Client for brain server
│       ├── getBrainContext()        # Unified context
│       ├── sendToBrain()            # Claude → Brain
│       ├── learnFromConnection()    # Self-learning
│       └── recordPredictions()      # Track predictions
│
└── mynd-brain/
    ├── server.py                    # FastAPI server
    │   ├── MYNDBrain                # ML models
    │   ├── unified_brain            # Brain orchestrator
    │   └── API endpoints            # All /brain/* routes
    │
    ├── brain/
    │   ├── __init__.py
    │   └── unified_brain.py         # THE BRAIN
    │       ├── UnifiedBrain         # Main class
    │       ├── SelfAwareness        # Identity & code
    │       ├── MemorySystem         # Short/working memory
    │       ├── PredictionTracker    # Self-learning
    │       └── KnowledgeDistiller   # Claude → Brain
    │
    └── models/
        ├── embeddings.py            # Sentence transformers
        ├── graph_transformer.py     # Connection prediction
        ├── voice.py                 # Whisper
        └── vision.py                # CLIP


┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CONTEXT FLOW                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

When frontend calls getBrainContext(), this is what happens:

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│  Request: {                                                                                 │
│    request_type: "chat",                                                                    │
│    user_message: "How does the neural net work?",                                          │
│    selected_node_id: "node_123",                                                           │
│    map_data: {...}                                                                          │
│  }                                                                                          │
│                                                                                             │
│         │                                                                                   │
│         ▼                                                                                   │
│                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         UnifiedBrain.get_context()                                   │   │
│  │                                                                                      │   │
│  │  1. SELF-AWARENESS                                                                   │   │
│  │     └─► SelfAwareness.get_identity_document()                                       │   │
│  │     └─► SelfAwareness.get_code_document()                                           │   │
│  │         "# I Am MYND... My Architecture... My Capabilities..."                       │   │
│  │                                                                                      │   │
│  │  2. MAP CONTEXT                                                                      │   │
│  │     └─► _build_map_context(request)                                                 │   │
│  │         "Selected: neural net node, 150 total nodes..."                             │   │
│  │                                                                                      │   │
│  │  3. MEMORIES                                                                         │   │
│  │     └─► MemorySystem.format_for_context()                                           │   │
│  │         "Recent: User asked about performance..."                                    │   │
│  │                                                                                      │   │
│  │  4. DISTILLED KNOWLEDGE (NEW!)                                                       │   │
│  │     └─► KnowledgeDistiller.get_relevant_knowledge("neural net")                     │   │
│  │         "Previously learned: neuralNet uses TensorFlow.js..."                       │   │
│  │                                                                                      │   │
│  │  5. REQUEST CONTEXT                                                                  │   │
│  │     └─► _build_request_context(request)                                             │   │
│  │         "Type: chat, User asks: How does neural net work?"                          │   │
│  │                                                                                      │   │
│  │  6. NEURAL INSIGHTS (if available)                                                   │   │
│  │     └─► ML Brain graph analysis                                                     │   │
│  │         "Graph Transformer sees: related nodes X, Y, Z"                             │   │
│  │                                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                             │
│         │                                                                                   │
│         ▼                                                                                   │
│                                                                                             │
│  Response: {                                                                                │
│    context_document: "# I Am MYND\n\n## Self-Awareness...\n## Map Context...",            │
│    token_count: 2500,                                                                       │
│    breakdown: {self_awareness: 1200, map: 500, memories: 300, knowledge: 400, ...},       │
│    brain_state: {accuracy: 0.73, distilled_facts: 45, ...}                                 │
│  }                                                                                          │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘


* = Phase 2 (persistent storage)
