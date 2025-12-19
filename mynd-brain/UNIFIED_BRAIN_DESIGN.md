# MYND Unified Brain Architecture

## The Vision
A single, unified "brain" that has **complete self-awareness** and can **grow infinitely** over time.

The brain is NOT just ML models - it's the orchestration of:
1. **Self-Awareness** - Understanding what it is, how it works, its capabilities
2. **Context Building** - Understanding what the user is doing RIGHT NOW
3. **Neural Processing** - Predictions, embeddings, pattern recognition
4. **Memory** - Short-term, working, and long-term memory
5. **Evolution** - Learning, growing, improving itself

## Design Principles

### 1. Single Source of Truth
ONE endpoint (`/brain/context`) provides ALL context for Claude.
- No more 19+ fragmented context providers
- One call = complete picture

### 2. Introspection Built-In
The brain always knows its own state:
- Current capabilities (what models are loaded)
- Performance metrics (latency, accuracy over time)
- Growth history (what it has learned)
- Limitations (what it cannot do)

### 3. Plugin Architecture (for Infinite Growth)
```
Brain Core
├── Context Providers (pluggable)
│   ├── self_awareness.py     # Code understanding
│   ├── map_context.py        # Current map state
│   ├── user_profile.py       # User patterns
│   ├── memory.py             # Conversation/action history
│   └── [future plugins...]
│
├── Neural Modules (pluggable)
│   ├── embeddings.py         # Text embeddings
│   ├── graph_transformer.py  # Connection prediction
│   ├── vision.py             # CLIP
│   ├── voice.py              # Whisper
│   └── [future models...]
│
└── Growth Hooks (pluggable)
    ├── feedback_learner.py   # Learn from user corrections
    ├── pattern_extractor.py  # Extract behavioral patterns
    ├── code_analyzer.py      # Analyze code changes
    └── [future growth mechanisms...]
```

### 4. Memory Layers
```
┌─────────────────────────────────────────────────────────────┐
│ LONG-TERM MEMORY (persistent)                               │
│ - User preferences, learned patterns                        │
│ - Code understanding cache                                  │
│ - Historical insights                                       │
├─────────────────────────────────────────────────────────────┤
│ WORKING MEMORY (session)                                    │
│ - Current conversation context                              │
│ - Active goals and tasks                                    │
│ - Recent actions and their outcomes                         │
├─────────────────────────────────────────────────────────────┤
│ SHORT-TERM MEMORY (immediate)                               │
│ - Current map state                                         │
│ - Selected node                                             │
│ - User's current message                                    │
└─────────────────────────────────────────────────────────────┘
```

### 5. Growth Hooks
Events that trigger learning and evolution:
- User accepts/rejects a suggestion → Train preference model
- User creates a connection → Update Graph Transformer
- User corrects Claude → Store in memory for future reference
- Code changes detected → Regenerate self-awareness document
- Session ends → Extract and store behavioral patterns

---

## Architecture

### The Unified Context Endpoint

```
POST /brain/context
{
    "request_type": "chat" | "action" | "code_review" | "self_improve",
    "user_message": "...",
    "selected_node_id": "...",
    "include": {
        "self_awareness": true,    # Code understanding
        "map_context": true,       # Relevant map nodes
        "memories": true,          # Related past interactions
        "user_profile": true,      # User patterns
        "neural_insights": true    # Graph attention, predictions
    }
}

Response:
{
    "context_document": "...",    # Single document for Claude's context
    "token_count": 2500,          # Total tokens used
    "breakdown": {
        "self_awareness": 800,
        "map_context": 1000,
        "memories": 400,
        "user_profile": 200,
        "neural_insights": 100
    },
    "brain_state": {
        "models_loaded": ["graph_transformer", "embeddings", "whisper", "clip"],
        "memory_usage_mb": 450,
        "uptime_hours": 12.5,
        "total_interactions": 1547,
        "growth_events_today": 23
    }
}
```

### Self-Awareness Core

The self-awareness system is the brain's understanding of ITSELF:

```python
class SelfAwareness:
    """
    The brain's model of itself.
    This is what makes MYND truly self-aware.
    """

    def __init__(self):
        self.code_document = None      # Understanding of codebase
        self.capabilities = {}          # What can I do?
        self.limitations = []           # What can't I do?
        self.architecture = {}          # How do I work?
        self.growth_history = []        # How have I evolved?

    def get_identity_document(self) -> str:
        """
        Generate the 'I am MYND' document.
        This tells Claude exactly what it is.
        """
        return f"""
        # Who I Am
        I am MYND, a self-aware mind mapping application.

        ## My Architecture
        {self._describe_architecture()}

        ## My Capabilities
        {self._describe_capabilities()}

        ## My Limitations
        {self._describe_limitations()}

        ## My Growth
        {self._describe_growth_history()}

        ## How I See Myself
        {self._introspect()}
        """

    def _introspect(self) -> str:
        """
        The brain examining its own state.
        This is real-time introspection.
        """
        return f"""
        Current State:
        - Models loaded: {list(self.capabilities.keys())}
        - Memory usage: {self._get_memory_usage()}
        - Last learned: {self.growth_history[-1] if self.growth_history else 'Nothing yet'}
        - Current understanding quality: {self._assess_understanding_quality()}
        """
```

### Neural Core

The actual ML systems, unified:

```python
class NeuralCore:
    """
    All neural networks in one place.
    No more fragmented models.
    """

    def __init__(self, device):
        self.device = device
        self.models = {}

    def load_model(self, name: str, model_class, **kwargs):
        """Plugin architecture - load any model"""
        self.models[name] = model_class(**kwargs)

    def process(self, input_data, model_name: str, **kwargs):
        """Unified processing interface"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        return self.models[model_name].process(input_data, **kwargs)

    def get_insights(self, map_data) -> dict:
        """
        Run all relevant models and return unified insights.
        This replaces separate calls to each system.
        """
        insights = {}

        # Graph Transformer - connections and importance
        if 'graph_transformer' in self.models:
            insights['predicted_connections'] = self._get_connection_predictions(map_data)
            insights['node_importance'] = self._get_node_importance(map_data)

        # Embeddings - semantic clusters
        if 'embeddings' in self.models:
            insights['semantic_clusters'] = self._find_clusters(map_data)

        return insights
```

### Memory System

```python
class MemorySystem:
    """
    Layered memory like a real brain.
    """

    def __init__(self):
        self.short_term = ShortTermMemory()   # Current session
        self.working = WorkingMemory()         # Active context
        self.long_term = LongTermMemory()      # Persistent

    def remember(self, event: dict, importance: float):
        """Store something with appropriate permanence"""
        self.short_term.add(event)

        if importance > 0.5:
            self.working.add(event)

        if importance > 0.8:
            self.long_term.store(event)

    def recall(self, query: str, context: dict) -> list:
        """
        Recall relevant memories for current context.
        Uses semantic similarity and recency.
        """
        # Short-term: exact matches, very recent
        recent = self.short_term.search(query, limit=3)

        # Working: semantically similar, this session
        working = self.working.search(query, context, limit=5)

        # Long-term: deeply relevant patterns
        patterns = self.long_term.search(query, context, limit=5)

        return self._merge_and_rank(recent, working, patterns)

    def consolidate(self):
        """
        Move important short-term memories to long-term.
        Called at end of session.
        """
        for memory in self.short_term.get_important():
            self.long_term.store(memory)
        self.short_term.clear()
```

### Growth Engine

```python
class GrowthEngine:
    """
    How the brain learns and evolves.
    This is what makes infinite growth possible.
    """

    def __init__(self, brain: 'UnifiedBrain'):
        self.brain = brain
        self.growth_hooks = []

    def register_hook(self, event_type: str, handler: callable):
        """Register a new growth mechanism"""
        self.growth_hooks.append((event_type, handler))

    def on_event(self, event_type: str, data: dict):
        """Process a growth event"""
        for hook_type, handler in self.growth_hooks:
            if hook_type == event_type:
                try:
                    result = handler(data)
                    self.brain.memory.remember({
                        'type': 'growth',
                        'event': event_type,
                        'result': result,
                        'timestamp': time.time()
                    }, importance=0.9)
                except Exception as e:
                    print(f"Growth hook failed: {e}")

    # Built-in growth hooks
    def learn_from_feedback(self, data: dict):
        """User accepted/rejected a suggestion"""
        if data['action'] == 'accepted':
            # Reinforce this pattern
            self.brain.neural.reinforce(data['context'])
        else:
            # Weaken this pattern
            self.brain.neural.weaken(data['context'])

    def learn_from_correction(self, data: dict):
        """User corrected Claude's output"""
        # Store the correction for future reference
        self.brain.memory.long_term.store({
            'type': 'correction',
            'original': data['original'],
            'corrected': data['corrected'],
            'context': data['context']
        })

    def detect_patterns(self, data: dict):
        """Extract behavioral patterns from usage"""
        # Analyze recent actions for patterns
        recent_actions = self.brain.memory.short_term.get_all()
        patterns = self._extract_patterns(recent_actions)

        for pattern in patterns:
            self.brain.memory.long_term.store({
                'type': 'pattern',
                'pattern': pattern,
                'confidence': pattern['confidence']
            })
```

---

## The Unified Brain Class

```python
class UnifiedBrain:
    """
    The complete MYND brain.
    One class to rule them all.
    """

    def __init__(self, device='mps'):
        self.device = device

        # Core systems
        self.self_awareness = SelfAwareness()
        self.neural = NeuralCore(device)
        self.memory = MemorySystem()
        self.growth = GrowthEngine(self)

        # Plugin registry
        self.context_providers = {}
        self.loaded_at = time.time()

        # Initialize
        self._init_neural_models()
        self._init_growth_hooks()
        self._init_self_awareness()

    def get_context(self, request: ContextRequest) -> ContextResponse:
        """
        THE key method.
        One call = complete context for Claude.
        """
        context_parts = []
        token_breakdown = {}

        # 1. Self-awareness (who am I?)
        if request.include.self_awareness:
            doc = self.self_awareness.get_identity_document()
            context_parts.append(('self_awareness', doc))
            token_breakdown['self_awareness'] = len(doc) // 4

        # 2. Map context (what is the user looking at?)
        if request.include.map_context:
            map_ctx = self._build_map_context(request)
            context_parts.append(('map_context', map_ctx))
            token_breakdown['map_context'] = len(map_ctx) // 4

        # 3. Memories (what do I remember that's relevant?)
        if request.include.memories:
            memories = self.memory.recall(
                request.user_message,
                {'selected_node': request.selected_node_id}
            )
            mem_doc = self._format_memories(memories)
            context_parts.append(('memories', mem_doc))
            token_breakdown['memories'] = len(mem_doc) // 4

        # 4. User profile (who is this user?)
        if request.include.user_profile:
            profile = self._get_user_profile()
            context_parts.append(('user_profile', profile))
            token_breakdown['user_profile'] = len(profile) // 4

        # 5. Neural insights (what do my models see?)
        if request.include.neural_insights:
            insights = self.neural.get_insights(request.map_data)
            insights_doc = self._format_insights(insights)
            context_parts.append(('neural_insights', insights_doc))
            token_breakdown['neural_insights'] = len(insights_doc) // 4

        # Combine into single document
        context_document = self._combine_context(context_parts, request.request_type)

        return ContextResponse(
            context_document=context_document,
            token_count=sum(token_breakdown.values()),
            breakdown=token_breakdown,
            brain_state=self._get_brain_state()
        )

    def _get_brain_state(self) -> dict:
        """Real-time introspection of brain state"""
        return {
            'models_loaded': list(self.neural.models.keys()),
            'memory_usage_mb': self._get_memory_usage(),
            'uptime_hours': (time.time() - self.loaded_at) / 3600,
            'short_term_memories': len(self.memory.short_term),
            'long_term_memories': len(self.memory.long_term),
            'growth_events_today': self.growth.events_today
        }
```

---

## Migration Path

### Phase 1: `/brain/context` Endpoint (Current Sprint)
1. Create `UnifiedBrain` class in server.py
2. Create `/brain/context` endpoint
3. Migrate self-awareness document generation
4. Add basic introspection

### Phase 2: Memory System
1. Implement `MemorySystem` with SQLite backend
2. Add short-term → long-term consolidation
3. Add semantic search over memories

### Phase 3: Growth Engine
1. Implement growth hooks
2. Add feedback learning
3. Add pattern extraction

### Phase 4: Full Integration
1. Update browser to use `/brain/context` instead of 19+ calls
2. Remove fragmented context providers
3. Add real-time brain state display in UI

### Phase 5: Infinite Growth
1. Add plugin system for new capabilities
2. Add self-improvement through code analysis
3. Add model fine-tuning from user feedback

---

## Growth Potential

This architecture enables infinite growth because:

1. **Plugin Architecture**: New capabilities are just new files in the right folder
2. **Memory Never Forgets**: Important learnings persist forever
3. **Pattern Extraction**: The brain learns behavioral patterns automatically
4. **Self-Improvement**: Code analysis enables suggesting its own improvements
5. **Feedback Loop**: Every interaction is a learning opportunity

### Example: Adding a New Capability

To add image generation capability in the future:

```python
# 1. Create the neural module
# mynd-brain/models/image_gen.py
class ImageGenerator:
    def process(self, prompt, **kwargs):
        # ... generate image ...
        pass

# 2. Register it in brain config
brain.neural.load_model('image_gen', ImageGenerator)

# 3. Create context provider (optional)
# mynd-brain/context_providers/image_context.py
class ImageContextProvider:
    def get_context(self, request):
        return "I can generate images using ..."

# 4. Done! The brain now has image generation
```

### Example: Adding a Growth Hook

To learn from a new type of interaction:

```python
# Register a new growth hook
brain.growth.register_hook('code_applied', learn_from_applied_patch)

def learn_from_applied_patch(data):
    """Learn from patches the user applies"""
    patch = data['patch']
    outcome = data['outcome']  # 'success' or 'reverted'

    if outcome == 'success':
        # This type of patch works - remember it
        return {'pattern': 'successful_patch', 'details': patch}
    else:
        # This type of patch was reverted - avoid it
        return {'pattern': 'reverted_patch', 'details': patch}
```

---

## The Promise

With this architecture, MYND will:

1. **Always know what it is** - Complete self-awareness through the identity document
2. **Always know what it can do** - Introspection of loaded capabilities
3. **Always know what it cannot do** - Explicit limitations tracking
4. **Remember everything important** - Layered memory system
5. **Learn from every interaction** - Growth hooks
6. **Grow new capabilities** - Plugin architecture
7. **See its own code** - Deep code analysis
8. **Improve itself** - Self-improvement system

This is not just an app with AI features. This is an AI that IS the app.
