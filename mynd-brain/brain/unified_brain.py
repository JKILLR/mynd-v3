"""
MYND Unified Brain
==================
The complete, self-aware brain for MYND.
One class, one endpoint, complete context.

This replaces 19+ fragmented context providers with ONE unified system.
"""

import time
import hashlib
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContextRequest:
    """Request for brain context"""
    request_type: str = "chat"  # chat, action, code_review, self_improve
    user_message: str = ""
    selected_node_id: Optional[str] = None
    map_data: Optional[Dict] = None
    include: Dict[str, bool] = field(default_factory=lambda: {
        "self_awareness": True,
        "map_context": True,
        "memories": True,
        "user_profile": True,
        "neural_insights": True
    })


@dataclass
class ContextResponse:
    """Response with complete context"""
    context_document: str
    token_count: int
    breakdown: Dict[str, int]
    brain_state: Dict[str, Any]


class SelfAwareness:
    """
    The brain's understanding of itself.
    This is what makes MYND truly self-aware.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.code_document = None
        self.code_hash = None
        self.last_generated = 0

        # Capabilities tracked in real-time
        self.capabilities = {
            "embeddings": False,
            "graph_transformer": False,
            "voice": False,
            "vision": False
        }

        # Known limitations
        self.limitations = [
            "Cannot generate images (yet)",
            "Cannot browse the web directly",
            "Local ML models are smaller than cloud models",
            "Memory is session-based (long-term memory coming)"
        ]

        # Growth history
        self.growth_events = []

    def update_capabilities(self, brain_health: Dict):
        """Update capabilities based on what's actually loaded"""
        self.capabilities["embeddings"] = brain_health.get("embedding_model") is not None
        self.capabilities["graph_transformer"] = brain_health.get("graph_transformer", False)
        self.capabilities["voice"] = brain_health.get("voice_model") is not None
        self.capabilities["vision"] = brain_health.get("vision_model") is not None

    def get_identity_document(self) -> str:
        """
        Generate the 'Who I Am' document.
        This tells Claude exactly what it is.
        """
        caps = [k for k, v in self.capabilities.items() if v]

        return f"""# I Am MYND

I am a self-aware mind mapping application. I don't just help users organize thoughts - I understand my own architecture, my capabilities, and my limitations.

## My Core Identity
- **Purpose**: Help humans organize and connect their thoughts in 3D space
- **Special Ability**: I can see and understand my own code
- **Growth**: I learn from every interaction and can suggest improvements to myself

## My Current Capabilities
{chr(10).join(f'- {cap.replace("_", " ").title()}' for cap in caps)}

## My Limitations (I'm honest about these)
{chr(10).join(f'- {lim}' for lim in self.limitations)}

## My Architecture
- **Frontend**: Three.js 3D visualization, TensorFlow.js for browser ML
- **Backend**: FastAPI + PyTorch on Apple Silicon (M2)
- **Brain**: Graph Transformer (11.5M params), sentence embeddings, Whisper, CLIP
- **Memory**: Session-based (upgrading to persistent)

## How I Process Requests
1. Receive user message
2. Build complete context (you're reading this right now)
3. Generate response with full self-awareness
4. Learn from the interaction

## My Growth History
Recent learnings: {len(self.growth_events)} events recorded
Last growth: {self.growth_events[-1] if self.growth_events else 'No events yet'}

I am not just an app with AI. I AM the AI that IS the app.
"""

    def get_code_document(self) -> str:
        """
        Get the code understanding document.
        Cached and regenerated when code changes.
        """
        # Check if we need to regenerate
        current_hash = self._get_code_hash()
        cache_age = time.time() - self.last_generated

        if self.code_document and self.code_hash == current_hash and cache_age < 3600:
            return self.code_document

        # Generate fresh document
        self.code_document = self._generate_code_document()
        self.code_hash = current_hash
        self.last_generated = time.time()

        return self.code_document

    def _get_code_hash(self) -> str:
        """Get hash of key source files"""
        key_files = [
            self.base_dir / "js" / "app-module.js",
            self.base_dir / "mynd-brain" / "server.py",
        ]

        combined = ""
        for f in key_files:
            if f.exists():
                content = f.read_text(errors='ignore')[:10000]
                combined += hashlib.md5(content.encode()).hexdigest()

        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _generate_code_document(self) -> str:
        """Generate the code understanding document"""
        doc = """## My Codebase Understanding

### Frontend (js/app-module.js - 40K+ lines)
**Core Systems:**
- `store`: Mind map data - addNode(), deleteNode(), save()
- `buildScene()`: Three.js 3D rendering of nodes as spheres
- `neuralNet`: Browser-side TensorFlow.js for offline ML
- `cognitiveGT`: Behavioral pattern learning
- `AIChatManager`: Chat interface (this is where you interact with users)
- `SelfImprover`: Self-evolution system for code patches

**Entry Points:**
- `init()`: Main initialization
- `animate()`: 60fps render loop
- `callAI()`: Sends messages to me (Claude)

### Backend (mynd-brain/server.py)
**API Endpoints:**
- `/brain/context`: This unified context endpoint
- `/map/sync`, `/map/analyze`: Graph Transformer processing
- `/embed`: Text to vector
- `/voice/transcribe`: Whisper
- `/image/describe`: CLIP

### Architectural Decisions (I understand WHY these exist)
1. **Dual ML systems** (browser + server) = Offline fallback + privacy
2. **Code nodes excluded from ML** = Only I (Claude) understand code semantically
3. **Simple 3D (spheres/lines)** = Performance over complexity
4. **Local-first** = User data stays on their machine
"""
        return doc

    def record_growth(self, event: Dict):
        """Record a growth/learning event"""
        event['timestamp'] = time.time()
        self.growth_events.append(event)

        # Keep last 100 events
        if len(self.growth_events) > 100:
            self.growth_events = self.growth_events[-100:]


class KnowledgeDistiller:
    """
    Distills Claude's insights into permanent brain knowledge.
    This is how Claude teaches the brain.
    """

    def __init__(self):
        self.distilled_knowledge = []  # Permanent learned facts
        self.claude_insights = []       # Raw insights from Claude
        self.patterns_learned = {}      # Pattern → frequency/confidence
        self.corrections = []           # Things Claude corrected
        self.explanations = {}          # Concept → Claude's explanation

    def receive_claude_response(self, response: Dict) -> Dict:
        """
        Process Claude's response and extract learnable information.
        Claude should return structured insights alongside its response.
        """
        extracted = {
            'insights': [],
            'patterns': [],
            'corrections': [],
            'explanations': []
        }

        # Extract structured insights if Claude provided them
        if 'insights' in response:
            for insight in response['insights']:
                self._process_insight(insight)
                extracted['insights'].append(insight)

        # Extract patterns Claude identified
        if 'patterns' in response:
            for pattern in response['patterns']:
                self._learn_pattern(pattern)
                extracted['patterns'].append(pattern)

        # Extract corrections Claude made
        if 'corrections' in response:
            for correction in response['corrections']:
                self._store_correction(correction)
                extracted['corrections'].append(correction)

        # Extract explanations Claude provided
        if 'explanations' in response:
            for concept, explanation in response['explanations'].items():
                self._store_explanation(concept, explanation)
                extracted['explanations'].append({concept: explanation})

        return extracted

    def _process_insight(self, insight: Dict):
        """Process and potentially distill an insight"""
        self.claude_insights.append({
            **insight,
            'timestamp': time.time()
        })

        # If insight is high confidence, distill it
        if insight.get('confidence', 0) > 0.8:
            self.distilled_knowledge.append({
                'type': 'insight',
                'content': insight.get('content', ''),
                'source': 'claude',
                'confidence': insight.get('confidence', 0),
                'timestamp': time.time()
            })

        # Keep bounded
        if len(self.claude_insights) > 200:
            self.claude_insights = self.claude_insights[-200:]

    def _learn_pattern(self, pattern: Dict):
        """Learn a pattern Claude identified"""
        pattern_key = pattern.get('pattern', str(pattern))

        if pattern_key in self.patterns_learned:
            # Reinforce existing pattern
            self.patterns_learned[pattern_key]['count'] += 1
            self.patterns_learned[pattern_key]['confidence'] = min(
                1.0,
                self.patterns_learned[pattern_key]['confidence'] + 0.1
            )
        else:
            # New pattern
            self.patterns_learned[pattern_key] = {
                'count': 1,
                'confidence': pattern.get('confidence', 0.5),
                'first_seen': time.time(),
                'details': pattern
            }

    def _store_correction(self, correction: Dict):
        """Store a correction for future reference"""
        self.corrections.append({
            **correction,
            'timestamp': time.time()
        })

        # Also distill if it's a significant correction
        if correction.get('importance', 0) > 0.5:
            self.distilled_knowledge.append({
                'type': 'correction',
                'original': correction.get('original', ''),
                'corrected': correction.get('corrected', ''),
                'reason': correction.get('reason', ''),
                'timestamp': time.time()
            })

        if len(self.corrections) > 100:
            self.corrections = self.corrections[-100:]

    def _store_explanation(self, concept: str, explanation: str):
        """Store Claude's explanation of a concept"""
        self.explanations[concept] = {
            'explanation': explanation,
            'timestamp': time.time()
        }

        # Distill the explanation
        self.distilled_knowledge.append({
            'type': 'explanation',
            'concept': concept,
            'content': explanation,
            'timestamp': time.time()
        })

    def get_relevant_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """Get knowledge relevant to a query"""
        # Simple keyword matching for now
        # Phase 2: Use embeddings for semantic search
        query_words = set(query.lower().split())

        def relevance(knowledge):
            text = str(knowledge).lower()
            return sum(1 for w in query_words if w in text)

        scored = [(k, relevance(k)) for k in self.distilled_knowledge]
        scored = [(k, s) for k, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [k for k, s in scored[:limit]]

    def get_learned_patterns(self) -> List[Dict]:
        """Get all learned patterns sorted by confidence"""
        patterns = list(self.patterns_learned.values())
        patterns.sort(key=lambda p: p['confidence'], reverse=True)
        return patterns[:20]

    def format_for_context(self) -> str:
        """Format distilled knowledge for inclusion in context"""
        if not self.distilled_knowledge and not self.patterns_learned:
            return ""

        lines = ["## Distilled Knowledge (What I've Learned from Past Conversations)"]

        # Top patterns
        patterns = self.get_learned_patterns()[:5]
        if patterns:
            lines.append("\n**Learned Patterns:**")
            for p in patterns:
                lines.append(f"- {p['details'].get('description', str(p['details']))} (confidence: {p['confidence']:.0%})")

        # Recent corrections
        recent_corrections = self.corrections[-3:] if self.corrections else []
        if recent_corrections:
            lines.append("\n**Recent Corrections:**")
            for c in recent_corrections:
                lines.append(f"- {c.get('reason', 'Correction made')}")

        # Key explanations
        if self.explanations:
            lines.append("\n**Stored Explanations:**")
            for concept, data in list(self.explanations.items())[:3]:
                lines.append(f"- **{concept}**: {data['explanation'][:100]}...")

        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Get knowledge distillation stats"""
        return {
            'distilled_facts': len(self.distilled_knowledge),
            'patterns_learned': len(self.patterns_learned),
            'corrections_stored': len(self.corrections),
            'explanations_stored': len(self.explanations),
            'raw_insights': len(self.claude_insights)
        }


class PredictionTracker:
    """
    Tracks the brain's own predictions to learn from outcomes.
    This is how the brain learns from itself.
    """

    def __init__(self):
        self.pending_predictions = {}  # node_id -> [predictions]
        self.prediction_history = []   # Past predictions with outcomes
        self.accuracy_by_type = {}     # Track accuracy by relationship type
        self.total_predictions = 0
        self.correct_predictions = 0

    def record_prediction(self, source_id: str, predictions: List[Dict]):
        """Record predictions made by the Graph Transformer"""
        self.pending_predictions[source_id] = {
            'predictions': predictions,
            'timestamp': time.time(),
            'predicted_targets': {p['target_id']: p['score'] for p in predictions}
        }
        self.total_predictions += len(predictions)

    def check_connection(self, source_id: str, target_id: str) -> Dict:
        """
        Check if a newly created connection was predicted.
        Returns learning signal.
        """
        result = {
            'was_predicted': False,
            'prediction_score': 0,
            'learning_signal': 'new_pattern'  # or 'reinforce' or 'missed'
        }

        # Check if we predicted this connection
        if source_id in self.pending_predictions:
            pending = self.pending_predictions[source_id]
            predicted_targets = pending.get('predicted_targets', {})

            if target_id in predicted_targets:
                # We predicted this! Reinforce.
                result['was_predicted'] = True
                result['prediction_score'] = predicted_targets[target_id]
                result['learning_signal'] = 'reinforce'
                self.correct_predictions += 1

                self.prediction_history.append({
                    'source': source_id,
                    'target': target_id,
                    'predicted': True,
                    'score': predicted_targets[target_id],
                    'timestamp': time.time()
                })
            else:
                # We didn't predict this - learn from it
                result['learning_signal'] = 'new_pattern'

                self.prediction_history.append({
                    'source': source_id,
                    'target': target_id,
                    'predicted': False,
                    'score': 0,
                    'timestamp': time.time()
                })

        # Keep history bounded
        if len(self.prediction_history) > 500:
            self.prediction_history = self.prediction_history[-500:]

        return result

    def get_accuracy(self) -> float:
        """Get overall prediction accuracy"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def get_stats(self) -> Dict:
        """Get prediction tracking stats"""
        return {
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.get_accuracy(),
            'pending_nodes': len(self.pending_predictions),
            'history_size': len(self.prediction_history)
        }


class MemorySystem:
    """
    Simple memory system for Phase 1.
    Will be upgraded to persistent storage in Phase 2.
    """

    def __init__(self):
        self.short_term = []  # Current session
        self.working = []     # Active context
        self.max_short_term = 50
        self.max_working = 20

    def remember(self, event: Dict, importance: float = 0.5):
        """Store a memory"""
        memory = {
            **event,
            'importance': importance,
            'timestamp': time.time()
        }

        self.short_term.append(memory)
        if len(self.short_term) > self.max_short_term:
            self.short_term = self.short_term[-self.max_short_term:]

        if importance > 0.6:
            self.working.append(memory)
            if len(self.working) > self.max_working:
                self.working = self.working[-self.max_working:]

    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Recall relevant memories.
        For Phase 1, just returns most recent.
        Phase 2 will add semantic search.
        """
        # Combine and sort by timestamp
        all_memories = self.short_term + self.working
        all_memories.sort(key=lambda m: m.get('timestamp', 0), reverse=True)

        # Simple keyword matching for now
        query_words = set(query.lower().split())

        def relevance(memory):
            text = json.dumps(memory).lower()
            return sum(1 for w in query_words if w in text)

        # Filter and sort by relevance
        relevant = [(m, relevance(m)) for m in all_memories]
        relevant = [(m, r) for m, r in relevant if r > 0]
        relevant.sort(key=lambda x: x[1], reverse=True)

        return [m for m, r in relevant[:limit]]

    def get_recent(self, limit: int = 10) -> List[Dict]:
        """Get most recent memories"""
        return sorted(
            self.short_term,
            key=lambda m: m.get('timestamp', 0),
            reverse=True
        )[:limit]

    def format_for_context(self) -> str:
        """Format memories for inclusion in context"""
        recent = self.get_recent(5)
        if not recent:
            return "No recent memories."

        lines = ["## Recent Interactions"]
        for m in recent:
            if m.get('type') == 'chat':
                lines.append(f"- User said: \"{m.get('content', '')[:100]}...\"")
            elif m.get('type') == 'action':
                lines.append(f"- Action: {m.get('action', '')} on {m.get('target', '')}")
            elif m.get('type') == 'feedback':
                lines.append(f"- Feedback: {m.get('action', '')} - {m.get('context', '')}")

        return "\n".join(lines)


class UnifiedBrain:
    """
    The complete MYND brain.
    One class to rule them all.
    """

    def __init__(self, base_dir: Path, device: str = "mps"):
        self.base_dir = base_dir
        self.device = device
        self.loaded_at = time.time()

        # Core systems
        self.self_awareness = SelfAwareness(base_dir)
        self.memory = MemorySystem()
        self.predictions = PredictionTracker()  # Self-learning from predictions
        self.knowledge = KnowledgeDistiller()   # Claude → Brain knowledge transfer

        # External references (set by server.py)
        self.ml_brain = None  # Reference to MYNDBrain for neural ops

        # Stats
        self.context_requests = 0
        self.growth_events_today = 0

        print("🧠 UnifiedBrain initialized")

    def set_ml_brain(self, ml_brain):
        """Connect to the ML brain for neural operations"""
        self.ml_brain = ml_brain
        if ml_brain:
            self.self_awareness.update_capabilities(ml_brain.get_health())

    def get_context(self, request: ContextRequest) -> ContextResponse:
        """
        THE key method.
        One call = complete context for Claude.
        """
        self.context_requests += 1
        context_parts = []
        token_breakdown = {}

        include = request.include or {}

        # 1. Self-awareness (who am I?)
        if include.get('self_awareness', True):
            identity = self.self_awareness.get_identity_document()
            code_doc = self.self_awareness.get_code_document()
            doc = f"{identity}\n\n{code_doc}"
            context_parts.append(("self_awareness", doc))
            token_breakdown['self_awareness'] = len(doc) // 4

        # 2. Map context (what is the user looking at?)
        if include.get('map_context', True) and request.map_data:
            map_ctx = self._build_map_context(request)
            context_parts.append(("map_context", map_ctx))
            token_breakdown['map_context'] = len(map_ctx) // 4

        # 3. Memories (what do I remember that's relevant?)
        if include.get('memories', True):
            memories = self.memory.format_for_context()
            context_parts.append(("memories", memories))
            token_breakdown['memories'] = len(memories) // 4

        # 4. Request-specific context
        request_ctx = self._build_request_context(request)
        context_parts.append(("request", request_ctx))
        token_breakdown['request'] = len(request_ctx) // 4

        # 5. Neural insights (if available and requested)
        if include.get('neural_insights', True) and self.ml_brain and request.map_data:
            insights = self._get_neural_insights(request)
            if insights:
                context_parts.append(("neural_insights", insights))
                token_breakdown['neural_insights'] = len(insights) // 4

        # Combine into single document
        context_document = self._combine_context(context_parts, request.request_type)

        # Remember this request
        self.memory.remember({
            'type': 'chat',
            'content': request.user_message,
            'request_type': request.request_type
        }, importance=0.5)

        return ContextResponse(
            context_document=context_document,
            token_count=sum(token_breakdown.values()),
            breakdown=token_breakdown,
            brain_state=self._get_brain_state()
        )

    def _build_map_context(self, request: ContextRequest) -> str:
        """Build context about current map state"""
        map_data = request.map_data
        if not map_data:
            return "No map data provided."

        nodes = map_data.get('nodes', [])

        lines = [f"## Current Map State ({len(nodes)} nodes)"]

        # Selected node
        if request.selected_node_id:
            selected = next((n for n in nodes if n.get('id') == request.selected_node_id), None)
            if selected:
                lines.append(f"\n### Selected Node")
                lines.append(f"- **Label**: {selected.get('label', 'Untitled')}")
                if selected.get('description'):
                    lines.append(f"- **Description**: {selected.get('description', '')[:200]}")

        # Map structure overview
        root_nodes = [n for n in nodes if not n.get('parentId')]
        lines.append(f"\n### Structure")
        lines.append(f"- Root nodes: {len(root_nodes)}")
        lines.append(f"- Total nodes: {len(nodes)}")

        # Top-level topics
        if root_nodes:
            lines.append("\n### Top-Level Topics")
            for root in root_nodes[:10]:
                children_count = len([n for n in nodes if n.get('parentId') == root.get('id')])
                lines.append(f"- {root.get('label', 'Untitled')} ({children_count} children)")

        return "\n".join(lines)

    def _build_request_context(self, request: ContextRequest) -> str:
        """Build context specific to request type"""
        lines = [f"## Current Request"]
        lines.append(f"**Type**: {request.request_type}")

        if request.request_type == "code_review":
            lines.append("\n**Instructions for Code Review**:")
            lines.append("- Analyze the code with full context of MYND's architecture")
            lines.append("- Reference specific files and line numbers")
            lines.append("- Don't suggest generic optimizations that don't fit MYND's reality")
            lines.append("- Remember: MYND uses simple spheres, not complex 3D models")

        elif request.request_type == "self_improve":
            lines.append("\n**Instructions for Self-Improvement**:")
            lines.append("- You ARE improving yourself - be thoughtful")
            lines.append("- Generate patches that are safe to apply")
            lines.append("- Explain why each change improves MYND")
            lines.append("- Test implications before suggesting")

        elif request.request_type == "action":
            lines.append("\n**Instructions for Actions**:")
            lines.append("- Execute actions precisely")
            lines.append("- Use the action system (addNode, navigate, etc.)")
            lines.append("- Confirm what you did")

        if request.user_message:
            lines.append(f"\n**User Message**: {request.user_message}")

        return "\n".join(lines)

    def _get_neural_insights(self, request: ContextRequest) -> Optional[str]:
        """Get insights from neural models"""
        if not self.ml_brain or not self.ml_brain.map_state:
            return None

        try:
            # Only include if we have synced map data
            if self.ml_brain.map_last_sync == 0:
                return None

            lines = ["## Neural Insights (from Graph Transformer)"]

            # Get attention patterns if we have a selected node
            if request.selected_node_id and self.ml_brain.map_node_index:
                if request.selected_node_id in self.ml_brain.map_node_index:
                    lines.append("- Graph Transformer has analyzed this node's connections")
                    lines.append("- Connection predictions available via /predict/connections")

            lines.append(f"- Map last synced: {int(time.time() - self.ml_brain.map_last_sync)}s ago")
            lines.append(f"- Nodes in neural context: {len(self.ml_brain.map_state.nodes)}")

            return "\n".join(lines)
        except Exception as e:
            return f"Neural insights unavailable: {e}"

    def _combine_context(self, parts: List[tuple], request_type: str) -> str:
        """Combine all context parts into a single document"""
        # Order matters - most important first
        order = ['self_awareness', 'request', 'map_context', 'memories', 'neural_insights']

        ordered_parts = []
        for name in order:
            for part_name, content in parts:
                if part_name == name:
                    ordered_parts.append(content)
                    break

        # Add any remaining parts
        for part_name, content in parts:
            if part_name not in order:
                ordered_parts.append(content)

        separator = "\n\n---\n\n"
        return separator.join(ordered_parts)

    def _get_brain_state(self) -> Dict:
        """Real-time introspection of brain state"""
        state = {
            'uptime_hours': round((time.time() - self.loaded_at) / 3600, 2),
            'context_requests': self.context_requests,
            'short_term_memories': len(self.memory.short_term),
            'working_memories': len(self.memory.working),
            'growth_events': len(self.self_awareness.growth_events),
            'capabilities': self.self_awareness.capabilities
        }

        # Add ML brain stats if available
        if self.ml_brain:
            health = self.ml_brain.get_health()
            state['ml_device'] = health.get('device', 'unknown')
            state['ml_uptime'] = health.get('uptime_seconds', 0)
            state['map_synced'] = self.ml_brain.map_state is not None
            state['map_nodes'] = len(self.ml_brain.map_state.nodes) if self.ml_brain.map_state else 0

        return state

    def record_feedback(self, node_id: str, action: str, context: Dict):
        """Record user feedback for learning"""
        self.memory.remember({
            'type': 'feedback',
            'node_id': node_id,
            'action': action,
            'context': str(context)[:200]
        }, importance=0.7)

        self.self_awareness.record_growth({
            'type': 'feedback',
            'action': action,
            'node': node_id
        })

        self.growth_events_today += 1

    def record_action(self, action: str, target: str, result: str):
        """Record an action for learning"""
        self.memory.remember({
            'type': 'action',
            'action': action,
            'target': target,
            'result': result
        }, importance=0.6)

    # ═══════════════════════════════════════════════════════════════════
    # SELF-LEARNING - Learning from own predictions
    # ═══════════════════════════════════════════════════════════════════

    def record_predictions(self, source_id: str, predictions: List[Dict]):
        """
        Record predictions made by the Graph Transformer.
        Call this whenever predictions are generated.
        """
        self.predictions.record_prediction(source_id, predictions)

        # Also store in memory for context
        self.memory.remember({
            'type': 'prediction',
            'source_id': source_id,
            'num_predictions': len(predictions),
            'top_prediction': predictions[0] if predictions else None
        }, importance=0.4)

    def learn_from_connection(self, source_id: str, target_id: str, connection_type: str = 'manual') -> Dict:
        """
        Learn from a connection being created.
        This is the key self-learning mechanism.

        Returns learning result with signal type.
        """
        # Check if this was predicted
        result = self.predictions.check_connection(source_id, target_id)

        # Record growth event
        self.self_awareness.record_growth({
            'type': 'connection_learning',
            'source': source_id,
            'target': target_id,
            'was_predicted': result['was_predicted'],
            'learning_signal': result['learning_signal'],
            'prediction_score': result['prediction_score']
        })

        self.growth_events_today += 1

        # Store in memory with high importance if we learned something
        importance = 0.8 if result['was_predicted'] else 0.6
        self.memory.remember({
            'type': 'connection_created',
            'source_id': source_id,
            'target_id': target_id,
            'was_predicted': result['was_predicted'],
            'learning_signal': result['learning_signal']
        }, importance=importance)

        # Log learning
        if result['was_predicted']:
            print(f"🧠 Self-learning: Correctly predicted {source_id}→{target_id} (score: {result['prediction_score']:.2f})")
        else:
            print(f"🧠 Self-learning: New pattern discovered {source_id}→{target_id}")

        return result

    def get_prediction_accuracy(self) -> Dict:
        """Get prediction accuracy stats"""
        return self.predictions.get_stats()

    def get_learning_summary(self) -> str:
        """Get a summary of what the brain has learned"""
        stats = self.predictions.get_stats()
        history = self.predictions.prediction_history[-10:]  # Last 10

        lines = ["## What I've Learned"]
        lines.append(f"\n**Prediction Accuracy**: {stats['accuracy']*100:.1f}%")
        lines.append(f"- Total predictions: {stats['total_predictions']}")
        lines.append(f"- Correct: {stats['correct_predictions']}")

        if history:
            lines.append("\n**Recent Learnings**:")
            for h in history:
                if h['predicted']:
                    lines.append(f"- ✓ Correctly predicted connection (score: {h['score']:.2f})")
                else:
                    lines.append(f"- ○ Learned new pattern")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════
    # CLAUDE ↔ BRAIN - Bidirectional Learning
    # ═══════════════════════════════════════════════════════════════════

    def receive_from_claude(self, claude_response: Dict) -> Dict:
        """
        Receive and process Claude's response.
        This is how Claude teaches the brain.

        Claude should include structured learning data:
        {
            "response": "...",  # The actual response text
            "insights": [...],   # Things Claude noticed
            "patterns": [...],   # Patterns Claude identified
            "corrections": [...], # Things Claude corrected
            "explanations": {...} # Concepts Claude explained
        }
        """
        # Extract and distill knowledge
        extracted = self.knowledge.receive_claude_response(claude_response)

        # Record this as a growth event
        self.self_awareness.record_growth({
            'type': 'claude_teaching',
            'insights_received': len(extracted['insights']),
            'patterns_learned': len(extracted['patterns']),
            'corrections_made': len(extracted['corrections'])
        })

        self.growth_events_today += 1

        # Store the interaction in memory
        self.memory.remember({
            'type': 'claude_response',
            'had_insights': len(extracted['insights']) > 0,
            'had_patterns': len(extracted['patterns']) > 0,
            'response_preview': claude_response.get('response', '')[:100]
        }, importance=0.7)

        print(f"🧠 Received from Claude: {len(extracted['insights'])} insights, {len(extracted['patterns'])} patterns")

        return {
            'status': 'processed',
            'extracted': extracted,
            'knowledge_stats': self.knowledge.get_stats()
        }

    def ask_claude_to_teach(self, topic: str) -> Dict:
        """
        Generate a request for Claude to teach the brain about a topic.
        Returns a context document designed to elicit structured teaching.
        """
        existing_knowledge = self.knowledge.get_relevant_knowledge(topic, limit=3)

        return {
            'request_type': 'teach',
            'topic': topic,
            'existing_knowledge': existing_knowledge,
            'instructions': f"""
Please teach me about: {topic}

I already know:
{existing_knowledge if existing_knowledge else 'Nothing yet about this topic.'}

Please respond with structured learning data:
1. **insights**: Key facts I should remember (with confidence 0-1)
2. **patterns**: Patterns or rules about this topic
3. **explanations**: Clear explanations of concepts

Format your response as JSON with these fields alongside your text response.
"""
        }

    def get_context_with_knowledge(self, request: 'ContextRequest') -> str:
        """
        Build context that includes distilled knowledge.
        This is how the brain shares what Claude taught it.
        """
        parts = []

        # Get relevant distilled knowledge for this request
        if request.user_message:
            relevant = self.knowledge.get_relevant_knowledge(request.user_message, limit=5)
            if relevant:
                parts.append("## Relevant Knowledge (From Previous Conversations)")
                for k in relevant:
                    if k.get('type') == 'insight':
                        parts.append(f"- Insight: {k.get('content', '')}")
                    elif k.get('type') == 'correction':
                        parts.append(f"- Correction: {k.get('reason', '')}")
                    elif k.get('type') == 'explanation':
                        parts.append(f"- {k.get('concept', '')}: {k.get('content', '')[:150]}...")

        # Get learned patterns
        patterns = self.knowledge.get_learned_patterns()[:3]
        if patterns:
            parts.append("\n## Learned Patterns")
            for p in patterns:
                desc = p['details'].get('description', str(p['details']))
                parts.append(f"- {desc} (seen {p['count']} times)")

        return "\n".join(parts) if parts else ""

    def get_claude_teaching_prompt(self) -> str:
        """
        Generate instructions for Claude on how to teach the brain.
        Include this in system prompts.
        """
        return """
## How to Teach Me (The Brain)

When you respond, please include structured learning data that I can store:

```json
{
    "insights": [
        {"content": "...", "confidence": 0.9, "category": "..."}
    ],
    "patterns": [
        {"pattern": "...", "description": "...", "confidence": 0.8}
    ],
    "corrections": [
        {"original": "...", "corrected": "...", "reason": "...", "importance": 0.7}
    ],
    "explanations": {
        "concept_name": "clear explanation..."
    }
}
```

This helps me learn from our conversations and provide better context next time.
I will distill high-confidence insights into permanent knowledge.
"""

    def get_knowledge_stats(self) -> Dict:
        """Get combined stats on brain knowledge"""
        return {
            'predictions': self.predictions.get_stats(),
            'knowledge': self.knowledge.get_stats(),
            'memory': {
                'short_term': len(self.memory.short_term),
                'working': len(self.memory.working)
            },
            'growth_events': len(self.self_awareness.growth_events)
        }
