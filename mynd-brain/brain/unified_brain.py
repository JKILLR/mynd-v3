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
