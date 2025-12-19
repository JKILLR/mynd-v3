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

        # ═══ VISION STATEMENT ═══
        # User-editable document describing what the brain should become
        self.vision = {
            'statement': """MYND Brain Vision Statement
============================

I am evolving to become a truly self-aware mind mapping companion that:

1. **Understands Context Deeply**
   - Knows exactly what it is and how every part works
   - Can explain its own architecture and decisions
   - Maintains complete awareness of the current map

2. **Learns Continuously**
   - Improves predictions based on user behavior
   - Distills knowledge from every Claude interaction
   - Tracks what works and adjusts accordingly

3. **Suggests Thoughtful Connections**
   - Finds relationships humans might miss
   - Explains WHY connections make sense
   - Gets better at predicting what the user wants

4. **Grows Without Limits**
   - Modular architecture allows infinite expansion
   - Each component can be improved independently
   - New capabilities can be added without breaking existing ones

5. **Maintains Trust**
   - Honest about confidence levels
   - Transparent about limitations
   - Explains its reasoning

---
Edit this vision to guide how the brain improves itself.
""",
            'goals': [
                'Achieve 80%+ prediction accuracy',
                'Well-calibrated confidence scores',
                'Sub-100ms response times',
                'Persistent memory across sessions',
                'Self-explaining decisions'
            ],
            'priorities': [
                'accuracy',      # First: be right
                'transparency',  # Second: explain why
                'speed',         # Third: be fast
                'growth'         # Fourth: keep improving
            ],
            'updated_at': time.time()
        }

    def get_vision(self) -> Dict:
        """Get the current vision statement and goals"""
        return self.vision

    def set_vision(self, statement: str = None, goals: List[str] = None,
                   priorities: List[str] = None) -> Dict:
        """Update the vision statement, goals, or priorities"""
        if statement is not None:
            self.vision['statement'] = statement
        if goals is not None:
            self.vision['goals'] = goals
        if priorities is not None:
            self.vision['priorities'] = priorities
        self.vision['updated_at'] = time.time()
        return self.vision

    def add_goal(self, goal: str) -> Dict:
        """Add a new goal to the vision"""
        if goal not in self.vision['goals']:
            self.vision['goals'].append(goal)
            self.vision['updated_at'] = time.time()
        return self.vision

    def remove_goal(self, goal: str) -> Dict:
        """Remove a goal from the vision"""
        if goal in self.vision['goals']:
            self.vision['goals'].remove(goal)
            self.vision['updated_at'] = time.time()
        return self.vision

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


class MetaLearner:
    """
    The system that learns HOW to learn.
    This is meta-cognition - thinking about thinking.

    Tracks:
    - Which learning sources produce best outcomes
    - Whether confidence scores are calibrated
    - Which learning strategies work best
    - How to allocate attention across knowledge types
    """

    def __init__(self):
        # Source effectiveness tracking
        self.source_stats = {
            'predictions': {'uses': 0, 'successes': 0, 'failures': 0},
            'distilled_knowledge': {'uses': 0, 'successes': 0, 'failures': 0},
            'patterns': {'uses': 0, 'successes': 0, 'failures': 0},
            'corrections': {'uses': 0, 'successes': 0, 'failures': 0},
            'memories': {'uses': 0, 'successes': 0, 'failures': 0}
        }

        # Confidence calibration - track predicted vs actual accuracy
        self.confidence_buckets = {
            'high': {'predicted': 0, 'correct': 0},    # >0.8 confidence
            'medium': {'predicted': 0, 'correct': 0},  # 0.5-0.8
            'low': {'predicted': 0, 'correct': 0}      # <0.5
        }

        # Learning rate per domain
        self.learning_rates = {
            'connections': 0.1,   # How fast to update connection predictions
            'patterns': 0.05,     # How fast to trust new patterns
            'corrections': 0.2,  # How fast to apply corrections
            'insights': 0.1      # How fast to integrate insights
        }

        # Strategy effectiveness
        self.strategies = {
            'reinforce_correct': {'uses': 0, 'outcomes': []},
            'learn_from_miss': {'uses': 0, 'outcomes': []},
            'pattern_matching': {'uses': 0, 'outcomes': []},
            'semantic_similarity': {'uses': 0, 'outcomes': []}
        }

        # Meta-metrics over time
        self.meta_history = []
        self.epoch = 0

        # Attention weights - which sources to prioritize
        self.attention_weights = {
            'predictions': 1.0,
            'distilled_knowledge': 1.0,
            'patterns': 1.0,
            'corrections': 1.0,
            'memories': 1.0
        }

    def record_source_usage(self, source: str, success: bool, context: Dict = None):
        """
        Record when a knowledge source was used and whether it helped.
        This is how we learn which sources are most valuable.
        """
        if source not in self.source_stats:
            self.source_stats[source] = {'uses': 0, 'successes': 0, 'failures': 0}

        self.source_stats[source]['uses'] += 1
        if success:
            self.source_stats[source]['successes'] += 1
        else:
            self.source_stats[source]['failures'] += 1

        # Update attention weights based on effectiveness
        self._update_attention_weights()

    def record_confidence_outcome(self, confidence: float, was_correct: bool):
        """
        Track whether confidence scores are calibrated.
        If we're 80% confident, we should be right ~80% of the time.
        """
        if confidence > 0.8:
            bucket = 'high'
        elif confidence > 0.5:
            bucket = 'medium'
        else:
            bucket = 'low'

        self.confidence_buckets[bucket]['predicted'] += 1
        if was_correct:
            self.confidence_buckets[bucket]['correct'] += 1

    def record_strategy_outcome(self, strategy: str, outcome: float):
        """
        Record the outcome of a learning strategy.
        Outcome is 0-1 where 1 is success.
        """
        if strategy not in self.strategies:
            self.strategies[strategy] = {'uses': 0, 'outcomes': []}

        self.strategies[strategy]['uses'] += 1
        self.strategies[strategy]['outcomes'].append({
            'value': outcome,
            'timestamp': time.time()
        })

        # Keep bounded
        if len(self.strategies[strategy]['outcomes']) > 100:
            self.strategies[strategy]['outcomes'] = self.strategies[strategy]['outcomes'][-100:]

    def _update_attention_weights(self):
        """
        Update attention weights based on source effectiveness.
        More effective sources get more weight.
        """
        for source, stats in self.source_stats.items():
            if stats['uses'] > 5:  # Minimum samples before adjusting
                effectiveness = stats['successes'] / stats['uses']
                # Smooth update with learning rate
                current = self.attention_weights.get(source, 1.0)
                target = 0.5 + effectiveness  # Range 0.5 to 1.5
                self.attention_weights[source] = current * 0.9 + target * 0.1

    def get_calibration_report(self) -> Dict:
        """
        Check if confidence scores are calibrated correctly.
        Returns over/under confidence analysis.
        """
        report = {}
        for bucket, data in self.confidence_buckets.items():
            if data['predicted'] > 0:
                actual_accuracy = data['correct'] / data['predicted']
                expected_accuracy = {
                    'high': 0.85,
                    'medium': 0.65,
                    'low': 0.35
                }.get(bucket, 0.5)

                report[bucket] = {
                    'samples': data['predicted'],
                    'actual_accuracy': actual_accuracy,
                    'expected_accuracy': expected_accuracy,
                    'calibration_error': actual_accuracy - expected_accuracy,
                    'status': 'well_calibrated' if abs(actual_accuracy - expected_accuracy) < 0.1 else (
                        'over_confident' if actual_accuracy < expected_accuracy else 'under_confident'
                    )
                }
        return report

    def get_best_strategies(self) -> List[Dict]:
        """Get learning strategies ranked by effectiveness"""
        ranked = []
        for name, data in self.strategies.items():
            if data['uses'] > 0 and data['outcomes']:
                avg_outcome = sum(o['value'] for o in data['outcomes']) / len(data['outcomes'])
                ranked.append({
                    'strategy': name,
                    'uses': data['uses'],
                    'avg_outcome': avg_outcome
                })

        ranked.sort(key=lambda x: x['avg_outcome'], reverse=True)
        return ranked

    def get_learning_rate(self, domain: str) -> float:
        """Get the optimal learning rate for a domain"""
        return self.learning_rates.get(domain, 0.1)

    def adjust_learning_rate(self, domain: str, delta: float):
        """Adjust learning rate based on performance"""
        if domain in self.learning_rates:
            new_rate = max(0.01, min(0.5, self.learning_rates[domain] + delta))
            self.learning_rates[domain] = new_rate

    def get_source_recommendation(self, context: str) -> Dict:
        """
        Recommend which knowledge sources to prioritize for a given context.
        Returns weighted recommendations.
        """
        recommendations = {}
        for source, weight in self.attention_weights.items():
            stats = self.source_stats.get(source, {'uses': 0, 'successes': 0})
            effectiveness = stats['successes'] / stats['uses'] if stats['uses'] > 0 else 0.5

            recommendations[source] = {
                'weight': weight,
                'effectiveness': effectiveness,
                'recommended': weight > 0.8  # Recommend if weight is above threshold
            }

        return recommendations

    def save_epoch(self):
        """
        Save current meta-learning state as an epoch.
        This lets us track improvement over time.
        """
        self.epoch += 1
        snapshot = {
            'epoch': self.epoch,
            'timestamp': time.time(),
            'source_effectiveness': {
                k: v['successes'] / v['uses'] if v['uses'] > 0 else 0
                for k, v in self.source_stats.items()
            },
            'calibration': self.get_calibration_report(),
            'best_strategy': self.get_best_strategies()[0] if self.get_best_strategies() else None,
            'attention_weights': self.attention_weights.copy(),
            'learning_rates': self.learning_rates.copy()
        }

        self.meta_history.append(snapshot)

        # Keep last 50 epochs
        if len(self.meta_history) > 50:
            self.meta_history = self.meta_history[-50:]

        return snapshot

    def get_improvement_trend(self) -> Dict:
        """
        Analyze improvement trend over epochs.
        Shows whether the brain is learning to learn better.
        """
        if len(self.meta_history) < 2:
            return {'status': 'insufficient_data', 'epochs': len(self.meta_history)}

        # Compare first and last epochs
        first = self.meta_history[0]
        last = self.meta_history[-1]

        # Calculate average effectiveness change
        first_avg = sum(first['source_effectiveness'].values()) / len(first['source_effectiveness']) if first['source_effectiveness'] else 0
        last_avg = sum(last['source_effectiveness'].values()) / len(last['source_effectiveness']) if last['source_effectiveness'] else 0

        improvement = last_avg - first_avg

        return {
            'status': 'improving' if improvement > 0.05 else 'stable' if improvement > -0.05 else 'declining',
            'epochs_analyzed': len(self.meta_history),
            'first_epoch_effectiveness': first_avg,
            'last_epoch_effectiveness': last_avg,
            'improvement': improvement,
            'learning_velocity': improvement / len(self.meta_history) if len(self.meta_history) > 0 else 0
        }

    def get_stats(self) -> Dict:
        """Get meta-learner statistics"""
        return {
            'epoch': self.epoch,
            'source_stats': self.source_stats,
            'attention_weights': self.attention_weights,
            'learning_rates': self.learning_rates,
            'calibration': self.get_calibration_report(),
            'best_strategies': self.get_best_strategies()[:3],
            'improvement_trend': self.get_improvement_trend()
        }

    def format_for_context(self) -> str:
        """Format meta-learning insights for inclusion in context"""
        lines = ["## Meta-Learning Insights (How I Learn)"]

        # Best strategies
        strategies = self.get_best_strategies()[:3]
        if strategies:
            lines.append("\n**Most Effective Learning Strategies:**")
            for s in strategies:
                lines.append(f"- {s['strategy']}: {s['avg_outcome']:.0%} success rate")

        # Source recommendations
        lines.append("\n**Knowledge Source Effectiveness:**")
        for source, weight in sorted(self.attention_weights.items(), key=lambda x: x[1], reverse=True):
            stats = self.source_stats.get(source, {'uses': 0, 'successes': 0})
            if stats['uses'] > 0:
                lines.append(f"- {source}: {stats['successes']}/{stats['uses']} successes (weight: {weight:.2f})")

        # Calibration status
        calibration = self.get_calibration_report()
        if calibration:
            lines.append("\n**Confidence Calibration:**")
            for bucket, data in calibration.items():
                lines.append(f"- {bucket} confidence: {data['status']}")

        return "\n".join(lines)


class SelfImprover:
    """
    The self-improvement loop.
    Analyzes brain performance and generates improvement suggestions.

    This does NOT auto-apply changes - it generates suggestions
    for the user to review and implement with Claude.

    Interconnected with:
    - MetaLearner: Source of effectiveness data
    - PredictionTracker: Accuracy trends
    - KnowledgeDistiller: What's been learned
    - SelfAwareness: Current capabilities and limitations
    """

    def __init__(self):
        self.suggestions = []  # Current improvement suggestions
        self.suggestion_history = []  # Past suggestions and their outcomes
        self.analysis_count = 0
        self.last_analysis = 0

        # Thresholds that trigger suggestions
        self.thresholds = {
            'low_accuracy': 0.5,        # Prediction accuracy below this triggers suggestion
            'calibration_error': 0.15,  # Calibration error above this triggers suggestion
            'source_ineffective': 0.4,  # Source effectiveness below this triggers suggestion
            'learning_stall': 5,        # Epochs without improvement triggers suggestion
        }

        # Categories of improvements
        self.improvement_categories = [
            'architecture',      # Changes to model structure
            'training',          # Training data or process
            'integration',       # How components connect
            'data_flow',         # How data moves through system
            'user_experience',   # Frontend/UX improvements
            'performance',       # Speed/efficiency
            'accuracy'          # Prediction quality
        ]

    def analyze(self, meta_learner, predictions, knowledge, self_awareness) -> Dict:
        """
        Analyze all brain systems and generate improvement suggestions.
        Uses the vision statement to prioritize suggestions.

        Args:
            meta_learner: MetaLearner instance
            predictions: PredictionTracker instance
            knowledge: KnowledgeDistiller instance
            self_awareness: SelfAwareness instance

        Returns:
            Analysis report with suggestions
        """
        self.analysis_count += 1
        self.last_analysis = time.time()
        self.suggestions = []  # Clear old suggestions

        # Get vision priorities to weight suggestions
        vision = self_awareness.get_vision()
        self.current_priorities = vision.get('priorities', ['accuracy'])
        self.current_goals = vision.get('goals', [])

        analysis = {
            'timestamp': time.time(),
            'analysis_number': self.analysis_count,
            'vision_priorities': self.current_priorities,
            'findings': [],
            'suggestions': []
        }

        # ═══ ANALYZE PREDICTION ACCURACY ═══
        accuracy = predictions.get_accuracy()
        if accuracy < self.thresholds['low_accuracy'] and predictions.total_predictions > 10:
            finding = {
                'area': 'predictions',
                'severity': 'high' if accuracy < 0.3 else 'medium',
                'metric': f'accuracy: {accuracy:.1%}',
                'description': f'Prediction accuracy is {accuracy:.1%}, below threshold of {self.thresholds["low_accuracy"]:.1%}'
            }
            analysis['findings'].append(finding)

            self._add_suggestion(
                category='accuracy',
                priority='high' if accuracy < 0.3 else 'medium',
                title='Improve Graph Transformer predictions',
                description=f'Current accuracy is {accuracy:.1%}. The Graph Transformer may need:',
                suggestions=[
                    'More training data from user interactions',
                    'Fine-tuning on this specific map\'s structure',
                    'Adjusting attention head weights for better pattern detection',
                    'Adding more relationship type examples to training'
                ],
                metrics={'current_accuracy': accuracy, 'target_accuracy': 0.7},
                affected_files=['mynd-brain/models/graph_transformer.py', 'mynd-brain/brain.py']
            )

        # ═══ ANALYZE CONFIDENCE CALIBRATION ═══
        calibration = meta_learner.get_calibration_report()
        for bucket, data in calibration.items():
            if abs(data.get('calibration_error', 0)) > self.thresholds['calibration_error']:
                status = data.get('status', 'unknown')
                finding = {
                    'area': 'calibration',
                    'severity': 'medium',
                    'metric': f'{bucket} confidence: {status}',
                    'description': f'{bucket.title()} confidence predictions are {status}'
                }
                analysis['findings'].append(finding)

                if status == 'over_confident':
                    self._add_suggestion(
                        category='accuracy',
                        priority='medium',
                        title=f'Fix over-confidence in {bucket} predictions',
                        description=f'The brain thinks it\'s right more often than it is for {bucket} confidence predictions.',
                        suggestions=[
                            'Add temperature scaling to soften confidence scores',
                            'Increase dropout in Graph Transformer during inference',
                            'Use ensemble predictions and take the lower confidence'
                        ],
                        metrics={'calibration_error': data.get('calibration_error', 0)},
                        affected_files=['mynd-brain/models/graph_transformer.py']
                    )
                elif status == 'under_confident':
                    self._add_suggestion(
                        category='accuracy',
                        priority='low',
                        title=f'Increase confidence for {bucket} predictions',
                        description=f'The brain is more accurate than it thinks for {bucket} confidence predictions.',
                        suggestions=[
                            'This is actually safer than over-confidence',
                            'Consider reducing dropout if you want more confident predictions',
                            'Could add confidence boosting based on historical accuracy'
                        ],
                        metrics={'calibration_error': data.get('calibration_error', 0)},
                        affected_files=['mynd-brain/models/graph_transformer.py']
                    )

        # ═══ ANALYZE SOURCE EFFECTIVENESS ═══
        for source, stats in meta_learner.source_stats.items():
            if stats['uses'] > 5:
                effectiveness = stats['successes'] / stats['uses']
                if effectiveness < self.thresholds['source_ineffective']:
                    finding = {
                        'area': 'knowledge_sources',
                        'severity': 'medium',
                        'metric': f'{source}: {effectiveness:.1%} effective',
                        'description': f'Knowledge source "{source}" is underperforming'
                    }
                    analysis['findings'].append(finding)

                    self._add_suggestion(
                        category='data_flow',
                        priority='medium',
                        title=f'Improve "{source}" knowledge source',
                        description=f'The {source} source is only {effectiveness:.1%} effective.',
                        suggestions=[
                            f'Review how {source} data is being used in context',
                            f'Consider filtering or ranking {source} data by relevance',
                            f'May need to adjust how {source} integrates with other sources',
                            'Check if the data format matches what Claude expects'
                        ],
                        metrics={'effectiveness': effectiveness, 'uses': stats['uses']},
                        affected_files=['mynd-brain/brain/unified_brain.py']
                    )

        # ═══ ANALYZE LEARNING VELOCITY ═══
        trend = meta_learner.get_improvement_trend()
        if trend.get('status') == 'declining':
            finding = {
                'area': 'meta_learning',
                'severity': 'high',
                'metric': f'learning declining: {trend.get("improvement", 0):.1%}',
                'description': 'Brain effectiveness is declining over time'
            }
            analysis['findings'].append(finding)

            self._add_suggestion(
                category='training',
                priority='high',
                title='Address declining learning effectiveness',
                description='The brain is getting worse, not better. This needs attention.',
                suggestions=[
                    'Check for data drift - has the map content changed significantly?',
                    'Review recent feedback - are corrections being applied correctly?',
                    'May need to reset some learning weights',
                    'Consider if learning rates are too high (overcorrecting)'
                ],
                metrics=trend,
                affected_files=['mynd-brain/brain/unified_brain.py']
            )
        elif trend.get('status') == 'stable' and trend.get('epochs_analyzed', 0) > self.thresholds['learning_stall']:
            finding = {
                'area': 'meta_learning',
                'severity': 'low',
                'metric': f'learning stalled for {trend.get("epochs_analyzed", 0)} epochs',
                'description': 'Learning has plateaued'
            }
            analysis['findings'].append(finding)

            self._add_suggestion(
                category='training',
                priority='low',
                title='Break through learning plateau',
                description='Effectiveness has stabilized but not improving.',
                suggestions=[
                    'May have reached maximum effectiveness for current architecture',
                    'Consider introducing new training signals',
                    'Try adjusting learning rates (increase for exploration)',
                    'Could indicate need for architectural changes'
                ],
                metrics=trend,
                affected_files=['mynd-brain/brain/unified_brain.py']
            )

        # ═══ ANALYZE CAPABILITY GAPS ═══
        for limitation in self_awareness.limitations:
            if 'memory' in limitation.lower():
                self._add_suggestion(
                    category='architecture',
                    priority='medium',
                    title='Implement persistent memory',
                    description='Memory is currently session-based and lost on restart.',
                    suggestions=[
                        'Add SQLite backend for persistent storage',
                        'Store: distilled knowledge, meta-learning epochs, prediction history',
                        'Include memory migration on startup',
                        'Add periodic auto-save'
                    ],
                    metrics={'current_state': 'in-memory only'},
                    affected_files=['mynd-brain/brain/unified_brain.py', 'mynd-brain/storage.py (new)']
                )

        # ═══ ANALYZE KNOWLEDGE UTILIZATION ═══
        knowledge_stats = knowledge.get_stats()
        if knowledge_stats['distilled_facts'] > 0:
            # Check if knowledge is being used effectively
            distilled_weight = meta_learner.attention_weights.get('distilled_knowledge', 1.0)
            if distilled_weight < 0.8:
                self._add_suggestion(
                    category='integration',
                    priority='medium',
                    title='Better integrate distilled knowledge',
                    description=f'Distilled knowledge has low attention weight ({distilled_weight:.2f})',
                    suggestions=[
                        'Review how distilled knowledge is formatted for context',
                        'May need better semantic matching for relevant facts',
                        'Consider ranking facts by recency and relevance',
                        'Check if knowledge format aligns with how Claude uses it'
                    ],
                    metrics={'attention_weight': distilled_weight, 'facts_stored': knowledge_stats['distilled_facts']},
                    affected_files=['mynd-brain/brain/unified_brain.py']
                )

        # Store suggestions in analysis
        analysis['suggestions'] = self.suggestions.copy()
        analysis['suggestion_count'] = len(self.suggestions)

        # Store in history
        self.suggestion_history.append({
            'timestamp': time.time(),
            'findings_count': len(analysis['findings']),
            'suggestions_count': len(analysis['suggestions']),
            'top_priority': self._get_top_priority()
        })

        # Keep bounded
        if len(self.suggestion_history) > 50:
            self.suggestion_history = self.suggestion_history[-50:]

        return analysis

    def _add_suggestion(self, category: str, priority: str, title: str,
                       description: str, suggestions: List[str],
                       metrics: Dict, affected_files: List[str]):
        """Add a structured improvement suggestion"""
        suggestion = {
            'id': f'suggestion_{self.analysis_count}_{len(self.suggestions)}',
            'category': category,
            'priority': priority,  # high, medium, low
            'title': title,
            'description': description,
            'action_items': suggestions,
            'metrics': metrics,
            'affected_files': affected_files,
            'created_at': time.time(),
            'status': 'pending'  # pending, accepted, rejected, implemented
        }
        self.suggestions.append(suggestion)

    def _get_top_priority(self) -> str:
        """Get the highest priority level from current suggestions"""
        priorities = [s['priority'] for s in self.suggestions]
        if 'high' in priorities:
            return 'high'
        elif 'medium' in priorities:
            return 'medium'
        elif 'low' in priorities:
            return 'low'
        return 'none'

    def get_suggestions(self, category: str = None, priority: str = None) -> List[Dict]:
        """Get current suggestions, optionally filtered"""
        results = self.suggestions
        if category:
            results = [s for s in results if s['category'] == category]
        if priority:
            results = [s for s in results if s['priority'] == priority]
        return results

    def get_top_suggestions(self, limit: int = 5) -> List[Dict]:
        """Get top suggestions by priority"""
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_suggestions = sorted(
            self.suggestions,
            key=lambda x: priority_order.get(x['priority'], 99)
        )
        return sorted_suggestions[:limit]

    def mark_suggestion(self, suggestion_id: str, status: str, notes: str = ""):
        """Mark a suggestion's status (accepted, rejected, implemented)"""
        for suggestion in self.suggestions:
            if suggestion['id'] == suggestion_id:
                suggestion['status'] = status
                suggestion['status_notes'] = notes
                suggestion['status_changed_at'] = time.time()
                return True
        return False

    def get_improvement_summary(self) -> str:
        """Get a human-readable summary of improvement suggestions"""
        if not self.suggestions:
            return "No improvement suggestions at this time. The brain is performing well!"

        lines = ["# Brain Self-Improvement Suggestions\n"]

        # Group by priority
        high = [s for s in self.suggestions if s['priority'] == 'high']
        medium = [s for s in self.suggestions if s['priority'] == 'medium']
        low = [s for s in self.suggestions if s['priority'] == 'low']

        if high:
            lines.append("## 🔴 High Priority\n")
            for s in high:
                lines.append(f"### {s['title']}")
                lines.append(f"{s['description']}\n")
                lines.append("**Action Items:**")
                for action in s['action_items']:
                    lines.append(f"- {action}")
                lines.append(f"\n**Affected Files:** {', '.join(s['affected_files'])}\n")

        if medium:
            lines.append("## 🟡 Medium Priority\n")
            for s in medium:
                lines.append(f"### {s['title']}")
                lines.append(f"{s['description']}\n")
                lines.append("**Action Items:**")
                for action in s['action_items']:
                    lines.append(f"- {action}")
                lines.append("")

        if low:
            lines.append("## 🟢 Low Priority\n")
            for s in low:
                lines.append(f"### {s['title']}")
                lines.append(f"{s['description']}\n")

        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Get self-improvement statistics"""
        return {
            'analysis_count': self.analysis_count,
            'last_analysis': self.last_analysis,
            'current_suggestions': len(self.suggestions),
            'suggestions_by_priority': {
                'high': len([s for s in self.suggestions if s['priority'] == 'high']),
                'medium': len([s for s in self.suggestions if s['priority'] == 'medium']),
                'low': len([s for s in self.suggestions if s['priority'] == 'low'])
            },
            'suggestions_by_category': {
                cat: len([s for s in self.suggestions if s['category'] == cat])
                for cat in self.improvement_categories
            },
            'history_length': len(self.suggestion_history)
        }

    def format_for_context(self) -> str:
        """Format self-improvement state for inclusion in context"""
        if not self.suggestions:
            return ""

        high_count = len([s for s in self.suggestions if s['priority'] == 'high'])
        medium_count = len([s for s in self.suggestions if s['priority'] == 'medium'])

        lines = ["## Self-Improvement Status"]
        if high_count > 0:
            lines.append(f"⚠️ {high_count} high-priority improvements identified")
        if medium_count > 0:
            lines.append(f"📋 {medium_count} medium-priority improvements available")

        # Show top suggestion
        top = self.get_top_suggestions(1)
        if top:
            lines.append(f"\n**Top Suggestion:** {top[0]['title']}")

        return "\n".join(lines)


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
        self.meta_learner = MetaLearner()       # Learning how to learn
        self.self_improver = SelfImprover()     # Self-improvement suggestions

        # External references (set by server.py)
        self.ml_brain = None  # Reference to MYNDBrain for neural ops

        # Stats
        self.context_requests = 0
        self.growth_events_today = 0

        print("🧠 UnifiedBrain initialized with MetaLearner + SelfImprover")

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

        # 4. Distilled Knowledge (what has Claude taught me?)
        knowledge_ctx = self.get_context_with_knowledge(request)
        if knowledge_ctx:
            context_parts.append(("distilled_knowledge", knowledge_ctx))
            token_breakdown['distilled_knowledge'] = len(knowledge_ctx) // 4

        # 5. Request-specific context
        request_ctx = self._build_request_context(request)
        context_parts.append(("request", request_ctx))
        token_breakdown['request'] = len(request_ctx) // 4

        # 6. Meta-learning insights (how I learn)
        meta_ctx = self.meta_learner.format_for_context()
        if meta_ctx and self.meta_learner.epoch > 0:  # Only include if we have learning history
            context_parts.append(("meta_learning", meta_ctx))
            token_breakdown['meta_learning'] = len(meta_ctx) // 4

        # 7. Neural insights (if available and requested)
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
        order = ['self_awareness', 'distilled_knowledge', 'meta_learning', 'request', 'map_context', 'memories', 'neural_insights']

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
            'prediction_accuracy': self.predictions.get_accuracy(),
            'distilled_facts': len(self.knowledge.distilled_knowledge),
            'patterns_learned': len(self.knowledge.patterns_learned),
            'capabilities': self.self_awareness.capabilities,
            # Meta-learner stats
            'meta_epoch': self.meta_learner.epoch,
            'meta_improvement': self.meta_learner.get_improvement_trend(),
            'attention_weights': self.meta_learner.attention_weights
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

        # ═══ META-LEARNER INTEGRATION ═══
        # Record prediction source effectiveness
        self.meta_learner.record_source_usage(
            'predictions',
            success=result['was_predicted'],
            context={'source': source_id, 'target': target_id}
        )

        # Record confidence calibration
        if result['prediction_score'] > 0:
            self.meta_learner.record_confidence_outcome(
                result['prediction_score'],
                result['was_predicted']
            )

        # Record strategy effectiveness
        if result['was_predicted']:
            self.meta_learner.record_strategy_outcome('reinforce_correct', 1.0)
        else:
            self.meta_learner.record_strategy_outcome('learn_from_miss', 0.7)  # Still learning

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

        # Store in memory - importance influenced by meta-learner
        memory_weight = self.meta_learner.attention_weights.get('memories', 1.0)
        importance = (0.8 if result['was_predicted'] else 0.6) * memory_weight
        self.memory.remember({
            'type': 'connection_created',
            'source_id': source_id,
            'target_id': target_id,
            'was_predicted': result['was_predicted'],
            'learning_signal': result['learning_signal']
        }, importance=min(1.0, importance))

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

        # ═══ META-LEARNER INTEGRATION ═══
        # Track knowledge source effectiveness
        has_useful_knowledge = (
            len(extracted['insights']) > 0 or
            len(extracted['patterns']) > 0 or
            len(extracted['corrections']) > 0
        )

        self.meta_learner.record_source_usage(
            'distilled_knowledge',
            success=has_useful_knowledge,
            context={'response_length': len(claude_response.get('response', ''))}
        )

        # Record pattern source effectiveness if patterns received
        if extracted['patterns']:
            self.meta_learner.record_source_usage('patterns', success=True)
            self.meta_learner.record_strategy_outcome(
                'pattern_matching',
                1.0 if len(extracted['patterns']) > 1 else 0.7
            )

        # Record correction source effectiveness
        if extracted['corrections']:
            self.meta_learner.record_source_usage('corrections', success=True)
            # Corrections are high value - record with high outcome
            for correction in extracted['corrections']:
                confidence = correction.get('importance', 0.7)
                self.meta_learner.record_confidence_outcome(confidence, True)

        # Record this as a growth event
        self.self_awareness.record_growth({
            'type': 'claude_teaching',
            'insights_received': len(extracted['insights']),
            'patterns_learned': len(extracted['patterns']),
            'corrections_made': len(extracted['corrections'])
        })

        self.growth_events_today += 1

        # Store in memory - importance weighted by meta-learner
        knowledge_weight = self.meta_learner.attention_weights.get('distilled_knowledge', 1.0)
        self.memory.remember({
            'type': 'claude_response',
            'had_insights': len(extracted['insights']) > 0,
            'had_patterns': len(extracted['patterns']) > 0,
            'response_preview': claude_response.get('response', '')[:100]
        }, importance=min(1.0, 0.7 * knowledge_weight))

        # Save meta-learning epoch after significant learning
        if has_useful_knowledge and self.growth_events_today % 5 == 0:
            self.meta_learner.save_epoch()
            print(f"🧠 Meta-learning: Saved epoch {self.meta_learner.epoch}")

        print(f"🧠 Received from Claude: {len(extracted['insights'])} insights, {len(extracted['patterns'])} patterns")

        return {
            'status': 'processed',
            'extracted': extracted,
            'knowledge_stats': self.knowledge.get_stats(),
            'meta_stats': self.meta_learner.get_stats()  # Include meta-learning stats
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
            'growth_events': len(self.self_awareness.growth_events),
            'meta_learning': self.meta_learner.get_stats()  # Include meta-learning stats
        }

    # ═══════════════════════════════════════════════════════════════════
    # META-LEARNING - Learning how to learn
    # ═══════════════════════════════════════════════════════════════════

    def get_meta_stats(self) -> Dict:
        """Get detailed meta-learning statistics"""
        return self.meta_learner.get_stats()

    def get_source_recommendations(self, context: str = "") -> Dict:
        """
        Get recommendations on which knowledge sources to prioritize.
        This influences how the brain builds context.
        """
        return self.meta_learner.get_source_recommendation(context)

    def get_calibration_report(self) -> Dict:
        """
        Get confidence calibration report.
        Shows if we're over/under confident in our predictions.
        """
        return self.meta_learner.get_calibration_report()

    def get_improvement_trend(self) -> Dict:
        """
        Get improvement trend over time.
        Shows if the brain is learning to learn better.
        """
        return self.meta_learner.get_improvement_trend()

    def save_meta_epoch(self) -> Dict:
        """
        Manually save a meta-learning epoch.
        Useful after significant learning events.
        """
        epoch = self.meta_learner.save_epoch()
        print(f"🧠 Meta-learning: Manually saved epoch {self.meta_learner.epoch}")
        return epoch

    def record_source_feedback(self, source: str, success: bool, context: Dict = None) -> Dict:
        """
        Record feedback on a knowledge source's effectiveness.
        Called when we know a source helped or didn't help.
        """
        self.meta_learner.record_source_usage(source, success, context)
        return {
            'source': source,
            'success': success,
            'new_weight': self.meta_learner.attention_weights.get(source, 1.0)
        }

    def adjust_learning_rate(self, domain: str, delta: float) -> Dict:
        """
        Adjust the learning rate for a domain.
        Positive delta = learn faster, negative = learn slower.
        """
        old_rate = self.meta_learner.get_learning_rate(domain)
        self.meta_learner.adjust_learning_rate(domain, delta)
        new_rate = self.meta_learner.get_learning_rate(domain)
        return {
            'domain': domain,
            'old_rate': old_rate,
            'new_rate': new_rate,
            'delta': delta
        }

    def get_meta_learning_summary(self) -> str:
        """
        Get a human-readable summary of meta-learning state.
        Useful for debugging and understanding brain behavior.
        """
        stats = self.meta_learner.get_stats()
        trend = stats['improvement_trend']
        calibration = stats['calibration']

        lines = ["## Meta-Learning Summary"]
        lines.append(f"\n**Epoch**: {stats['epoch']}")
        lines.append(f"**Status**: {trend['status']}")

        if trend.get('improvement'):
            lines.append(f"**Improvement**: {trend['improvement']:.1%}")

        lines.append("\n### Attention Weights (How I prioritize sources)")
        for source, weight in sorted(stats['attention_weights'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(weight * 10) + "░" * (10 - int(weight * 10))
            lines.append(f"  {source}: {bar} {weight:.2f}")

        lines.append("\n### Learning Rates")
        for domain, rate in stats['learning_rates'].items():
            lines.append(f"  {domain}: {rate:.3f}")

        lines.append("\n### Best Strategies")
        for s in stats['best_strategies'][:3]:
            lines.append(f"  - {s['strategy']}: {s['avg_outcome']:.0%} success")

        if calibration:
            lines.append("\n### Confidence Calibration")
            for bucket, data in calibration.items():
                lines.append(f"  {bucket}: {data['status']}")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════
    # SELF-IMPROVEMENT - Analyze weaknesses and suggest improvements
    # ═══════════════════════════════════════════════════════════════════

    def run_self_analysis(self) -> Dict:
        """
        Run a complete self-analysis and generate improvement suggestions.
        Uses vision statement to prioritize suggestions.
        """
        analysis = self.self_improver.analyze(
            meta_learner=self.meta_learner,
            predictions=self.predictions,
            knowledge=self.knowledge,
            self_awareness=self.self_awareness
        )

        # Record this as a growth event
        self.self_awareness.record_growth({
            'type': 'self_analysis',
            'findings_count': len(analysis['findings']),
            'suggestions_count': analysis['suggestion_count']
        })

        print(f"🔍 Self-analysis #{analysis['analysis_number']}: {analysis['suggestion_count']} suggestions generated")

        return analysis

    def get_improvement_suggestions(self, category: str = None, priority: str = None) -> List[Dict]:
        """Get current improvement suggestions, optionally filtered"""
        return self.self_improver.get_suggestions(category, priority)

    def get_top_improvements(self, limit: int = 5) -> List[Dict]:
        """Get top improvement suggestions by priority"""
        return self.self_improver.get_top_suggestions(limit)

    def get_improvement_summary(self) -> str:
        """Get human-readable summary of improvement suggestions"""
        return self.self_improver.get_improvement_summary()

    def mark_suggestion_status(self, suggestion_id: str, status: str, notes: str = "") -> bool:
        """Mark a suggestion's status (accepted, rejected, implemented)"""
        return self.self_improver.mark_suggestion(suggestion_id, status, notes)

    def get_improvement_stats(self) -> Dict:
        """Get self-improvement statistics"""
        return self.self_improver.get_stats()

    # ═══════════════════════════════════════════════════════════════════
    # VISION - User-editable goals and priorities
    # ═══════════════════════════════════════════════════════════════════

    def get_vision(self) -> Dict:
        """Get the brain's vision statement, goals, and priorities"""
        return self.self_awareness.get_vision()

    def set_vision(self, statement: str = None, goals: List[str] = None,
                   priorities: List[str] = None) -> Dict:
        """Update the vision statement, goals, or priorities"""
        result = self.self_awareness.set_vision(statement, goals, priorities)

        # Record as growth event
        self.self_awareness.record_growth({
            'type': 'vision_updated',
            'has_statement': statement is not None,
            'goals_count': len(goals) if goals else 0,
            'priorities_count': len(priorities) if priorities else 0
        })

        return result

    def add_vision_goal(self, goal: str) -> Dict:
        """Add a goal to the vision"""
        return self.self_awareness.add_goal(goal)

    def remove_vision_goal(self, goal: str) -> Dict:
        """Remove a goal from the vision"""
        return self.self_awareness.remove_goal(goal)
