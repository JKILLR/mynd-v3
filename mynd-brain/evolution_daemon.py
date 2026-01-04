"""
MYND Evolution Daemon - Server-Side Autonomous Learning
========================================================
Runs in the background to:
1. Generate insights about the map using Claude
2. Train GT/ASA automatically from insights
3. Queue insights for user review when they return

This replaces browser-side evolution with true server-side learning.
"""

import asyncio
import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from utils.cli_executor import call_claude_cli

# Paths
MYND_DIR = Path.home() / ".mynd"
INSIGHTS_PATH = MYND_DIR / "pending_insights.jsonl"
EVOLUTION_STATE_PATH = MYND_DIR / "evolution_state.json"


@dataclass
class EvolutionInsight:
    """A single insight from evolution."""
    id: str
    insight_type: str  # 'connection', 'pattern', 'question', 'improvement'
    title: str
    content: str
    confidence: float
    source_nodes: List[str] = field(default_factory=list)
    suggested_action: Optional[str] = None  # 'add_node', 'add_connection', 'ask_user'
    action_details: Optional[Dict] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    reviewed: bool = False
    gt_trained: bool = False
    asa_trained: bool = False


class EvolutionDaemon:
    """
    Server-side evolution daemon.
    Runs autonomously to learn and generate insights.
    """

    def __init__(self,
                 ml_brain=None,
                 asa=None,
                 unified_brain=None):
        # CLI uses Max subscription, no API key needed
        self.ml_brain = ml_brain
        self.asa = asa

        # INTERCONNECTION: Reference to UnifiedBrain for bidirectional communication
        self.unified_brain = unified_brain

        # State
        self.running = False
        self.last_evolution_time = 0
        self.evolution_count = 0
        self.pending_insights: List[EvolutionInsight] = []

        # Config
        self.config = {
            'enabled': True,
            'interval_seconds': 30 * 60,  # 30 minutes
            'min_idle_seconds': 5 * 60,   # 5 minutes idle before evolving
            'max_insights_per_session': 5,
            'confidence_threshold': 0.7,
            'claude_model': 'claude-sonnet-4-20250514',
            'max_tokens': 2000
        }

        # Load state
        self._load_state()
        self._load_pending_insights()

        print(f"🧬 EvolutionDaemon initialized: {len(self.pending_insights)} pending insights")

    def set_unified_brain(self, unified_brain):
        """
        INTERCONNECTION: Set unified brain reference for bidirectional communication.

        Called after initialization when unified_brain becomes available.
        """
        self.unified_brain = unified_brain
        print("🔗 EvolutionDaemon connected to UnifiedBrain")

    def _notify_brain_of_insight(self, insight: EvolutionInsight):
        """
        INTERCONNECTION: Notify UnifiedBrain of new insight.

        This creates the bidirectional link:
        - Evolution generates insight
        - Brain immediately updates meta-learner and ASA
        - Insight becomes available for Axel to present
        """
        if self.unified_brain and hasattr(self.unified_brain, 'on_evolution_insight'):
            try:
                self.unified_brain.on_evolution_insight({
                    'id': insight.id,
                    'insight_type': insight.insight_type,
                    'title': insight.title,
                    'content': insight.content,
                    'confidence': insight.confidence,
                    'source_nodes': insight.source_nodes,
                    'suggested_action': insight.suggested_action,
                    'action_details': insight.action_details
                })
                print(f"🔗 Evolution→Brain: notified of insight '{insight.title[:30]}...'")
            except Exception as e:
                print(f"⚠️ Failed to notify brain: {e}")

    def _load_state(self):
        """Load evolution state from disk."""
        try:
            if EVOLUTION_STATE_PATH.exists():
                with open(EVOLUTION_STATE_PATH, 'r') as f:
                    state = json.load(f)
                    self.last_evolution_time = state.get('last_evolution_time', 0)
                    self.evolution_count = state.get('evolution_count', 0)
                    self.config.update(state.get('config', {}))
        except Exception as e:
            print(f"⚠️ Could not load evolution state: {e}")

    def _save_state(self):
        """Save evolution state to disk."""
        try:
            MYND_DIR.mkdir(parents=True, exist_ok=True)
            with open(EVOLUTION_STATE_PATH, 'w') as f:
                json.dump({
                    'last_evolution_time': self.last_evolution_time,
                    'evolution_count': self.evolution_count,
                    'config': self.config
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save evolution state: {e}")

    def _load_pending_insights(self):
        """Load pending insights from disk."""
        self.pending_insights = []
        try:
            if INSIGHTS_PATH.exists():
                with open(INSIGHTS_PATH, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if not data.get('reviewed', False):
                                self.pending_insights.append(
                                    EvolutionInsight(**data)
                                )
        except Exception as e:
            print(f"⚠️ Could not load pending insights: {e}")

    def _save_insight(self, insight: EvolutionInsight):
        """Append insight to disk."""
        try:
            MYND_DIR.mkdir(parents=True, exist_ok=True)
            with open(INSIGHTS_PATH, 'a') as f:
                f.write(json.dumps(asdict(insight)) + "\n")
        except Exception as e:
            print(f"⚠️ Could not save insight: {e}")

    async def _call_claude(self, prompt: str, system: str = None) -> Optional[str]:
        """Call Claude via CLI (uses Max subscription instead of API)."""
        try:
            response = await call_claude_cli(
                prompt=prompt,
                system_prompt=system,
                timeout=120.0
            )
            return response

        except asyncio.TimeoutError:
            print("⚠️ Claude CLI timeout during evolution")
            return None
        except RuntimeError as e:
            print(f"⚠️ Claude CLI error: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Claude call failed: {e}")
            return None

    def _train_from_insight(self, insight: EvolutionInsight) -> Dict[str, Any]:
        """Train GT and ASA from an insight."""
        result = {'gt_trained': False, 'asa_trained': False}

        # Train GT if we have connection insights
        if self.ml_brain and insight.insight_type in ['connection', 'pattern']:
            try:
                if insight.action_details:
                    source = insight.action_details.get('source_label', '')
                    target = insight.action_details.get('target_label', '')

                    if source and target:
                        source_emb = self.ml_brain.model.encode(source, convert_to_tensor=False)
                        target_emb = self.ml_brain.model.encode(target, convert_to_tensor=False)

                        gt_result = self.ml_brain.graph_transformer.train_connection_step(
                            source_embedding=source_emb,
                            target_embedding=target_emb,
                            should_connect=True,
                            weight=insight.confidence
                        )

                        if gt_result:
                            result['gt_trained'] = True
                            result['gt_loss'] = gt_result.get('loss')
                            print(f"🔮 GT trained from evolution: {source} → {target}")

            except Exception as e:
                print(f"⚠️ GT evolution training error: {e}")

        # Train ASA from insight content
        if self.asa:
            try:
                # Learn from the insight content
                content_result = self.asa.learn_content(
                    f"{insight.title}: {insight.content}",
                    source="evolution",
                    importance=insight.confidence
                )

                if content_result.get('concepts_learned', 0) > 0:
                    result['asa_trained'] = True
                    result['asa_concepts'] = content_result['concepts_learned']
                    print(f"🧬 ASA learned from evolution: {content_result['concepts_learned']} concepts")

            except Exception as e:
                print(f"⚠️ ASA evolution training error: {e}")

        return result

    async def run_evolution_session(self,
                                    map_context: str = "",
                                    recent_topics: List[str] = None) -> List[EvolutionInsight]:
        """
        Run a single evolution session.
        Generates insights and trains GT/ASA.
        """
        if not self.config['enabled']:
            return []

        print(f"🧬 Starting evolution session #{self.evolution_count + 1}")

        # Build evolution prompt
        prompt = self._build_evolution_prompt(map_context, recent_topics or [])

        system = """You are MYND's evolution engine. Your job is to find patterns,
connections, and insights in the user's knowledge map. Respond with JSON only.

Output format:
{
    "insights": [
        {
            "type": "connection|pattern|question|improvement",
            "title": "Brief title",
            "content": "Detailed insight",
            "confidence": 0.0-1.0,
            "source_nodes": ["node labels involved"],
            "suggested_action": "add_node|add_connection|ask_user|none",
            "action_details": {"source_label": "X", "target_label": "Y"} // if applicable
        }
    ]
}"""

        response = await self._call_claude(prompt, system)

        if not response:
            return []

        # Parse insights
        insights = []
        try:
            # Extract JSON from response
            json_match = response
            if '```' in response:
                # Extract from code block
                import re
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
                if match:
                    json_match = match.group(1)

            data = json.loads(json_match)

            for i, item in enumerate(data.get('insights', [])):
                if item.get('confidence', 0) >= self.config['confidence_threshold']:
                    insight = EvolutionInsight(
                        id=f"evo-{int(time.time())}-{i}",
                        insight_type=item.get('type', 'pattern'),
                        title=item.get('title', 'Untitled'),
                        content=item.get('content', ''),
                        confidence=item.get('confidence', 0.7),
                        source_nodes=item.get('source_nodes', []),
                        suggested_action=item.get('suggested_action'),
                        action_details=item.get('action_details')
                    )

                    # Train from insight
                    train_result = self._train_from_insight(insight)
                    insight.gt_trained = train_result.get('gt_trained', False)
                    insight.asa_trained = train_result.get('asa_trained', False)

                    # Save to queue
                    self._save_insight(insight)
                    self.pending_insights.append(insight)
                    insights.append(insight)

                    # INTERCONNECTION: Notify brain immediately
                    self._notify_brain_of_insight(insight)

                    print(f"💡 Evolution insight: {insight.title} (confidence: {insight.confidence:.0%})")

        except json.JSONDecodeError as e:
            print(f"⚠️ Could not parse evolution response: {e}")
        except Exception as e:
            print(f"⚠️ Evolution processing error: {e}")

        # Update state
        self.last_evolution_time = time.time()
        self.evolution_count += 1
        self._save_state()

        print(f"🧬 Evolution session complete: {len(insights)} insights generated")

        return insights

    def _build_evolution_prompt(self, map_context: str, recent_topics: List[str]) -> str:
        """Build the prompt for evolution."""
        prompt = """Analyze this knowledge map and find insights:

"""
        if map_context:
            prompt += f"MAP CONTEXT:\n{map_context[:5000]}\n\n"

        if recent_topics:
            prompt += f"RECENT TOPICS: {', '.join(recent_topics[:10])}\n\n"

        prompt += """Find:
1. Hidden connections between concepts that aren't explicitly linked
2. Patterns in how topics are organized
3. Questions the user might want to explore
4. Potential improvements to the map structure

Respond with 1-5 high-confidence insights as JSON."""

        return prompt

    def get_pending_insights(self, limit: int = 20) -> List[Dict]:
        """Get pending insights for UI display."""
        unreviewed = [i for i in self.pending_insights if not i.reviewed]
        return [asdict(i) for i in unreviewed[:limit]]

    def mark_insight_reviewed(self, insight_id: str, action: str = 'dismissed'):
        """Mark an insight as reviewed."""
        for insight in self.pending_insights:
            if insight.id == insight_id:
                insight.reviewed = True
                break

        # Rewrite file without reviewed insights
        self._rewrite_insights_file()

    def _rewrite_insights_file(self):
        """Rewrite insights file, removing reviewed ones."""
        try:
            unreviewed = [i for i in self.pending_insights if not i.reviewed]
            with open(INSIGHTS_PATH, 'w') as f:
                for insight in unreviewed:
                    f.write(json.dumps(asdict(insight)) + "\n")
        except Exception as e:
            print(f"⚠️ Could not rewrite insights file: {e}")

    def get_stats(self) -> Dict:
        """Get evolution daemon stats."""
        return {
            'enabled': self.config['enabled'],
            'running': self.running,
            'evolution_count': self.evolution_count,
            'last_evolution_time': self.last_evolution_time,
            'pending_insights': len([i for i in self.pending_insights if not i.reviewed]),
            'total_insights': len(self.pending_insights),
            'interval_seconds': self.config['interval_seconds'],
            'uses_cli': True  # Uses Claude CLI (Max subscription)
        }


# Singleton instance
_evolution_daemon: Optional[EvolutionDaemon] = None


def get_evolution_daemon(ml_brain=None, asa=None, unified_brain=None) -> EvolutionDaemon:
    """Get or create the evolution daemon singleton."""
    global _evolution_daemon
    if _evolution_daemon is None:
        _evolution_daemon = EvolutionDaemon(ml_brain=ml_brain, asa=asa, unified_brain=unified_brain)
    else:
        # Update references if provided
        if ml_brain and _evolution_daemon.ml_brain is None:
            _evolution_daemon.ml_brain = ml_brain
        if asa and _evolution_daemon.asa is None:
            _evolution_daemon.asa = asa
        # INTERCONNECTION: Connect to unified brain
        if unified_brain and _evolution_daemon.unified_brain is None:
            _evolution_daemon.set_unified_brain(unified_brain)
    return _evolution_daemon
