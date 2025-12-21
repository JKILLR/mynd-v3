"""
Context Synthesizer
===================
Unifies all context sources into ONE coherent context block.

Instead of 19 separate layers dumping into the prompt, this synthesizer:
1. Uses HYBRID SEARCH (vector similarity + BM25 keyword matching)
2. Ranks results by relevance (not by source type)
3. Uses EXPONENTIAL DECAY for recency (half-life formula, not linear)
4. Positions high-relevance items at START and END (avoids "Lost in the Middle")
5. Links to active goals automatically

This is the "funnel" that turns fragmented data into focused understanding.
"""

import time
import math
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
import numpy as np


@dataclass
class ContextItem:
    """A single piece of context from any source"""
    content: str
    source_type: str  # 'map_node', 'ai_memory', 'conversation', 'pattern', 'goal'
    relevance_score: float  # 0-1, from embedding similarity
    importance: float  # 0-1, from source (e.g., memory importance)
    recency: float  # 0-1, how recent (1 = now, 0 = old)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def combined_score(self) -> float:
        """Combined ranking score: relevance * importance * recency_boost"""
        recency_boost = 0.7 + (0.3 * self.recency)  # Recent items get 30% boost max
        return self.relevance_score * self.importance * recency_boost


@dataclass
class SynthesizedContext:
    """The unified context output"""
    items: List[ContextItem]  # Ranked by combined_score
    active_goal: Optional[Dict] = None  # Most relevant goal if any
    contradictions: List[Dict] = field(default_factory=list)  # Detected conflicts
    session_insights: List[str] = field(default_factory=list)  # What's been learned
    token_estimate: int = 0
    synthesis_time_ms: float = 0


class ContextSynthesizer:
    """
    Unifies all context sources with intelligent ranking.

    This replaces fragmented context gathering with ONE unified search
    that ranks ALL sources by relevance to the current query.
    """

    def __init__(self,
                 embedding_engine=None,
                 map_vector_db=None,
                 conversation_archive=None,
                 knowledge_distiller=None,
                 memory_system=None,
                 supabase_client=None):
        """
        Initialize with references to existing brain components.

        Args:
            embedding_engine: EmbeddingEngine instance for text→vector
            map_vector_db: MapVectorDB instance for node search
            conversation_archive: ConversationArchive for past conversations
            knowledge_distiller: KnowledgeDistiller for Claude-taught knowledge
            memory_system: MemorySystem for session memories
            supabase_client: Optional Supabase client for AI memories
        """
        self.embedder = embedding_engine
        self.map_db = map_vector_db
        self.conversations = conversation_archive
        self.knowledge = knowledge_distiller
        self.memory = memory_system
        self.supabase = supabase_client

        # Cache for query embeddings (avoid recomputing)
        self._embedding_cache = {}
        self._cache_max_size = 100

        # Source weights (learned over time via meta-learner)
        self.source_weights = {
            'map_node': 1.0,
            'ai_memory': 1.0,
            'conversation': 0.9,
            'pattern': 0.8,
            'goal': 1.1,  # Goals get slight boost
            'distilled': 0.9
        }

        # Hybrid search weights: vector similarity + BM25 keyword matching
        self.vector_weight = 0.7
        self.bm25_weight = 0.3

        # Recency decay half-life in days (exponential decay)
        self.recency_half_life_days = 7.0

        # Common stop words to filter from BM25 scoring
        self._stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'how', 'when',
            'where', 'why', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there'
        }

        print("🔀 ContextSynthesizer initialized (hybrid search enabled)")

    def set_source_weights(self, weights: Dict[str, float]):
        """Update source weights from meta-learner"""
        self.source_weights.update(weights)

    def synthesize(self,
                   query: str,
                   user_id: Optional[str] = None,
                   map_data: Optional[Dict] = None,
                   goals: Optional[List[Dict]] = None,
                   max_items: int = 20,
                   max_tokens: int = 4000) -> SynthesizedContext:
        """
        Synthesize context from all sources for a query.

        Args:
            query: The user's message/query
            user_id: User ID for Supabase queries (optional)
            map_data: Current map state (optional)
            goals: Active goals from Goal Wizard (optional)
            max_items: Maximum context items to return
            max_tokens: Token budget for context

        Returns:
            SynthesizedContext with ranked items from all sources
        """
        start_time = time.time()
        all_items: List[ContextItem] = []

        # Get query embedding (cached if repeated)
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            # Fallback: return empty context if no embedder
            return SynthesizedContext(
                items=[],
                token_estimate=0,
                synthesis_time_ms=(time.time() - start_time) * 1000
            )

        # ═══ SEARCH ALL SOURCES IN PARALLEL (conceptually) ═══

        # 1. Search map nodes (always use hybrid scoring now)
        if map_data:
            map_items = self._search_map_nodes(query_embedding, query, map_data)
            all_items.extend(map_items)

        # 2. Search AI memories (Supabase)
        if self.supabase and user_id:
            memory_items = self._search_ai_memories(query_embedding, query, user_id)
            all_items.extend(memory_items)

        # 3. Search conversation archive
        if self.conversations:
            conv_items = self._search_conversations(query_embedding, query)
            all_items.extend(conv_items)

        # 4. Search distilled knowledge
        if self.knowledge:
            knowledge_items = self._search_knowledge(query_embedding, query)
            all_items.extend(knowledge_items)

        # 5. Search session memories
        if self.memory:
            session_items = self._search_session_memories(query)
            all_items.extend(session_items)

        # 6. Match goals
        active_goal = None
        if goals:
            goal_items, active_goal = self._match_goals(query_embedding, query, goals)
            all_items.extend(goal_items)

        # ═══ RANK ALL ITEMS UNIFORMLY ═══
        # Apply source weights
        for item in all_items:
            weight = self.source_weights.get(item.source_type, 1.0)
            item.relevance_score *= weight

        # Sort by combined score
        all_items.sort(key=lambda x: x.combined_score, reverse=True)

        # ═══ DETECT CONTRADICTIONS ═══
        contradictions = self._detect_contradictions(all_items[:max_items])

        # ═══ BUILD WITHIN TOKEN BUDGET ═══
        final_items, token_count = self._fit_to_budget(all_items, max_items, max_tokens)

        # ═══ EXTRACT SESSION INSIGHTS ═══
        session_insights = self._extract_session_insights(final_items)

        elapsed = (time.time() - start_time) * 1000

        return SynthesizedContext(
            items=final_items,
            active_goal=active_goal,
            contradictions=contradictions,
            session_insights=session_insights,
            token_estimate=token_count,
            synthesis_time_ms=elapsed
        )

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text, with caching"""
        if not self.embedder:
            return None

        # Check cache
        cache_key = hash(text[:200])  # Use first 200 chars as key
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Generate embedding
        try:
            embedding = self.embedder.embed(text)

            # Cache it
            if len(self._embedding_cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest]
            self._embedding_cache[cache_key] = embedding

            return embedding
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25 scoring with stop-word filtering"""
        # Lowercase, split on non-alphanumeric, filter short tokens and stop words
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return [t for t in tokens if len(t) > 2 and t not in self._stop_words]

    def _calculate_bm25_score(self, query: str, document: str,
                               corpus_size: int = 10,
                               avg_doc_len: int = 100,
                               k1: float = 1.5, b: float = 0.75) -> float:
        """
        Calculate BM25 score between query and document.

        BM25 is the industry-standard keyword ranking algorithm used by
        Elasticsearch, Lucene, etc. It handles exact term matching better
        than pure vector similarity.

        Args:
            query: Search query
            document: Document text to score
            corpus_size: Number of documents in corpus (for IDF calculation)
            avg_doc_len: Average document length in corpus
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Length normalization parameter (0.75 typical)

        Returns:
            BM25 score normalized to 0-1 range
        """
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)

        if not query_tokens or not doc_tokens:
            return 0.0

        # Build document term frequencies
        doc_tf = Counter(doc_tokens)
        doc_len = len(doc_tokens)

        # Use provided corpus stats (local, not shared instance state)
        num_docs = max(corpus_size, 10)  # Assume at least 10 docs

        score = 0.0
        for term in query_tokens:
            if term not in doc_tf:
                continue

            tf = doc_tf[term]

            # IDF: log((N - n + 0.5) / (n + 0.5))
            # Estimate term document frequency as fraction of corpus
            # (simplified - in production would track actual term frequencies)
            doc_freq = max(1, num_docs * 0.1)  # Assume term in 10% of docs
            idf = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

            # BM25 term score
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1)))
            score += idf * (numerator / denominator)

        # Normalize to 0-1 range (cap at reasonable max)
        # Typical BM25 scores range 0-10+, normalize by query length
        max_possible = len(query_tokens) * 3  # Rough estimate
        normalized = min(1.0, max(0.0, score / max(max_possible, 1)))

        return normalized

    def _hybrid_score(self, vector_score: float, bm25_score: float) -> float:
        """
        Combine vector similarity and BM25 into hybrid relevance score.

        This fixes the "exact keyword match fails" problem by ensuring
        keyword matches contribute to relevance even when semantic
        similarity is lower.

        Formula: (vector * 0.7) + (bm25 * 0.3)
        """
        return (vector_score * self.vector_weight) + (bm25_score * self.bm25_weight)

    def _search_map_nodes(self, query_embedding: np.ndarray, query: str, map_data: Dict) -> List[ContextItem]:
        """Search map nodes using HYBRID scoring (vector + BM25)"""
        items = []
        nodes = map_data.get('nodes', [])

        # Calculate local corpus stats for BM25 (not shared instance state)
        corpus_size = len(nodes)
        avg_doc_len = 50  # Default
        if nodes:
            total_len = sum(len(n.get('label', '')) + len(n.get('description', '')) for n in nodes)
            avg_doc_len = max(20, total_len // len(nodes))

        for node in nodes:
            # Skip if no meaningful content
            label = node.get('label', '')
            description = node.get('description', '')
            if not label and not description:
                continue

            node_text = f"{label}. {description}" if description else label

            # Get node embedding for vector similarity
            node_embedding = node.get('embedding')
            if node_embedding:
                node_embedding = np.array(node_embedding)
            else:
                node_embedding = self._get_embedding(node_text)

            # Calculate HYBRID score: vector + BM25
            vector_score = 0.0
            if node_embedding is not None:
                vector_score = self._cosine_similarity(query_embedding, node_embedding)

            bm25_score = self._calculate_bm25_score(query, node_text, corpus_size, avg_doc_len)
            hybrid_score = self._hybrid_score(vector_score, bm25_score)

            if hybrid_score > 0.25:  # Lower threshold for hybrid
                content = f"[{label}]"
                if description:
                    content += f": {description[:200]}"

                items.append(ContextItem(
                    content=content,
                    source_type='map_node',
                    relevance_score=hybrid_score,
                    importance=0.7,  # Map nodes are moderately important
                    recency=1.0,  # Current map is always "now"
                    metadata={
                        'node_id': node.get('id'),
                        'label': label,
                        'parent_id': node.get('parentId'),
                        'vector_score': vector_score,
                        'bm25_score': bm25_score
                    }
                ))

        return items[:15]  # Limit map nodes

    def _search_map_simple(self, query: str, map_data: Dict) -> List[ContextItem]:
        """Simple keyword search on map nodes (fallback)"""
        items = []
        nodes = map_data.get('nodes', [])
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for node in nodes:
            label = node.get('label', '').lower()
            description = node.get('description', '').lower()

            # Count matching words
            text = f"{label} {description}"
            matches = sum(1 for w in query_words if w in text)

            if matches > 0:
                relevance = min(1.0, matches / len(query_words))
                content = f"[{node.get('label', '')}]"
                if node.get('description'):
                    content += f": {node.get('description', '')[:200]}"

                items.append(ContextItem(
                    content=content,
                    source_type='map_node',
                    relevance_score=relevance,
                    importance=0.6,
                    recency=1.0,
                    metadata={'node_id': node.get('id'), 'label': node.get('label', '')}
                ))

        return sorted(items, key=lambda x: x.relevance_score, reverse=True)[:10]

    def _search_ai_memories(self, query_embedding: np.ndarray, query: str, user_id: str) -> List[ContextItem]:
        """Search Supabase AI memories using HYBRID scoring (vector + BM25)"""
        items = []

        try:
            # Query Supabase for user's memories
            response = self.supabase.table('ai_memory') \
                .select('*') \
                .eq('user_id', user_id) \
                .order('importance', desc=True) \
                .limit(30) \
                .execute()

            memories = response.data if response.data else []

            # Calculate local corpus stats for BM25 (not shared instance state)
            corpus_size = len(memories)
            avg_doc_len = 100  # Default
            if memories:
                total_len = sum(len(m.get('content', '')) for m in memories)
                avg_doc_len = max(50, total_len // len(memories))

            for mem in memories:
                content = mem.get('content', '')
                if not content:
                    continue

                # Calculate HYBRID score: vector + BM25
                vector_score = 0.0
                mem_embedding = mem.get('embedding')
                if mem_embedding:
                    mem_embedding = np.array(mem_embedding)
                    vector_score = self._cosine_similarity(query_embedding, mem_embedding)

                bm25_score = self._calculate_bm25_score(query, content, corpus_size, avg_doc_len)
                hybrid_score = self._hybrid_score(vector_score, bm25_score)

                if hybrid_score > 0.15:  # Lower threshold for memories
                    # Calculate recency (evergreen memories never decay)
                    is_evergreen = mem.get('evergreen', False)
                    created = mem.get('created_at', '')
                    recency = self._calculate_recency(created, evergreen=is_evergreen)

                    # Format content with type indicator
                    mem_type = mem.get('memory_type', 'memory')
                    evergreen_marker = "⚓" if is_evergreen else ""
                    formatted = f"[{mem_type}]{evergreen_marker} {content[:300]}"

                    items.append(ContextItem(
                        content=formatted,
                        source_type='ai_memory',
                        relevance_score=hybrid_score,
                        importance=mem.get('importance', 0.5),
                        recency=recency,
                        metadata={
                            'memory_id': mem.get('id'),
                            'memory_type': mem_type,
                            'evergreen': is_evergreen,
                            'related_nodes': mem.get('related_nodes', []),
                            'vector_score': vector_score,
                            'bm25_score': bm25_score
                        }
                    ))

        except Exception as e:
            print(f"⚠️ AI memory search error: {e}")

        return items

    def _search_conversations(self, query_embedding: np.ndarray, query: str) -> List[ContextItem]:
        """Search conversation archive"""
        items = []

        if not self.conversations:
            return items

        try:
            # Use archive's search if available
            results = self.conversations.search(query, limit=10)

            for conv in results:
                content = conv.get('text', conv.get('content', ''))[:300]

                # Calculate relevance
                conv_embedding = conv.get('embedding')
                if conv_embedding:
                    relevance = self._cosine_similarity(query_embedding, np.array(conv_embedding))
                else:
                    relevance = conv.get('score', 0.5)

                items.append(ContextItem(
                    content=f"[past conversation] {content}",
                    source_type='conversation',
                    relevance_score=relevance,
                    importance=0.6,
                    recency=self._calculate_recency(conv.get('timestamp', '')),
                    metadata={'conversation_id': conv.get('id')}
                ))

        except Exception as e:
            print(f"⚠️ Conversation search error: {e}")

        return items

    def _search_knowledge(self, query_embedding: np.ndarray, query: str) -> List[ContextItem]:
        """Search distilled knowledge from Claude"""
        items = []

        if not self.knowledge:
            return items

        try:
            # Get relevant knowledge
            relevant = self.knowledge.get_relevant_knowledge(query, limit=10)

            for k in relevant:
                content = k.get('content', str(k))[:300]
                k_type = k.get('type', 'knowledge')

                items.append(ContextItem(
                    content=f"[{k_type}] {content}",
                    source_type='distilled',
                    relevance_score=0.7,  # Keyword-matched, so moderate relevance
                    importance=k.get('confidence', 0.7),
                    recency=self._calculate_recency(k.get('timestamp', '')),
                    metadata={'knowledge_type': k_type}
                ))

            # Also get learned patterns
            patterns = self.knowledge.get_learned_patterns()[:5]
            for p in patterns:
                desc = p.get('details', {}).get('description', str(p.get('details', '')))
                items.append(ContextItem(
                    content=f"[pattern] {desc}",
                    source_type='pattern',
                    relevance_score=0.6,
                    importance=p.get('confidence', 0.5),
                    recency=0.5,
                    metadata={'pattern_count': p.get('count', 1)}
                ))

        except Exception as e:
            print(f"⚠️ Knowledge search error: {e}")

        return items

    def _search_session_memories(self, query: str) -> List[ContextItem]:
        """Search current session memories"""
        items = []

        if not self.memory:
            return items

        try:
            relevant = self.memory.recall(query, limit=5)

            for mem in relevant:
                mem_type = mem.get('type', 'memory')
                content = mem.get('content', str(mem))[:200]

                items.append(ContextItem(
                    content=f"[session:{mem_type}] {content}",
                    source_type='session',
                    relevance_score=0.8,  # Session memories are highly relevant
                    importance=mem.get('importance', 0.6),
                    recency=1.0,  # Current session
                    metadata={'memory_type': mem_type}
                ))

        except Exception as e:
            print(f"⚠️ Session memory search error: {e}")

        return items

    def _match_goals(self, query_embedding: np.ndarray, query: str, goals: List[Dict]) -> Tuple[List[ContextItem], Optional[Dict]]:
        """Match query to user goals"""
        items = []
        best_goal = None
        best_score = 0

        for goal in goals:
            title = goal.get('title', goal.get('label', ''))
            description = goal.get('description', '')
            goal_text = f"{title}. {description}"

            # Get goal embedding
            goal_embedding = self._get_embedding(goal_text)
            if goal_embedding is not None:
                relevance = self._cosine_similarity(query_embedding, goal_embedding)
            else:
                # Keyword fallback
                query_words = set(query.lower().split())
                goal_lower = goal_text.lower()
                matches = sum(1 for w in query_words if w in goal_lower)
                relevance = min(1.0, matches / max(len(query_words), 1))

            if relevance > 0.3:
                priority = goal.get('priority', 'medium')
                importance = {'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(priority, 0.7)

                items.append(ContextItem(
                    content=f"[goal:{priority}] {title}",
                    source_type='goal',
                    relevance_score=relevance,
                    importance=importance,
                    recency=1.0,
                    metadata={'goal_id': goal.get('id'), 'priority': priority}
                ))

                if relevance > best_score:
                    best_score = relevance
                    best_goal = goal

        return items, best_goal if best_score > 0.4 else None

    def _detect_contradictions(self, items: List[ContextItem]) -> List[Dict]:
        """Detect potential contradictions between context items"""
        contradictions = []

        # Simple heuristic: look for items with similar topics but different sources
        # that might conflict. This is a basic implementation - could be enhanced
        # with NLI models in the future.

        # For now, just flag when memory and map node say different things about same topic
        memory_items = [i for i in items if i.source_type == 'ai_memory']
        map_items = [i for i in items if i.source_type == 'map_node']

        # Could add semantic contradiction detection here
        # For now, return empty list

        return contradictions

    def _fit_to_budget(self, items: List[ContextItem], max_items: int, max_tokens: int) -> Tuple[List[ContextItem], int]:
        """Fit items within token budget"""
        selected = []
        total_tokens = 0

        for item in items[:max_items * 2]:  # Check more items than needed
            # Estimate tokens (roughly 4 chars per token)
            item_tokens = len(item.content) // 4 + 10  # +10 for formatting

            if total_tokens + item_tokens <= max_tokens:
                selected.append(item)
                total_tokens += item_tokens

                if len(selected) >= max_items:
                    break

        return selected, total_tokens

    def _extract_session_insights(self, items: List[ContextItem]) -> List[str]:
        """Extract key insights from session items"""
        insights = []

        for item in items:
            if item.source_type == 'session' and 'insight' in item.metadata.get('memory_type', ''):
                insights.append(item.content)

        return insights[:5]

    def _calculate_recency(self, timestamp, evergreen: bool = False) -> float:
        """
        Calculate recency score using EXPONENTIAL DECAY (half-life formula).

        Formula: S(t) = S₀ × 2^(-t/h)
        Where:
            - S₀ = 1.0 (initial score)
            - t = age in days
            - h = half-life in days (default: 7 days)

        This is more biomimetic than linear decay - memories fade quickly
        at first, then level off (you don't completely forget old things).

        Args:
            timestamp: Creation or last_accessed time
            evergreen: If True, always returns 1.0 (foundational knowledge never decays)

        Returns:
            Recency score from 0.1 (floor) to 1.0 (brand new)
        """
        # Evergreen memories never decay
        if evergreen:
            return 1.0

        if not timestamp:
            return 0.5

        try:
            from datetime import datetime, timezone

            # Handle various timestamp formats
            if isinstance(timestamp, (int, float)):
                age_seconds = time.time() - timestamp
            else:
                # Parse ISO string
                ts_str = str(timestamp)
                if 'T' in ts_str:
                    # Replace Z with +00:00 for proper parsing
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

                    # Handle naive timestamps (no timezone info)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                        now = datetime.now(timezone.utc)
                    else:
                        now = datetime.now(dt.tzinfo)

                    age_seconds = (now - dt).total_seconds()
                else:
                    return 0.5

            # EXPONENTIAL DECAY with half-life
            # S(t) = 2^(-t/h) where h = half-life in days
            days_old = age_seconds / 86400
            half_life = self.recency_half_life_days  # Default: 7 days

            # After 1 half-life (7 days): score = 0.5
            # After 2 half-lives (14 days): score = 0.25
            # After 3 half-lives (21 days): score = 0.125
            # After 4 half-lives (28 days): score = 0.0625
            recency = math.pow(2, -days_old / half_life)

            # Floor at 0.1 to prevent memories from becoming completely invisible
            return max(0.1, min(1.0, recency))

        except Exception as e:
            print(f"⚠️ Recency calculation error: {e}")
            return 0.5

    def _reorder_for_attention(self, items: List[ContextItem]) -> List[ContextItem]:
        """
        Reorder items to combat "Lost in the Middle" problem.

        LLMs pay more attention to the START and END of context windows.
        This method places high-relevance items at both positions:

        [HIGH relevance] → [MEDIUM relevance] → [HIGH relevance]

        Research: https://arxiv.org/abs/2307.03172 "Lost in the Middle"
        """
        if len(items) <= 3:
            return items  # Too few to reorder

        # Sort by combined_score descending
        sorted_items = sorted(items, key=lambda x: x.combined_score, reverse=True)

        # Split into high (top 40%) and medium/low (bottom 60%)
        split_point = max(2, len(sorted_items) * 2 // 5)  # At least 2 in high
        high_relevance = sorted_items[:split_point]
        medium_relevance = sorted_items[split_point:]

        # Further split high relevance: half at start, half at end
        mid_high = len(high_relevance) // 2
        start_items = high_relevance[:mid_high]
        end_items = high_relevance[mid_high:]

        # Final order: [start_high] + [medium] + [end_high]
        reordered = start_items + medium_relevance + end_items

        return reordered

    def format_for_prompt(self, synthesized: SynthesizedContext) -> str:
        """
        Format synthesized context for inclusion in Claude prompt.

        This is the final output that goes to Claude - ONE unified context block.

        IMPORTANT: Uses "Lost in the Middle" mitigation - high-relevance items
        are positioned at START and END of context for better LLM attention.
        Items are output in reordered sequence (not grouped by source).
        """
        lines = ["## Relevant Context (Synthesized from All Sources)"]

        if not synthesized.items:
            lines.append("No specifically relevant context found.")
            return "\n".join(lines)

        # ═══ LOST IN THE MIDDLE FIX ═══
        # Reorder items so high-relevance are at START and END
        reordered_items = self._reorder_for_attention(synthesized.items)

        # Source labels for inline display
        source_labels = {
            'map_node': '📍',
            'ai_memory': '🧠',
            'conversation': '💬',
            'goal': '🎯',
            'distilled': '📚',
            'pattern': '🔄',
            'session': '⚡'
        }

        # Output items in reordered sequence (preserving Lost in the Middle fix)
        # DO NOT regroup by source - that would undo the attention optimization
        for item in reordered_items[:20]:  # Max 20 items total
            # Show relevance indicator
            if item.combined_score > 0.8:
                indicator = "★"  # High relevance
            elif item.combined_score > 0.5:
                indicator = "◆"  # Medium relevance
            else:
                indicator = "○"  # Lower relevance

            source_icon = source_labels.get(item.source_type, '•')
            lines.append(f"{indicator}{source_icon} {item.content}")

        # Add active goal highlight if present (goals are important - at end)
        if synthesized.active_goal:
            lines.append(f"\n### 🎯 Active Goal: {synthesized.active_goal.get('title', 'Unknown')}")
            if synthesized.active_goal.get('description'):
                lines.append(f"   {synthesized.active_goal['description'][:200]}")

        # Add contradictions warning if any
        if synthesized.contradictions:
            lines.append("\n### ⚠️ Potential Contradictions")
            for c in synthesized.contradictions[:3]:
                lines.append(f"- {c.get('description', 'Conflict detected')}")

        # Token count note
        lines.append(f"\n_Context: {len(reordered_items[:20])} items, ~{synthesized.token_estimate} tokens_")

        return "\n".join(lines)
