"""
Context Synthesizer + Context Lens
===================================
Unifies all context sources into ONE coherent context block,
then applies the Context Lens to transform fragments into understanding.

RETRIEVAL LAYER (ContextSynthesizer):
1. Uses HYBRID SEARCH (vector similarity + BM25 keyword matching)
2. Ranks results by relevance (not by source type)
3. Uses EXPONENTIAL DECAY for recency (half-life formula, not linear)
4. Positions high-relevance items at START and END (avoids "Lost in the Middle")
5. Links to active goals automatically

COMPREHENSION LAYER (Context Lens) - NEW:
6. FocusDetector: Uses ASA neuron energy to detect current focus
7. ThematicClusterer: Groups related items into coherent themes
8. NarrativeTracker: Tracks how topics evolve over time
9. InsightDistiller: Produces the "so what?" from raw context

This transforms "here are 20 relevant items" into
"here's what these items mean together about what you're trying to understand."
"""

import time
import math
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

# Token counting - use tiktoken if available, fallback to approximation
try:
    import tiktoken
    _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")  # Claude's encoding
    def count_tokens(text: str) -> int:
        """Count tokens accurately using tiktoken"""
        return len(_tiktoken_encoder.encode(text))
    print("✓ tiktoken loaded for accurate token counting")
except ImportError:
    def count_tokens(text: str) -> int:
        """Approximate token count (fallback when tiktoken unavailable)"""
        return len(text) // 4 + 1
    print("⚠️ tiktoken not available, using approximate token counting")


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


# =============================================================================
# CONTEXT LENS DATA STRUCTURES
# =============================================================================

@dataclass
class FocusState:
    """What the user is currently trying to understand"""
    primary_focus: str  # Main topic/question being explored
    focus_confidence: float  # 0-1 how clear the focus is
    supporting_topics: List[str]  # Related topics that support the focus
    energy_sources: List[Dict]  # ASA atoms with high energy (working memory)
    detected_intent: str  # 'exploring', 'deciding', 'learning', 'creating', 'reflecting'


@dataclass
class Theme:
    """A coherent grouping of related context items"""
    name: str  # Theme title
    description: str  # What this theme represents
    items: List[ContextItem]  # Items belonging to this theme
    sources: List[str]  # Source types contributing (map_node, ai_memory, etc.)
    coherence_score: float  # 0-1 how well items relate to each other
    relevance_to_focus: float  # 0-1 how relevant to current focus


@dataclass
class NarrativeThread:
    """How a topic has evolved over time"""
    topic: str
    evolution: List[Dict]  # [{timestamp, state, source}] - timeline of changes
    current_state: str  # Where this topic stands now
    trajectory: str  # 'growing', 'shifting', 'stabilizing', 'questioning'
    key_transitions: List[str]  # Major shifts in thinking


@dataclass
class DistilledInsight:
    """The "so what?" extracted from context"""
    insight: str  # The key understanding
    confidence: float  # 0-1 how confident
    supporting_evidence: List[str]  # What supports this insight
    implications: List[str]  # What this means for the user
    source_themes: List[str]  # Which themes contributed


@dataclass
class ContextLens:
    """
    The complete Context Lens output.
    Transforms raw context items into coherent understanding.
    """
    # Focus: What is the user trying to understand?
    focus: FocusState

    # Themes: How do the items group together?
    themes: List[Theme]

    # Narrative: How have topics evolved?
    narratives: List[NarrativeThread]

    # Insights: What does this all mean?
    insights: List[DistilledInsight]

    # Meta: Understanding about the understanding
    understanding_quality: float  # 0-1 overall coherence
    gaps_detected: List[str]  # What's missing from the picture
    suggested_explorations: List[str]  # What might help deepen understanding


@dataclass
class EnhancedSynthesizedContext(SynthesizedContext):
    """
    Extended SynthesizedContext with Context Lens output.
    Backward compatible - still has all original fields.
    """
    lens: Optional[ContextLens] = None  # The Context Lens analysis


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
                 supabase_client=None,
                 asa_system=None):
        """
        Initialize with references to existing brain components.

        Args:
            embedding_engine: EmbeddingEngine instance for text→vector
            map_vector_db: MapVectorDB instance for node search
            conversation_archive: ConversationArchive for past conversations
            knowledge_distiller: KnowledgeDistiller for Claude-taught knowledge
            memory_system: MemorySystem for session memories
            supabase_client: Optional Supabase client for AI memories
            asa_system: Optional LivingASA instance for focus detection
        """
        self.embedder = embedding_engine
        self.map_db = map_vector_db
        self.conversations = conversation_archive
        self.knowledge = knowledge_distiller
        self.memory = memory_system
        self.supabase = supabase_client
        self.asa = asa_system  # For Context Lens focus detection

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

        # ═══ CONTEXT LENS STATE ═══
        # Track topic evolution for narrative threading
        self._topic_history = []  # [{topic, timestamp, state, source}]
        self._max_topic_history = 200

        # Intent detection patterns
        self._intent_patterns = {
            'exploring': ['what', 'how does', 'tell me about', 'explain', 'curious'],
            'deciding': ['should i', 'which', 'compare', 'better', 'choose', 'option'],
            'learning': ['learn', 'understand', 'study', 'practice', 'improve'],
            'creating': ['create', 'build', 'make', 'design', 'write', 'develop'],
            'reflecting': ['why did', 'what if', 'looking back', 'realize', 'think about']
        }

        print("🔀 ContextSynthesizer initialized (hybrid search + Context Lens enabled)")

    def set_source_weights(self, weights: Dict[str, float]):
        """Update source weights from meta-learner"""
        self.source_weights.update(weights)

    def synthesize(self,
                   query: str,
                   user_id: Optional[str] = None,
                   map_data: Optional[Dict] = None,
                   goals: Optional[List[Dict]] = None,
                   max_items: int = 50,
                   max_tokens: int = 20000) -> SynthesizedContext:
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
        nodes = map_data.get('nodes') or []

        # Calculate local corpus stats for BM25 (not shared instance state)
        corpus_size = len(nodes)
        avg_doc_len = 50  # Default
        if nodes:
            total_len = sum(len(n.get('label') or '') + len(n.get('description') or '') for n in nodes if n)
            avg_doc_len = max(20, total_len // len(nodes)) if nodes else 50

        for node in nodes:
            if not node:
                continue
            # Skip if no meaningful content
            label = node.get('label') or ''
            description = node.get('description') or ''
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
            results = self.conversations.search(query, top_k=10)

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
        """Fit items within token budget using accurate token counting"""
        selected = []
        total_tokens = 0

        for item in items[:max_items * 2]:  # Check more items than needed
            # Use accurate token counting (tiktoken if available)
            item_tokens = count_tokens(item.content) + 15  # +15 for formatting overhead

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

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT LENS - Transforms raw context into coherent understanding
    # ═══════════════════════════════════════════════════════════════════════════

    def apply_context_lens(self, query: str, items: List[ContextItem],
                           goals: Optional[List[Dict]] = None) -> ContextLens:
        """
        Apply the Context Lens to transform raw context items into coherent understanding.

        This is the "comprehension layer" that sits on top of retrieval.

        Args:
            query: The user's current message/question
            items: Context items from synthesis
            goals: Active goals if any

        Returns:
            ContextLens with focus, themes, narratives, and insights
        """
        # 1. Detect focus - what is the user trying to understand?
        focus = self._detect_focus(query, items, goals)

        # 2. Cluster into themes - how do items relate?
        themes = self._cluster_into_themes(items, focus)

        # 3. Track narrative evolution - how have topics changed?
        narratives = self._track_narratives(themes, query)

        # 4. Distill insights - what does this all mean?
        insights = self._distill_insights(themes, focus, narratives)

        # 5. Assess understanding quality
        understanding_quality = self._assess_understanding_quality(themes, focus)

        # 6. Detect gaps
        gaps = self._detect_gaps(themes, focus, items)

        # 7. Suggest explorations
        explorations = self._suggest_explorations(gaps, focus, themes)

        return ContextLens(
            focus=focus,
            themes=themes,
            narratives=narratives,
            insights=insights,
            understanding_quality=understanding_quality,
            gaps_detected=gaps,
            suggested_explorations=explorations
        )

    def _detect_focus(self, query: str, items: List[ContextItem],
                      goals: Optional[List[Dict]] = None) -> FocusState:
        """
        Detect what the user is currently trying to understand.

        Integrates:
        - Query analysis (what they're asking)
        - ASA neuron energy (what's active in working memory)
        - Goal context (what they're working toward)
        """
        query_lower = query.lower()

        # 1. Detect intent from query patterns
        detected_intent = 'exploring'  # default
        max_matches = 0
        for intent, patterns in self._intent_patterns.items():
            matches = sum(1 for p in patterns if p in query_lower)
            if matches > max_matches:
                max_matches = matches
                detected_intent = intent

        # 2. Get working memory from ASA (high energy topics)
        energy_sources = []
        if self.asa:
            try:
                working_memory = self.asa.get_working_memory(threshold=0.2)
                energy_sources = working_memory[:10]  # Top 10 active topics
            except Exception as e:
                print(f"⚠️ ASA working memory error: {e}")

        # 3. Extract key topics from query
        query_tokens = self._tokenize(query)
        key_topics = [t for t in query_tokens if len(t) > 3][:5]

        # 4. Cross-reference with context items to find primary focus
        topic_scores = {}
        for item in items[:20]:
            item_tokens = set(self._tokenize(item.content))
            for topic in key_topics:
                if topic in item_tokens:
                    topic_scores[topic] = topic_scores.get(topic, 0) + item.combined_score

        # 5. Boost topics that are in ASA working memory
        for energy_item in energy_sources:
            topic_name = energy_item.get('name', '').lower()
            for topic in key_topics:
                if topic in topic_name:
                    topic_scores[topic] = topic_scores.get(topic, 0) + energy_item.get('energy', 0) * 0.5

        # 6. Determine primary focus
        if topic_scores:
            primary_topic = max(topic_scores, key=topic_scores.get)
            focus_confidence = min(1.0, topic_scores[primary_topic] / 2.0)
        else:
            primary_topic = query[:50] if query else "general exploration"
            focus_confidence = 0.3

        # 7. Get supporting topics
        supporting = [t for t in topic_scores.keys() if t != primary_topic][:5]

        # 8. If there's an active goal, boost focus confidence
        if goals:
            for goal in goals:
                goal_text = (goal.get('title', '') + ' ' + goal.get('description', '')).lower()
                if any(t in goal_text for t in key_topics):
                    focus_confidence = min(1.0, focus_confidence + 0.2)
                    break

        return FocusState(
            primary_focus=primary_topic,
            focus_confidence=focus_confidence,
            supporting_topics=supporting,
            energy_sources=energy_sources,
            detected_intent=detected_intent
        )

    def _cluster_into_themes(self, items: List[ContextItem],
                             focus: FocusState) -> List[Theme]:
        """
        Group context items into coherent themes.

        Uses semantic similarity and source correlation to find
        natural groupings that tell a coherent story.
        """
        if not items:
            return []

        themes = []
        assigned_items = set()

        # Strategy 1: Source-based initial grouping
        source_groups = {}
        for i, item in enumerate(items):
            source = item.source_type
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append((i, item))

        # Strategy 2: Semantic clustering within and across sources
        # Use embedding similarity if available, else keyword overlap

        # First pass: Create themes from high-density keyword clusters
        keyword_clusters = self._find_keyword_clusters(items)

        for cluster_name, cluster_items in keyword_clusters.items():
            if len(cluster_items) >= 2:  # Need at least 2 items for a theme
                # Calculate coherence
                coherence = self._calculate_coherence(cluster_items)

                # Calculate relevance to focus
                focus_relevance = self._calculate_focus_relevance(cluster_items, focus)

                # Get sources
                sources = list(set(item.source_type for item in cluster_items))

                # Create theme
                theme = Theme(
                    name=cluster_name.replace('_', ' ').title(),
                    description=self._generate_theme_description(cluster_items, cluster_name),
                    items=cluster_items,
                    sources=sources,
                    coherence_score=coherence,
                    relevance_to_focus=focus_relevance
                )
                themes.append(theme)

                # Mark items as assigned
                for item in cluster_items:
                    assigned_items.add(id(item))

        # Second pass: Group remaining items by source
        for source, source_items in source_groups.items():
            unassigned = [(i, item) for i, item in source_items if id(item) not in assigned_items]

            if len(unassigned) >= 2:
                items_only = [item for _, item in unassigned]
                coherence = self._calculate_coherence(items_only)
                focus_relevance = self._calculate_focus_relevance(items_only, focus)

                theme = Theme(
                    name=f"{source.replace('_', ' ').title()} Context",
                    description=f"Information from {source} sources",
                    items=items_only,
                    sources=[source],
                    coherence_score=coherence,
                    relevance_to_focus=focus_relevance
                )
                themes.append(theme)

        # Sort themes by relevance to focus
        themes.sort(key=lambda t: t.relevance_to_focus, reverse=True)

        return themes[:5]  # Return top 5 themes

    def _find_keyword_clusters(self, items: List[ContextItem]) -> Dict[str, List[ContextItem]]:
        """Find natural keyword-based clusters in items"""
        # Extract significant keywords from each item
        item_keywords = []
        for item in items:
            tokens = self._tokenize(item.content)
            # Keep meaningful tokens
            keywords = [t for t in tokens if len(t) > 4]
            item_keywords.append((item, set(keywords)))

        # Find overlapping keyword groups
        clusters = {}
        for item, keywords in item_keywords:
            for kw in keywords:
                if kw not in clusters:
                    clusters[kw] = []
                clusters[kw].append(item)

        # Keep only clusters with multiple items from different sources
        filtered_clusters = {}
        for kw, cluster_items in clusters.items():
            if len(cluster_items) >= 2:
                sources = set(item.source_type for item in cluster_items)
                if len(sources) >= 1:  # At least 1 source type
                    filtered_clusters[kw] = cluster_items

        # Merge overlapping clusters
        merged = {}
        used_keywords = set()
        for kw, items in sorted(filtered_clusters.items(), key=lambda x: len(x[1]), reverse=True):
            if kw in used_keywords:
                continue

            # Find similar keywords to merge
            similar = [k for k in filtered_clusters if k not in used_keywords
                       and self._keyword_similarity(kw, k) > 0.5]
            used_keywords.add(kw)
            used_keywords.update(similar)

            # Merge items
            merged_items = list(items)
            for sk in similar:
                merged_items.extend([i for i in filtered_clusters[sk] if i not in merged_items])

            merged[kw] = merged_items

        return merged

    def _keyword_similarity(self, kw1: str, kw2: str) -> float:
        """Simple keyword similarity (could use embeddings for better results)"""
        if kw1 == kw2:
            return 1.0
        # Check if one contains the other
        if kw1 in kw2 or kw2 in kw1:
            return 0.7
        # Check character overlap
        set1, set2 = set(kw1), set(kw2)
        overlap = len(set1 & set2) / max(len(set1 | set2), 1)
        return overlap

    def _calculate_coherence(self, items: List[ContextItem]) -> float:
        """Calculate how well items in a theme relate to each other"""
        if len(items) < 2:
            return 1.0

        # Strategy: Check keyword overlap between items
        all_keywords = []
        for item in items:
            tokens = set(self._tokenize(item.content))
            all_keywords.append(tokens)

        # Calculate average pairwise overlap
        overlaps = []
        for i in range(len(all_keywords)):
            for j in range(i + 1, len(all_keywords)):
                intersection = len(all_keywords[i] & all_keywords[j])
                union = len(all_keywords[i] | all_keywords[j])
                if union > 0:
                    overlaps.append(intersection / union)

        return sum(overlaps) / max(len(overlaps), 1) if overlaps else 0.5

    def _calculate_focus_relevance(self, items: List[ContextItem],
                                    focus: FocusState) -> float:
        """Calculate how relevant a theme is to the current focus"""
        focus_keywords = set(focus.supporting_topics + [focus.primary_focus])

        relevance_scores = []
        for item in items:
            item_tokens = set(self._tokenize(item.content))
            overlap = len(focus_keywords & item_tokens)
            relevance_scores.append(min(1.0, overlap / max(len(focus_keywords), 1)))

        # Also consider item relevance scores
        avg_item_relevance = sum(item.combined_score for item in items) / max(len(items), 1)

        keyword_relevance = sum(relevance_scores) / max(len(relevance_scores), 1)

        return (keyword_relevance * 0.4) + (avg_item_relevance * 0.6)

    def _generate_theme_description(self, items: List[ContextItem], cluster_name: str) -> str:
        """Generate a description for a theme"""
        sources = set(item.source_type for item in items)
        source_str = ', '.join(sources)

        return f"A cluster of {len(items)} related items about '{cluster_name}' from {source_str}"

    def _track_narratives(self, themes: List[Theme], query: str) -> List[NarrativeThread]:
        """
        Track how topics have evolved over time.

        Uses stored topic history to identify trajectories.
        """
        narratives = []
        now = time.time()

        # Add current topics to history
        for theme in themes:
            self._topic_history.append({
                'topic': theme.name,
                'timestamp': now,
                'state': 'active',
                'source': 'current_session',
                'relevance': theme.relevance_to_focus
            })

        # Trim history
        if len(self._topic_history) > self._max_topic_history:
            self._topic_history = self._topic_history[-self._max_topic_history:]

        # Group history by topic
        topic_timelines = {}
        for entry in self._topic_history:
            topic = entry['topic']
            if topic not in topic_timelines:
                topic_timelines[topic] = []
            topic_timelines[topic].append(entry)

        # Analyze each topic's trajectory
        for topic, timeline in topic_timelines.items():
            if len(timeline) < 2:
                continue

            # Sort by time
            timeline.sort(key=lambda x: x['timestamp'])

            # Determine trajectory
            recent_relevance = [e['relevance'] for e in timeline[-3:]]
            if len(recent_relevance) >= 2:
                trend = recent_relevance[-1] - recent_relevance[0]
                if trend > 0.2:
                    trajectory = 'growing'
                elif trend < -0.2:
                    trajectory = 'declining'
                elif abs(trend) < 0.1:
                    trajectory = 'stabilizing'
                else:
                    trajectory = 'shifting'
            else:
                trajectory = 'emerging'

            # Find key transitions
            transitions = []
            for i in range(1, len(timeline)):
                prev = timeline[i - 1]
                curr = timeline[i]
                if abs(curr.get('relevance', 0) - prev.get('relevance', 0)) > 0.3:
                    transitions.append(f"Shifted at {int((now - curr['timestamp']) / 60)} minutes ago")

            # Current state
            current_state = timeline[-1].get('state', 'active')

            narrative = NarrativeThread(
                topic=topic,
                evolution=[{
                    'timestamp': e['timestamp'],
                    'state': e['state'],
                    'source': e['source']
                } for e in timeline[-5:]],  # Last 5 entries
                current_state=current_state,
                trajectory=trajectory,
                key_transitions=transitions[:3]
            )
            narratives.append(narrative)

        # Sort by recency
        narratives.sort(key=lambda n: n.evolution[-1]['timestamp'] if n.evolution else 0, reverse=True)

        return narratives[:5]  # Top 5 narratives

    def _distill_insights(self, themes: List[Theme], focus: FocusState,
                          narratives: List[NarrativeThread]) -> List[DistilledInsight]:
        """
        Distill the "so what?" from themes, focus, and narratives.

        This is the core of the Context Lens - turning fragments into understanding.
        """
        insights = []

        # Insight 1: Cross-theme patterns
        if len(themes) >= 2:
            # Find keywords that appear across multiple themes
            theme_keywords = [set(self._tokenize(t.name + ' ' + t.description)) for t in themes]
            common_keywords = set.intersection(*theme_keywords) if theme_keywords else set()

            if common_keywords:
                common_str = ', '.join(list(common_keywords)[:3])
                insight = DistilledInsight(
                    insight=f"Multiple themes connect around: {common_str}",
                    confidence=min(0.9, len(common_keywords) * 0.15 + 0.4),
                    supporting_evidence=[t.name for t in themes[:3]],
                    implications=["These concepts are central to your current thinking",
                                  "Consider how they relate to your focus"],
                    source_themes=[t.name for t in themes[:3]]
                )
                insights.append(insight)

        # Insight 2: Focus-based insight
        if focus.focus_confidence > 0.5:
            supporting_count = len(focus.supporting_topics)
            if supporting_count > 0:
                insight = DistilledInsight(
                    insight=f"Your focus on '{focus.primary_focus}' is supported by {supporting_count} related concepts",
                    confidence=focus.focus_confidence,
                    supporting_evidence=focus.supporting_topics[:3],
                    implications=[
                        f"You appear to be {focus.detected_intent} this topic",
                        "The context strongly supports this direction"
                    ],
                    source_themes=[focus.primary_focus]
                )
                insights.append(insight)

        # Insight 3: Narrative-based insight
        growing_topics = [n for n in narratives if n.trajectory == 'growing']
        if growing_topics:
            topic_names = [n.topic for n in growing_topics[:3]]
            insight = DistilledInsight(
                insight=f"These topics are gaining momentum: {', '.join(topic_names)}",
                confidence=0.7,
                supporting_evidence=[f"{n.topic} shows {n.trajectory} pattern" for n in growing_topics[:2]],
                implications=["Your attention is increasingly drawn here",
                              "Consider deepening exploration of these areas"],
                source_themes=topic_names
            )
            insights.append(insight)

        # Insight 4: Source diversity insight
        if themes:
            all_sources = set()
            for theme in themes:
                all_sources.update(theme.sources)

            if len(all_sources) >= 3:
                insight = DistilledInsight(
                    insight=f"Your understanding is informed by {len(all_sources)} different sources",
                    confidence=0.8,
                    supporting_evidence=[f"Includes {s}" for s in list(all_sources)[:3]],
                    implications=["Multi-source triangulation strengthens understanding",
                                  "Consider if any source is overrepresented"],
                    source_themes=list(all_sources)
                )
                insights.append(insight)

        # Insight 5: ASA energy-based insight (working memory)
        if focus.energy_sources:
            hot_topics = [e.get('name', 'unknown') for e in focus.energy_sources[:3]]
            insight = DistilledInsight(
                insight=f"Currently active in working memory: {', '.join(hot_topics)}",
                confidence=0.85,
                supporting_evidence=[f"{e.get('name')} (energy: {e.get('energy', 0):.0%})"
                                     for e in focus.energy_sources[:3]],
                implications=["These are top-of-mind concepts",
                              "They may influence how you interpret new information"],
                source_themes=hot_topics
            )
            insights.append(insight)

        return insights[:5]  # Top 5 insights

    def _assess_understanding_quality(self, themes: List[Theme],
                                       focus: FocusState) -> float:
        """
        Assess how coherent and complete the current understanding is.

        Returns 0-1 score.
        """
        scores = []

        # Factor 1: Focus clarity
        scores.append(focus.focus_confidence)

        # Factor 2: Theme coherence
        if themes:
            avg_coherence = sum(t.coherence_score for t in themes) / len(themes)
            scores.append(avg_coherence)

        # Factor 3: Theme-focus alignment
        if themes:
            avg_relevance = sum(t.relevance_to_focus for t in themes) / len(themes)
            scores.append(avg_relevance)

        # Factor 4: Source diversity
        if themes:
            all_sources = set()
            for theme in themes:
                all_sources.update(theme.sources)
            source_diversity = min(1.0, len(all_sources) / 5.0)  # 5 sources = max
            scores.append(source_diversity)

        # Factor 5: Working memory support
        if focus.energy_sources:
            energy_support = min(1.0, len(focus.energy_sources) / 5.0)
            scores.append(energy_support)

        return sum(scores) / max(len(scores), 1)

    def _detect_gaps(self, themes: List[Theme], focus: FocusState,
                     items: List[ContextItem]) -> List[str]:
        """
        Detect what's missing from the current understanding.
        """
        gaps = []

        # Gap 1: Low focus confidence
        if focus.focus_confidence < 0.4:
            gaps.append("Focus is unclear - the query may need clarification")

        # Gap 2: Single source
        if themes:
            all_sources = set()
            for theme in themes:
                all_sources.update(theme.sources)
            if len(all_sources) == 1:
                gaps.append(f"Only {list(all_sources)[0]} sources - consider other perspectives")

        # Gap 3: Low coherence themes
        low_coherence = [t for t in themes if t.coherence_score < 0.3]
        if low_coherence:
            gaps.append(f"Theme '{low_coherence[0].name}' has low internal coherence")

        # Gap 4: Missing goal alignment
        if focus.detected_intent in ['deciding', 'creating'] and not any(
            'goal' in t.sources for t in themes
        ):
            gaps.append("No active goals connected - consider setting a goal")

        # Gap 5: No recent context
        if items:
            recent_count = sum(1 for i in items if i.recency > 0.7)
            if recent_count < 2:
                gaps.append("Limited recent context - older information may be outdated")

        return gaps[:5]

    def _suggest_explorations(self, gaps: List[str], focus: FocusState,
                              themes: List[Theme]) -> List[str]:
        """
        Suggest what might help deepen understanding.
        """
        suggestions = []

        # Based on intent
        if focus.detected_intent == 'exploring':
            suggestions.append(f"Ask a specific question about '{focus.primary_focus}'")
        elif focus.detected_intent == 'deciding':
            suggestions.append("List pros and cons explicitly")
            suggestions.append("Consider what success looks like")
        elif focus.detected_intent == 'learning':
            suggestions.append("Try explaining this to someone else")
            suggestions.append("Connect to something you already know")
        elif focus.detected_intent == 'creating':
            suggestions.append("Start with the smallest viable version")
            suggestions.append("Identify the first concrete step")
        elif focus.detected_intent == 'reflecting':
            suggestions.append("What would you do differently?")
            suggestions.append("What pattern do you see emerging?")

        # Based on gaps
        if "Focus is unclear" in str(gaps):
            suggestions.append("Try rephrasing your question more specifically")

        # Based on themes
        if themes:
            weakest = min(themes, key=lambda t: t.relevance_to_focus)
            if weakest.relevance_to_focus < 0.3:
                suggestions.append(f"The '{weakest.name}' theme seems tangential - is it relevant?")

        # Based on supporting topics
        if len(focus.supporting_topics) > 3:
            suggestions.append("Consider focusing on fewer topics for deeper understanding")

        return suggestions[:5]

    def format_lens_for_prompt(self, lens: ContextLens) -> str:
        """
        Format Context Lens output for inclusion in Claude prompt.

        This produces the coherent understanding section.
        """
        lines = ["## Current Understanding (Context Lens)"]

        # Focus section
        lines.append(f"\n### Your Focus")
        lines.append(f"**Primary**: {lens.focus.primary_focus}")
        lines.append(f"**Intent**: {lens.focus.detected_intent.title()}")
        lines.append(f"**Clarity**: {lens.focus.focus_confidence:.0%}")

        if lens.focus.supporting_topics:
            lines.append(f"**Supporting concepts**: {', '.join(lens.focus.supporting_topics[:5])}")

        # Themes section
        if lens.themes:
            lines.append(f"\n### Key Themes")
            for theme in lens.themes[:3]:
                relevance_icon = "★" if theme.relevance_to_focus > 0.6 else "◆" if theme.relevance_to_focus > 0.3 else "○"
                lines.append(f"{relevance_icon} **{theme.name}** ({len(theme.items)} items from {', '.join(theme.sources)})")

        # Insights section
        if lens.insights:
            lines.append(f"\n### Key Insights")
            for insight in lens.insights[:3]:
                conf_icon = "✓" if insight.confidence > 0.7 else "~"
                lines.append(f"{conf_icon} {insight.insight}")
                if insight.implications:
                    lines.append(f"  → {insight.implications[0]}")

        # Working memory section (from ASA)
        if lens.focus.energy_sources:
            lines.append(f"\n### Active in Mind")
            for src in lens.focus.energy_sources[:3]:
                energy = src.get('energy', 0)
                bar = "█" * int(energy * 5) + "░" * (5 - int(energy * 5))
                lines.append(f"  [{bar}] {src.get('name', 'unknown')}")

        # Narrative threads
        growing = [n for n in lens.narratives if n.trajectory == 'growing']
        if growing:
            lines.append(f"\n### Emerging Topics")
            for n in growing[:2]:
                lines.append(f"↗ {n.topic} (gaining momentum)")

        # Gaps and suggestions
        if lens.gaps_detected:
            lines.append(f"\n### Potential Gaps")
            for gap in lens.gaps_detected[:2]:
                lines.append(f"⚠ {gap}")

        if lens.suggested_explorations:
            lines.append(f"\n### To Deepen Understanding")
            for suggestion in lens.suggested_explorations[:2]:
                lines.append(f"→ {suggestion}")

        # Quality indicator
        quality_label = "Strong" if lens.understanding_quality > 0.7 else "Moderate" if lens.understanding_quality > 0.4 else "Developing"
        lines.append(f"\n_Understanding quality: {quality_label} ({lens.understanding_quality:.0%})_")

        return "\n".join(lines)

    def synthesize_with_lens(self,
                             query: str,
                             user_id: Optional[str] = None,
                             map_data: Optional[Dict] = None,
                             goals: Optional[List[Dict]] = None,
                             max_items: int = 50,
                             max_tokens: int = 20000) -> EnhancedSynthesizedContext:
        """
        Enhanced synthesis that includes Context Lens analysis.

        This is the new primary entry point that returns both
        raw context items AND coherent understanding.
        """
        # First, do standard synthesis
        base_result = self.synthesize(
            query=query,
            user_id=user_id,
            map_data=map_data,
            goals=goals,
            max_items=max_items,
            max_tokens=max_tokens
        )

        # Then, apply Context Lens
        lens = self.apply_context_lens(query, base_result.items, goals)

        # Return enhanced result
        return EnhancedSynthesizedContext(
            items=base_result.items,
            active_goal=base_result.active_goal,
            contradictions=base_result.contradictions,
            session_insights=base_result.session_insights,
            token_estimate=base_result.token_estimate,
            synthesis_time_ms=base_result.synthesis_time_ms,
            lens=lens
        )

    def format_enhanced_for_prompt(self, enhanced: EnhancedSynthesizedContext) -> str:
        """
        Format enhanced synthesis with both raw context and lens understanding.
        """
        parts = []

        # First: Context Lens understanding (the "so what")
        if enhanced.lens:
            parts.append(self.format_lens_for_prompt(enhanced.lens))

        # Then: Raw synthesized context (the evidence)
        parts.append(self.format_for_prompt(enhanced))

        return "\n\n---\n\n".join(parts)
