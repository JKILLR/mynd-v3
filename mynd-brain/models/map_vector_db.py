"""
MYND Brain - Unified Map Vector Database
=========================================
Single source of truth for the knowledge graph.
Combines map structure + embeddings + source tracking.

The visual mind map IS the vector database.
"""

import json
import time
import pathlib
import numpy as np
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class SourceRef:
    """Reference to where knowledge came from."""
    conversation_id: str
    excerpt: str  # Relevant snippet
    extracted_at: str  # ISO timestamp

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SourceRef':
        return cls(**data)


@dataclass
class UnifiedNode:
    """
    A node in the unified knowledge graph.
    Combines visual map node + vector embedding + provenance.
    """
    # Identity
    id: str
    type: str = 'concept'  # 'concept' | 'conversation' | 'memory' | 'insight'

    # Content
    label: str = ''
    description: str = ''
    content: str = ''  # Full text (for large content)
    summary: str = ''  # AI-generated summary

    # Vector
    embedding: Optional[List[float]] = None
    embedding_model: str = 'BAAI/bge-small-en-v1.5'
    embedded_at: Optional[str] = None

    # Graph structure
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    links: List[Dict] = field(default_factory=list)  # Semantic cross-links

    # Visual (for rendering)
    visual: Dict = field(default_factory=dict)

    # Source tracking
    sources: List[SourceRef] = field(default_factory=list)

    # Metadata
    source_info: Optional[Dict] = None  # Where this node came from
    created_at: str = ''
    updated_at: str = ''
    enriched_at: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[str] = None
    confidence: float = 1.0  # How sure are we about this?

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert SourceRef objects
        data['sources'] = [s.to_dict() if isinstance(s, SourceRef) else s for s in self.sources]
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'UnifiedNode':
        """Create from dictionary."""
        # Convert sources back to SourceRef objects
        if 'sources' in data and data['sources']:
            data['sources'] = [
                SourceRef.from_dict(s) if isinstance(s, dict) else s
                for s in data['sources']
            ]
        return cls(**data)

    def get_text_for_embedding(self) -> str:
        """Get the text that should be embedded."""
        parts = [self.label]
        if self.description:
            parts.append(self.description)
        if self.summary:
            parts.append(self.summary)
        return '. '.join(parts)


class MapVectorDB:
    """
    Unified storage for the knowledge graph.

    The map structure and vector database are ONE system.
    Every node has: visual properties + semantic embedding + source tracking.
    """

    def __init__(self, data_dir: pathlib.Path, embedder=None):
        """
        Initialize the unified map vector database.

        Args:
            data_dir: Base directory for data storage
            embedder: EmbeddingEngine instance for generating vectors
        """
        self.data_dir = data_dir / "graph"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder
        self.embedding_model = 'BAAI/bge-small-en-v1.5'

        # In-memory graph
        self.nodes: Dict[str, UnifiedNode] = {}
        self.root_id: Optional[str] = None

        # Embedding cache for fast similarity search
        self._embedding_matrix: Optional[np.ndarray] = None
        self._node_id_index: List[str] = []  # Maps matrix row → node_id
        self._cache_valid = False

        # Load existing graph
        self._load()

    # ═══════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════

    def _load(self):
        """Load graph from disk."""
        graph_path = self.data_dir / "graph.json"

        if graph_path.exists():
            try:
                with open(graph_path, 'r') as f:
                    data = json.load(f)

                self.root_id = data.get('root_id')

                for node_id, node_data in data.get('nodes', {}).items():
                    self.nodes[node_id] = UnifiedNode.from_dict(node_data)

                print(f"✅ Loaded unified graph: {len(self.nodes)} nodes")
                self._invalidate_cache()

            except Exception as e:
                print(f"⚠️ Error loading graph: {e}")
                self.nodes = {}
                self.root_id = None
        else:
            print("📝 No existing graph found, starting fresh")

    def save(self):
        """Save graph to disk."""
        graph_path = self.data_dir / "graph.json"

        data = {
            'version': 1,
            'root_id': self.root_id,
            'nodes': {
                node_id: node.to_dict()
                for node_id, node in self.nodes.items()
            },
            'stats': {
                'total_nodes': len(self.nodes),
                'embedded_nodes': sum(1 for n in self.nodes.values() if n.embedding),
                'last_saved': datetime.utcnow().isoformat()
            }
        }

        with open(graph_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Saved unified graph: {len(self.nodes)} nodes")

    # ═══════════════════════════════════════════════════════════════════
    # NODE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    def add_node(self, node: UnifiedNode, embed: bool = True) -> UnifiedNode:
        """
        Add a node to the graph.

        Args:
            node: The node to add
            embed: Whether to generate embedding

        Returns:
            The added node (with embedding if requested)
        """
        # Generate embedding if needed
        if embed and self.embedder and not node.embedding:
            text = node.get_text_for_embedding()
            if text:
                node.embedding = self.embedder.embed(text).tolist()
                node.embedded_at = datetime.utcnow().isoformat()
                node.embedding_model = self.embedding_model

        # Add to graph
        self.nodes[node.id] = node

        # Update parent's children list
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children:
                parent.children.append(node.id)

        # Set root if this is the first node
        if self.root_id is None:
            self.root_id = node.id

        self._invalidate_cache()
        return node

    def update_node(self, node_id: str, updates: Dict, re_embed: bool = False) -> Optional[UnifiedNode]:
        """
        Update a node's properties.

        Args:
            node_id: ID of node to update
            updates: Dictionary of updates
            re_embed: Whether to regenerate embedding

        Returns:
            Updated node or None if not found
        """
        if node_id not in self.nodes:
            return None

        node = self.nodes[node_id]

        # Apply updates
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)

        node.updated_at = datetime.utcnow().isoformat()

        # Re-embed if content changed
        if re_embed and self.embedder:
            text = node.get_text_for_embedding()
            if text:
                node.embedding = self.embedder.embed(text).tolist()
                node.embedded_at = datetime.utcnow().isoformat()

        self._invalidate_cache()
        return node

    def delete_node(self, node_id: str, recursive: bool = True) -> bool:
        """
        Delete a node from the graph.

        Args:
            node_id: ID of node to delete
            recursive: Whether to delete children

        Returns:
            True if deleted
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        # Recursively delete children if requested
        if recursive:
            for child_id in list(node.children):
                self.delete_node(child_id, recursive=True)
        else:
            # Reparent children to this node's parent
            for child_id in node.children:
                if child_id in self.nodes:
                    self.nodes[child_id].parent_id = node.parent_id
                    if node.parent_id and node.parent_id in self.nodes:
                        self.nodes[node.parent_id].children.append(child_id)

        # Remove from parent's children
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node_id in parent.children:
                parent.children.remove(node_id)

        # Remove node
        del self.nodes[node_id]

        # Update root if needed
        if node_id == self.root_id:
            self.root_id = None

        self._invalidate_cache()
        return True

    def get_node(self, node_id: str) -> Optional[UnifiedNode]:
        """Get a node by ID."""
        node = self.nodes.get(node_id)
        if node:
            node.access_count += 1
            node.last_accessed = datetime.utcnow().isoformat()
        return node

    def get_path(self, node_id: str) -> List[UnifiedNode]:
        """Get path from root to node."""
        path = []
        current_id = node_id

        while current_id and current_id in self.nodes:
            node = self.nodes[current_id]
            path.insert(0, node)
            current_id = node.parent_id

        return path

    def get_children(self, node_id: str) -> List[UnifiedNode]:
        """Get all children of a node."""
        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]
        return [self.nodes[cid] for cid in node.children if cid in self.nodes]

    # ═══════════════════════════════════════════════════════════════════
    # SEMANTIC SEARCH
    # ═══════════════════════════════════════════════════════════════════

    def _invalidate_cache(self):
        """Invalidate the embedding cache."""
        self._cache_valid = False
        self._embedding_matrix = None
        self._node_id_index = []

    def _build_cache(self):
        """Build the embedding matrix for fast similarity search."""
        if self._cache_valid:
            return

        embeddings = []
        node_ids = []

        for node_id, node in self.nodes.items():
            if node.embedding:
                embeddings.append(node.embedding)
                node_ids.append(node_id)

        if embeddings:
            self._embedding_matrix = np.array(embeddings)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
            self._embedding_matrix = self._embedding_matrix / (norms + 1e-8)
        else:
            self._embedding_matrix = None

        self._node_id_index = node_ids
        self._cache_valid = True

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.3,
        node_types: Optional[List[str]] = None,
        parent_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Semantic search across the graph.

        Args:
            query: Search query
            top_k: Maximum results
            threshold: Minimum similarity
            node_types: Filter by node types
            parent_id: Only search under this node

        Returns:
            List of {node, similarity} dicts
        """
        if not self.embedder:
            return []

        self._build_cache()

        if self._embedding_matrix is None or len(self._node_id_index) == 0:
            return []

        # Get query embedding
        query_embedding = self.embedder.embed(query)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Compute similarities
        similarities = np.dot(self._embedding_matrix, query_embedding)

        # Build results
        results = []
        for i, (node_id, similarity) in enumerate(zip(self._node_id_index, similarities)):
            if similarity < threshold:
                continue

            node = self.nodes.get(node_id)
            if not node:
                continue

            # Apply filters
            if node_types and node.type not in node_types:
                continue

            if parent_id:
                path = self.get_path(node_id)
                if not any(n.id == parent_id for n in path):
                    continue

            results.append({
                'node': node,
                'node_id': node_id,
                'similarity': float(similarity)
            })

        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:top_k]

    def find_similar_nodes(
        self,
        node_id: str,
        top_k: int = 5,
        exclude_ancestors: bool = True,
        exclude_descendants: bool = True
    ) -> List[Dict]:
        """
        Find nodes similar to a given node.

        Args:
            node_id: Source node ID
            top_k: Maximum results
            exclude_ancestors: Exclude ancestor nodes
            exclude_descendants: Exclude descendant nodes

        Returns:
            List of {node, similarity} dicts
        """
        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]
        if not node.embedding:
            return []

        self._build_cache()

        if self._embedding_matrix is None:
            return []

        # Get excluded node IDs
        excluded: Set[str] = {node_id}

        if exclude_ancestors:
            path = self.get_path(node_id)
            excluded.update(n.id for n in path)

        if exclude_descendants:
            def get_descendants(nid: str) -> Set[str]:
                if nid not in self.nodes:
                    return set()
                result = {nid}
                for child_id in self.nodes[nid].children:
                    result.update(get_descendants(child_id))
                return result
            excluded.update(get_descendants(node_id))

        # Compute similarities
        query_embedding = np.array(node.embedding)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        similarities = np.dot(self._embedding_matrix, query_embedding)

        # Build results
        results = []
        for i, (nid, similarity) in enumerate(zip(self._node_id_index, similarities)):
            if nid in excluded:
                continue

            results.append({
                'node': self.nodes[nid],
                'node_id': nid,
                'similarity': float(similarity)
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    # ═══════════════════════════════════════════════════════════════════
    # KNOWLEDGE INTEGRATION
    # ═══════════════════════════════════════════════════════════════════

    def find_or_create_node(
        self,
        label: str,
        description: str = '',
        similarity_threshold: float = 0.85,
        source: Optional[SourceRef] = None
    ) -> tuple[UnifiedNode, bool]:
        """
        Find existing similar node or create new one.
        Used for integrating extracted knowledge.

        Args:
            label: Node label
            description: Node description
            similarity_threshold: Threshold for considering nodes as same
            source: Source reference for provenance

        Returns:
            Tuple of (node, is_new)
        """
        # Search for similar existing node
        search_text = f"{label}. {description}" if description else label
        similar = self.search(search_text, top_k=3, threshold=similarity_threshold)

        if similar:
            # Found similar node - enrich it
            existing = similar[0]['node']

            # Add source reference
            if source:
                existing.sources.append(source)

            existing.enriched_at = datetime.utcnow().isoformat()
            existing.access_count += 1

            self._invalidate_cache()
            return existing, False

        # No similar node - find best parent and create
        parent_id = self._find_best_parent(label, description)

        new_node = UnifiedNode(
            id=self._generate_id(),
            type='concept',
            label=label,
            description=description,
            parent_id=parent_id,
            sources=[source] if source else []
        )

        self.add_node(new_node, embed=True)
        return new_node, True

    def _find_best_parent(self, label: str, description: str) -> Optional[str]:
        """Find the best parent node for new content."""
        if not self.nodes or not self.root_id:
            return None

        # Search for semantically related nodes
        search_text = f"{label}. {description}" if description else label
        similar = self.search(search_text, top_k=5, threshold=0.4)

        if not similar:
            return self.root_id

        # Find the most relevant node that could be a parent
        # Prefer nodes at depth 1-2 (main categories/subcategories)
        for result in similar:
            node = result['node']
            path = self.get_path(node.id)
            depth = len(path)

            # Good parent candidates are at depth 1-3
            if 1 <= depth <= 3:
                return node.id

        # Fallback to parent of most similar node
        best_match = similar[0]['node']
        return best_match.parent_id or self.root_id

    def _generate_id(self) -> str:
        """Generate a unique node ID."""
        import uuid
        return f"node_{uuid.uuid4().hex[:12]}"

    # ═══════════════════════════════════════════════════════════════════
    # SYNC WITH BROWSER MAP
    # ═══════════════════════════════════════════════════════════════════

    def import_from_browser_map(self, map_data: Dict, re_embed_all: bool = False):
        """
        Import/sync from browser map format.

        Args:
            map_data: Map data from browser (recursive node structure)
            re_embed_all: Whether to regenerate all embeddings
        """
        def import_node(node_data: Dict, parent_id: Optional[str] = None, depth: int = 0):
            node_id = node_data.get('id')

            # Check if node exists
            existing = self.nodes.get(node_id)

            if existing:
                # Update existing node
                existing.label = node_data.get('label', existing.label)
                existing.description = node_data.get('description', existing.description)
                existing.parent_id = parent_id
                existing.visual = node_data.get('visual', existing.visual)
                existing.updated_at = datetime.utcnow().isoformat()

                if re_embed_all and self.embedder:
                    text = existing.get_text_for_embedding()
                    if text:
                        existing.embedding = self.embedder.embed(text).tolist()
                        existing.embedded_at = datetime.utcnow().isoformat()
            else:
                # Create new node
                new_node = UnifiedNode(
                    id=node_id,
                    type='concept',
                    label=node_data.get('label', ''),
                    description=node_data.get('description', ''),
                    parent_id=parent_id,
                    visual=node_data.get('visual', {})
                )
                self.add_node(new_node, embed=True)

            # Set as root if depth 0
            if depth == 0:
                self.root_id = node_id

            # Import children
            children = node_data.get('children', [])
            child_ids = []
            for child_data in children:
                import_node(child_data, parent_id=node_id, depth=depth + 1)
                child_ids.append(child_data.get('id'))

            # Update children list
            if node_id in self.nodes:
                self.nodes[node_id].children = child_ids

        # Start import from root
        import_node(map_data)
        self._invalidate_cache()
        self.save()

        print(f"✅ Imported browser map: {len(self.nodes)} nodes")

    def export_to_browser_map(self) -> Optional[Dict]:
        """
        Export to browser map format.

        Returns:
            Recursive map structure for browser
        """
        if not self.root_id or self.root_id not in self.nodes:
            return None

        def export_node(node_id: str) -> Dict:
            node = self.nodes[node_id]

            return {
                'id': node.id,
                'label': node.label,
                'description': node.description,
                'visual': node.visual,
                'children': [
                    export_node(child_id)
                    for child_id in node.children
                    if child_id in self.nodes
                ]
            }

        return export_node(self.root_id)

    # ═══════════════════════════════════════════════════════════════════
    # CONTEXT RETRIEVAL FOR RAG
    # ═══════════════════════════════════════════════════════════════════

    def get_context(
        self,
        query: str,
        max_tokens: int = 8000,
        include_sources: bool = False
    ) -> Dict:
        """
        Get relevant context for RAG.

        Args:
            query: Query to find relevant context for
            max_tokens: Maximum tokens to return
            include_sources: Whether to include source excerpts

        Returns:
            Dict with context document and metadata
        """
        results = self.search(query, top_k=20, threshold=0.35)

        if not results:
            return {'context': '', 'nodes_used': 0, 'chars': 0}

        context_parts = []
        chars_used = 0
        max_chars = max_tokens * 4  # Rough estimate
        nodes_used = 0

        for result in results:
            node = result['node']
            similarity = result['similarity']

            # Build context for this node
            path = self.get_path(node.id)
            path_str = ' > '.join(n.label for n in path)

            parts = [f"[{path_str}] (relevance: {similarity:.0%})"]

            if node.description:
                parts.append(node.description)

            if node.summary:
                parts.append(f"Summary: {node.summary}")

            if include_sources and node.sources:
                source_texts = [f"- {s.excerpt[:200]}" for s in node.sources[:3]]
                if source_texts:
                    parts.append("Sources:\n" + "\n".join(source_texts))

            node_context = "\n".join(parts)

            if chars_used + len(node_context) > max_chars:
                break

            context_parts.append(node_context)
            chars_used += len(node_context)
            nodes_used += 1

        return {
            'context': "\n\n---\n\n".join(context_parts),
            'nodes_used': nodes_used,
            'chars': chars_used
        }

    # ═══════════════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Get database statistics."""
        embedded_count = sum(1 for n in self.nodes.values() if n.embedding)
        type_counts = {}
        for n in self.nodes.values():
            type_counts[n.type] = type_counts.get(n.type, 0) + 1

        return {
            'total_nodes': len(self.nodes),
            'embedded_nodes': embedded_count,
            'embedding_coverage': embedded_count / max(len(self.nodes), 1),
            'type_counts': type_counts,
            'root_id': self.root_id,
            'embedding_model': self.embedding_model
        }
