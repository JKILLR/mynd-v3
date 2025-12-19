"""
MYND Brain - Conversation Archive
==================================
Stores raw AI conversations as source material.
Conversations are processed by the Knowledge Extractor to become map nodes.
"""

import json
import pathlib
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib


@dataclass
class ArchivedConversation:
    """A stored conversation from any AI source."""
    id: str
    source: str  # 'claude', 'chatgpt', 'grok', 'other'
    title: str
    text: str  # Full conversation text

    # Vector for finding related conversations
    embedding: Optional[List[float]] = None
    embedding_model: str = 'BAAI/bge-small-en-v1.5'

    # Processing status
    processed: bool = False
    processed_at: Optional[str] = None
    concepts_extracted: List[str] = field(default_factory=list)  # Node IDs created

    # Metadata
    created_at: str = ''
    char_count: int = 0
    message_count: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.char_count:
            self.char_count = len(self.text)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ArchivedConversation':
        return cls(**data)

    def get_summary_text(self, max_chars: int = 2000) -> str:
        """Get text suitable for embedding (truncated if needed)."""
        # Use title + beginning and end of conversation
        if len(self.text) <= max_chars:
            return f"{self.title}. {self.text}"

        half = (max_chars - len(self.title) - 50) // 2
        return f"{self.title}. {self.text[:half]}... {self.text[-half:]}"


class ConversationArchive:
    """
    Archive for raw AI conversations.

    Conversations are stored as source material and processed
    by the Knowledge Extractor to create map nodes.
    """

    def __init__(self, data_dir: pathlib.Path, embedder=None):
        """
        Initialize the conversation archive.

        Args:
            data_dir: Base directory for data storage
            embedder: EmbeddingEngine instance for generating vectors
        """
        self.data_dir = data_dir / "conversations"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder
        self.index_path = self.data_dir / "index.json"

        # In-memory index
        self.index: Dict[str, Dict] = {}  # id -> metadata
        self._embedding_cache: Dict[str, np.ndarray] = {}

        self._load_index()

    def _load_index(self):
        """Load the conversation index."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                self.index = data.get('conversations', {})
                print(f"✅ Loaded conversation archive: {len(self.index)} conversations")
            except Exception as e:
                print(f"⚠️ Error loading conversation index: {e}")
                self.index = {}
        else:
            print("📝 No existing conversation archive found")

    def _save_index(self):
        """Save the conversation index."""
        data = {
            'version': 1,
            'conversations': self.index,
            'stats': {
                'total': len(self.index),
                'processed': sum(1 for c in self.index.values() if c.get('processed')),
                'last_updated': datetime.utcnow().isoformat()
            }
        }

        with open(self.index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_id(self, text: str, source: str) -> str:
        """Generate a unique ID for a conversation."""
        content = f"{source}:{text[:500]}:{datetime.utcnow().isoformat()}"
        return f"conv_{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    # ═══════════════════════════════════════════════════════════════════
    # CRUD OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    def add(
        self,
        text: str,
        source: str = 'unknown',
        title: Optional[str] = None
    ) -> ArchivedConversation:
        """
        Add a conversation to the archive.

        Args:
            text: Full conversation text
            source: Source AI ('claude', 'chatgpt', 'grok', etc.)
            title: Optional title (auto-generated if not provided)

        Returns:
            The archived conversation
        """
        # Generate ID
        conv_id = self._generate_id(text, source)

        # Auto-generate title if not provided
        if not title:
            # Use first meaningful line
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            title = lines[0][:100] if lines else 'Untitled Conversation'

        # Count messages (rough estimate based on role markers)
        message_markers = ['Human:', 'User:', 'Assistant:', 'Claude:', 'ChatGPT:', 'Grok:']
        message_count = sum(text.count(marker) for marker in message_markers)

        # Create conversation object
        conv = ArchivedConversation(
            id=conv_id,
            source=source,
            title=title,
            text=text,
            message_count=max(message_count, 1)
        )

        # Generate embedding
        if self.embedder:
            summary_text = conv.get_summary_text()
            conv.embedding = self.embedder.embed(summary_text).tolist()

        # Save conversation file
        conv_path = self.data_dir / f"{conv_id}.json"
        with open(conv_path, 'w') as f:
            json.dump(conv.to_dict(), f, indent=2)

        # Update index
        self.index[conv_id] = {
            'id': conv_id,
            'source': source,
            'title': title,
            'created_at': conv.created_at,
            'char_count': conv.char_count,
            'message_count': conv.message_count,
            'processed': False
        }
        self._save_index()

        print(f"📥 Archived conversation: {title[:50]}... ({conv.char_count} chars)")
        return conv

    def get(self, conv_id: str) -> Optional[ArchivedConversation]:
        """Get a conversation by ID."""
        conv_path = self.data_dir / f"{conv_id}.json"

        if not conv_path.exists():
            return None

        try:
            with open(conv_path, 'r') as f:
                data = json.load(f)
            return ArchivedConversation.from_dict(data)
        except Exception as e:
            print(f"⚠️ Error loading conversation {conv_id}: {e}")
            return None

    def update(self, conv: ArchivedConversation):
        """Update a conversation in the archive."""
        conv_path = self.data_dir / f"{conv.id}.json"

        with open(conv_path, 'w') as f:
            json.dump(conv.to_dict(), f, indent=2)

        # Update index
        self.index[conv.id] = {
            'id': conv.id,
            'source': conv.source,
            'title': conv.title,
            'created_at': conv.created_at,
            'char_count': conv.char_count,
            'message_count': conv.message_count,
            'processed': conv.processed,
            'processed_at': conv.processed_at,
            'concepts_count': len(conv.concepts_extracted)
        }
        self._save_index()

    def delete(self, conv_id: str) -> bool:
        """Delete a conversation from the archive."""
        conv_path = self.data_dir / f"{conv_id}.json"

        if conv_path.exists():
            conv_path.unlink()

        if conv_id in self.index:
            del self.index[conv_id]
            self._save_index()
            return True

        return False

    def list_all(
        self,
        source: Optional[str] = None,
        processed: Optional[bool] = None
    ) -> List[Dict]:
        """
        List all conversations.

        Args:
            source: Filter by source
            processed: Filter by processed status

        Returns:
            List of conversation metadata
        """
        results = list(self.index.values())

        if source:
            results = [c for c in results if c.get('source') == source]

        if processed is not None:
            results = [c for c in results if c.get('processed') == processed]

        # Sort by created_at descending
        results.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return results

    # ═══════════════════════════════════════════════════════════════════
    # SEMANTIC SEARCH
    # ═══════════════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        top_k: int = 5,
        source: Optional[str] = None
    ) -> List[Dict]:
        """
        Search conversations by semantic similarity.

        Args:
            query: Search query
            top_k: Maximum results
            source: Filter by source

        Returns:
            List of {conversation, similarity} dicts
        """
        if not self.embedder:
            return []

        # Get query embedding
        query_embedding = self.embedder.embed(query)

        results = []

        for conv_id, meta in self.index.items():
            if source and meta.get('source') != source:
                continue

            # Get conversation embedding
            if conv_id in self._embedding_cache:
                conv_embedding = self._embedding_cache[conv_id]
            else:
                conv = self.get(conv_id)
                if not conv or not conv.embedding:
                    continue
                conv_embedding = np.array(conv.embedding)
                self._embedding_cache[conv_id] = conv_embedding

            # Compute similarity
            similarity = float(np.dot(query_embedding, conv_embedding) /
                             (np.linalg.norm(query_embedding) * np.linalg.norm(conv_embedding) + 1e-8))

            results.append({
                'id': conv_id,
                'title': meta.get('title'),
                'source': meta.get('source'),
                'similarity': similarity,
                'processed': meta.get('processed', False)
            })

        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:top_k]

    def get_unprocessed(self, limit: int = 10) -> List[ArchivedConversation]:
        """Get conversations that haven't been processed yet."""
        unprocessed = [
            meta for meta in self.index.values()
            if not meta.get('processed')
        ]

        # Sort by created_at (oldest first)
        unprocessed.sort(key=lambda x: x.get('created_at', ''))

        conversations = []
        for meta in unprocessed[:limit]:
            conv = self.get(meta['id'])
            if conv:
                conversations.append(conv)

        return conversations

    def mark_processed(
        self,
        conv_id: str,
        concepts_extracted: List[str]
    ):
        """Mark a conversation as processed."""
        conv = self.get(conv_id)
        if not conv:
            return

        conv.processed = True
        conv.processed_at = datetime.utcnow().isoformat()
        conv.concepts_extracted = concepts_extracted

        self.update(conv)

    # ═══════════════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Get archive statistics."""
        sources = {}
        total_chars = 0
        processed_count = 0

        for meta in self.index.values():
            source = meta.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            total_chars += meta.get('char_count', 0)
            if meta.get('processed'):
                processed_count += 1

        return {
            'total_conversations': len(self.index),
            'processed': processed_count,
            'unprocessed': len(self.index) - processed_count,
            'total_chars': total_chars,
            'total_mb': round(total_chars / 1024 / 1024, 2),
            'sources': sources
        }
