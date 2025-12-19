"""
MYND Brain - Embedding Engine
=============================
Generates semantic embeddings for text using sentence-transformers.
Optimized for Apple Silicon via MPS.
"""

import numpy as np
import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """
    Embedding engine using sentence-transformers.
    Much more powerful than browser-based Universal Sentence Encoder.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: torch.device = None):
        """
        Initialize the embedding engine.

        Args:
            model_name: HuggingFace model name. Options:
                - "BAAI/bge-small-en-v1.5": Fast, 384 dims, best retrieval (default)
                - "BAAI/bge-base-en-v1.5": Higher quality, 768 dims
                - "all-MiniLM-L6-v2": Legacy, 384 dims
                - "all-mpnet-base-v2": Good quality, 768 dims
            device: torch device (mps, cuda, or cpu)
        """
        self.model_name = model_name
        self.device = device or torch.device("cpu")

        print(f"📦 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        print(f"✅ Embedding model loaded ({self.model.get_sentence_embedding_dimension()} dims)")

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding

        Returns:
            2D numpy array of embeddings (n_texts x embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return

        Returns:
            List of (text, similarity_score) tuples
        """
        query_emb = self.embed(query)
        candidate_embs = self.embed_batch(candidates)

        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb) / (
            np.linalg.norm(candidate_embs, axis=1) * np.linalg.norm(query_emb)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(candidates[i], float(similarities[i])) for i in top_indices]

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
