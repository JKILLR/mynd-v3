"""
MYND Brain - Graph Transformer v2
==================================
Upgraded architecture with:
- 8 attention heads (discover different relationship types)
- 512 hidden dimension (more representational capacity)
- Edge-aware attention (adjacency matrix influences attention)
- Graph positional encoding (depth/structure awareness)

"Multi-head attention computes multiple attention weights in parallel.
Each attention head learns its own set of query, key, and value transformations,
allowing the model to attend to different types of relationships simultaneously."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict


class GraphPositionalEncoding(nn.Module):
    """
    Encodes graph structure into node representations.
    - Depth encoding: How deep in the hierarchy
    - Degree encoding: How many connections
    - Centrality encoding: How important in the graph
    """

    def __init__(self, hidden_dim: int, max_depth: int = 20, max_degree: int = 50):
        super().__init__()
        self.depth_embed = nn.Embedding(max_depth, hidden_dim)
        self.degree_embed = nn.Embedding(max_degree, hidden_dim)
        self.centrality_proj = nn.Linear(1, hidden_dim)

        # Learnable combination weights
        self.combine = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        depths: torch.Tensor = None,
        degrees: torch.Tensor = None,
        centrality: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Add positional encoding to node features.

        Args:
            x: Node features (batch, num_nodes, hidden_dim)
            depths: Node depths in tree (batch, num_nodes)
            degrees: Node degrees (batch, num_nodes)
            centrality: Node centrality scores (batch, num_nodes, 1)
        """
        batch_size, num_nodes, _ = x.shape
        device = x.device

        # Default values if not provided
        if depths is None:
            depths = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)
        if degrees is None:
            degrees = torch.ones(batch_size, num_nodes, dtype=torch.long, device=device)
        if centrality is None:
            centrality = torch.ones(batch_size, num_nodes, 1, device=device) * 0.5

        # Clamp to valid ranges
        depths = depths.clamp(0, self.depth_embed.num_embeddings - 1)
        degrees = degrees.clamp(0, self.degree_embed.num_embeddings - 1)

        # Get embeddings
        depth_enc = self.depth_embed(depths)
        degree_enc = self.degree_embed(degrees)
        centrality_enc = self.centrality_proj(centrality)

        # Combine all positional info
        pos_combined = torch.cat([depth_enc, degree_enc, centrality_enc], dim=-1)
        pos_encoding = self.combine(pos_combined)

        return x + pos_encoding


class EdgeAwareMultiHeadAttention(nn.Module):
    """
    Multi-head attention that incorporates graph structure.

    Each of the 8 heads learns different relationship patterns:
    - Heads 1-2: Structural (parent-child, siblings)
    - Heads 3-4: Semantic (similar meaning)
    - Heads 5-6: Sequential (temporal/logical flow)
    - Heads 7-8: Emergent (hidden patterns)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: int = 16,  # Dimension for edge type embeddings
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.edge_dim = edge_dim

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Edge-aware components
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, num_heads)
        )

        # Learnable edge bias per head (direct connections get attention boost)
        self.edge_bias = nn.Parameter(torch.zeros(num_heads))

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Node embeddings (batch, num_nodes, embed_dim)
            adjacency: Adjacency matrix (batch, num_nodes, num_nodes) or (num_nodes, num_nodes)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            output: Transformed embeddings (batch, num_nodes, embed_dim)
            attention: Optional attention weights (batch, num_heads, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch, num_nodes, num_heads, head_dim) -> (batch, num_heads, num_nodes, head_dim)
        q = q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: (batch, num_heads, num_nodes, num_nodes)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add edge-aware bias
        if adjacency is not None:
            # Ensure adjacency has batch dimension
            if adjacency.dim() == 2:
                adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)

            # Encode edges and add bias per head
            # adjacency: (batch, num_nodes, num_nodes) -> edge features
            edge_features = adjacency.unsqueeze(-1)  # (batch, n, n, 1)
            edge_attn = self.edge_encoder(edge_features)  # (batch, n, n, num_heads)
            edge_attn = edge_attn.permute(0, 3, 1, 2)  # (batch, num_heads, n, n)

            # Add direct connection bias
            direct_connection_boost = adjacency.unsqueeze(1) * self.edge_bias.view(1, -1, 1, 1)

            attn_scores = attn_scores + edge_attn + direct_connection_boost

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = torch.matmul(attn_probs, v)

        # Reshape back: (batch, num_heads, num_nodes, head_dim) -> (batch, num_nodes, embed_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.embed_dim)
        out = self.out_proj(out)

        if return_attention:
            return out, attn_probs
        return out, None


class GraphTransformerLayer(nn.Module):
    """
    A single Graph Transformer layer with:
    - Edge-aware multi-head self-attention (global + structure-aware)
    - Feed-forward network with GELU activation
    - Pre-norm architecture (more stable training)
    - Residual connections
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1
    ):
        super().__init__()

        ff_dim = ff_dim or embed_dim * 4  # 2048 for 512 embed

        self.attention = EdgeAwareMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(
            normed,
            adjacency=adjacency,
            mask=mask,
            return_attention=return_attention
        )
        x = x + self.dropout(attn_out)

        # Pre-norm feed-forward with residual
        x = x + self.ff(self.norm2(x))

        return x, attn_weights


class MYNDGraphTransformer(nn.Module):
    """
    MYND Graph Transformer v2 - Enhanced Architecture

    Upgrades from v1:
    - 8 attention heads (was 4) - discovers more relationship types
    - 512 hidden dim (was 256) - more representational capacity
    - Edge-aware attention - adjacency matrix influences attention
    - Graph positional encoding - depth/structure awareness
    - 3 transformer layers (was 2) - deeper reasoning

    Each attention head specializes in different relationship patterns:
    - Sequential/temporal (goals → steps)
    - Categorical clustering (health topics)
    - Semantic similarity (related concepts)
    - Hierarchical (parent-child)
    - Hidden emergent patterns

    Expected ~6.7M parameters (was 1.68M)
    """

    def __init__(
        self,
        input_dim: int = 384,      # Sentence embedding dimension (MiniLM)
        hidden_dim: int = 512,     # Upgraded from 256
        num_heads: int = 8,        # Upgraded from 4
        num_layers: int = 3,       # Upgraded from 2
        dropout: float = 0.1,
        device: torch.device = None
    ):
        super().__init__()

        self.device = device or torch.device("cpu")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Input projection (embedding dim -> hidden dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Graph positional encoding
        self.pos_encoding = GraphPositionalEncoding(hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output heads
        self.connection_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Relationship type prediction (more nuanced)
        self.relationship_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8)  # More relationship types
        )

        # Node importance scoring
        self.importance_head = nn.Linear(hidden_dim, 1)

        self.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔮 Graph Transformer v2 initialized:")
        print(f"   - {total_params:,} total parameters")
        print(f"   - {num_heads} attention heads")
        print(f"   - {hidden_dim} hidden dimension")
        print(f"   - {num_layers} transformer layers")

    def forward(
        self,
        node_embeddings: np.ndarray,
        adjacency: Optional[np.ndarray] = None,
        depths: Optional[np.ndarray] = None,
        degrees: Optional[np.ndarray] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the transformer.

        Args:
            node_embeddings: numpy array of shape (num_nodes, embedding_dim)
            adjacency: Optional adjacency matrix (num_nodes, num_nodes)
            depths: Optional node depths in hierarchy
            degrees: Optional node degrees (connection counts)
            return_attention: whether to return attention weights

        Returns:
            output: Enhanced node representations (num_nodes, hidden_dim)
            attention: Optional attention weights from last layer
        """
        # Convert to tensor
        x = torch.tensor(node_embeddings, dtype=torch.float32, device=self.device)

        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, num_nodes, embed_dim)

        batch_size, num_nodes, _ = x.shape

        # Convert adjacency to tensor
        adj_tensor = None
        if adjacency is not None:
            adj_tensor = torch.tensor(adjacency, dtype=torch.float32, device=self.device)
            if adj_tensor.dim() == 2:
                adj_tensor = adj_tensor.unsqueeze(0)

        # Convert positional info to tensors
        depth_tensor = None
        degree_tensor = None
        if depths is not None:
            depth_tensor = torch.tensor(depths, dtype=torch.long, device=self.device)
            if depth_tensor.dim() == 1:
                depth_tensor = depth_tensor.unsqueeze(0)
        if degrees is not None:
            degree_tensor = torch.tensor(degrees, dtype=torch.long, device=self.device)
            if degree_tensor.dim() == 1:
                degree_tensor = degree_tensor.unsqueeze(0)

        # Project to hidden dim
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoding(x, depths=depth_tensor, degrees=degree_tensor)

        # Pass through transformer layers
        attention = None
        for layer in self.layers:
            x, attention = layer(
                x,
                adjacency=adj_tensor,
                return_attention=return_attention
            )

        # Final normalization
        x = self.final_norm(x)

        return x.squeeze(0), attention

    def forward_with_attention(
        self,
        node_embeddings: np.ndarray,
        source_idx: int,
        adjacency: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Forward pass that returns attention weights from source node to all others.

        Args:
            node_embeddings: numpy array of shape (num_nodes, embedding_dim)
            source_idx: Index of the source node
            adjacency: Optional adjacency matrix

        Returns:
            output: Enhanced node representations
            attention_weights: Attention from source to all nodes (num_nodes,)
        """
        output, attention = self.forward(
            node_embeddings,
            adjacency=adjacency,
            return_attention=True
        )

        # attention shape: (1, num_heads, num_nodes, num_nodes)
        # Average across heads and get source node's attention
        if attention is not None:
            attn_weights = attention[0].mean(dim=0)[source_idx].detach().cpu().numpy()
        else:
            # Fallback: use cosine similarity
            output_np = output.detach().cpu().numpy()
            source_emb = output_np[source_idx]
            attn_weights = np.dot(output_np, source_emb) / (
                np.linalg.norm(output_np, axis=1) * np.linalg.norm(source_emb) + 1e-8
            )

        return output, attn_weights

    def get_head_attention(
        self,
        node_embeddings: np.ndarray,
        source_idx: int,
        adjacency: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get attention weights from each head separately.
        Useful for understanding what each head learned.

        Returns:
            Dictionary mapping head names to attention weights
        """
        output, attention = self.forward(
            node_embeddings,
            adjacency=adjacency,
            return_attention=True
        )

        if attention is None:
            return {}

        # attention: (1, num_heads, num_nodes, num_nodes)
        head_names = [
            "structural_1", "structural_2",
            "semantic_1", "semantic_2",
            "sequential_1", "sequential_2",
            "emergent_1", "emergent_2"
        ]

        result = {}
        for i, name in enumerate(head_names[:self.num_heads]):
            result[name] = attention[0, i, source_idx].detach().cpu().numpy()

        return result

    def predict_connection_score(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> float:
        """Predict likelihood of connection between two nodes."""
        if source_embedding.dim() == 1:
            source_embedding = source_embedding.unsqueeze(0)
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)

        combined = torch.cat([source_embedding, target_embedding], dim=-1)
        score = torch.sigmoid(self.connection_head(combined))
        return float(score.squeeze().item())

    def predict_relationship_type(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> Tuple[str, float]:
        """
        Predict relationship type between two nodes.

        Returns:
            (relationship_type, confidence)
        """
        if source_embedding.dim() == 1:
            source_embedding = source_embedding.unsqueeze(0)
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)

        combined = torch.cat([source_embedding, target_embedding], dim=-1)
        logits = self.relationship_head(combined)
        probs = F.softmax(logits, dim=-1)

        type_idx = torch.argmax(probs).item()
        confidence = probs[0, type_idx].item()

        relationship_types = [
            'relates_to',      # General semantic relation
            'leads_to',        # Causal/sequential
            'supports',        # Evidence/backing
            'conflicts_with',  # Contradiction
            'is_part_of',      # Hierarchical
            'is_example_of',   # Instance
            'contrasts_with',  # Comparison
            'depends_on'       # Dependency
        ]

        return relationship_types[type_idx], confidence

    def get_node_importance(
        self,
        node_embeddings: np.ndarray,
        adjacency: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Score each node's importance based on transformer representations.

        Returns:
            importance_scores: (num_nodes,) array of importance scores
        """
        output, _ = self.forward(node_embeddings, adjacency=adjacency)
        importance = torch.sigmoid(self.importance_head(output)).squeeze(-1)
        return importance.detach().cpu().numpy()

    def find_missing_connections(
        self,
        node_embeddings: np.ndarray,
        adjacency: np.ndarray,
        threshold: float = 0.7,
        top_k: int = 10
    ) -> List[Tuple[int, int, float]]:
        """
        Find node pairs that should probably be connected but aren't.

        Args:
            node_embeddings: Node embeddings
            adjacency: Current adjacency matrix
            threshold: Minimum connection score to suggest
            top_k: Maximum number of suggestions

        Returns:
            List of (source_idx, target_idx, score) tuples
        """
        output, _ = self.forward(node_embeddings, adjacency=adjacency)
        num_nodes = output.shape[0]

        suggestions = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Skip if already connected
                if adjacency[i, j] > 0 or adjacency[j, i] > 0:
                    continue

                score = self.predict_connection_score(output[i], output[j])
                if score >= threshold:
                    suggestions.append((i, j, score))

        # Sort by score and return top_k
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions[:top_k]
