"""
MYND Brain - Graph Transformer
==============================
A Graph Transformer that enables nodes to attend to the ENTIRE map,
not just adjacent nodes. This is the key upgrade from browser ML.

"Graph Transformers combine both local message passing and global multi-head attention"
- This lets MYND see connections humans might miss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head attention for graph nodes.
    Each node can attend to ALL other nodes regardless of graph structure.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Node embeddings (batch, num_nodes, embed_dim)
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
        q = q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = torch.matmul(attn_probs, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.embed_dim)
        out = self.out_proj(out)

        if return_attention:
            return out, attn_probs
        return out, None


class GraphTransformerLayer(nn.Module):
    """
    A single Graph Transformer layer with:
    - Multi-head self-attention (global, across all nodes)
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ff_dim: int = None,
        dropout: float = 0.1
    ):
        super().__init__()

        ff_dim = ff_dim or embed_dim * 4

        self.attention = MultiHeadGraphAttention(
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
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with residual
        attn_out, attn_weights = self.attention(
            self.norm1(x),
            mask=mask,
            return_attention=return_attention
        )
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        return x, attn_weights


class MYNDGraphTransformer(nn.Module):
    """
    The full MYND Graph Transformer.

    Takes node embeddings and produces:
    1. Enhanced node representations (each node "sees" all others)
    2. Attention weights showing which nodes attend to which

    This enables:
    - Connection prediction across the entire map
    - Finding hidden relationships
    - Semantic clustering
    """

    def __init__(
        self,
        input_dim: int = 384,      # Sentence embedding dimension
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: torch.device = None
    ):
        super().__init__()

        self.device = device or torch.device("cpu")

        # Input projection (embedding dim -> hidden dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output heads
        self.connection_head = nn.Linear(hidden_dim * 2, 1)  # Predict connection score
        self.relationship_head = nn.Linear(hidden_dim * 2, 4)  # Predict relationship type

        self.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔮 Graph Transformer initialized: {total_params:,} parameters")

    def forward(
        self,
        node_embeddings: np.ndarray,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the transformer.

        Args:
            node_embeddings: numpy array of shape (num_nodes, embedding_dim)
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

        # Project to hidden dim
        x = self.input_proj(x)

        # Pass through transformer layers
        attention = None
        for layer in self.layers:
            x, attention = layer(x, return_attention=return_attention)

        return x.squeeze(0), attention

    def forward_with_attention(
        self,
        node_embeddings: np.ndarray,
        source_idx: int
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Forward pass that returns attention weights from source node to all others.

        Args:
            node_embeddings: numpy array of shape (num_nodes, embedding_dim)
            source_idx: Index of the source node

        Returns:
            output: Enhanced node representations
            attention_weights: Attention from source to all nodes (num_nodes,)
        """
        output, attention = self.forward(node_embeddings, return_attention=True)

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

    def predict_connection_score(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> float:
        """Predict likelihood of connection between two nodes."""
        combined = torch.cat([source_embedding, target_embedding], dim=-1)
        score = torch.sigmoid(self.connection_head(combined))
        return float(score.item())

    def predict_relationship_type(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> str:
        """Predict relationship type between two nodes."""
        combined = torch.cat([source_embedding, target_embedding], dim=-1)
        logits = self.relationship_head(combined)
        type_idx = torch.argmax(logits).item()

        relationship_types = ['relates_to', 'leads_to', 'supports', 'conflicts_with']
        return relationship_types[type_idx]
