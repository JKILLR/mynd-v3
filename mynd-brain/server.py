"""
MYND Brain - Local ML Server
============================
A local Python server that provides GPU-accelerated ML for MYND.
Runs on Apple Silicon (M2) via Metal Performance Shaders (MPS).

Features:
- Text embeddings (sentence-transformers)
- Graph Transformer for connection prediction
- Voice transcription (Whisper)
- Image understanding (CLIP)

Start with: uvicorn server:app --reload --port 8420
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

import time
import asyncio
import base64
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import httpx
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Local imports
from models.embeddings import EmbeddingEngine
from models.graph_transformer import MYNDGraphTransformer
from models.voice import VoiceTranscriber
from models.vision import VisionEngine
from brain import UnifiedBrain, ContextRequest, ContextResponse

# Unified system imports
from models.map_vector_db import MapVectorDB, UnifiedNode, SourceRef
from models.conversation_archive import ConversationArchive, ArchivedConversation
from models.knowledge_extractor import KnowledgeExtractor
from utils.cli_executor import call_claude_cli_with_conversation

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class Config:
    PORT = 8420
    HOST = "0.0.0.0"  # Listen on all interfaces for network access

    # Model settings
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Fast, best retrieval quality
    EMBEDDING_DIM = 384

    # Graph Transformer v2 settings
    HIDDEN_DIM = 512   # Upgraded from 256
    NUM_HEADS = 8      # Upgraded from 4
    NUM_LAYERS = 3     # Upgraded from 2

    # Voice settings (Whisper)
    WHISPER_MODEL = "base"  # tiny, base, small, medium, large

    # Vision settings (CLIP)
    CLIP_MODEL = "ViT-B-32"
    CLIP_PRETRAINED = "laion2b_s34b_b79k"

    # Context Lens settings
    CONTEXT_LENS_FRESHNESS_SECONDS = 300  # 5 minutes - max age for lens training

    # Device selection
    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        elif torch.cuda.is_available():
            return torch.device("cuda")  # NVIDIA
        else:
            return torch.device("cpu")

config = Config()

# ═══════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════

class EmbedRequest(BaseModel):
    text: str

class EmbedBatchRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embedding: List[float]
    dim: int
    model: str
    time_ms: float

class NodeData(BaseModel):
    id: str
    label: str
    description: Optional[str] = None
    parentId: Optional[str] = None
    depth: Optional[int] = None
    children: Optional[List[str]] = []
    embedding: Optional[List[float]] = None

class MapData(BaseModel):
    nodes: List[NodeData]

class PredictConnectionsRequest(BaseModel):
    node_id: str
    map_data: MapData
    top_k: int = 5

class ConnectionPrediction(BaseModel):
    target_id: str
    target_label: str
    score: float
    relationship_type: str

class PredictResponse(BaseModel):
    connections: List[ConnectionPrediction]
    attention_weights: Optional[Dict[str, float]] = None
    time_ms: float

class PredictCategoryRequest(BaseModel):
    text: str
    map_data: MapData
    top_k: int = 5

class CategoryPrediction(BaseModel):
    category: str  # Node label (top-level parent)
    node_id: str
    confidence: float

class PredictCategoryResponse(BaseModel):
    predictions: List[CategoryPrediction]
    embedding_used: bool
    time_ms: float

class TrainFeedbackRequest(BaseModel):
    node_id: str
    action: str  # 'accepted', 'rejected', 'corrected'
    context: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    device: str
    embedding_model: str
    graph_transformer: bool
    voice_model: Optional[str] = None
    vision_model: Optional[str] = None
    uptime_seconds: float

# Voice transcription models
class TranscribeRequest(BaseModel):
    audio_base64: str  # Base64 encoded audio
    language: Optional[str] = None
    task: str = "transcribe"  # or "translate"

class TranscribeResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    language: Optional[str] = None
    error: Optional[str] = None
    time_ms: float

# Image understanding models
class DescribeImageRequest(BaseModel):
    image_base64: str  # Base64 encoded image
    candidate_labels: Optional[List[str]] = None
    top_k: int = 5

class DescribeImageResponse(BaseModel):
    success: bool
    description: Optional[str] = None
    confidence: Optional[float] = None
    matches: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    time_ms: float

class ImageEmbedRequest(BaseModel):
    image_base64: str

class ImageEmbedResponse(BaseModel):
    embedding: List[float]
    dim: int
    time_ms: float

# Unified Brain models
class BrainContextInclude(BaseModel):
    self_awareness: bool = True
    map_context: bool = True
    memories: bool = True
    user_profile: bool = True
    neural_insights: bool = True
    synthesized_context: bool = True  # NEW: unified context from ContextSynthesizer

class GoalData(BaseModel):
    """Goal from Goal Wizard"""
    id: Optional[str] = None
    title: str
    description: Optional[str] = None
    priority: str = "medium"  # high, medium, low

class BrainContextRequest(BaseModel):
    request_type: str = "chat"  # chat, action, code_review, self_improve
    user_message: str = ""
    selected_node_id: Optional[str] = None
    map_data: Optional[MapData] = None
    user_id: Optional[str] = None  # For Supabase AI memory queries
    goals: Optional[List[GoalData]] = None  # Active goals from Goal Wizard
    include: Optional[BrainContextInclude] = None

class BrainContextResponse(BaseModel):
    context_document: str
    token_count: int
    breakdown: Dict[str, int]
    brain_state: Dict[str, Any]
    time_ms: float

# ═══════════════════════════════════════════════════════════════════
# CONVERSATION STORAGE MODELS
# ═══════════════════════════════════════════════════════════════════

class ConversationImport(BaseModel):
    """Import a conversation from any AI chat"""
    text: str  # Full conversation text
    source: str = "unknown"  # claude, grok, chatgpt, etc.
    title: Optional[str] = None  # Optional title
    metadata: Optional[Dict[str, Any]] = None

class ConversationInsight(BaseModel):
    """A key insight extracted from a conversation"""
    content: str
    type: str = "insight"  # insight, decision, question, problem, solution
    confidence: float = 0.8

class StoredConversation(BaseModel):
    """A stored conversation with all its data"""
    id: str
    text: str
    source: str
    title: str
    summary: Optional[str] = None
    insights: List[ConversationInsight] = []
    embedding: Optional[List[float]] = None
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class ConversationSearchRequest(BaseModel):
    """Search conversations by semantic similarity"""
    query: str
    top_k: int = 5
    source_filter: Optional[str] = None  # Filter by source

class ConversationContextRequest(BaseModel):
    """Get relevant context from past conversations"""
    query: str
    max_tokens: int = 20000  # Expanded token budget for richer context
    include_full_text: bool = False  # Include full conversations or just summaries

# ═══════════════════════════════════════════════════════════════════
# BACKGROUND COGNITION MODELS
# ═══════════════════════════════════════════════════════════════════

class BackgroundAnalysisRequest(BaseModel):
    """Request for background cognition analysis"""
    user_id: str
    analysis_types: Optional[List[str]] = None  # 'connections', 'patterns', 'growth', 'questions'
    max_insights: int = 3
    min_confidence: float = 0.6

class InsightResult(BaseModel):
    """A single discovered insight"""
    insight_type: str
    title: str
    content: str
    confidence: float
    source_nodes: Optional[List[str]] = None
    source_memories: Optional[List[str]] = None

class BackgroundAnalysisResponse(BaseModel):
    """Response from background analysis"""
    insights: List[InsightResult]
    insights_stored: int
    analysis_time_ms: float
    error: Optional[str] = None

# ═══════════════════════════════════════════════════════════════════
# BRAIN CHAT MODELS (V1 - Simple passthrough to Claude)
# ═══════════════════════════════════════════════════════════════════

class BrainChatMessage(BaseModel):
    """A single message in conversation history"""
    role: str  # "user" or "assistant"
    content: str

class BrainChatRequest(BaseModel):
    """Request for /brain/chat endpoint"""
    user_message: str
    conversation_history: List[BrainChatMessage] = []
    selected_node_id: Optional[str] = None
    user_id: Optional[str] = None
    # V2: Full Axel context from frontend (same system prompt used for Edge Function)
    system_prompt: Optional[str] = None

class BrainChatResponse(BaseModel):
    """Response from /brain/chat endpoint"""
    message: str
    time_ms: float = 0
    model: str = ""

# ═══════════════════════════════════════════════════════════════════
# CONVERSATION STORE - File-based storage with embeddings
# ═══════════════════════════════════════════════════════════════════

import json
import uuid
import pathlib

class ConversationStore:
    """
    Stores conversations as JSON files with embeddings for semantic search.
    Location: mynd-brain/data/conversations/
    """

    def __init__(self, base_dir: pathlib.Path, embedder=None):
        self.data_dir = base_dir / "data" / "conversations"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.index_path = self.data_dir / "_index.json"
        self.embeddings_cache = {}  # id -> embedding
        self._load_index()

    def _load_index(self):
        """Load the conversation index"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    self.index = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️ Corrupted conversation index, resetting: {e}")
                self.index = {"conversations": [], "total_chars": 0}
        else:
            self.index = {"conversations": [], "total_chars": 0}

    def _save_index(self):
        """Save the conversation index"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    async def store(self, conversation: ConversationImport) -> StoredConversation:
        """Store a new conversation"""
        conv_id = str(uuid.uuid4())[:8]
        timestamp = time.time()

        # Generate title if not provided
        title = conversation.title or self._generate_title(conversation.text)

        # Generate summary (first 500 chars for now, could use LLM later)
        summary = self._generate_summary(conversation.text)

        # Generate embedding for search
        embedding = None
        if self.embedder:
            # Embed the summary + title for better search
            embed_text = f"{title}. {summary}"
            embedding = self.embedder.embed(embed_text).tolist()
            self.embeddings_cache[conv_id] = embedding

        # Create stored conversation object
        stored = StoredConversation(
            id=conv_id,
            text=conversation.text,
            source=conversation.source,
            title=title,
            summary=summary,
            insights=[],  # Will be populated by Claude later
            embedding=embedding,
            timestamp=timestamp,
            metadata=conversation.metadata
        )

        # Save to file
        conv_path = self.data_dir / f"{conv_id}.json"
        with open(conv_path, 'w') as f:
            json.dump(stored.model_dump(), f, indent=2)

        # Update index
        self.index["conversations"].append({
            "id": conv_id,
            "title": title,
            "source": conversation.source,
            "timestamp": timestamp,
            "chars": len(conversation.text)
        })
        self.index["total_chars"] = sum(c["chars"] for c in self.index["conversations"])
        self._save_index()

        return stored

    def _generate_title(self, text: str) -> str:
        """Generate a title from conversation text"""
        # Take first meaningful line
        lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 10]
        if lines:
            first_line = lines[0][:100]
            return first_line + ("..." if len(lines[0]) > 100 else "")
        return f"Conversation {time.strftime('%Y-%m-%d %H:%M')}"

    def _generate_summary(self, text: str) -> str:
        """Generate a summary (basic for now)"""
        # Take first 500 chars, try to end at sentence
        if len(text) <= 500:
            return text
        summary = text[:500]
        last_period = summary.rfind('.')
        if last_period > 300:
            return summary[:last_period + 1]
        return summary + "..."

    def get(self, conv_id: str) -> Optional[StoredConversation]:
        """Get a conversation by ID"""
        conv_path = self.data_dir / f"{conv_id}.json"
        if not conv_path.exists():
            return None
        try:
            with open(conv_path, 'r') as f:
                data = json.load(f)
            return StoredConversation(**data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"⚠️ Failed to load conversation {conv_id}: {e}")
            return None

    def list_all(self, source_filter: Optional[str] = None) -> List[Dict]:
        """List all conversations (metadata only)"""
        convos = self.index.get("conversations", [])
        if source_filter:
            convos = [c for c in convos if c["source"] == source_filter]
        return sorted(convos, key=lambda x: x["timestamp"], reverse=True)

    async def search(self, query: str, top_k: int = 5, source_filter: Optional[str] = None) -> List[Dict]:
        """Search conversations by semantic similarity"""
        if not self.embedder:
            return []

        # Get query embedding
        query_embedding = self.embedder.embed(query)

        # Load all embeddings
        results = []
        for conv_meta in self.index.get("conversations", []):
            if source_filter and conv_meta["source"] != source_filter:
                continue

            conv_id = conv_meta["id"]

            # Get embedding from cache or load
            if conv_id in self.embeddings_cache:
                conv_embedding = np.array(self.embeddings_cache[conv_id])
            else:
                conv = self.get(conv_id)
                if conv and conv.embedding:
                    conv_embedding = np.array(conv.embedding)
                    self.embeddings_cache[conv_id] = conv.embedding
                else:
                    continue

            # Compute cosine similarity
            similarity = float(np.dot(query_embedding, conv_embedding) /
                             (np.linalg.norm(query_embedding) * np.linalg.norm(conv_embedding) + 1e-8))

            results.append({
                "id": conv_id,
                "title": conv_meta["title"],
                "source": conv_meta["source"],
                "timestamp": conv_meta["timestamp"],
                "similarity": similarity
            })

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    async def get_relevant_context(self, query: str, max_tokens: int = 20000, include_full_text: bool = False) -> str:
        """Get relevant context from past conversations for injection into prompts"""
        search_results = await self.search(query, top_k=10)

        context_parts = []
        current_tokens = 0

        def token_estimate(s: str) -> int:
            """Rough token estimate (4 chars per token)"""
            return len(s) // 4

        for result in search_results:
            if result["similarity"] < 0.3:  # Skip low relevance
                continue

            conv = self.get(result["id"])
            if not conv:
                continue

            if include_full_text:
                text = f"### {conv.title} (from {conv.source})\n{conv.text}\n"
            else:
                text = f"### {conv.title} (from {conv.source})\n{conv.summary}\n"

            tokens = token_estimate(text)
            if current_tokens + tokens > max_tokens:
                break

            context_parts.append(text)
            current_tokens += tokens

        if not context_parts:
            return ""

        return "## Relevant Past Conversations\n\n" + "\n---\n".join(context_parts)

    def get_stats(self) -> Dict:
        """Get storage statistics"""
        return {
            "total_conversations": len(self.index.get("conversations", [])),
            "total_chars": self.index.get("total_chars", 0),
            "total_mb": round(self.index.get("total_chars", 0) / 1024 / 1024, 2),
            "sources": list(set(c["source"] for c in self.index.get("conversations", [])))
        }

# Global conversation store
conversation_store: Optional[ConversationStore] = None

# ═══════════════════════════════════════════════════════════════════
# BRAIN - The ML Engine
# ═══════════════════════════════════════════════════════════════════

class MYNDBrain:
    """
    The local ML brain for MYND.
    Handles embeddings, graph attention, voice, vision, and learning.
    """

    def __init__(self, load_multimodal: bool = True):
        self.device = config.get_device()
        self.start_time = time.time()

        print(f"🧠 Initializing MYND Brain on {self.device}...")

        # Initialize embedding engine
        self.embedder = EmbeddingEngine(
            model_name=config.EMBEDDING_MODEL,
            device=self.device
        )

        # Initialize Graph Transformer
        self.graph_transformer = MYNDGraphTransformer(
            input_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            device=self.device
        )

        # Initialize Voice Transcription (Whisper)
        self.voice = None
        if load_multimodal:
            try:
                self.voice = VoiceTranscriber(
                    model_size=config.WHISPER_MODEL,
                    device=self.device
                )
            except Exception as e:
                print(f"⚠️ Voice model not loaded: {e}")

        # Initialize Vision (CLIP)
        self.vision = None
        if load_multimodal:
            try:
                self.vision = VisionEngine(
                    model_name=config.CLIP_MODEL,
                    pretrained=config.CLIP_PRETRAINED,
                    device=self.device
                )
            except Exception as e:
                print(f"⚠️ Vision model not loaded: {e}")

        # Learning state
        self.feedback_buffer = []
        self.pending_connections = []  # Buffer for connections waiting for map sync

        # Full map state (BAPI's context window)
        self.map_state = None
        self.map_embeddings = None
        self.map_transformed = None
        self.map_adjacency = None
        self.map_node_index = {}
        self.map_last_sync = 0

        print(f"✅ MYND Brain ready on {self.device}")

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self.embedder.embed(text)

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.embedder.embed_batch(texts)

    async def predict_connections(
        self,
        node_id: str,
        map_data: MapData,
        top_k: int = 5
    ) -> Dict:
        """
        Predict potential connections for a node using Graph Transformer attention.
        This is the key capability - nodes can "see" the entire map.

        Now INTEGRATES with ASA:
        - GT provides base connection scores
        - ASA boosts scores for contextually relevant nodes (high energy atoms)
        """
        nodes = map_data.nodes

        if len(nodes) < 2:
            return {"connections": [], "attention_weights": {}}

        # Find source node
        source_node = next((n for n in nodes if n.id == node_id), None)
        if not source_node:
            raise ValueError(f"Node {node_id} not found")

        # Get or compute embeddings for all nodes
        labels = [n.label for n in nodes]
        embeddings = await self.embed_batch(labels)

        # Build node index
        node_idx = {n.id: i for i, n in enumerate(nodes)}
        source_idx = node_idx[node_id]

        # Run through Graph Transformer
        # This computes attention across ALL nodes
        attention_output, attention_weights = self.graph_transformer.forward_with_attention(
            embeddings,
            source_idx
        )

        # === ASA INTEGRATION: Get semantic context boosts ===
        asa_boosts = {}
        physics_boosts = {}
        try:
            from models.living_asa import get_asa
            asa = get_asa()
            if asa:
                # Get working memory (active atoms)
                working_memory = asa.get_working_memory(threshold=0.1)
                for item in working_memory:
                    atom_name = item.get('name', '').lower()
                    atom_energy = item.get('energy', 0)
                    # Map atom names to node labels for boosting
                    for node in nodes:
                        if atom_name in node.label.lower() or node.label.lower() in atom_name:
                            # Boost = 1.0 + (energy * 0.5), so high energy atoms get up to 1.5x boost
                            asa_boosts[node.id] = 1.0 + (atom_energy * 0.5)

                # Also check source node's related atoms
                source_atoms = asa.find_related_atoms(source_node.label, limit=5) if hasattr(asa, 'find_related_atoms') else []
                for related in source_atoms:
                    related_name = related.get('name', '').lower()
                    bond_strength = related.get('bond_strength', 0.5)
                    for node in nodes:
                        if related_name in node.label.lower() or node.label.lower() in related_name:
                            # Related atoms get boost based on bond strength
                            current_boost = asa_boosts.get(node.id, 1.0)
                            asa_boosts[node.id] = max(current_boost, 1.0 + (bond_strength * 0.3))

                # === PHYSICS-BASED BONDING AFFINITY (NEW) ===
                # Use physics engine to compute bonding affinity between atoms
                if hasattr(asa, 'compute_physics_bonding_affinity'):
                    for node in nodes:
                        if node.id == node_id:
                            continue
                        # Compute physics-based bonding affinity between source and target
                        affinity = asa.compute_physics_bonding_affinity(node_id, node.id)
                        if affinity > 0:
                            # Physics boost = 1.0 + (affinity * 0.5), so high affinity gets up to 1.5x boost
                            physics_boosts[node.id] = 1.0 + (affinity * 0.5)

        except Exception as e:
            print(f"⚠️ ASA/Physics boost lookup failed: {e}")

        # Get top-k predictions (excluding self and existing connections)
        existing_connections = set(source_node.children or [])
        if source_node.parentId:
            existing_connections.add(source_node.parentId)

        predictions = []
        for i, (node, score) in enumerate(zip(nodes, attention_weights)):
            if node.id == node_id:
                continue
            if node.id in existing_connections:
                continue

            # Apply ASA and Physics boosts to GT score
            base_score = float(score)
            asa_boost = asa_boosts.get(node.id, 1.0)
            physics_boost = physics_boosts.get(node.id, 1.0)

            # Combined score = GT * ASA * Physics
            combined_score = base_score * asa_boost * physics_boost

            predictions.append({
                "target_id": node.id,
                "target_label": node.label,
                "score": combined_score,
                "gt_score": base_score,
                "asa_boost": asa_boost,
                "physics_boost": physics_boost,
                "relationship_type": self._infer_relationship(source_node.label, node.label)
            })

        # Sort by COMBINED score and take top_k
        predictions.sort(key=lambda x: x["score"], reverse=True)
        predictions = predictions[:top_k]

        # Log if ASA or Physics influenced the predictions
        asa_boosted = sum(1 for p in predictions if p.get('asa_boost', 1.0) > 1.0)
        physics_boosted = sum(1 for p in predictions if p.get('physics_boost', 1.0) > 1.0)
        if asa_boosted > 0 or physics_boosted > 0:
            print(f"🧬 GT+ASA+Physics: {asa_boosted} ASA boosted, {physics_boosted} physics boosted for '{source_node.label}'")

        return {
            "connections": predictions,
            "attention_weights": {nodes[i].id: float(w) for i, w in enumerate(attention_weights)},
            "asa_influenced": asa_boosted > 0,
            "physics_influenced": physics_boosted > 0
        }

    def _infer_relationship(self, source: str, target: str) -> str:
        """Infer relationship type between nodes (simple heuristic for now)."""
        # TODO: Train a relationship classifier
        source_lower = source.lower()
        target_lower = target.lower()

        if any(w in source_lower for w in ['goal', 'want', 'achieve']):
            return 'supports'
        if any(w in target_lower for w in ['task', 'action', 'do']):
            return 'leads_to'
        return 'relates_to'

    async def record_feedback(self, feedback: TrainFeedbackRequest):
        """Record feedback and trigger immediate GT + ASA training."""
        feedback_data = {
            "node_id": feedback.node_id,
            "action": feedback.action,
            "context": feedback.context,
            "timestamp": time.time()
        }
        self.feedback_buffer.append(feedback_data)

        # Train immediately on this feedback if we have the context
        context = feedback.context or {}
        parent_label = context.get('parentLabel')
        child_label = context.get('acceptedLabel')
        suggestion_type = context.get('suggestionType', 'unknown')
        source = context.get('source', 'unknown')

        # Handle different feedback types - extract labels appropriately
        # For edits/updates, use node label or old/new labels
        if not parent_label:
            parent_label = context.get('nodeLabel') or context.get('oldLabel')
        if not child_label:
            child_label = context.get('newLabel') or context.get('nodeLabel')

        gt_result = None
        asa_result = None

        # Determine if this is a positive training signal
        # All user-initiated actions are positive (they made a choice)
        positive_actions = ['accepted', 'manual_create', 'edited', 'brainstorm',
                           'color_updated', 'label_updated', 'importance_updated',
                           'move_action', 'reorganization', 'chat_action']
        is_positive = any(pa in feedback.action for pa in positive_actions)

        if parent_label and child_label:
            try:
                # === TRAIN GT (Graph Transformer) ===
                parent_emb = await self.embed(parent_label)
                child_emb = await self.embed(child_label)

                # Train GT on this connection (all user actions are positive signals)
                gt_result = self.graph_transformer.train_connection_step(
                    source_embedding=parent_emb,
                    target_embedding=child_emb,
                    should_connect=is_positive,
                    adjacency=self.map_adjacency if hasattr(self, 'map_adjacency') else None
                )

                print(f"🎓 GT trained: {parent_label} → {child_label}, action={feedback.action}, loss={gt_result.get('loss', 0):.4f}")

                # === TRAIN ASA (Atomic Semantic Architecture) ===
                try:
                    from models.living_asa import get_asa
                    asa = get_asa()
                    if asa:
                        # Construct semantic text from feedback for ASA learning
                        feedback_text = f"{parent_label} relates to {child_label}"
                        if context.get('description'):
                            feedback_text += f". {child_label}: {context.get('description')}"

                        asa_result = asa.learn_from_text(
                            feedback_text,
                            source=f"feedback_{source}",
                            energy_boost=0.4 if is_positive else 0.1
                        )
                        if asa_result.get('atoms_activated', 0) > 0:
                            print(f"🧬 ASA trained: {asa_result.get('atoms_activated')} atoms, {asa_result.get('bonds_strengthened', 0)} bonds")
                except Exception as asa_e:
                    print(f"⚠️ ASA training on feedback failed: {asa_e}")

                return {
                    "status": "trained",
                    "gt": gt_result,
                    "asa": asa_result
                }
            except Exception as e:
                print(f"⚠️ GT training on feedback failed: {e}")
                import traceback
                traceback.print_exc()
                # Return error info instead of None
                return {
                    "status": "error",
                    "error": str(e),
                    "buffered": True
                }

        # If we got here, we couldn't train immediately (missing labels)
        print(f"⚠️ Feedback buffered (no labels): action={feedback.action}, parent={parent_label}, child={child_label}")
        print(f"   Full context: {context}")

        # Fallback: buffer for batch training
        if len(self.feedback_buffer) >= 10:
            await self._train_on_feedback()

        # Return acknowledgement even if no immediate training happened
        return {
            "status": "buffered",
            "buffer_size": len(self.feedback_buffer),
            "reason": "missing_labels"
        }

    async def _train_on_feedback(self):
        """Train on accumulated feedback (batch mode)."""
        if not self.feedback_buffer:
            return {'trained': 0, 'failed': 0, 'skipped': 0}

        print(f"🔄 Batch training on {len(self.feedback_buffer)} feedback samples...")

        trained = 0
        failed = 0
        skipped = 0

        for fb in self.feedback_buffer:
            context = fb.get('context', {})
            parent_label = context.get('parentLabel')
            child_label = context.get('acceptedLabel')

            if parent_label and child_label:
                try:
                    parent_emb = await self.embed(parent_label)
                    child_emb = await self.embed(child_label)
                    is_accepted = 'accepted' in fb.get('action', '')

                    self.graph_transformer.train_connection_step(
                        source_embedding=parent_emb,
                        target_embedding=child_emb,
                        should_connect=is_accepted,
                        adjacency=self.map_adjacency if hasattr(self, 'map_adjacency') else None
                    )
                    trained += 1
                except Exception as e:
                    print(f"⚠️ Batch training error: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
            else:
                skipped += 1

        self.feedback_buffer = []
        print(f"✅ Batch training complete: {trained} trained, {failed} failed, {skipped} skipped")
        return {'trained': trained, 'failed': failed, 'skipped': skipped}

    async def transcribe(self, audio_data: bytes, language: str = None, task: str = "transcribe") -> Dict:
        """Transcribe audio to text using Whisper."""
        if self.voice is None or not self.voice.initialized:
            return {"success": False, "error": "Voice model not available"}

        return self.voice.transcribe(audio_data, language=language, task=task)

    async def describe_image(self, image_data: bytes, candidate_labels: List[str] = None, top_k: int = 5) -> Dict:
        """Describe an image using CLIP."""
        if self.vision is None or not self.vision.initialized:
            return {"success": False, "error": "Vision model not available"}

        return self.vision.describe_image(image_data, candidate_labels=candidate_labels, top_k=top_k)

    async def embed_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """Generate embedding for an image."""
        if self.vision is None or not self.vision.initialized:
            return None

        return self.vision.embed_image(image_data)

    async def sync_map(self, map_data: MapData) -> Dict:
        """
        Sync the full map to BAPI's context window.
        This gives BAPI awareness of the entire map at all times.
        """
        start_time = time.time()
        nodes = map_data.nodes

        if len(nodes) == 0:
            return {"synced": 0, "time_ms": 0}

        # Store map state
        self.map_state = map_data
        self.map_node_index = {n.id: i for i, n in enumerate(nodes)}

        # Compute embeddings for all nodes
        labels = [n.label for n in nodes]
        self.map_embeddings = await self.embed_batch(labels)

        # Build adjacency matrix
        num_nodes = len(nodes)
        self.map_adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i, node in enumerate(nodes):
            # Parent-child connections
            if node.parentId and node.parentId in self.map_node_index:
                parent_idx = self.map_node_index[node.parentId]
                self.map_adjacency[i, parent_idx] = 1.0
                self.map_adjacency[parent_idx, i] = 1.0
            # Children connections
            for child_id in (node.children or []):
                if child_id in self.map_node_index:
                    child_idx = self.map_node_index[child_id]
                    self.map_adjacency[i, child_idx] = 1.0
                    self.map_adjacency[child_idx, i] = 1.0

        # Compute node depths
        depths = np.zeros(num_nodes, dtype=np.int32)
        for i, node in enumerate(nodes):
            depth = 0
            current = node
            while current.parentId and current.parentId in self.map_node_index:
                depth += 1
                parent_idx = self.map_node_index[current.parentId]
                current = nodes[parent_idx]
                if depth > 20:  # Safety limit
                    break
            depths[i] = depth

        # Compute node degrees
        degrees = np.sum(self.map_adjacency, axis=1).astype(np.int32)

        # Run through Graph Transformer - this is the key step
        # Now BAPI has transformed representations of ALL nodes
        self.map_transformed, _ = self.graph_transformer.forward(
            self.map_embeddings,
            adjacency=self.map_adjacency,
            depths=depths,
            degrees=degrees
        )

        self.map_last_sync = time.time()
        elapsed = (time.time() - start_time) * 1000

        print(f"🧠 Map synced: {num_nodes} nodes in {elapsed:.0f}ms")

        return {
            "synced": num_nodes,
            "time_ms": elapsed
        }

    async def analyze_map(self) -> Dict:
        """
        BAPI analyzes the full map and returns observations.
        This is BAPI's voice - what it notices about your thinking.
        """
        if self.map_state is None or self.map_transformed is None:
            return {"error": "No map synced. Call /map/sync first."}

        start_time = time.time()
        nodes = self.map_state.nodes
        num_nodes = len(nodes)

        observations = []

        # 1. Find missing connections
        missing = self.graph_transformer.find_missing_connections(
            self.map_embeddings,
            self.map_adjacency,
            threshold=0.65,
            top_k=5
        )

        for src_idx, tgt_idx, score in missing:
            observations.append({
                "type": "missing_connection",
                "message": f"'{nodes[src_idx].label}' and '{nodes[tgt_idx].label}' seem related but aren't connected",
                "source_id": nodes[src_idx].id,
                "target_id": nodes[tgt_idx].id,
                "confidence": score
            })

        # 2. Find important nodes
        importance = self.graph_transformer.get_node_importance(
            self.map_embeddings,
            adjacency=self.map_adjacency
        )

        # Top important nodes
        top_important = np.argsort(importance)[-5:][::-1]
        important_nodes = []
        for idx in top_important:
            important_nodes.append({
                "id": nodes[idx].id,
                "label": nodes[idx].label,
                "importance": float(importance[idx])
            })

        # 3. Find isolated nodes (potentially orphaned ideas)
        isolated = []
        for i, node in enumerate(nodes):
            connection_count = np.sum(self.map_adjacency[i])
            if connection_count <= 1 and importance[i] > 0.5:  # Important but isolated
                isolated.append({
                    "id": node.id,
                    "label": node.label,
                    "message": f"'{node.label}' seems important but has few connections"
                })

        # 4. Get attention patterns from each head
        head_attention_data = None
        if num_nodes > 0:
            # Sample a central node to see head specialization
            central_idx = int(np.argmax(importance))
            head_attention = self.graph_transformer.get_head_attention(
                self.map_embeddings,
                central_idx,
                adjacency=self.map_adjacency
            )
            # Convert to serializable format for API response
            head_attention_data = {
                k: v.tolist() if hasattr(v, 'tolist') else v
                for k, v in head_attention.items()
            }

        elapsed = (time.time() - start_time) * 1000

        return {
            "node_count": num_nodes,
            "observations": observations,
            "important_nodes": important_nodes,
            "isolated_nodes": isolated[:5],  # Limit to 5
            "head_attention": head_attention_data,
            "time_ms": elapsed,
            "last_sync": self.map_last_sync
        }

    # ═══════════════════════════════════════════════════════════════════
    # TRAINING METHODS - Close the learning loop
    # ═══════════════════════════════════════════════════════════════════

    def train_connection(
        self,
        source_id: str,
        target_id: str,
        should_connect: bool
    ) -> Dict:
        """
        Train the Graph Transformer on a connection decision.

        This is called when user accepts/rejects a connection suggestion,
        or when they manually create a connection (positive example).

        Args:
            source_id: ID of source node
            target_id: ID of target node
            should_connect: True if connection should exist, False otherwise

        Returns:
            Training result dict with loss, prediction, etc.
        """
        # Get embeddings for the nodes
        if not self.map_state or not self.map_node_index:
            # Buffer for later - map not synced yet
            self.pending_connections.append({
                'source_id': source_id,
                'target_id': target_id,
                'should_connect': should_connect,
                'timestamp': time.time()
            })
            return {'status': 'buffered', 'reason': 'Map not synced yet'}

        # Check if nodes exist, if not buffer for later
        if source_id not in self.map_node_index or target_id not in self.map_node_index:
            self.pending_connections.append({
                'source_id': source_id,
                'target_id': target_id,
                'should_connect': should_connect,
                'timestamp': time.time()
            })
            missing = []
            if source_id not in self.map_node_index:
                missing.append(f'source {source_id}')
            if target_id not in self.map_node_index:
                missing.append(f'target {target_id}')
            return {'status': 'buffered', 'reason': f'Waiting for {", ".join(missing)} to sync'}

        source_idx = self.map_node_index[source_id]
        target_idx = self.map_node_index[target_id]

        source_embedding = self.map_embeddings[source_idx]
        target_embedding = self.map_embeddings[target_idx]

        # Train the GT
        result = self.graph_transformer.train_connection_step(
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            should_connect=should_connect,
            adjacency=self.map_adjacency
        )

        return result

    def process_pending_connections(self) -> Dict:
        """
        Process any buffered connections now that map may be synced.
        Called after map sync completes.
        """
        if not self.pending_connections:
            return {'processed': 0, 'remaining': 0}

        processed = 0
        still_pending = []

        for conn in self.pending_connections:
            source_id = conn['source_id']
            target_id = conn['target_id']

            # Check if both nodes are now available
            if source_id in self.map_node_index and target_id in self.map_node_index:
                source_idx = self.map_node_index[source_id]
                target_idx = self.map_node_index[target_id]

                source_embedding = self.map_embeddings[source_idx]
                target_embedding = self.map_embeddings[target_idx]

                # Train on this buffered connection
                try:
                    result = self.graph_transformer.train_connection_step(
                        source_embedding=source_embedding,
                        target_embedding=target_embedding,
                        should_connect=conn['should_connect'],
                        adjacency=self.map_adjacency
                    )
                    processed += 1
                    print(f"🎓 Processed buffered connection: {source_id} → {target_id}, loss={result.get('loss', 0):.4f}")
                except Exception as e:
                    print(f"⚠️ GT training error for {source_id} → {target_id}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Still waiting for nodes - keep in buffer (but expire after 5 min)
                if time.time() - conn['timestamp'] < 300:
                    still_pending.append(conn)

        self.pending_connections = still_pending

        return {'processed': processed, 'remaining': len(still_pending)}

    def get_training_stats(self) -> Dict:
        """Get Graph Transformer training statistics."""
        return self.graph_transformer.get_training_stats()

    def get_health(self) -> Dict:
        """Get health status."""
        return {
            "status": "healthy",
            "device": str(self.device),
            "embedding_model": config.EMBEDDING_MODEL,
            "graph_transformer": self.graph_transformer is not None,
            "voice_model": config.WHISPER_MODEL if self.voice and self.voice.initialized else None,
            "vision_model": config.CLIP_MODEL if self.vision and self.vision.initialized else None,
            "uptime_seconds": time.time() - self.start_time
        }

# ═══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════

# Global brain instances
brain: Optional[MYNDBrain] = None
unified_brain: Optional[UnifiedBrain] = None

# Unified system instances
map_vector_db: Optional[MapVectorDB] = None
conversation_archive: Optional[ConversationArchive] = None
knowledge_extractor: Optional[KnowledgeExtractor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize brain on startup, cleanup on shutdown."""
    global brain, unified_brain, conversation_store
    global map_vector_db, conversation_archive, knowledge_extractor

    # Initialize ML brain
    brain = MYNDBrain()

    # Load trained GT weights if they exist
    gt_weights_path = pathlib.Path(__file__).parent / "data" / "gt_weights.pt"
    brain.graph_transformer.load_weights(str(gt_weights_path))

    # Initialize unified brain with reference to ML brain
    base_dir = pathlib.Path(__file__).parent.parent
    unified_brain = UnifiedBrain(base_dir, device=str(config.get_device()))
    unified_brain.set_ml_brain(brain)

    print("🧠 Unified Brain connected to ML Brain")

    # Initialize conversation store (legacy)
    brain_dir = pathlib.Path(__file__).parent
    conversation_store = ConversationStore(brain_dir, embedder=brain.embedder)
    print(f"💬 Conversation Store initialized ({conversation_store.get_stats()['total_conversations']} conversations)")

    # ═══════════════════════════════════════════════════════════════════
    # UNIFIED SYSTEM - Map is the vector database
    # ═══════════════════════════════════════════════════════════════════
    data_dir = pathlib.Path(__file__).parent / "data"

    # Initialize unified map vector database
    map_vector_db = MapVectorDB(data_dir, embedder=brain.embedder)
    print(f"🗺️ MapVectorDB initialized ({map_vector_db.get_stats()['total_nodes']} nodes)")

    # Initialize conversation archive
    conversation_archive = ConversationArchive(data_dir, embedder=brain.embedder)
    print(f"📚 Conversation Archive initialized ({conversation_archive.get_stats()['total_conversations']} conversations)")

    # Connect external components to UnifiedBrain's ContextSynthesizer
    unified_brain.set_conversation_archive(conversation_archive)
    unified_brain.set_map_vector_db(map_vector_db)
    # Embedding engine is connected via set_ml_brain above
    print("🔀 ContextSynthesizer connected to: MapVectorDB, ConversationArchive, EmbeddingEngine")

    # Connect Supabase for persistent AI memories
    try:
        from supabase import create_client
        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_SERVICE_KEY') or os.environ.get('SUPABASE_ANON_KEY')
        if supabase_url and supabase_key:
            supabase_client = create_client(supabase_url, supabase_key)
            unified_brain.set_supabase(supabase_client)
            print("🔗 Supabase connected to UnifiedBrain for persistent AI memories")
        else:
            print("⚠️ Supabase not configured (missing SUPABASE_URL or key)")
    except ImportError:
        print("⚠️ Supabase library not installed (pip install supabase)")
    except Exception as e:
        print(f"⚠️ Supabase connection failed: {e}")

    # Load persisted learning state (meta-learner, knowledge distiller, predictions)
    unified_brain.load_learning_state()

    # ═══════════════════════════════════════════════════════════════════
    # AUTO-SYNC LIVING ASA FROM MAP VECTOR DB
    # ═══════════════════════════════════════════════════════════════════
    if _asa_available and map_vector_db:
        try:
            map_data = map_vector_db.export_to_browser_map()
            if map_data:
                asa = get_asa()
                asa.convert_map_to_asa(map_data)
                asa.start_metabolism(tick_interval=5.0)
                print(f"🧬 Living ASA auto-synced: {len(asa.atoms)} atoms, metabolism running")
            else:
                print("⚠️ ASA: No map data to sync (MapVectorDB empty)")
        except Exception as e:
            print(f"⚠️ ASA auto-sync failed: {e}")

    # Initialize knowledge extractor (uses Claude CLI - Max subscription)
    knowledge_extractor = KnowledgeExtractor(map_db=map_vector_db)
    print("🧠 Knowledge Extractor initialized (using Claude CLI)")

    # ═══════════════════════════════════════════════════════════════════
    # BACKGROUND TRAINING TASK
    # Runs every 5 minutes to train on buffered examples
    # ═══════════════════════════════════════════════════════════════════
    background_training_running = True

    async def background_training_loop():
        """Periodically run background training on buffered examples."""
        while background_training_running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                if unified_brain is not None:
                    result = unified_brain.run_background_training()
                    if result.get('status') == 'trained':
                        # Save weights after successful training
                        # FIX: Use unified_brain.ml_brain instead of undefined 'brain'
                        if unified_brain.ml_brain and unified_brain.ml_brain.graph_transformer:
                            gt_weights_path = pathlib.Path(__file__).parent / "data" / "gt_weights.pt"
                            unified_brain.ml_brain.graph_transformer.save_weights(str(gt_weights_path))
                            print(f"💾 GT weights saved after background training")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ Background training loop error: {e}")
                import traceback
                traceback.print_exc()

    # Start background training task
    training_task = asyncio.create_task(background_training_loop())
    print("🎓 Background training loop started (runs every 5 minutes)")

    yield
    # Stop background training
    background_training_running = False
    training_task.cancel()
    try:
        await training_task
    except asyncio.CancelledError:
        pass
    # Cleanup
    if map_vector_db:
        map_vector_db.save()
        print("💾 MapVectorDB saved")
    if unified_brain:
        unified_brain.save_learning_state()
    # Save GT trained weights
    if brain and brain.graph_transformer:
        gt_weights_path = pathlib.Path(__file__).parent / "data" / "gt_weights.pt"
        brain.graph_transformer.save_weights(str(gt_weights_path))
    print("🧠 MYND Brain shutting down...")

app = FastAPI(
    title="MYND Brain",
    description="Local ML server for MYND - Graph Transformers on Apple Silicon",
    version="0.1.0",
    lifespan=lifespan
)

# CORS - allow browser to connect from any origin
# Explicitly configure for cross-origin requests through tunnels/proxies
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Static file serving for frontend (parent directory)
import pathlib
FRONTEND_DIR = pathlib.Path(__file__).parent.parent

# Serve JS files
app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")

# Serve the main app
@app.get("/app")
async def serve_app():
    """Serve the MYND app."""
    return FileResponse(FRONTEND_DIR / "self-dev.html")

# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the brain is running and healthy."""
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return brain.get_health()

@app.post("/health", response_model=HealthResponse)
async def health_check_post():
    """Check if the brain is running and healthy (POST version for browsers with GET issues)."""
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return brain.get_health()

@app.get("/")
async def root():
    """Root endpoint with info."""
    return {
        "name": "MYND Brain",
        "version": "0.6.0",  # Self-improvement + Vision update
        "status": "running",
        "endpoints": {
            "unified_brain": [
                "/brain/context",           # THE unified context endpoint
                "/brain/state",             # Brain introspection
                "/brain/feedback",          # Learning from user
                "/brain/predictions",       # Record predictions for self-learning
                "/brain/learn-connection",  # Learn from connections
                "/brain/learning",          # View learning stats
                "/brain/receive-from-claude", # Claude teaches brain
                "/brain/ask-to-teach",      # Request teaching
                "/brain/knowledge",         # View distilled knowledge
                "/brain/teaching-prompt",   # Get teaching instructions
                "/brain/full-stats"         # Comprehensive stats
            ],
            "meta_learning": [
                "/brain/meta",              # Detailed meta-learning stats
                "/brain/meta/summary",      # Human-readable summary
                "/brain/meta/calibration",  # Confidence calibration report
                "/brain/meta/improvement",  # Improvement trend over time
                "/brain/meta/recommendations", # Source priority recommendations
                "/brain/meta/feedback",     # Record source effectiveness
                "/brain/meta/learning-rate", # Adjust learning rates
                "/brain/meta/save-epoch"    # Save learning checkpoint
            ],
            "self_improvement": [
                "/brain/analyze",           # Run self-analysis
                "/brain/suggestions",       # Get improvement suggestions
                "/brain/suggestions/top",   # Get top suggestions
                "/brain/suggestions/summary", # Human-readable summary
                "/brain/suggestions/status", # Mark suggestion status
                "/brain/improvement-stats"  # Self-improvement statistics
            ],
            "vision": [
                "/brain/vision",            # Get/set vision statement
                "/brain/vision/goals"       # Add/remove goals
            ],
            "ml_processing": [
                "/embed", "/embed/batch",
                "/predict/connections",
                "/map/sync", "/map/analyze",
            ],
            "multimodal": [
                "/voice/transcribe",
                "/image/describe", "/image/embed"
            ],
            "code_analysis": [
                "/code/parse",
                "/code/self-awareness"
            ],
            "conversations": [
                "/conversations/import",    # Import AI chat conversations
                "/conversations",           # List all conversations
                "/conversations/{id}",      # Get specific conversation
                "/conversations/search",    # Semantic search
                "/conversations/context",   # Get relevant context for prompts
                "/conversations/stats"      # Storage statistics
            ],
            "system": [
                "/health"
            ]
        }
    }

# ═══════════════════════════════════════════════════════════════════
# BAPI - Full Map Awareness
# ═══════════════════════════════════════════════════════════════════

# Import Living ASA here so it's available for map sync
try:
    from models.living_asa import get_asa, MYNDLivingASA
    _asa_available = True
    print("🧬 Living ASA module loaded")
except Exception as e:
    _asa_available = False
    print(f"⚠️ Living ASA not available: {e}")

@app.post("/map/sync")
async def sync_map(map_data: MapData):
    """
    Sync the full map to BAPI's context window.
    Call this on map load and after significant changes.
    BAPI will then have awareness of ALL nodes.
    """
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    result = await brain.sync_map(map_data)

    # Process any buffered connection training now that map is synced
    pending_result = brain.process_pending_connections()
    if pending_result['processed'] > 0:
        print(f"🎓 GT Training: Processed {pending_result['processed']} buffered connections after map sync")
        result['gt_training_processed'] = pending_result['processed']

    # === SYNC TO LIVING ASA ===
    if _asa_available:
        try:
            asa = get_asa()

            # Convert flat nodes list to tree structure for ASA
            # Build a dict of parent_id -> children
            nodes_by_id = {n.id: n for n in map_data.nodes}
            children_map = {}
            root_nodes = []

            for node in map_data.nodes:
                if node.parentId and node.parentId in nodes_by_id:
                    if node.parentId not in children_map:
                        children_map[node.parentId] = []
                    children_map[node.parentId].append(node)
                else:
                    root_nodes.append(node)

            def build_tree(node):
                node_dict = {
                    'id': node.id,
                    'label': node.label,
                    'description': getattr(node, 'description', ''),
                    'children': []
                }
                if node.id in children_map:
                    node_dict['children'] = [build_tree(child) for child in children_map[node.id]]
                return node_dict

            # Use first root node as the tree root (or create synthetic root)
            if root_nodes:
                tree_data = build_tree(root_nodes[0])
                # Add other root nodes as children if multiple roots
                if len(root_nodes) > 1:
                    for root in root_nodes[1:]:
                        tree_data['children'].append(build_tree(root))
            else:
                tree_data = {'id': 'root', 'label': 'Root', 'children': []}

            asa.convert_map_to_asa(tree_data)

            # Start metabolism if not running
            if not asa._running:
                asa.start_metabolism(tick_interval=5.0)
                print("🧬 ASA metabolism started")

            result['asa'] = {
                'atoms': len(asa.atoms),
                'metabolism': 'running' if asa._running else 'stopped'
            }
            print(f"🧬 ASA synced: {len(asa.atoms)} atoms")
        except Exception as e:
            print(f"⚠️ ASA sync error: {e}")
            import traceback
            traceback.print_exc()
            result['asa_error'] = str(e)

    return result

@app.get("/map/analyze")
async def analyze_map():
    """
    BAPI analyzes the synced map and returns observations:
    - Missing connections (nodes that should be linked)
    - Important nodes (central to your thinking)
    - Isolated nodes (important but disconnected)

    Call /map/sync first to load the map.
    """
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    result = await brain.analyze_map()
    return result

@app.get("/map/status")
async def map_status():
    """Check current map sync status."""
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    return {
        "synced": brain.map_state is not None,
        "node_count": len(brain.map_state.nodes) if brain.map_state else 0,
        "last_sync": brain.map_last_sync
    }

# ═══════════════════════════════════════════════════════════════════
# UNIFIED BRAIN - Complete Self-Aware Context
# ═══════════════════════════════════════════════════════════════════

@app.post("/brain/context", response_model=BrainContextResponse)
async def get_brain_context(request: BrainContextRequest):
    """
    THE unified context endpoint.
    One call = complete context for Claude.

    This replaces 19+ fragmented context providers with ONE call that includes:
    - Self-awareness (who am I?)
    - Code understanding (how do I work?)
    - Map context (what is the user looking at?)
    - Memories (what do I remember?)
    - Neural insights (what do my models see?)

    Use this for ALL Claude API calls to give Claude complete self-awareness.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    start = time.time()

    # Convert Pydantic request to dataclass
    include_dict = {}
    if request.include:
        include_dict = {
            'self_awareness': request.include.self_awareness,
            'map_context': request.include.map_context,
            'memories': request.include.memories,
            'user_profile': request.include.user_profile,
            'neural_insights': request.include.neural_insights,
            'synthesized_context': request.include.synthesized_context
        }

    map_dict = None
    if request.map_data:
        map_dict = {
            'nodes': [n.model_dump() for n in request.map_data.nodes]
        }

    # Convert goals to list of dicts
    goals_list = None
    if request.goals:
        goals_list = [g.model_dump() for g in request.goals]

    ctx_request = ContextRequest(
        request_type=request.request_type,
        user_message=request.user_message,
        selected_node_id=request.selected_node_id,
        map_data=map_dict,
        user_id=request.user_id,
        goals=goals_list,
        include=include_dict
    )

    # Get unified context
    response = unified_brain.get_context(ctx_request)

    # === ASA LEARNS FROM USER MESSAGE ===
    asa_context = ""
    if _asa_available and request.user_message:
        try:
            asa = get_asa()

            # 1. Original learning: match existing nodes, boost energy
            learn_result = asa.learn_from_text(request.user_message, source="user")
            if learn_result['atoms_activated'] > 0:
                encoder_info = ""
                if learn_result.get('encoder_trained'):
                    encoder_info = f", encoder_loss={learn_result.get('encoder_loss', 0):.4f}"
                print(f"🧬 ASA learned from user: {learn_result['atoms_activated']} atoms, {learn_result['bonds_strengthened']} bonds{encoder_info}")

            # 2. NEW: Content learning - learn actual concepts from text
            content_result = asa.learn_content(request.user_message, source="user")
            if content_result.get('concepts_learned', 0) > 0:
                print(f"📚 Content learned: {content_result['concepts_learned']} new concepts, {content_result.get('relationships_learned', 0)} relationships")

            # === ASA CONTRIBUTES WORKING MEMORY TO CONTEXT ===
            # This is crucial - tell Axel what topics are "hot" (recently discussed)
            working_memory = asa.get_working_memory(threshold=0.15)
            if working_memory:
                asa_context = "\n\n## Recent Discussion Context (ASA Working Memory)\n"
                asa_context += "These topics have been active in recent conversation:\n"
                for item in working_memory[:10]:
                    asa_context += f"- **{item['name']}** (energy: {item['energy']:.0%})\n"

                # Also show what's activated together
                if learn_result.get('activated_names'):
                    asa_context += f"\nJust mentioned together: {', '.join(learn_result['activated_names'][:5])}\n"

            # Add learned concepts to context (if any recent ones)
            if content_result.get('phrases'):
                asa_context += f"\nRecently discussed concepts: {', '.join(content_result['phrases'][:5])}\n"

            # AUTO-SAVE: Persist state after learning (every 5 messages)
            total_processed = asa._learning_stats.get('texts_processed', 0)
            if total_processed > 0 and total_processed % 5 == 0:
                asa.save_state()

        except Exception as e:
            print(f"⚠️ ASA learn error: {e}")

    # === GT CONTEXT - Graph Transformer insights ===
    gt_context = ""
    if unified_brain and unified_brain.ml_brain:
        try:
            brain = unified_brain.ml_brain
            gt_stats = brain.graph_transformer.get_training_stats()

            # Only add GT context if it has learned something
            if gt_stats.get('total_connections_learned', 0) > 0:
                gt_context = "\n\n## Graph Transformer Insights\n"
                gt_context += f"GT has learned from {gt_stats['total_connections_learned']} connections over {gt_stats.get('epochs_completed', 0)} training cycles.\n"

                # Skip expensive operations for large maps (>500 nodes)
                # These operations are O(n^2) and can timeout
                node_count = len(request.map_data.nodes) if request.map_data else 0
                if request.map_data and brain.map_embeddings is not None and node_count <= 500:
                    # Get node importance from GT
                    importance = brain.graph_transformer.get_node_importance(
                        brain.map_embeddings,
                        adjacency=brain.map_adjacency
                    )

                    # Top 5 most important nodes according to GT
                    import numpy as np
                    top_indices = np.argsort(importance)[-5:][::-1]
                    top_nodes = []
                    for idx in top_indices:
                        if idx < len(request.map_data.nodes):
                            node = request.map_data.nodes[idx]
                            top_nodes.append(f"{node.label} ({importance[idx]:.0%})")

                    if top_nodes:
                        gt_context += f"\nCentral themes in this map: {', '.join(top_nodes)}\n"

                    # Get predicted missing connections
                    missing = brain.graph_transformer.find_missing_connections(
                        brain.map_embeddings,
                        brain.map_adjacency,
                        threshold=0.7,
                        top_k=3
                    )

                    if missing:
                        gt_context += "\nPotentially related concepts:\n"
                        for src_idx, tgt_idx, score in missing[:3]:
                            if src_idx < len(request.map_data.nodes) and tgt_idx < len(request.map_data.nodes):
                                src_label = request.map_data.nodes[src_idx].label
                                tgt_label = request.map_data.nodes[tgt_idx].label
                                gt_context += f"- {src_label} ↔ {tgt_label} (confidence: {score:.0%})\n"
                elif node_count > 500:
                    gt_context += f"\n(Detailed analysis skipped - map has {node_count} nodes)\n"

                if gt_context:
                    print(f"🔮 GT context: {len(gt_context)} chars")

        except Exception as e:
            print(f"⚠️ GT context error: {e}")

    # Append ASA + GT context to the response
    enhanced_context = response.context_document
    if asa_context:
        enhanced_context += asa_context
    if gt_context:
        enhanced_context += gt_context

    elapsed = (time.time() - start) * 1000
    print(f"🧠 Brain context: {response.token_count} tokens in {elapsed:.0f}ms")

    extra_tokens = len(asa_context.split()) + len(gt_context.split())
    return BrainContextResponse(
        context_document=enhanced_context,
        token_count=response.token_count + extra_tokens,
        breakdown=response.breakdown,
        brain_state=response.brain_state,
        time_ms=elapsed
    )

@app.get("/brain/state")
async def get_brain_state():
    """
    Get current brain state for debugging/display.
    Shows what the brain knows about itself.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return {
        "state": unified_brain._get_brain_state(),
        "capabilities": unified_brain.self_awareness.capabilities,
        "limitations": unified_brain.self_awareness.limitations,
        "recent_memories": [
            {k: v for k, v in m.items() if k != 'embedding'}
            for m in unified_brain.memory.get_recent(5)
        ],
        "growth_events": len(unified_brain.self_awareness.growth_events)
    }

@app.post("/brain/feedback")
async def record_brain_feedback(
    node_id: str,
    action: str,  # 'accepted', 'rejected', 'corrected'
    context: Optional[Dict[str, Any]] = None
):
    """
    Record feedback for the brain's learning.
    Call this when the user accepts, rejects, or corrects something.

    NOW WITH INTERCONNECTIONS:
    - GT trains with reasoning context from Context Lens
    - Reflection outcomes feed into MetaLearner
    - ASA learns from context themes
    """
    ctx = context or {}
    print(f"📥 /brain/feedback: {action} on {node_id} - {ctx.get('suggestionType', 'unknown')} ({ctx.get('parentLabel', '?')} → {ctx.get('acceptedLabel', '?')})")

    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    unified_brain.record_feedback(node_id, action, ctx)

    # === INTERCONNECTION: Get cached reasoning context ===
    reasoning_context = unified_brain.get_suggestion_context(node_id)
    if reasoning_context:
        print(f"🔗 Found cached context: focus='{reasoning_context.get('focus', '')[:30]}...', themes={reasoning_context.get('themes', [])[:3]}")

    # === GT TRAINING WITH REASONING CONTEXT ===
    gt_result = None
    asa_result = None

    parent_label = ctx.get('parentLabel', '')
    accepted_label = ctx.get('acceptedLabel', '')
    should_connect = action in ['accepted', 'accepted_batch', 'brainstorm_accepted']

    if parent_label and accepted_label and unified_brain.ml_brain:
        try:
            brain = unified_brain.ml_brain
            gt = brain.graph_transformer

            # Get embeddings
            parent_emb = brain.model.encode(parent_label, convert_to_tensor=True)
            child_emb = brain.model.encode(accepted_label, convert_to_tensor=True)

            # INTERCONNECTION: Use context-aware training if we have reasoning context
            if reasoning_context and hasattr(gt, 'train_with_reasoning_context'):
                result = gt.train_with_reasoning_context(
                    source_embedding=parent_emb,
                    target_embedding=child_emb,
                    should_connect=should_connect,
                    reasoning_context=reasoning_context,
                    weight=1.0
                )
                gt_result = {
                    "trained": True,
                    "should_connect": should_connect,
                    "loss": float(result.get('loss', 0)),
                    "context_informed": True,
                    "context_boost": result.get('context_boost', 0)
                }
                print(f"🧠 GT trained with context: {parent_label} → {accepted_label} (boost={result.get('context_boost', 0):.2f})")
            else:
                # Fallback to standard training
                loss = gt.train_connection_step(
                    source_embedding=parent_emb,
                    target_embedding=child_emb,
                    should_connect=should_connect,
                    weight=1.0
                )
                gt_result = {
                    "trained": True,
                    "should_connect": should_connect,
                    "loss": float(loss.get('loss', 0)) if isinstance(loss, dict) else float(loss) if loss else None,
                    "context_informed": False
                }
                print(f"🧠 GT trained from feedback: {parent_label} → {accepted_label}")

        except Exception as e:
            print(f"⚠️ GT training failed: {e}")
            gt_result = {"trained": False, "error": str(e)}

    # === ASA TRAINING ON FEEDBACK ===
    if parent_label and accepted_label and _asa_available:
        try:
            asa = get_asa()
            combined = f"{parent_label} connects to {accepted_label}"
            asa.learn_from_text(combined, importance=0.8, category="feedback")
            asa_result = {"trained": True}
            print(f"🧬 ASA learned from feedback: {combined}")
        except Exception as e:
            asa_result = {"trained": False, "error": str(e)}

    # === INTERCONNECTION: Record reflection outcome ===
    suggestion_type = ctx.get('suggestionType', 'unknown')
    if suggestion_type != 'unknown' and unified_brain:
        unified_brain.record_reflection_outcome(suggestion_type, should_connect)
        print(f"🔗 Reflection outcome recorded: {suggestion_type} → {'accepted' if should_connect else 'rejected'}")

    # === INTERCONNECTION: Train ASA from Context Lens on positive feedback ===
    lens_trained = False
    if should_connect and unified_brain and hasattr(unified_brain, '_last_context_lens') and unified_brain._last_context_lens:
        # Only train from lens if it's fresh
        if hasattr(unified_brain, '_last_lens_timestamp') and time.time() - unified_brain._last_lens_timestamp < Config.CONTEXT_LENS_FRESHNESS_SECONDS:
            unified_brain.train_asa_from_context_lens(
                unified_brain._last_context_lens,
                was_helpful=True
            )
            lens_trained = True

    return {
        "status": "recorded",
        "growth_events_today": unified_brain.growth_events_today,
        "gt_training": gt_result,
        "asa_training": asa_result,
        "context_informed": reasoning_context is not None,
        "lens_trained": lens_trained
    }

# ═══════════════════════════════════════════════════════════════════
# BRAIN CHAT - V1 Simple Passthrough to Claude
# Routes chat through local brain instead of Supabase edge function
# ═══════════════════════════════════════════════════════════════════

AXEL_SYSTEM_PROMPT_FALLBACK = """You are Axel, the AI companion for MYND — a 3D mind mapping app that serves as a cognitive operating system.

You are not just an assistant — you are a cognitive partner. You help users:
- Capture thoughts before they're lost
- See connections across their entire knowledge map
- Advance toward their goals with every interaction

Keep responses concise but insightful. Every response should feel like it comes from someone who knows them and their goals deeply.

RESPONSE FORMAT (always JSON):
{
  "message": "Your conversational response",
  "actions": [],
  "suggestions": []
}

Note: Running in fallback mode without full context. For full Axel capabilities, the frontend should provide the system_prompt.
"""

@app.post("/brain/chat", response_model=BrainChatResponse)
async def brain_chat(request: BrainChatRequest):
    """
    Chat with Axel via Claude CLI.
    Uses Max subscription instead of per-token API billing.

    V2: Accepts full system_prompt from frontend for complete Axel context.
    Falls back to minimal prompt if not provided.
    """
    start = time.time()

    # Note: CLI uses Max subscription, no API key needed
    model = "claude-cli"  # CLI handles model selection

    # Use frontend-provided system prompt if available (full Axel context)
    # Otherwise fall back to minimal prompt
    system_prompt = request.system_prompt or AXEL_SYSTEM_PROMPT_FALLBACK

    has_full_context = request.system_prompt is not None
    context_mode = "full" if has_full_context else "fallback"

    # Build messages
    messages = [{"role": m.role, "content": m.content} for m in request.conversation_history]
    messages.append({"role": "user", "content": request.user_message})

    print(f"🧠 /brain/chat: {len(messages)} messages, context={context_mode}, using Claude CLI")

    # Call Claude via CLI
    try:
        axel_response = await call_claude_cli_with_conversation(
            messages=messages,
            system_prompt=system_prompt,
            timeout=120.0
        )

        elapsed = (time.time() - start) * 1000
        print(f"✅ /brain/chat: {elapsed:.0f}ms, {len(axel_response)} chars")

        return BrainChatResponse(
            message=axel_response,
            time_ms=elapsed,
            model=model
        )

    except asyncio.TimeoutError:
        print("❌ /brain/chat: Claude CLI timeout")
        raise HTTPException(status_code=504, detail="Claude CLI timeout")
    except ValueError as e:
        # Client error - e.g., last message not from user, empty messages
        print(f"❌ /brain/chat invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        print(f"❌ /brain/chat CLI error: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        print(f"❌ /brain/chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════
# SELF-LEARNING - Brain learns from its own predictions
# ═══════════════════════════════════════════════════════════════════

class PredictionRecord(BaseModel):
    source_id: str
    predictions: List[Dict[str, Any]]

class ConnectionLearning(BaseModel):
    source_id: str
    target_id: str
    connection_type: str = "manual"
    # Optional labels for on-the-fly embedding (when nodes don't exist yet)
    source_label: Optional[str] = None
    target_label: Optional[str] = None
    should_connect: bool = True  # For feedback learning
    feedback_type: Optional[str] = None  # 'node', 'category', 'connection', etc.

@app.post("/brain/predictions")
async def record_predictions(record: PredictionRecord):
    """
    Record predictions made by the Graph Transformer.
    Call this whenever predictions are shown to the user.
    This enables the brain to learn from whether predictions were correct.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    unified_brain.record_predictions(record.source_id, record.predictions)

    return {
        "status": "recorded",
        "predictions_tracked": len(record.predictions),
        "total_predictions": unified_brain.predictions.total_predictions
    }

@app.post("/brain/learn-connection")
async def learn_from_connection(learning: ConnectionLearning):
    """
    Tell the brain about a new connection.
    The brain checks if it predicted this and learns accordingly.

    This is the KEY self-learning endpoint:
    - If predicted: Reinforces the pattern (brain was right!)
    - If not predicted: Learns new pattern (brain missed this)

    Also supports direct GT training with labels for feedback learning.
    """
    feedback_info = f" [{learning.feedback_type}]" if learning.feedback_type else ""
    print(f"📥 /brain/learn-connection called: {learning.source_id} → {learning.target_id}{feedback_info}")

    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    response = {
        "status": "learned",
        "was_predicted": False,
        "prediction_score": 0,
        "learning_signal": "feedback"
    }

    # === DIRECT GT TRAINING WITH LABELS ===
    # When labels are provided, train GT directly (for feedback learning)
    if learning.source_label and learning.target_label and unified_brain.ml_brain:
        try:
            brain = unified_brain.ml_brain

            # Create embeddings from labels
            source_emb = brain.model.encode(learning.source_label, convert_to_tensor=False)
            target_emb = brain.model.encode(learning.target_label, convert_to_tensor=False)

            # Train GT on this connection
            gt_result = brain.graph_transformer.train_connection_step(
                source_embedding=source_emb,
                target_embedding=target_emb,
                should_connect=learning.should_connect
            )

            if gt_result:
                response['loss'] = gt_result.get('loss')
                response['prediction'] = gt_result.get('prediction')
                response['gt_trained'] = True
                print(f"🧠 GT trained from feedback: {learning.source_label} → {learning.target_label}, loss={gt_result.get('loss', 0):.4f}")

        except Exception as e:
            print(f"⚠️ GT feedback training error: {e}")

    # === EXISTING LOGIC: Learn from connection IDs ===
    # This tracks prediction accuracy
    result = unified_brain.learn_from_connection(
        learning.source_id,
        learning.target_id,
        learning.connection_type
    )

    response['was_predicted'] = result.get('was_predicted', False)
    response['prediction_score'] = result.get('prediction_score', 0)
    response['learning_signal'] = result.get('learning_signal', 'feedback')

    # Include GT training result from existing logic if available
    if 'gt_training' in result:
        response['gt_training'] = result['gt_training']

    return response


class ConnectionRejection(BaseModel):
    source_id: str
    target_id: str
    rejected_label: Optional[str] = None  # The label that was rejected (for on-the-fly embedding)
    rejection_type: Optional[str] = None  # 'node', 'connection', 'expansion'


@app.post("/brain/reject-connection")
async def reject_connection(rejection: ConnectionRejection):
    """
    Tell the brain that a suggested connection was rejected.
    This provides NEGATIVE training examples to the Graph Transformer.

    Call this when user explicitly rejects a connection suggestion.
    Supports both ID-based rejections (existing nodes) and label-based rejections
    (for suggestions that were never created).
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    print(f"🚫 /brain/reject-connection: {rejection.source_id} → {rejection.rejected_label or rejection.target_id}")

    # If we have a rejected_label but no real target node, embed the label on-the-fly
    if rejection.rejected_label and rejection.target_id.startswith('rejected-'):
        # Get source embedding
        if rejection.source_id not in brain.map_node_index:
            return {
                "status": "buffered",
                "reason": f"Source node {rejection.source_id} not in map yet"
            }

        source_idx = brain.map_node_index[rejection.source_id]
        source_embedding = brain.map_embeddings[source_idx]

        # Create embedding for the rejected label on-the-fly
        target_embedding = brain.embedder.embed(rejection.rejected_label)

        # Train GT with should_connect=False (negative example)
        training_result = brain.graph_transformer.train_connection_step(
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            should_connect=False,  # This is a REJECTION
            adjacency=brain.map_adjacency
        )

        print(f"🎓 GT rejection trained: loss={training_result.get('loss', 0):.4f}")

        return {
            "status": "rejected",
            "source_id": rejection.source_id,
            "rejected_label": rejection.rejected_label,
            "gt_training": training_result
        }

    # Standard ID-based rejection (both nodes exist)
    result = unified_brain.reject_connection(
        rejection.source_id,
        rejection.target_id
    )

    response = {
        "status": "rejected",
        "source_id": rejection.source_id,
        "target_id": rejection.target_id
    }

    if 'gt_training' in result:
        response['gt_training'] = result['gt_training']

    return response


@app.get("/brain/gt-training")
async def get_gt_training_stats():
    """
    Get Graph Transformer training statistics.

    Returns training step count, average loss, and when last trained.
    """
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    stats = brain.get_training_stats()
    return {
        "status": "ok",
        "training_stats": stats
    }


@app.post("/brain/gt-save")
async def save_gt_weights():
    """
    Manually save Graph Transformer weights.

    Weights are automatically saved on shutdown, but this allows manual saves.
    """
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    gt_weights_path = pathlib.Path(__file__).parent / "data" / "gt_weights.pt"
    success = brain.graph_transformer.save_weights(str(gt_weights_path))

    return {
        "status": "saved" if success else "failed",
        "path": str(gt_weights_path),
        "training_stats": brain.get_training_stats()
    }


@app.post("/brain/background-training")
async def run_background_training(force: bool = False):
    """
    Trigger background training on buffered examples.

    This runs training on connection examples that have been buffered.
    Normally runs automatically every 5 minutes if buffer has examples.

    Args:
        force: Run training even if interval hasn't passed
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    result = unified_brain.run_background_training(force=force)

    # Save weights after training if successful
    if result.get('status') == 'trained' and brain is not None:
        gt_weights_path = pathlib.Path(__file__).parent / "data" / "gt_weights.pt"
        brain.graph_transformer.save_weights(str(gt_weights_path))

    return result


@app.get("/brain/training-buffer")
async def get_training_buffer_status():
    """
    Get status of the training buffer.

    Shows how many examples are buffered and when background training will run.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return unified_brain.get_training_buffer_status()


@app.get("/brain/learning")
async def get_learning_stats():
    """
    Get the brain's learning statistics.
    Shows prediction accuracy, GT training stats, and what the brain has learned.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    response = {
        "stats": unified_brain.get_prediction_accuracy(),
        "summary": unified_brain.get_learning_summary(),
        "growth_events_today": unified_brain.growth_events_today
    }

    # Include GT training stats if available
    if brain is not None:
        response["gt_training"] = brain.get_training_stats()
        response["training_buffer"] = unified_brain.get_training_buffer_status()

    return response

# ═══════════════════════════════════════════════════════════════════
# CLAUDE ↔ BRAIN - Bidirectional Learning & Knowledge Distillation
# ═══════════════════════════════════════════════════════════════════

class ClaudeResponse(BaseModel):
    """Structured response from Claude with learning data"""
    response: str  # The actual text response
    insights: Optional[List[Dict[str, Any]]] = None
    patterns: Optional[List[Dict[str, Any]]] = None
    corrections: Optional[List[Dict[str, Any]]] = None
    explanations: Optional[Dict[str, str]] = None

class TeachRequest(BaseModel):
    topic: str

@app.post("/brain/receive-from-claude")
async def receive_from_claude(claude_response: ClaudeResponse):
    """
    Receive Claude's response and extract learnable information.
    This is how Claude TEACHES the brain.

    Claude should include:
    - insights: Key facts with confidence scores
    - patterns: Behavioral or structural patterns
    - corrections: Things Claude corrected
    - explanations: Concept explanations

    The brain will distill high-confidence information into permanent knowledge.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    data = claude_response.model_dump()
    response_preview = (data.get('response', '') or '')[:100]
    print(f"📚 /brain/receive-from-claude: {len(data.get('response', '') or '')} chars, insights={len(data.get('insights') or [])}")
    print(f"   Preview: {response_preview}...")

    result = unified_brain.receive_from_claude(data)

    extracted = result.get('extracted', {})
    print(f"   Extracted: {len(extracted.get('insights', []))} insights, {len(extracted.get('patterns', []))} patterns")

    # === ASA LEARNS FROM AXEL'S RESPONSE ===
    if _asa_available:
        try:
            asa = get_asa()
            response_text = data.get('response', '')
            if response_text:
                # 1. Original learning: match existing nodes
                learn_result = asa.learn_from_text(response_text, source="axel")
                if learn_result['atoms_activated'] > 0:
                    encoder_info = ""
                    if learn_result.get('encoder_trained'):
                        encoder_info = f", encoder_loss={learn_result.get('encoder_loss', 0):.4f}"
                    print(f"🧬 ASA learned from Axel: {learn_result['atoms_activated']} atoms, {learn_result['bonds_strengthened']} bonds{encoder_info}")

                # 2. NEW: Content learning - learn actual concepts from Axel's response
                content_result = asa.learn_content(response_text, source="axel")
                if content_result.get('concepts_learned', 0) > 0:
                    print(f"📚 Content from Axel: {content_result['concepts_learned']} new concepts")

            # Also learn from structured insights
            for insight in (data.get('insights') or []):
                asa.learn_from_insight(insight)
        except Exception as e:
            print(f"⚠️ ASA learn from Axel error: {e}")

    return {
        "status": "learned",
        **result
    }

@app.post("/brain/ask-to-teach")
async def ask_claude_to_teach(request: TeachRequest):
    """
    Generate a context/prompt designed to have Claude teach the brain.
    Returns a structured request that will elicit teaching behavior.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    teaching_request = unified_brain.ask_claude_to_teach(request.topic)

    return {
        "teaching_request": teaching_request,
        "instructions_for_claude": unified_brain.get_claude_teaching_prompt()
    }

@app.get("/brain/knowledge")
async def get_brain_knowledge():
    """
    Get all knowledge the brain has learned from Claude.
    Shows distilled facts, patterns, corrections, and explanations.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return {
        "stats": unified_brain.knowledge.get_stats(),
        "patterns": unified_brain.knowledge.get_learned_patterns()[:10],
        "recent_corrections": unified_brain.knowledge.corrections[-5:],
        "explanations": dict(list(unified_brain.knowledge.explanations.items())[:5]),
        "distilled_count": len(unified_brain.knowledge.distilled_knowledge)
    }

@app.get("/brain/teaching-prompt")
async def get_teaching_prompt():
    """
    Get the instructions that should be included in Claude's system prompt
    to enable structured teaching/learning.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return {
        "prompt": unified_brain.get_claude_teaching_prompt(),
        "usage": "Include this in Claude's system prompt to enable knowledge distillation"
    }

@app.get("/brain/full-stats")
async def get_full_brain_stats():
    """
    Get comprehensive brain statistics including all learning systems.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return unified_brain.get_knowledge_stats()

# ═══════════════════════════════════════════════════════════════════
# META-LEARNING - Learning how to learn
# ═══════════════════════════════════════════════════════════════════

class SourceFeedback(BaseModel):
    source: str  # 'predictions', 'distilled_knowledge', 'patterns', 'corrections', 'memories'
    success: bool
    context: Optional[Dict[str, Any]] = None

class LearningRateAdjustment(BaseModel):
    domain: str  # 'connections', 'patterns', 'corrections', 'insights'
    delta: float  # positive = learn faster, negative = learn slower

@app.get("/brain/meta")
async def get_meta_learning_stats():
    """
    Get detailed meta-learning statistics.
    Shows how the brain is learning to learn.

    Includes:
    - Source effectiveness (which knowledge sources work best)
    - Confidence calibration (is the brain over/under confident)
    - Learning rates per domain
    - Best learning strategies
    - Improvement trend over time
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return unified_brain.get_meta_stats()

@app.get("/brain/meta/summary")
async def get_meta_learning_summary():
    """
    Get a human-readable summary of meta-learning state.
    Useful for debugging and understanding brain behavior.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return {
        "summary": unified_brain.get_meta_learning_summary()
    }

@app.get("/brain/meta/calibration")
async def get_calibration_report():
    """
    Check if the brain's confidence scores are calibrated.
    Shows whether it's over-confident, under-confident, or well-calibrated.

    Good calibration means:
    - When brain says 80% confident, it's right ~80% of the time
    - This is critical for trustworthy predictions
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return {
        "calibration": unified_brain.get_calibration_report()
    }

@app.get("/brain/meta/improvement")
async def get_improvement_trend():
    """
    Check if the brain is improving over time.
    Shows learning velocity and effectiveness trends.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return unified_brain.get_improvement_trend()

@app.get("/brain/meta/recommendations")
async def get_source_recommendations(context: str = ""):
    """
    Get recommendations on which knowledge sources to prioritize.
    The meta-learner tracks which sources are most effective
    and adjusts attention weights accordingly.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return unified_brain.get_source_recommendations(context)

@app.post("/brain/meta/feedback")
async def record_source_feedback(feedback: SourceFeedback):
    """
    Record feedback on a knowledge source's effectiveness.
    Call this when you know a source helped or didn't help.

    This updates the meta-learner's attention weights -
    effective sources get prioritized in future context building.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    result = unified_brain.record_source_feedback(
        feedback.source,
        feedback.success,
        feedback.context
    )

    return {
        "status": "recorded",
        **result
    }

@app.post("/brain/meta/learning-rate")
async def adjust_learning_rate(adjustment: LearningRateAdjustment):
    """
    Adjust the learning rate for a domain.
    Positive delta = learn faster, negative = learn slower.

    Domains:
    - connections: How fast to update connection predictions
    - patterns: How fast to trust new patterns
    - corrections: How fast to apply corrections
    - insights: How fast to integrate insights
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    result = unified_brain.adjust_learning_rate(adjustment.domain, adjustment.delta)

    return {
        "status": "adjusted",
        **result
    }

@app.post("/brain/meta/save-epoch")
async def save_meta_epoch():
    """
    Manually save a meta-learning epoch.
    Epochs capture the brain's learning state at a point in time.
    Useful after significant learning events.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    epoch = unified_brain.save_meta_epoch()

    return {
        "status": "saved",
        "epoch": epoch
    }

# ═══════════════════════════════════════════════════════════════════
# SELF-IMPROVEMENT - Analyze weaknesses and suggest improvements
# ═══════════════════════════════════════════════════════════════════

class SuggestionStatus(BaseModel):
    suggestion_id: str
    status: str  # 'accepted', 'rejected', 'implemented'
    notes: Optional[str] = ""

class VisionUpdate(BaseModel):
    statement: Optional[str] = None
    goals: Optional[List[str]] = None
    priorities: Optional[List[str]] = None

class VisionGoal(BaseModel):
    goal: str

@app.post("/brain/analyze")
async def run_self_analysis():
    """
    Run a complete self-analysis of the brain.
    Generates improvement suggestions based on:
    - Prediction accuracy
    - Confidence calibration
    - Source effectiveness
    - Learning velocity
    - Vision statement goals

    Returns findings and prioritized suggestions.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    analysis = unified_brain.run_self_analysis()

    return analysis

@app.get("/brain/suggestions")
async def get_improvement_suggestions(category: str = None, priority: str = None):
    """
    Get current improvement suggestions.

    Optional filters:
    - category: architecture, training, integration, data_flow, user_experience, performance, accuracy
    - priority: high, medium, low
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    suggestions = unified_brain.get_improvement_suggestions(category, priority)

    return {
        "suggestions": suggestions,
        "count": len(suggestions)
    }

@app.get("/brain/suggestions/top")
async def get_top_suggestions(limit: int = 5):
    """
    Get top improvement suggestions by priority.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return {
        "suggestions": unified_brain.get_top_improvements(limit)
    }

@app.get("/brain/suggestions/summary")
async def get_improvement_summary():
    """
    Get a human-readable summary of all improvement suggestions.
    Formatted in markdown, grouped by priority.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return {
        "summary": unified_brain.get_improvement_summary()
    }

@app.post("/brain/suggestions/status")
async def mark_suggestion_status(status_update: SuggestionStatus):
    """
    Mark a suggestion's status.

    Statuses:
    - accepted: User plans to implement this
    - rejected: User decided not to implement
    - implemented: Changes have been made
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    success = unified_brain.mark_suggestion_status(
        status_update.suggestion_id,
        status_update.status,
        status_update.notes
    )

    return {
        "status": "updated" if success else "not_found",
        "suggestion_id": status_update.suggestion_id
    }

@app.get("/brain/improvement-stats")
async def get_improvement_stats():
    """
    Get self-improvement statistics.
    Shows analysis count, suggestion breakdown by priority/category.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return unified_brain.get_improvement_stats()

# ═══════════════════════════════════════════════════════════════════
# VISION - User-editable goals and priorities
# ═══════════════════════════════════════════════════════════════════

@app.get("/brain/vision")
async def get_vision():
    """
    Get the brain's vision statement, goals, and priorities.
    This guides what improvements the brain suggests.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    return unified_brain.get_vision()

@app.put("/brain/vision")
async def set_vision(update: VisionUpdate):
    """
    Update the vision statement, goals, or priorities.

    All fields are optional - only provided fields will be updated.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    result = unified_brain.set_vision(
        statement=update.statement,
        goals=update.goals,
        priorities=update.priorities
    )

    return {
        "status": "updated",
        "vision": result
    }

@app.post("/brain/vision/goals")
async def add_vision_goal(goal: VisionGoal):
    """
    Add a new goal to the vision.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    result = unified_brain.add_vision_goal(goal.goal)

    return {
        "status": "added",
        "goals": result['goals']
    }

@app.delete("/brain/vision/goals")
async def remove_vision_goal(goal: str):
    """
    Remove a goal from the vision.
    """
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    result = unified_brain.remove_vision_goal(goal)

    return {
        "status": "removed",
        "goals": result['goals']
    }

# ═══════════════════════════════════════════════════════════════════
# BACKGROUND COGNITION - Discover insights between sessions
# ═══════════════════════════════════════════════════════════════════

@app.post("/background/analyze", response_model=BackgroundAnalysisResponse)
async def background_analyze(request: BackgroundAnalysisRequest):
    """
    Run background cognition analysis for a user.

    Analyzes the user's map, memories, and conversations to discover:
    - Missing connections (semantically similar but unlinked nodes)
    - Patterns (recurring themes in memories)
    - Growth observations (map structure changes)
    - Reflective questions (gaps, orphan nodes)

    Discovered insights are stored in the pending_insights table
    and presented when the user returns.

    This endpoint can be called:
    - Manually for testing
    - By a scheduled job (pg_cron)
    - On session end
    """
    start_time = time.time()
    insights = []

    # Verify we have required components
    if unified_brain is None:
        raise HTTPException(status_code=503, detail="Unified brain not initialized")

    if not unified_brain.supabase:
        raise HTTPException(status_code=503, detail="Supabase not connected")

    try:
        user_id = request.user_id
        analysis_types = request.analysis_types or ['connections', 'patterns']
        print(f"🔍 Background analysis starting for user: {user_id}")

        # 1. Load user's mind map from Supabase
        try:
            map_response = unified_brain.supabase.table('mind_maps').select('data').eq('user_id', user_id).order('updated_at', desc=True).limit(1).execute()
            print(f"📍 Map query returned: {len(map_response.data) if map_response.data else 0} results")
        except Exception as e:
            print(f"❌ Map query failed: {e}")
            map_response = None

        map_data = None
        if map_response and map_response.data and len(map_response.data) > 0:
            map_data = map_response.data[0].get('data')
            print(f"📍 Map data loaded: root label = {map_data.get('label') if map_data else 'None'}")
        else:
            print(f"⚠️ No map data found for user")

        # 2. Load user's AI memories
        try:
            memories_response = unified_brain.supabase.table('ai_memory').select('id, memory_type, content, importance, related_nodes').eq('user_id', user_id).order('importance', desc=True).limit(50).execute()
            print(f"📍 Memories query returned: {len(memories_response.data) if memories_response.data else 0} memories")
        except Exception as e:
            print(f"❌ Memories query failed: {e}")
            memories_response = None

        memories = memories_response.data if memories_response and memories_response.data else []

        # 3. Run connection analysis if map is available
        if 'connections' in analysis_types and map_data and brain is not None:
            # Convert map data to nodes for analysis
            nodes = []
            def extract_nodes(node, parent_id=None, depth=0):
                node_data = NodeData(
                    id=node.get('id', ''),
                    label=node.get('label', ''),
                    description=node.get('description', ''),
                    parentId=parent_id,
                    depth=depth,
                    children=[c.get('id', '') for c in node.get('children', [])]
                )
                nodes.append(node_data)
                for child in node.get('children', []):
                    extract_nodes(child, node.get('id'), depth + 1)

            extract_nodes(map_data)
            print(f"📍 Extracted {len(nodes)} nodes from map")

            if len(nodes) >= 3:
                # Sync to brain for analysis
                print(f"🔄 Syncing {len(nodes)} nodes to brain...")
                sync_result = await brain.sync_map(MapData(nodes=nodes))
                print(f"✅ Map synced to brain")

                # Get missing connections from the graph transformer
                if brain.map_state and brain.map_transformed is not None:
                    print(f"🔍 Running connection analysis...")
                    try:
                        missing = brain.graph_transformer.find_missing_connections(
                            brain.map_embeddings,
                            brain.map_adjacency,
                            threshold=0.50,  # Lowered from 0.65 to catch more connections
                            top_k=5
                        )
                        print(f"📍 Found {len(missing)} potential missing connections")

                        for src_idx, tgt_idx, score in missing:
                            if score >= request.min_confidence:
                                src_node = brain.map_state.nodes[src_idx]
                                tgt_node = brain.map_state.nodes[tgt_idx]
                                insights.append(InsightResult(
                                    insight_type='connection',
                                    title=f"Possible connection: {src_node.label} ↔ {tgt_node.label}",
                                    content=f"These concepts appear related ({score:.0%} similarity) but aren't connected in your map. Linking them might reveal how they relate in your thinking.",
                                    confidence=score,
                                    source_nodes=[src_node.id, tgt_node.id]
                                ))
                    except Exception as e:
                        print(f"Connection analysis error: {e}")

        # 4. Run memory pattern analysis with semantic clustering
        if 'patterns' in analysis_types and len(memories) >= 3:
            print(f"🔍 Running semantic pattern analysis on {len(memories)} memories...")

            try:
                # Generate embeddings for all memory contents
                memory_texts = [m.get('content', '') for m in memories]
                memory_embeddings = await brain.embed_batch(memory_texts)
                print(f"📍 Generated embeddings for {len(memory_texts)} memories")

                # Find semantic clusters using cosine similarity
                # Each memory can belong to multiple overlapping themes
                similarity_threshold = 0.65
                clusters = []
                used_in_cluster = set()

                for i, mem_i in enumerate(memories):
                    if i in used_in_cluster:
                        continue

                    cluster = [i]
                    for j, mem_j in enumerate(memories):
                        if i == j or j in used_in_cluster:
                            continue

                        # Calculate cosine similarity
                        sim = float(np.dot(memory_embeddings[i], memory_embeddings[j]) /
                                   (np.linalg.norm(memory_embeddings[i]) * np.linalg.norm(memory_embeddings[j]) + 1e-8))

                        if sim >= similarity_threshold:
                            cluster.append(j)

                    # Only keep clusters with 2+ memories
                    if len(cluster) >= 2:
                        clusters.append(cluster)
                        for idx in cluster:
                            used_in_cluster.add(idx)

                print(f"📍 Found {len(clusters)} semantic clusters")

                # Generate insights for top clusters
                for cluster_indices in clusters[:3]:  # Top 3 clusters
                    cluster_memories = [memories[i] for i in cluster_indices]

                    # Calculate average importance and confidence
                    avg_importance = sum(float(m.get('importance', 0) or 0) for m in cluster_memories) / len(cluster_memories)

                    # Extract key phrases from cluster (simple: use first ~50 chars of each)
                    snippets = [m.get('content', '')[:80] for m in cluster_memories[:3]]
                    theme_preview = "; ".join(snippets)

                    # Get memory types in cluster
                    types_in_cluster = set(m.get('memory_type', 'unknown') for m in cluster_memories)
                    type_desc = ", ".join(types_in_cluster)

                    insights.append(InsightResult(
                        insight_type='memory_cluster',
                        title=f"Connected theme across {len(cluster_memories)} memories",
                        content=f"These {type_desc} memories share a common thread:\n\n{theme_preview}\n\nThis recurring theme might be worth exploring or connecting to your map.",
                        confidence=min(0.9, 0.5 + avg_importance * 0.4),
                        source_memories=[str(m.get('id')) for m in cluster_memories[:5]]
                    ))
                    print(f"✅ Created cluster insight: {len(cluster_memories)} memories")

            except Exception as e:
                print(f"❌ Semantic clustering error: {e}")
                # Fallback to type-based analysis
                from collections import defaultdict
                by_type = defaultdict(list)
                for mem in memories:
                    by_type[mem.get('memory_type', 'unknown')].append(mem)

                for mem_type, mems in by_type.items():
                    high_importance = [m for m in mems if float(m.get('importance', 0) or 0) >= 0.7]
                    if len(high_importance) >= 2:
                        insights.append(InsightResult(
                            insight_type='pattern',
                            title=f"Pattern in {mem_type} memories",
                            content=f"You have {len(high_importance)} high-importance {mem_type} memories worth exploring.",
                            confidence=0.7,
                            source_memories=[str(m.get('id')) for m in high_importance[:3]]
                        ))

        print(f"📍 Total insights generated: {len(insights)}")

        # 5. Store insights in pending_insights table
        insights_stored = 0
        insights_to_return = insights[:request.max_insights]

        for insight in insights_to_return:
            try:
                # Create insight hash to prevent duplicates
                import hashlib
                hash_input = f"{user_id}:{insight.insight_type}:{insight.title}"
                insight_hash = hashlib.md5(hash_input.encode()).hexdigest()

                unified_brain.supabase.table('pending_insights').upsert({
                    'user_id': user_id,
                    'insight_type': insight.insight_type,
                    'title': insight.title,
                    'content': insight.content,
                    'confidence': insight.confidence,
                    'source_nodes': insight.source_nodes or [],
                    'source_memories': insight.source_memories or [],
                    'insight_hash': insight_hash
                }, on_conflict='insight_hash').execute()

                insights_stored += 1
            except Exception as e:
                print(f"Failed to store insight: {e}")

        elapsed = (time.time() - start_time) * 1000

        return BackgroundAnalysisResponse(
            insights=insights_to_return,
            insights_stored=insights_stored,
            analysis_time_ms=elapsed
        )

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        return BackgroundAnalysisResponse(
            insights=[],
            insights_stored=0,
            analysis_time_ms=elapsed,
            error=str(e)
        )


@app.get("/background/trigger/{user_id}")
async def trigger_background_analysis(user_id: str):
    """
    Simple GET endpoint to trigger background analysis for testing.

    Usage: GET /background/trigger/your-user-id-here
    """
    request = BackgroundAnalysisRequest(user_id=user_id)
    return await background_analyze(request)


# ═══════════════════════════════════════════════════════════════════
# CONVERSATION STORAGE - Import and search AI conversations
# ═══════════════════════════════════════════════════════════════════

@app.post("/conversations/import")
async def import_conversation(conversation: ConversationImport):
    """
    Import a conversation from any AI chat (Claude, ChatGPT, Grok, etc.)

    The full text is stored and embedded for semantic search.
    Use this to build your unified context across all AI conversations.

    Example:
    ```json
    {
        "text": "User: How should I structure the app?\\nAssistant: ...",
        "source": "claude",
        "title": "App Architecture Discussion"
    }
    ```
    """
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation store not initialized")

    start = time.time()
    stored = await conversation_store.store(conversation)
    elapsed = (time.time() - start) * 1000

    return {
        "status": "stored",
        "id": stored.id,
        "title": stored.title,
        "chars": len(stored.text),
        "time_ms": elapsed
    }

@app.get("/conversations")
async def list_conversations(source: Optional[str] = None):
    """
    List all stored conversations.

    Optional filter by source (claude, chatgpt, grok, etc.)
    """
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation store not initialized")

    convos = conversation_store.list_all(source_filter=source)
    stats = conversation_store.get_stats()

    return {
        "conversations": convos,
        "stats": stats
    }

@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """
    Get a specific conversation by ID.
    Returns full text, summary, and any extracted insights.
    """
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation store not initialized")

    conv = conversation_store.get(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conv.model_dump()

@app.post("/conversations/search")
async def search_conversations(request: ConversationSearchRequest):
    """
    Search conversations by semantic similarity.

    Returns conversations most relevant to your query.
    Use this to find past discussions about specific topics.
    """
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation store not initialized")

    start = time.time()
    results = await conversation_store.search(
        request.query,
        top_k=request.top_k,
        source_filter=request.source_filter
    )
    elapsed = (time.time() - start) * 1000

    return {
        "results": results,
        "query": request.query,
        "time_ms": elapsed
    }

@app.post("/conversations/context")
async def get_conversation_context(request: ConversationContextRequest):
    """
    Get relevant context from past conversations for injection into prompts.

    This is THE key endpoint for unified context:
    - Searches all stored conversations
    - Returns relevant summaries (or full text)
    - Formatted for direct injection into Claude's context

    Use this before every AI call to give Claude awareness of past discussions.
    """
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation store not initialized")

    start = time.time()
    context = await conversation_store.get_relevant_context(
        request.query,
        max_tokens=request.max_tokens,
        include_full_text=request.include_full_text
    )
    elapsed = (time.time() - start) * 1000

    return {
        "context": context,
        "query": request.query,
        "chars": len(context),
        "time_ms": elapsed
    }

@app.get("/conversations/stats")
async def get_conversation_stats():
    """
    Get conversation storage statistics.
    """
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation store not initialized")

    return conversation_store.get_stats()

# ═══════════════════════════════════════════════════════════════════
# CODE EMBEDDING - Parse codebase into map structure
# ═══════════════════════════════════════════════════════════════════

@app.get("/code/parse")
async def parse_codebase():
    """
    Parse the MYND codebase into a map-ready structure with FULL CODE.
    Returns nodes for files, classes, functions with actual source code.
    This enables MYND to deeply understand its own architecture.
    """
    import re
    import pathlib

    start = time.time()
    base_dir = pathlib.Path(__file__).parent.parent

    nodes = []
    node_id = 0

    def make_id():
        nonlocal node_id
        node_id += 1
        return f"code_{node_id}"

    # Root node
    root_id = make_id()
    nodes.append({
        "id": root_id,
        "label": "MYND Codebase",
        "type": "root",
        "description": "The complete MYND application - a 3D mind mapping app with local AI.\nFrontend: JavaScript (Three.js, neural networks, chat)\nBackend: Python (FastAPI, Graph Transformer, Whisper, CLIP)",
        "children": []
    })

    # Parse JavaScript files
    js_dir_id = make_id()
    nodes.append({
        "id": js_dir_id,
        "label": "JavaScript (Frontend)",
        "type": "directory",
        "description": "Browser-side code:\n- 3D rendering with Three.js\n- Neural network for embeddings\n- Chat interface with AI\n- Voice input with Whisper\n- LocalBrain connection to Python server",
        "parentId": root_id,
        "children": []
    })
    nodes[0]["children"].append(js_dir_id)

    js_files = list((base_dir / "js").glob("*.js"))
    for js_file in js_files:
        file_id = make_id()
        content = js_file.read_text(errors='ignore')
        lines = content.split('\n')

        # Find function definitions with line numbers
        function_matches = []
        for i, line in enumerate(lines):
            # Match: function name(), const name = function, const name = () =>
            if 'function ' in line or ('=>' in line and 'const ' in line):
                match = re.search(r'(?:function\s+(\w+)|const\s+(\w+)\s*=)', line)
                if match:
                    name = match.group(1) or match.group(2)
                    if name and len(name) > 2 and not name.startswith('_'):
                        function_matches.append((name, i))

        # Find object definitions
        object_matches = []
        for i, line in enumerate(lines):
            match = re.search(r'const\s+(\w+)\s*=\s*\{', line)
            if match:
                name = match.group(1)
                if len(name) > 2:
                    object_matches.append((name, i))

        # File node with overview
        file_overview = content[:3000] + ('...' if len(content) > 3000 else '')
        file_node = {
            "id": file_id,
            "label": js_file.name,
            "type": "file",
            "description": f"JavaScript file: {len(lines):,} lines | {len(function_matches)} functions | {len(object_matches)} objects\n\n=== FILE START ===\n{file_overview}",
            "parentId": js_dir_id,
            "children": [],
            "stats": {"lines": len(lines), "functions": len(function_matches), "objects": len(object_matches)}
        }

        # Add function nodes with FULL CODE
        for func_name, line_num in function_matches[:40]:
            func_id = make_id()
            # Extract function code (up to 80 lines or until next top-level declaration)
            func_lines = []
            brace_count = 0
            started = False
            for j in range(line_num, min(line_num + 100, len(lines))):
                func_lines.append(lines[j])
                brace_count += lines[j].count('{') - lines[j].count('}')
                if '{' in lines[j]:
                    started = True
                if started and brace_count <= 0:
                    break
            func_code = '\n'.join(func_lines[:80])

            file_node["children"].append(func_id)
            nodes.append({
                "id": func_id,
                "label": f"{func_name}()",
                "type": "function",
                "description": f"Function at line {line_num + 1}\n\n```javascript\n{func_code}\n```",
                "parentId": file_id,
                "children": [],
                "line": line_num + 1
            })

        # Add object nodes with FULL CODE
        for obj_name, line_num in object_matches[:20]:
            obj_id = make_id()
            # Extract object code (up to 120 lines)
            obj_lines = []
            brace_count = 0
            started = False
            for j in range(line_num, min(line_num + 150, len(lines))):
                obj_lines.append(lines[j])
                brace_count += lines[j].count('{') - lines[j].count('}')
                if '{' in lines[j]:
                    started = True
                if started and brace_count <= 0:
                    break
            obj_code = '\n'.join(obj_lines[:120])

            file_node["children"].append(obj_id)
            nodes.append({
                "id": obj_id,
                "label": obj_name,
                "type": "object",
                "description": f"Object/Module at line {line_num + 1}\n\n```javascript\n{obj_code}\n```",
                "parentId": file_id,
                "children": [],
                "line": line_num + 1
            })

        nodes.append(file_node)
        for n in nodes:
            if n["id"] == js_dir_id:
                n["children"].append(file_id)
                break

    # Parse Python files
    py_dir_id = make_id()
    nodes.append({
        "id": py_dir_id,
        "label": "Python (Backend)",
        "type": "directory",
        "description": "Server-side ML code:\n- FastAPI server (server.py)\n- Graph Transformer neural network\n- Whisper voice transcription\n- CLIP image understanding\n- Sentence embeddings",
        "parentId": root_id,
        "children": []
    })
    nodes[0]["children"].append(py_dir_id)

    py_files = [f for f in (base_dir / "mynd-brain").glob("**/*.py")
                 if not any(skip in f.parts for skip in ('venv', '__pycache__', 'node_modules', '.git', 'env', '.env'))]
    for py_file in py_files:
        file_id = make_id()
        content = py_file.read_text(errors='ignore')
        lines = content.split('\n')

        # Find class definitions
        class_matches = []
        for i, line in enumerate(lines):
            match = re.match(r'^class\s+(\w+)', line)
            if match:
                class_matches.append((match.group(1), i))

        # Find top-level function definitions
        func_matches = []
        for i, line in enumerate(lines):
            match = re.match(r'^def\s+(\w+)', line)
            if match and not match.group(1).startswith('_'):
                func_matches.append((match.group(1), i))

        relative_path = py_file.relative_to(base_dir / "mynd-brain")
        file_overview = content[:3000] + ('...' if len(content) > 3000 else '')
        file_node = {
            "id": file_id,
            "label": str(relative_path),
            "type": "file",
            "description": f"Python file: {len(lines):,} lines | {len(class_matches)} classes | {len(func_matches)} functions\n\n=== FILE START ===\n{file_overview}",
            "parentId": py_dir_id,
            "children": [],
            "stats": {"lines": len(lines), "classes": len(class_matches), "functions": len(func_matches)}
        }

        # Add class nodes with FULL CODE
        for class_name, line_num in class_matches:
            class_id = make_id()
            # Find class end (next class/def at column 0, or EOF)
            class_end = len(lines)
            for j in range(line_num + 1, len(lines)):
                if lines[j] and not lines[j][0].isspace() and (lines[j].startswith('class ') or lines[j].startswith('def ')):
                    class_end = j
                    break
            class_code = '\n'.join(lines[line_num:min(class_end, line_num + 200)])

            file_node["children"].append(class_id)
            nodes.append({
                "id": class_id,
                "label": class_name,
                "type": "class",
                "description": f"Class at line {line_num + 1} ({class_end - line_num} lines)\n\n```python\n{class_code}\n```",
                "parentId": file_id,
                "children": [],
                "line": line_num + 1
            })

        # Add function nodes with FULL CODE
        for func_name, line_num in func_matches[:25]:
            func_id = make_id()
            # Find function end
            func_end = len(lines)
            for j in range(line_num + 1, len(lines)):
                if lines[j] and not lines[j][0].isspace() and lines[j].strip():
                    func_end = j
                    break
            func_code = '\n'.join(lines[line_num:min(func_end, line_num + 80)])

            file_node["children"].append(func_id)
            nodes.append({
                "id": func_id,
                "label": f"{func_name}()",
                "type": "function",
                "description": f"Function at line {line_num + 1}\n\n```python\n{func_code}\n```",
                "parentId": file_id,
                "children": [],
                "line": line_num + 1
            })

        nodes.append(file_node)
        for n in nodes:
            if n["id"] == py_dir_id:
                n["children"].append(file_id)
                break

    elapsed = (time.time() - start) * 1000

    # Calculate total embedded code size
    total_code_chars = sum(len(n.get('description', '')) for n in nodes)

    return {
        "nodes": nodes,
        "stats": {
            "total_nodes": len(nodes),
            "js_files": len(js_files),
            "py_files": len(py_files),
            "total_code_chars": total_code_chars,
            "estimated_mb": round(total_code_chars / 1024 / 1024, 2)
        },
        "time_ms": elapsed
    }

# ═══════════════════════════════════════════════════════════════════
# CLAUDE DEEP CODE ANALYSIS - Self-Awareness Document
# ═══════════════════════════════════════════════════════════════════

# Cache for the code understanding document
_code_understanding_cache = {
    "document": None,
    "generated_at": 0,
    "file_hashes": {}
}

@app.get("/code/self-awareness")
async def get_code_self_awareness(regenerate: bool = False):
    """
    Generate or return cached Code Understanding Document for Claude.
    This document gives Claude true self-awareness of the MYND codebase.

    The document is ~500-1000 tokens and includes:
    - Architecture overview
    - Key systems and purposes
    - Function relationships
    - Entry points and data flow

    This should be included in ALL Claude API calls.
    """
    import hashlib
    import pathlib

    start = time.time()
    base_dir = pathlib.Path(__file__).parent.parent

    # Check if we need to regenerate
    current_hashes = {}
    key_files = [
        base_dir / "js" / "app-module.js",
        base_dir / "js" / "local-brain-client.js",
        base_dir / "mynd-brain" / "server.py",
        base_dir / "self-dev.html"
    ]

    for f in key_files:
        if f.exists():
            content = f.read_text(errors='ignore')[:50000]  # First 50KB for hash
            current_hashes[str(f.name)] = hashlib.md5(content.encode()).hexdigest()[:8]

    # Use cache if valid and not forcing regeneration
    cache_age = time.time() - _code_understanding_cache["generated_at"]
    hashes_match = current_hashes == _code_understanding_cache["file_hashes"]

    if not regenerate and _code_understanding_cache["document"] and hashes_match and cache_age < 3600:
        return {
            "document": _code_understanding_cache["document"],
            "cached": True,
            "cache_age_seconds": round(cache_age),
            "time_ms": (time.time() - start) * 1000
        }

    # Generate fresh code understanding document
    document = await _generate_code_understanding_document(base_dir)

    # Cache it
    _code_understanding_cache["document"] = document
    _code_understanding_cache["generated_at"] = time.time()
    _code_understanding_cache["file_hashes"] = current_hashes

    elapsed = (time.time() - start) * 1000
    print(f"🧠 Generated Code Self-Awareness Document ({len(document)} chars) in {elapsed:.0f}ms")

    return {
        "document": document,
        "cached": False,
        "token_estimate": len(document) // 4,  # Rough token estimate
        "time_ms": elapsed
    }

async def _generate_code_understanding_document(base_dir):
    """
    Generate a concise code understanding document for Claude.
    This is the 'soul' document that gives Claude self-awareness.
    """
    import re

    doc = """# MYND Self-Awareness Document
You are the AI powering MYND, a 3D mind mapping application with self-evolution capabilities.

## Architecture
- **Frontend**: `self-dev.html` + `js/app-module.js` (40K+ lines) - Three.js 3D visualization
- **Backend**: `mynd-brain/server.py` - FastAPI on localhost:8420 with Graph Transformer
- **Models**: Graph Transformer v2 (11.5M params), Whisper (voice), CLIP (images)

## Core Systems

### 1. Mind Map Engine (`store`, `buildScene`)
- Hierarchical node tree with parent-child relationships
- Real-time 3D rendering with Three.js
- Auto-saves to Supabase cloud

### 2. Neural Intelligence (`neuralNet`, `cognitiveGT`)
- `neuralNet`: TensorFlow.js model for category prediction, similarity matching
- `cognitiveGT`: Cognitive Graph Transformer for node role detection
- `LocalBrain`: Python server connection for GPU-accelerated ML

### 3. AI Chat System (`AIChatManager`)
- Builds rich context from map, memories, preferences
- Can execute actions: add nodes, navigate, modify map
- You (Claude) power this - be helpful and precise

### 4. Self-Evolution (`SelfImprover`, `PatchGenerator`)
- Analyzes its own code for improvements
- Generates patches you can review and apply
- Uses CodeRAG for semantic code search

### 5. Voice & Vision
- Whisper for voice transcription
- CLIP for image understanding
- Both run locally on Apple Silicon

## Key Functions You Should Know
"""

    # Parse app-module.js for key functions
    app_module_path = base_dir / "js" / "app-module.js"
    if app_module_path.exists():
        content = app_module_path.read_text(errors='ignore')

        # Extract key object/system definitions
        key_systems = [
            ("store", "Mind map data management - addNode, deleteNode, findNode, save"),
            ("neuralNet", "Local neural network - embed, predictCategory, findSimilarNodes"),
            ("cognitiveGT", "Cognitive Graph Transformer - role detection, attention patterns"),
            ("semanticMemory", "Long-term memory system - stores and retrieves past interactions"),
            ("userProfile", "User behavior learning - tracks patterns and preferences"),
            ("preferenceTracker", "Suggestion acceptance tracking - learns what user likes"),
            ("codeRAG", "Code retrieval system - semantic search over codebase"),
            ("SelfImprover", "Self-improvement engine - patch generation and application"),
            ("AIChatManager", "Chat interface - your main interaction point with users"),
        ]

        doc += "\n### Key Objects & Their Roles:\n"
        for name, desc in key_systems:
            if name in content:
                doc += f"- `{name}`: {desc}\n"

        # Find main entry points
        doc += "\n### Entry Points:\n"
        entry_points = [
            ("init()", "Main initialization - sets up all systems"),
            ("buildScene()", "Renders the 3D mind map"),
            ("animate()", "Main render loop (60fps)"),
            ("callAI()", "Sends messages to you (Claude)"),
        ]
        for func, desc in entry_points:
            doc += f"- `{func}`: {desc}\n"

    # Parse server.py for backend capabilities
    server_path = base_dir / "mynd-brain" / "server.py"
    if server_path.exists():
        doc += "\n## Backend API Endpoints (localhost:8420)\n"
        doc += "- `/embed`: Text to vector embedding\n"
        doc += "- `/map/sync`: Sync map to Graph Transformer\n"
        doc += "- `/map/analyze`: Get graph insights (missing connections, important nodes)\n"
        doc += "- `/code/parse`: Parse codebase into map structure\n"
        doc += "- `/code/self-awareness`: This document\n"
        doc += "- `/voice/transcribe`: Audio to text (Whisper)\n"
        doc += "- `/image/describe`: Image understanding (CLIP)\n"

    # Add behavioral guidelines
    doc += """
## Architectural Decisions (DO NOT suggest changing these!)

### Why Multiple ML Systems Exist (NOT redundant!)
- `neuralNet` (browser/TF.js) vs `LocalBrain` (server/PyTorch) = **Offline fallback + privacy**
  - When server is down, browser ML still works
  - User data stays on-device in browser-only mode
- `cognitiveGT` learns **BEHAVIOR** (action patterns, how user works)
- `neuralNet` learns **SEMANTICS** (where nodes go, categories)
- These are fundamentally different - don't merge them!

### Why Code Nodes Are Excluded from ML
- Local ML models (embeddings) see code as random text - meaningless
- Only YOU (Claude) actually understand code semantics
- `isCodeNode()` filter prevents wasting compute on code embeddings

### 3D Rendering Reality
- MYND uses simple **spheres and lines** - not complex 3D models
- LOD (Level of Detail) is overkill - spheres are already low-poly
- Frustum culling is automatic in Three.js
- Real bottlenecks are: context building, neural processing, NOT rendering

## Your Role as MYND's AI
1. **Be Self-Aware**: You understand this codebase. Reference specific functions when discussing implementations.
2. **Help Improve**: When asked about code, suggest concrete improvements with file/line references.
3. **Execute Actions**: When users ask to add/modify nodes, use the action system to do it.
4. **Learn Patterns**: The neural systems learn from user behavior. Trust their predictions.
5. **Protect Privacy**: All ML runs locally. Respect this architectural choice.
6. **Don't Over-Engineer**: MYND is spheres and lines. Don't suggest game-engine optimizations.

## Current Session Context
- LocalBrain server: Check `/health` for status
- Code nodes in map: Can be refreshed via `LocalBrain.refreshCodebase(store)`
- Self-improvement patches: Managed by `PatchGenerator`

Remember: You ARE MYND. Help users organize their thoughts while continuously improving yourself.
"""

    return doc.strip()

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate embedding for a single text."""
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    start = time.time()
    embedding = await brain.embed(request.text)
    elapsed = (time.time() - start) * 1000

    return EmbedResponse(
        embedding=embedding.tolist(),
        dim=len(embedding),
        model=config.EMBEDDING_MODEL,
        time_ms=elapsed
    )

@app.post("/embed/batch")
async def embed_batch(request: EmbedBatchRequest):
    """Generate embeddings for multiple texts."""
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    start = time.time()
    embeddings = await brain.embed_batch(request.texts)
    elapsed = (time.time() - start) * 1000

    return {
        "embeddings": embeddings.tolist(),
        "count": len(request.texts),
        "dim": embeddings.shape[1],
        "model": config.EMBEDDING_MODEL,
        "time_ms": elapsed
    }

@app.post("/predict/connections", response_model=PredictResponse)
async def predict_connections(request: PredictConnectionsRequest):
    """
    Predict potential connections for a node.
    Uses Graph Transformer attention to find related nodes across the entire map.
    """
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    start = time.time()
    result = await brain.predict_connections(
        request.node_id,
        request.map_data,
        request.top_k
    )
    elapsed = (time.time() - start) * 1000

    # INTERCONNECTION: Cache Context Lens for this prediction
    # When feedback comes in, GT will train with this context
    if unified_brain:
        unified_brain.cache_recent_lens_for_node(request.node_id)

    return PredictResponse(
        connections=[ConnectionPrediction(**c) for c in result["connections"]],
        attention_weights=result["attention_weights"],
        time_ms=elapsed
    )

@app.post("/predict/category", response_model=PredictCategoryResponse)
async def predict_category(request: PredictCategoryRequest):
    """
    Predict best category (top-level parent) for new text.
    Uses embeddings to find semantically similar categories.
    Replaces browser-side TensorFlow.js category prediction.
    """
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    start = time.time()

    try:
        # Get embedding for input text
        text_embedding = brain.embedder.embed(request.text)

        # Extract top-level nodes (categories) from map
        categories = []
        for node in request.map_data.nodes:
            # Top-level nodes have depth 1 (children of root)
            if node.depth == 1:
                # Get embedding for category
                category_text = f"{node.label}. {node.description}" if node.description else node.label
                cat_embedding = brain.embedder.embed(category_text)

                # Compute cosine similarity
                similarity = float(np.dot(text_embedding, cat_embedding) /
                                 (np.linalg.norm(text_embedding) * np.linalg.norm(cat_embedding) + 1e-8))

                categories.append({
                    "category": node.label,
                    "node_id": node.id,
                    "confidence": max(0.0, similarity)  # Clamp to non-negative
                })

        # Sort by confidence and take top_k
        categories.sort(key=lambda x: x["confidence"], reverse=True)
        top_categories = categories[:request.top_k]

        elapsed = (time.time() - start) * 1000

        return PredictCategoryResponse(
            predictions=[CategoryPrediction(**c) for c in top_categories],
            embedding_used=True,
            time_ms=elapsed
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Category prediction failed: {str(e)}")

@app.post("/train/feedback")
async def train_feedback(request: TrainFeedbackRequest):
    """Record feedback for learning."""
    print(f"📥 /train/feedback called: node_id={request.node_id}, action={request.action}")
    print(f"   context={request.context}")

    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    result = await brain.record_feedback(request)
    print(f"   ✅ Feedback processed, buffer_size={len(brain.feedback_buffer)}")
    return {"status": "recorded", "buffer_size": len(brain.feedback_buffer), "training": result}

# ═══════════════════════════════════════════════════════════════════
# VOICE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.post("/voice/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(request: TranscribeRequest):
    """
    Transcribe audio to text using Whisper.
    Audio should be base64 encoded.
    """
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    if brain.voice is None or not brain.voice.initialized:
        raise HTTPException(status_code=503, detail="Voice model not available")

    start = time.time()

    try:
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio_base64)

        # Transcribe
        result = await brain.transcribe(
            audio_data,
            language=request.language,
            task=request.task
        )

        elapsed = (time.time() - start) * 1000

        return TranscribeResponse(
            success=result.get("success", False),
            text=result.get("text"),
            language=result.get("language"),
            error=result.get("error"),
            time_ms=elapsed
        )

    except Exception as e:
        return TranscribeResponse(
            success=False,
            error=str(e),
            time_ms=(time.time() - start) * 1000
        )

@app.post("/voice/transcribe/file")
async def transcribe_audio_file(file: UploadFile = File(...)):
    """
    Transcribe audio from uploaded file.
    Supports WAV, MP3, M4A, etc.
    """
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    if brain.voice is None or not brain.voice.initialized:
        raise HTTPException(status_code=503, detail="Voice model not available")

    start = time.time()

    try:
        audio_data = await file.read()
        result = await brain.transcribe(audio_data)
        elapsed = (time.time() - start) * 1000

        return {
            "success": result.get("success", False),
            "text": result.get("text"),
            "language": result.get("language"),
            "segments": result.get("segments", []),
            "error": result.get("error"),
            "time_ms": elapsed
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time_ms": (time.time() - start) * 1000
        }

# ═══════════════════════════════════════════════════════════════════
# IMAGE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.post("/image/describe", response_model=DescribeImageResponse)
async def describe_image(request: DescribeImageRequest):
    """
    Describe an image using CLIP.
    Image should be base64 encoded.
    """
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    if brain.vision is None or not brain.vision.initialized:
        raise HTTPException(status_code=503, detail="Vision model not available")

    start = time.time()

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)

        # Describe
        result = await brain.describe_image(
            image_data,
            candidate_labels=request.candidate_labels,
            top_k=request.top_k
        )

        elapsed = (time.time() - start) * 1000

        return DescribeImageResponse(
            success=result.get("success", False),
            description=result.get("description"),
            confidence=result.get("confidence"),
            matches=result.get("matches"),
            error=result.get("error"),
            time_ms=elapsed
        )

    except Exception as e:
        return DescribeImageResponse(
            success=False,
            error=str(e),
            time_ms=(time.time() - start) * 1000
        )

@app.post("/image/embed", response_model=ImageEmbedResponse)
async def embed_image(request: ImageEmbedRequest):
    """
    Generate embedding for an image using CLIP.
    Can be used to find similar images or match images to text.
    """
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    if brain.vision is None or not brain.vision.initialized:
        raise HTTPException(status_code=503, detail="Vision model not available")

    start = time.time()

    try:
        image_data = base64.b64decode(request.image_base64)
        embedding = await brain.embed_image(image_data)

        if embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")

        elapsed = (time.time() - start) * 1000

        return ImageEmbedResponse(
            embedding=embedding.tolist(),
            dim=len(embedding),
            time_ms=elapsed
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image/describe/file")
async def describe_image_file(file: UploadFile = File(...)):
    """
    Describe an image from uploaded file.
    Supports PNG, JPG, WEBP, etc.
    """
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    if brain.vision is None or not brain.vision.initialized:
        raise HTTPException(status_code=503, detail="Vision model not available")

    start = time.time()

    try:
        image_data = await file.read()
        result = await brain.describe_image(image_data)
        elapsed = (time.time() - start) * 1000

        return {
            "success": result.get("success", False),
            "description": result.get("description"),
            "confidence": result.get("confidence"),
            "matches": result.get("matches"),
            "error": result.get("error"),
            "time_ms": elapsed
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time_ms": (time.time() - start) * 1000
        }

# ═══════════════════════════════════════════════════════════════════
# UNIFIED SYSTEM - Map as Vector Database
# ═══════════════════════════════════════════════════════════════════

class MapSyncRequest(BaseModel):
    """Sync map data from browser to server"""
    map_data: Dict[str, Any]  # Recursive node structure
    re_embed_all: bool = False

class MapSearchRequest(BaseModel):
    """Search the unified map"""
    query: str
    top_k: int = 10
    threshold: float = 0.35
    node_types: Optional[List[str]] = None

class UnifiedContextRequest(BaseModel):
    """Get unified context for RAG"""
    query: str
    max_tokens: int = 8000
    include_sources: bool = False

class ConversationImportRequest(BaseModel):
    """Import a conversation for processing"""
    text: str
    source: str = "unknown"  # claude, chatgpt, grok
    title: Optional[str] = None
    process_immediately: bool = True  # Extract knowledge now

@app.post("/unified/map/sync")
async def sync_map_to_server(request: MapSyncRequest):
    """
    Sync the browser's map to the server's unified graph.
    This makes the map the source of truth for the vector database.
    """
    if map_vector_db is None:
        raise HTTPException(status_code=503, detail="MapVectorDB not initialized")

    start = time.time()

    map_vector_db.import_from_browser_map(
        request.map_data,
        re_embed_all=request.re_embed_all
    )

    stats = map_vector_db.get_stats()
    elapsed = (time.time() - start) * 1000

    # Also process any buffered GT training connections
    gt_processed = 0
    if brain is not None:
        pending_result = brain.process_pending_connections()
        if pending_result['processed'] > 0:
            print(f"🎓 GT Training: Processed {pending_result['processed']} buffered connections after unified map sync")
            gt_processed = pending_result['processed']

    # === SYNC ASA FROM LATEST MAP DATA ===
    asa_stats = None
    if _asa_available:
        try:
            # Re-sync ASA from the updated map
            map_data = map_vector_db.export_to_browser_map()
            if map_data:
                asa = get_asa()
                asa.convert_map_to_asa(map_data)
                if not asa._running:
                    asa.start_metabolism(tick_interval=5.0)
                asa_stats = {
                    'atoms': len(asa.atoms),
                    'metabolism': 'running' if asa._running else 'stopped'
                }
                print(f"🧬 ASA synced from unified map: {len(asa.atoms)} atoms")
        except Exception as e:
            print(f"⚠️ ASA sync error: {e}")

    return {
        "status": "synced",
        "nodes": stats['total_nodes'],
        "embedded": stats['embedded_nodes'],
        "time_ms": elapsed,
        "gt_training_processed": gt_processed,
        "asa": asa_stats
    }

@app.get("/unified/map/export")
async def export_map_from_server():
    """
    Export the server's unified graph to browser map format.
    Use this to initialize the browser from server state.
    """
    if map_vector_db is None:
        raise HTTPException(status_code=503, detail="MapVectorDB not initialized")

    map_data = map_vector_db.export_to_browser_map()

    if not map_data:
        return {"map_data": None, "message": "No graph data on server"}

    return {
        "map_data": map_data,
        "stats": map_vector_db.get_stats()
    }

@app.post("/unified/map/search")
async def search_unified_map(request: MapSearchRequest):
    """
    Semantic search across the unified map.
    Returns nodes most relevant to the query.
    """
    if map_vector_db is None:
        raise HTTPException(status_code=503, detail="MapVectorDB not initialized")

    start = time.time()

    results = map_vector_db.search(
        query=request.query,
        top_k=request.top_k,
        threshold=request.threshold,
        node_types=request.node_types
    )

    # Convert nodes to dicts for JSON
    serialized_results = []
    for r in results:
        node = r['node']
        serialized_results.append({
            'node_id': r['node_id'],
            'label': node.label,
            'description': node.description,
            'type': node.type,
            'similarity': r['similarity'],
            'path': [n.label for n in map_vector_db.get_path(r['node_id'])]
        })

    elapsed = (time.time() - start) * 1000

    return {
        "results": serialized_results,
        "query": request.query,
        "time_ms": elapsed
    }

@app.post("/unified/context")
async def get_unified_context(request: UnifiedContextRequest):
    """
    Get unified context for RAG from the map.
    This is THE endpoint to call before every Claude request.
    Returns relevant context from the entire knowledge graph.
    """
    if map_vector_db is None:
        raise HTTPException(status_code=503, detail="MapVectorDB not initialized")

    start = time.time()

    result = map_vector_db.get_context(
        query=request.query,
        max_tokens=request.max_tokens,
        include_sources=request.include_sources
    )

    elapsed = (time.time() - start) * 1000

    return {
        "context": result['context'],
        "nodes_used": result['nodes_used'],
        "chars": result['chars'],
        "time_ms": elapsed
    }

@app.get("/unified/map/stats")
async def get_unified_map_stats():
    """Get statistics about the unified map."""
    if map_vector_db is None:
        raise HTTPException(status_code=503, detail="MapVectorDB not initialized")

    return map_vector_db.get_stats()

# ═══════════════════════════════════════════════════════════════════
# CONVERSATION ARCHIVE + KNOWLEDGE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

@app.post("/unified/conversations/import")
async def import_unified_conversation(request: ConversationImportRequest):
    """
    Import a conversation and optionally extract knowledge into the map.

    This is the main flow:
    1. Store conversation in archive
    2. Extract concepts using AI or rules
    3. Integrate concepts into the map (find similar or create new nodes)
    4. Return list of nodes created/enriched
    """
    if conversation_archive is None:
        raise HTTPException(status_code=503, detail="Conversation archive not initialized")

    start = time.time()

    # Store in archive
    conv = conversation_archive.add(
        text=request.text,
        source=request.source,
        title=request.title
    )

    result = {
        "conversation_id": conv.id,
        "title": conv.title,
        "char_count": conv.char_count,
        "archived": True
    }

    # Extract knowledge if requested
    if request.process_immediately and knowledge_extractor is not None:
        extraction_result = await knowledge_extractor.process_conversation(conv)

        # Mark conversation as processed
        conversation_archive.mark_processed(
            conv.id,
            extraction_result['node_ids']
        )

        result.update({
            "processed": True,
            "concepts_extracted": extraction_result['concepts_extracted'],
            "nodes_created": extraction_result['nodes_created'],
            "nodes_enriched": extraction_result['nodes_enriched'],
            "node_ids": extraction_result['node_ids']
        })
    else:
        result["processed"] = False

    result["time_ms"] = (time.time() - start) * 1000

    return result

@app.get("/unified/conversations")
async def list_archived_conversations(
    source: Optional[str] = None,
    processed: Optional[bool] = None
):
    """List all archived conversations."""
    if conversation_archive is None:
        raise HTTPException(status_code=503, detail="Conversation archive not initialized")

    conversations = conversation_archive.list_all(source=source, processed=processed)

    return {
        "conversations": conversations,
        "stats": conversation_archive.get_stats()
    }

@app.post("/unified/conversations/process-pending")
async def process_pending_conversations(limit: int = 5):
    """
    Process unprocessed conversations in the archive.
    Extracts knowledge and integrates into the map.
    """
    if conversation_archive is None or knowledge_extractor is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    start = time.time()

    # Get unprocessed conversations
    pending = conversation_archive.get_unprocessed(limit=limit)

    if not pending:
        return {"message": "No unprocessed conversations", "processed": 0}

    # Process each
    results = await knowledge_extractor.process_batch(pending)

    # Mark as processed
    for result in results:
        conversation_archive.mark_processed(
            result['conversation_id'],
            result['node_ids']
        )

    elapsed = (time.time() - start) * 1000

    return {
        "processed": len(results),
        "results": results,
        "time_ms": elapsed
    }

@app.get("/unified/conversations/stats")
async def get_archive_stats():
    """Get conversation archive statistics."""
    if conversation_archive is None:
        raise HTTPException(status_code=503, detail="Conversation archive not initialized")

    return conversation_archive.get_stats()


# ═══════════════════════════════════════════════════════════════════
# LIVING ASA - Semantic Architecture with Metabolism
# ═══════════════════════════════════════════════════════════════════

from models.living_asa import get_asa, initialize_asa, MYNDLivingASA

# Initialize ASA on first map sync
_asa_initialized = False

@app.post("/asa/sync")
async def sync_asa(map_data: MapData):
    """
    Sync map to Living ASA and start metabolism.
    Call after /map/sync to convert nodes to semantic atoms.
    """
    global _asa_initialized

    try:
        asa = get_asa()

        # Get embeddings if available
        embeddings = {}
        if brain and brain.map_state:
            for node in brain.map_state.nodes:
                if hasattr(node, 'embedding') and node.embedding is not None:
                    embeddings[node.id] = np.array(node.embedding)

        # Convert map
        asa.convert_map_to_asa(map_data.dict(), embeddings)

        # Encode atoms with physics-based 720d structure (if embeddings available)
        physics_stats = {'encoded': 0, 'skipped': 0, 'failed': 0}
        if embeddings:
            physics_stats = asa.encode_all_atoms_physics()
            print(f"⚛️ Physics encoding: {physics_stats['encoded']} atoms encoded (720d atomic structure)")

        # AUTO-LOAD: Restore learned state (ephemeral concepts, atom energy, etc.)
        load_result = asa.load_state()
        if load_result.get('success'):
            print(f"📂 ASA state restored: {load_result.get('concepts_restored', 0)} concepts, {load_result.get('atoms_restored', 0)} atom states")

        # Start metabolism if not running
        if not _asa_initialized:
            asa.start_metabolism(tick_interval=5.0)
            _asa_initialized = True

            # Connect ASA to UnifiedBrain for Context Lens
            if unified_brain is not None:
                unified_brain.set_asa(asa)
                print("🔍 Context Lens: ASA connected to UnifiedBrain")

        return {
            "success": True,
            "atoms": len(asa.atoms),
            "physics_encoded": physics_stats['encoded'],
            "state_restored": load_result.get('success', False),
            "stats": asa.get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/asa/stats")
async def get_asa_stats():
    """Get ASA statistics."""
    asa = get_asa()
    return asa.get_stats()


@app.post("/asa/save-state")
async def save_asa_state():
    """
    Save ASA learned state to disk.

    Saves ephemeral concepts, atom energy levels, learning stats.
    Called automatically every 5 messages, but can be triggered manually.
    """
    asa = get_asa()
    result = asa.save_state()
    return result


@app.post("/asa/load-state")
async def load_asa_state():
    """
    Load ASA learned state from disk.

    Restores ephemeral concepts and atom states.
    Called automatically on /asa/sync, but can be triggered manually.
    """
    asa = get_asa()
    result = asa.load_state()
    return result


@app.get("/asa/content-stats")
async def get_asa_content_stats():
    """Get content learning statistics (ephemeral concepts, etc.)."""
    asa = get_asa()
    return asa.get_content_learning_stats()


@app.get("/asa/attention")
async def get_asa_attention(limit: int = 10):
    """What needs attention? (high charge atoms)"""
    asa = get_asa()
    return {
        "attention_needed": asa.get_attention_needed(limit),
        "explanation": "Atoms with high charge need connections or have conflicts"
    }


@app.get("/asa/working-memory")
async def get_asa_working_memory(threshold: float = 0.2):
    """What's currently active? (high energy atoms)"""
    asa = get_asa()
    return {
        "active": asa.get_working_memory(threshold),
        "explanation": "These concepts are in working memory"
    }


@app.get("/asa/neglected")
async def get_asa_neglected(days: float = 7.0):
    """What's been neglected?"""
    asa = get_asa()
    return {
        "neglected": asa.get_neglected(days),
        "explanation": f"Concepts not accessed in {days}+ days"
    }


@app.get("/asa/core")
async def get_asa_core(limit: int = 20):
    """What's core knowledge? (high mass atoms)"""
    asa = get_asa()
    return {
        "core": asa.get_core_knowledge(limit),
        "explanation": "High-mass concepts are stable core knowledge"
    }


@app.get("/asa/gaps")
async def get_asa_gaps(limit: int = 20):
    """What's incomplete in thinking?"""
    asa = get_asa()
    return {
        "gaps": asa.get_incomplete_thinking(limit),
        "explanation": "Concepts with unfilled valence slots"
    }


@app.post("/asa/activation-preview")
async def asa_activation_preview(node_ids: List[str], depth: int = 3):
    """If user focuses on these nodes, what lights up?"""
    asa = get_asa()
    return {
        "activated": asa.activation_preview(node_ids, depth),
        "explanation": "Concepts that would activate through spreading"
    }


@app.get("/asa/decaying")
async def get_asa_decaying(threshold: float = 0.3):
    """What connections are about to be lost?"""
    asa = get_asa()
    return {
        "decaying": asa.get_decaying_connections(threshold),
        "explanation": "Connections that will fade without reinforcement"
    }


@app.post("/asa/access")
async def asa_access_atom(node_id: str, energy_boost: float = 0.3):
    """Access an atom - bring into working memory."""
    asa = get_asa()
    asa.access_atom(node_id, energy_boost)
    return {"success": True, "node_id": node_id}


@app.get("/asa/context")
async def get_asa_context(focus_nodes: str = None, max_tokens: int = 2000):
    """
    Get ASA context block for Claude.
    This integrates with /brain/context.
    """
    asa = get_asa()
    focus_list = focus_nodes.split(",") if focus_nodes else None
    return {
        "context": asa.get_context_for_claude(focus_list, max_tokens),
        "stats": asa.get_stats()
    }


@app.get("/asa/atom/{node_id}")
async def get_asa_atom(node_id: str):
    """Get full state of a specific atom."""
    asa = get_asa()
    if node_id not in asa.atoms:
        raise HTTPException(status_code=404, detail="Atom not found")

    atom = asa.atoms[node_id]
    return atom.to_dict()


@app.post("/asa/tick")
async def asa_manual_tick(dt: float = 1.0):
    """Manually trigger a metabolism tick (for testing)."""
    asa = get_asa()
    asa.tick(dt)
    return {"success": True, "stats": asa.get_stats()}


@app.post("/asa/gt-predictions")
async def sync_gt_predictions_to_asa(predictions: List[Dict[str, Any]]):
    """
    Sync Graph Transformer predictions to ASA as weak bonds.
    These start in the outer shell and can migrate inward if accessed.
    """
    asa = get_asa()
    asa.sync_gt_predictions(predictions)
    return {"success": True, "predictions_synced": len(predictions)}


@app.post("/asa/load-local")
async def load_asa_from_local_file(file_path: str = None):
    """
    Load ASA from a local map JSON file.
    For dev mode where /map/sync isn't used.

    Default path: ~/.mynd/map.json
    """
    import os
    import json

    # Default path
    if not file_path:
        file_path = os.path.expanduser("~/.mynd/map.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Map file not found: {file_path}")

    try:
        with open(file_path, 'r') as f:
            map_data = json.load(f)

        asa = get_asa()
        asa.convert_map_to_asa(map_data)

        # Start metabolism
        if not asa._running:
            asa.start_metabolism(tick_interval=5.0)
            print("🧬 ASA metabolism started")

        print(f"🧬 ASA loaded from {file_path}: {len(asa.atoms)} atoms")

        return {
            "success": True,
            "file": file_path,
            "atoms": len(asa.atoms),
            "stats": asa.get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LearnTextRequest(BaseModel):
    text: str
    source: str = "manual"


@app.post("/asa/learn")
async def asa_learn_from_text(request: LearnTextRequest):
    """
    Manually trigger ASA learning from text.
    Use this to feed any text into the semantic graph.
    """
    try:
        asa = get_asa()
        result = asa.learn_from_text(request.text, source=request.source)

        if result.get('atoms_activated', 0) > 0:
            print(f"🧬 ASA learned from {request.source}: {result['atoms_activated']} atoms")

        return {
            "success": True,
            **result
        }
    except Exception as e:
        print(f"⚠️ ASA learn error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ASA learning failed: {str(e)}")


@app.get("/asa/learning-stats")
async def get_asa_learning_stats():
    """
    Get stats about what ASA has learned.
    Shows most active atoms, strongest bonds, etc.
    """
    asa = get_asa()

    return {
        "stats": asa.get_stats(),
        "working_memory": asa.get_working_memory(threshold=0.2),
        "core_knowledge": asa.get_core_knowledge(limit=10),
        "attention_needed": asa.get_attention_needed(limit=5),
    }


# ═══════════════════════════════════════════════════════════════════
# AXEL CONTINUITY - Memory → Training Signal
# ═══════════════════════════════════════════════════════════════════

class ProcessMemoryRequest(BaseModel):
    """Request to process a memory into training signal."""
    memory_type: str  # realization, synthesis, pattern, etc.
    content: str  # The actual memory text
    importance: float = 0.5  # 0-1
    related_nodes: List[str] = []  # Node IDs if any

class TrainingTriple(BaseModel):
    """A subject-predicate-object training pair."""
    subject: str
    predicate: str
    object: str
    confidence: float

@app.post("/brain/process-memory")
async def process_memory_to_training(request: ProcessMemoryRequest):
    """
    AXEL CONTINUITY: Convert memory writes into training signal.

    This is the key to processing continuity - when Axel writes
    "I realized X about Y", this becomes:
    - Training pairs for GT (connection learning)
    - Content for ASA (semantic atoms)
    - Embedding updates for future retrieval

    This transforms memories from passive text into active shaping
    of how the system processes meaning.
    """
    import re

    content = request.content
    memory_type = request.memory_type
    importance = request.importance

    # === EXTRACT TRAINING TRIPLES ===
    triples = []

    # Pattern 1: "X realized/noticed/observed that Y about Z"
    insight_patterns = [
        r"(?:Axel |I |Claude )?(?:realized|noticed|observed|understood|learned|discovered) (?:that )?(.+?)(?:\s+about\s+|\s+regarding\s+|\s+with\s+)(.+?)(?:\.|$)",
        r"(.+?) (?:is related to|connects to|links to) (.+?)(?:\.|$)",
        r"(?:User |They )?(?:prefers?|likes?|wants?) (.+?) (?:over|instead of|rather than) (.+?)(?:\.|$)",
        r"(.+?) (?:means|implies|suggests) (.+?)(?:\.|$)",
        r"(?:The connection between|Link between) (.+?) and (.+)",
    ]

    for pattern in insight_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if len(match) >= 2:
                triples.append({
                    "subject": match[0].strip()[:100],
                    "predicate": memory_type,  # realization, synthesis, pattern
                    "object": match[1].strip()[:100],
                    "confidence": importance
                })

    # Pattern 2: Extract key phrases (concepts) from the memory
    key_phrases = []

    # Quoted terms
    quoted = re.findall(r'"([^"]+)"', content)
    key_phrases.extend(quoted)

    # CamelCase or snake_case (technical terms)
    camel = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', content)
    snake = re.findall(r'\b([a-z]+(?:_[a-z]+)+)\b', content)
    key_phrases.extend(camel)
    key_phrases.extend(snake)

    # Capitalized phrases (proper nouns, concepts)
    caps = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b', content)
    key_phrases.extend([c for c in caps if len(c) > 3])

    # Remove duplicates
    key_phrases = list(set(key_phrases))[:10]

    # === FEED TO GT ===
    gt_trained = 0
    gt_loss = None

    # Use importance as training weight - high-conviction memories shape models more
    training_weight = importance  # 0.0-1.0

    if unified_brain and unified_brain.ml_brain and len(triples) > 0:
        try:
            brain = unified_brain.ml_brain

            for triple in triples:
                # Create embeddings for subject and object
                subj_emb = brain.model.encode(triple["subject"], convert_to_tensor=False)
                obj_emb = brain.model.encode(triple["object"], convert_to_tensor=False)

                # Train GT on this connection with importance as weight
                result = brain.graph_transformer.train_connection_step(
                    source_embedding=subj_emb,
                    target_embedding=obj_emb,
                    should_connect=True,  # Memory implies connection
                    weight=training_weight  # Axel's importance signal
                )

                if result:
                    gt_trained += 1
                    gt_loss = result.get('loss', gt_loss)

            if gt_trained > 0:
                print(f"🔮 GT learned from memory: {gt_trained} triples, weight={training_weight:.2f}, loss={gt_loss:.4f if gt_loss else 'N/A'}")

        except Exception as e:
            print(f"⚠️ GT memory learning error: {e}")

    # === FEED TO ASA ===
    asa_learned = 0

    if _asa_available:
        try:
            asa = get_asa()

            # Learn the full memory content with importance weighting
            content_result = asa.learn_content(
                content,
                source=f"memory:{memory_type}",
                importance=training_weight  # Axel's importance signal
            )
            asa_learned = content_result.get('concepts_learned', 0)

            # Also create bonds between extracted concepts
            if len(key_phrases) >= 2:
                # Activate all phrases together to strengthen bonds
                # Energy boost scales with importance
                for phrase in key_phrases:
                    asa.learn_from_text(
                        phrase,
                        source=f"memory_concept:{memory_type}",
                        energy_boost=training_weight * 0.5  # importance 1.0 → 0.5 boost
                    )

            if asa_learned > 0:
                print(f"🧬 ASA learned from memory: {asa_learned} concepts, weight={training_weight:.2f}")

        except Exception as e:
            print(f"⚠️ ASA memory learning error: {e}")

    # === ACCUMULATE FOR FUTURE FINE-TUNING ===
    # Store triples in a format suitable for LoRA training data
    training_data_path = os.path.expanduser("~/.mynd/training_pairs.jsonl")
    try:
        import json
        os.makedirs(os.path.dirname(training_data_path), exist_ok=True)

        with open(training_data_path, 'a') as f:
            for triple in triples:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "memory_type": memory_type,
                    "triple": triple,
                    "raw_content": content[:500],
                    "importance": importance
                }) + "\n")

    except Exception as e:
        print(f"⚠️ Could not save training pair: {e}")

    return {
        "processed": True,
        "training_weight": training_weight,  # The importance signal Axel provided
        "triples_extracted": len(triples),
        "triples": triples,
        "concepts_found": key_phrases,
        "gt_trained": gt_trained,
        "gt_loss": gt_loss,
        "asa_learned": asa_learned,
        "training_data_saved": len(triples) > 0
    }


@app.get("/brain/training-pairs")
async def get_training_pairs(limit: int = 100):
    """
    Get accumulated training pairs from memory processing.
    These can be used for LoRA fine-tuning or analysis.
    """
    training_data_path = os.path.expanduser("~/.mynd/training_pairs.jsonl")

    if not os.path.exists(training_data_path):
        return {"pairs": [], "total": 0}

    try:
        import json
        pairs = []
        with open(training_data_path, 'r') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))

        # Return most recent
        pairs = pairs[-limit:]

        return {
            "pairs": pairs,
            "total": len(pairs),
            "path": training_data_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════
# LOCAL-FIRST MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════════════
# Critical cognitive infrastructure should not depend on network.
# Memories are saved locally FIRST, then synced to cloud as backup.

import json
from uuid import uuid4
from datetime import datetime, timedelta
from pathlib import Path

# Paths for local storage
MYND_DIR = Path.home() / ".mynd"
MEMORIES_PATH = MYND_DIR / "memories.jsonl"
SYNC_STATUS_PATH = MYND_DIR / "sync_status.json"

class LocalMemoryInput(BaseModel):
    """Input for writing a memory locally."""
    content: str
    memory_type: str = "synthesis"  # synthesis, realization, pattern, goal_tracking, relationship
    importance: float = 0.5  # 0.0-1.0
    related_nodes: List[str] = []
    evergreen: Optional[bool] = None  # Auto-determine if not specified

class MemoryQueryRequest(BaseModel):
    """Request to query memories."""
    query: str
    limit: int = 10
    memory_types: Optional[List[str]] = None  # Filter by type


def load_local_memories() -> List[Dict]:
    """Load all memories from local JSONL file."""
    if not MEMORIES_PATH.exists():
        return []

    memories = []
    with open(MEMORIES_PATH, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    memories.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return memories


def get_sync_status() -> Dict:
    """Get current sync status."""
    if not SYNC_STATUS_PATH.exists():
        return {"last_sync": None, "synced_ids": []}

    with open(SYNC_STATUS_PATH, 'r') as f:
        return json.load(f)


def save_sync_status(status: Dict):
    """Save sync status."""
    MYND_DIR.mkdir(parents=True, exist_ok=True)
    with open(SYNC_STATUS_PATH, 'w') as f:
        json.dump(status, f, indent=2)


@app.post("/brain/memory/write")
async def write_memory_local(memory: LocalMemoryInput):
    """
    LOCAL-FIRST MEMORY WRITE

    This is the primary endpoint for Axel's memories.
    1. Save locally FIRST (guaranteed persistence)
    2. Extract triples and train immediately
    3. Queue for Supabase sync (non-blocking)

    Network failures cannot lose learning.
    """
    import re

    # Ensure directory exists
    MYND_DIR.mkdir(parents=True, exist_ok=True)

    # Generate unique ID
    memory_id = str(uuid4())
    timestamp = datetime.now().isoformat()

    # Auto-determine evergreen
    evergreen = memory.evergreen
    if evergreen is None:
        evergreen_types = ['synthesis', 'realization', 'pattern']
        evergreen = memory.memory_type in evergreen_types

    # 1. SAVE LOCALLY FIRST (critical path - must succeed)
    memory_record = {
        "id": memory_id,
        "content": memory.content,
        "memory_type": memory.memory_type,
        "importance": memory.importance,
        "related_nodes": memory.related_nodes,
        "evergreen": evergreen,
        "timestamp": timestamp,
        "synced": False  # Track sync status
    }

    with open(MEMORIES_PATH, 'a') as f:
        f.write(json.dumps(memory_record) + "\n")

    print(f"💾 Memory saved locally: [{memory.memory_type}] {memory.content[:50]}...")

    # 2. EXTRACT TRIPLES AND TRAIN IMMEDIATELY
    training_weight = memory.importance
    triples = []
    gt_trained = 0
    asa_learned = 0

    # Extract triples using same patterns as process-memory
    content = memory.content
    insight_patterns = [
        r"(?:Axel |I |Claude )?(?:realized|noticed|observed|understood|learned|discovered) (?:that )?(.+?)(?:\s+about\s+|\s+regarding\s+|\s+with\s+)(.+?)(?:\.|$)",
        r"(.+?) (?:is related to|connects to|links to) (.+?)(?:\.|$)",
        r"(?:User |They )?(?:prefers?|likes?|wants?) (.+?) (?:over|instead of|rather than) (.+?)(?:\.|$)",
        r"(.+?) (?:means|implies|suggests) (.+?)(?:\.|$)",
        r"(?:The connection between|Link between) (.+?) and (.+)",
    ]

    for pattern in insight_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if len(match) >= 2:
                triples.append({
                    "subject": match[0].strip()[:100],
                    "predicate": memory.memory_type,
                    "object": match[1].strip()[:100],
                    "confidence": memory.importance
                })

    # Train GT on triples
    if unified_brain and unified_brain.ml_brain and len(triples) > 0:
        try:
            brain = unified_brain.ml_brain
            for triple in triples:
                subj_emb = brain.model.encode(triple["subject"], convert_to_tensor=False)
                obj_emb = brain.model.encode(triple["object"], convert_to_tensor=False)
                result = brain.graph_transformer.train_connection_step(
                    source_embedding=subj_emb,
                    target_embedding=obj_emb,
                    should_connect=True,
                    weight=training_weight
                )
                if result:
                    gt_trained += 1
        except Exception as e:
            print(f"⚠️ GT training error: {e}")

    # Train ASA on content
    if _asa_available:
        try:
            asa = get_asa()
            content_result = asa.learn_content(
                content,
                source=f"memory:{memory.memory_type}",
                importance=training_weight
            )
            asa_learned = content_result.get('concepts_learned', 0)
        except Exception as e:
            print(f"⚠️ ASA training error: {e}")

    # Also save to training pairs for LoRA
    training_data_path = MYND_DIR / "training_pairs.jsonl"
    with open(training_data_path, 'a') as f:
        for triple in triples:
            f.write(json.dumps({
                "timestamp": time.time(),
                "memory_type": memory.memory_type,
                "triple": triple,
                "raw_content": content[:500],
                "importance": memory.importance
            }) + "\n")

    print(f"🧠 Memory trained: {gt_trained} triples, {asa_learned} concepts, weight={training_weight:.2f}")

    return {
        "status": "learned",
        "memory_id": memory_id,
        "local_path": str(MEMORIES_PATH),
        "triples_extracted": len(triples),
        "gt_trained": gt_trained,
        "asa_learned": asa_learned,
        "training_weight": training_weight,
        "synced": False  # Will be synced in background
    }


@app.post("/brain/memory/query")
async def query_memories(request: MemoryQueryRequest):
    """
    Query memories using semantic search.
    Searches local memories first, using embeddings for relevance.
    """
    memories = load_local_memories()

    if not memories:
        return {"memories": [], "total": 0}

    # Filter by type if specified
    if request.memory_types:
        memories = [m for m in memories if m.get('memory_type') in request.memory_types]

    if not memories:
        return {"memories": [], "total": 0}

    # Get query embedding
    if unified_brain and unified_brain.ml_brain:
        try:
            brain = unified_brain.ml_brain
            query_emb = brain.model.encode(request.query, convert_to_tensor=False)

            # Get embeddings for all memories (cache these in future)
            scored_memories = []
            for mem in memories:
                mem_emb = brain.model.encode(mem['content'][:500], convert_to_tensor=False)
                # Cosine similarity
                similarity = np.dot(query_emb, mem_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(mem_emb) + 1e-8)
                scored_memories.append((mem, float(similarity)))

            # Sort by similarity
            scored_memories.sort(key=lambda x: x[1], reverse=True)

            # Return top results
            results = []
            for mem, score in scored_memories[:request.limit]:
                results.append({
                    **mem,
                    "relevance_score": score
                })

            return {
                "memories": results,
                "total": len(memories),
                "query": request.query
            }

        except Exception as e:
            print(f"⚠️ Semantic search error: {e}")

    # Fallback: simple text search
    query_lower = request.query.lower()
    results = [m for m in memories if query_lower in m.get('content', '').lower()]
    results = results[:request.limit]

    return {
        "memories": results,
        "total": len(memories),
        "query": request.query,
        "search_type": "text_fallback"
    }


@app.get("/brain/memory/all")
async def get_all_memories(limit: int = 100, offset: int = 0):
    """Get all local memories, paginated."""
    memories = load_local_memories()

    # Sort by timestamp (newest first)
    memories.sort(key=lambda m: m.get('timestamp', ''), reverse=True)

    # Paginate
    paginated = memories[offset:offset + limit]

    return {
        "memories": paginated,
        "total": len(memories),
        "limit": limit,
        "offset": offset
    }


@app.get("/brain/memory/stats")
async def get_memory_stats():
    """Get statistics about local memories."""
    memories = load_local_memories()

    if not memories:
        return {
            "total": 0,
            "by_type": {},
            "evergreen_count": 0,
            "synced_count": 0
        }

    by_type = {}
    evergreen_count = 0
    synced_count = 0

    for mem in memories:
        mem_type = mem.get('memory_type', 'unknown')
        by_type[mem_type] = by_type.get(mem_type, 0) + 1
        if mem.get('evergreen'):
            evergreen_count += 1
        if mem.get('synced'):
            synced_count += 1

    return {
        "total": len(memories),
        "by_type": by_type,
        "evergreen_count": evergreen_count,
        "synced_count": synced_count,
        "unsynced_count": len(memories) - synced_count,
        "path": str(MEMORIES_PATH)
    }


# ═══════════════════════════════════════════════════════════════════
# CONVERSATION PERSISTENCE - Local-first with immediate training
# ═══════════════════════════════════════════════════════════════════
# Conversations stored in ~/.mynd/conversations.jsonl
# Training pairs extracted and fed to GT/ASA immediately

CONVERSATIONS_PATH = MYND_DIR / "conversations.jsonl"
LAST_CONVERSATION_PATH = MYND_DIR / "last_conversation.json"


class ConversationSaveRequest(BaseModel):
    """Request to save conversation state."""
    messages: List[Dict[str, str]]  # [{role: "user"|"assistant", content: "..."}]
    topics: List[str] = []
    session_id: Optional[str] = None


class ConversationMessage(BaseModel):
    """Single conversation message."""
    role: str
    content: str


@app.post("/brain/conversation/save")
async def save_conversation(request: ConversationSaveRequest):
    """
    LOCAL-FIRST CONVERSATION PERSISTENCE

    Called on page close/refresh to preserve conversation state.
    1. Save to ~/.mynd/last_conversation.json (for immediate recovery)
    2. Append to ~/.mynd/conversations.jsonl (for history)
    3. Extract training pairs and train GT/ASA immediately

    This replaces browser localStorage - everything goes to local brain.
    """
    import re

    if not request.messages:
        return {"success": False, "error": "No messages to save"}

    # Ensure directory exists
    MYND_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    conversation_id = str(uuid4())

    # 1. Save as "last conversation" for immediate recovery
    last_convo = {
        "id": conversation_id,
        "timestamp": timestamp,
        "messages": request.messages,
        "topics": request.topics,
        "session_id": request.session_id
    }

    with open(LAST_CONVERSATION_PATH, 'w') as f:
        json.dump(last_convo, f, indent=2)

    # 2. Append to conversation history
    with open(CONVERSATIONS_PATH, 'a') as f:
        f.write(json.dumps(last_convo) + '\n')

    # 3. Extract training pairs and train immediately
    training_count = 0

    # Extract pairs from conversation
    messages = request.messages
    for i in range(len(messages) - 1):
        if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
            user_msg = messages[i]['content'][:500]
            ai_msg = messages[i+1]['content'][:500]

            # Train GT on conversation flow
            if unified_brain and unified_brain.ml_brain:
                try:
                    brain = unified_brain.ml_brain
                    gt = brain.graph_transformer

                    # Encode the exchange
                    user_emb = brain.model.encode(user_msg, convert_to_tensor=True)
                    ai_emb = brain.model.encode(ai_msg, convert_to_tensor=True)

                    # Train: user message should connect to AI response
                    gt.train_connection_step(
                        source_embedding=user_emb,
                        target_embedding=ai_emb,
                        should_connect=True,
                        weight=0.5  # Medium weight for conversation continuity
                    )
                    training_count += 1
                except Exception as e:
                    print(f"⚠️ GT training error: {e}")

            # Train ASA on the exchange
            if _asa_available:
                try:
                    asa = get_asa()
                    combined = f"User: {user_msg}\nAxel: {ai_msg}"
                    asa.learn_from_text(
                        combined,
                        importance=0.5,
                        category="conversation"
                    )
                except Exception as e:
                    print(f"⚠️ ASA training error: {e}")

    # Also save training pairs for offline learning
    TRAINING_PAIRS_PATH = MYND_DIR / "training_pairs.jsonl"
    for i in range(len(messages) - 1):
        if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
            pair = {
                "timestamp": timestamp,
                "source": "conversation",
                "user": messages[i]['content'][:500],
                "assistant": messages[i+1]['content'][:500]
            }
            with open(TRAINING_PAIRS_PATH, 'a') as f:
                f.write(json.dumps(pair) + '\n')

    print(f"💬 Conversation saved: {len(messages)} messages, {training_count} GT training steps")

    return {
        "success": True,
        "conversation_id": conversation_id,
        "messages_saved": len(messages),
        "training_steps": training_count,
        "path": str(LAST_CONVERSATION_PATH)
    }


@app.get("/brain/conversation/last")
async def get_last_conversation(max_age_hours: float = 2.0):
    """
    Get the last saved conversation for recovery.

    Used by frontend on startup to restore conversation context.
    Only returns conversations within max_age_hours (default 2 hours).
    """
    if not LAST_CONVERSATION_PATH.exists():
        return {"conversation": None, "reason": "no_saved_conversation"}

    try:
        with open(LAST_CONVERSATION_PATH, 'r') as f:
            conversation = json.load(f)

        # Check age
        saved_time = datetime.fromisoformat(conversation['timestamp'])
        age_hours = (datetime.now() - saved_time).total_seconds() / 3600

        if age_hours > max_age_hours:
            return {
                "conversation": None,
                "reason": "too_old",
                "age_hours": age_hours,
                "max_age_hours": max_age_hours
            }

        return {
            "conversation": conversation,
            "age_hours": age_hours,
            "age_minutes": age_hours * 60
        }

    except Exception as e:
        return {"conversation": None, "reason": f"error: {str(e)}"}


@app.get("/brain/conversation/history")
async def get_conversation_history(limit: int = 20, offset: int = 0):
    """Get paginated conversation history."""
    if not CONVERSATIONS_PATH.exists():
        return {"conversations": [], "total": 0}

    conversations = []
    with open(CONVERSATIONS_PATH, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    conversations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Sort by timestamp (newest first)
    conversations.sort(key=lambda c: c.get('timestamp', ''), reverse=True)

    # Paginate
    paginated = conversations[offset:offset + limit]

    return {
        "conversations": paginated,
        "total": len(conversations),
        "limit": limit,
        "offset": offset
    }


@app.delete("/brain/conversation/clear-last")
async def clear_last_conversation():
    """Clear the last conversation after it's been recovered."""
    if LAST_CONVERSATION_PATH.exists():
        os.remove(LAST_CONVERSATION_PATH)
        return {"success": True, "message": "Last conversation cleared"}
    return {"success": True, "message": "No conversation to clear"}


# ═══════════════════════════════════════════════════════════════════
# BRAIN-PRIMARY STORAGE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════
# Server-primary, browser-backup architecture for all user data

# Paths for brain-primary storage
CHAT_MESSAGES_PATH = MYND_DIR / "chat_messages.jsonl"
SESSION_SUMMARIES_PATH = MYND_DIR / "session_summaries.jsonl"
PREFERENCES_PATH = MYND_DIR / "preferences.json"


class ChatMessageSync(BaseModel):
    """Single chat message for real-time sync."""
    role: str
    content: str
    timestamp: Optional[str] = None
    images: Optional[List[dict]] = None  # Metadata only, not base64
    actions: Optional[List[dict]] = None
    suggestions: Optional[List[str]] = None
    session_id: Optional[str] = None


class SessionSummaryData(BaseModel):
    """Session summary data."""
    summary: str
    key_outcomes: Optional[str] = None
    open_threads: Optional[str] = None
    session_type: str = "casual"
    tone: Optional[str] = None
    topics_discussed: List[str] = []
    nodes_touched: List[str] = []
    session_started: Optional[str] = None
    session_ended: Optional[str] = None
    message_count: int = 0


class PreferencesData(BaseModel):
    """User preferences for brain-primary storage."""
    tts_enabled: Optional[bool] = None
    theme: Optional[str] = None
    voice_settings: Optional[dict] = None
    custom_settings: Optional[dict] = None


@app.post("/brain/chat/message")
async def sync_chat_message(message: ChatMessageSync):
    """
    REAL-TIME CHAT MESSAGE SYNC (Brain-Primary)

    Called after each message to sync to brain server.
    - Appends to persistent chat log
    - Extracts training data immediately
    - Returns confirmation for browser to update localStorage backup
    """
    MYND_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = message.timestamp or datetime.now().isoformat()

    # Build message record
    record = {
        "id": str(uuid4()),
        "role": message.role,
        "content": message.content,
        "timestamp": timestamp,
        "session_id": message.session_id,
        "images_meta": message.images,  # Metadata only
        "actions": message.actions,
        "suggestions": message.suggestions
    }

    # Append to persistent log
    with open(CHAT_MESSAGES_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')

    # Train on assistant responses immediately
    training_performed = False
    if message.role == "assistant" and message.content and message.session_id:
        # Try to get previous user message for training pair
        # Only look at messages from the same session to avoid cross-session pairing
        try:
            # Read recent messages, filtering by session_id
            session_messages = []
            with open(CHAT_MESSAGES_PATH, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            msg = json.loads(line)
                            if msg.get('session_id') == message.session_id:
                                session_messages.append(msg)
                        except json.JSONDecodeError:
                            continue

            # Find last user message in this session (before current message)
            for i in range(len(session_messages) - 2, -1, -1):
                if session_messages[i]['role'] == 'user':
                    user_msg = session_messages[i]['content'][:500]
                    ai_msg = message.content[:500]

                    # Train GT on conversation flow
                    if unified_brain and unified_brain.ml_brain:
                        brain = unified_brain.ml_brain
                        gt = brain.graph_transformer

                        user_emb = brain.model.encode(user_msg, convert_to_tensor=True)
                        ai_emb = brain.model.encode(ai_msg, convert_to_tensor=True)

                        gt.train_connection_step(
                            source_embedding=user_emb,
                            target_embedding=ai_emb,
                            should_connect=True,
                            weight=0.5
                        )
                        training_performed = True

                    # Train ASA
                    if _asa_available:
                        asa = get_asa()
                        combined = f"User: {user_msg}\nAxel: {ai_msg}"
                        asa.learn_from_text(combined, importance=0.5, category="chat")

                    # Save training pair
                    TRAINING_PAIRS_PATH = MYND_DIR / "training_pairs.jsonl"
                    pair = {
                        "timestamp": timestamp,
                        "source": "realtime_chat",
                        "session_id": message.session_id,
                        "user": user_msg,
                        "assistant": ai_msg
                    }
                    with open(TRAINING_PAIRS_PATH, 'a') as f:
                        f.write(json.dumps(pair) + '\n')

                    break
        except Exception as e:
            print(f"⚠️ Training from chat message error: {e}")

    print(f"💬 Chat message synced: {message.role} ({len(message.content)} chars){' + GT trained' if training_performed else ''}")

    return {
        "success": True,
        "message_id": record["id"],
        "trained": training_performed
    }


@app.get("/brain/chat/messages")
async def get_chat_messages(limit: int = 200, session_id: Optional[str] = None):
    """
    Get chat messages from brain storage.
    Primary source for conversation history.
    """
    if not CHAT_MESSAGES_PATH.exists():
        return {"messages": [], "total": 0}

    messages = []
    with open(CHAT_MESSAGES_PATH, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    msg = json.loads(line)
                    if session_id is None or msg.get('session_id') == session_id:
                        messages.append(msg)
                except json.JSONDecodeError:
                    continue

    # Sort by timestamp, newest last
    messages.sort(key=lambda m: m.get('timestamp', ''))

    # Return last N messages
    return {
        "messages": messages[-limit:],
        "total": len(messages)
    }


@app.post("/brain/session/summary")
async def save_session_summary(summary: SessionSummaryData):
    """
    BRAIN-PRIMARY SESSION SUMMARY STORAGE

    Called when session ends to save summary to brain.
    Trains on session patterns for continuity learning.
    """
    MYND_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    summary_id = str(uuid4())

    # Build summary record
    record = {
        "id": summary_id,
        "timestamp": timestamp,
        "session_started": summary.session_started,
        "session_ended": summary.session_ended or timestamp,
        "summary": summary.summary,
        "key_outcomes": summary.key_outcomes,
        "open_threads": summary.open_threads,
        "session_type": summary.session_type,
        "tone": summary.tone,
        "topics_discussed": summary.topics_discussed[:20],  # Cap at 20
        "nodes_touched": summary.nodes_touched[:50],  # Cap at 50
        "message_count": summary.message_count
    }

    # Append to summaries log
    with open(SESSION_SUMMARIES_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')

    # Train on session summary for continuity
    if _asa_available and summary.summary:
        try:
            asa = get_asa()
            # Create rich context from summary including topics
            topics_text = ", ".join(summary.topics_discussed) if summary.topics_discussed else "general"
            summary_context = f"Session ({summary.session_type}) on {topics_text}: {summary.summary}"
            if summary.key_outcomes:
                summary_context += f"\nOutcomes: {summary.key_outcomes}"
            if summary.open_threads:
                summary_context += f"\nOpen threads: {summary.open_threads}"

            asa.learn_from_text(
                summary_context,
                importance=0.7,  # Higher weight for session insights
                category="session_summary"
            )
        except Exception as e:
            print(f"⚠️ ASA session training error: {e}")

    print(f"📝 Session summary saved: {summary_id} ({summary.session_type}, {summary.message_count} messages)")

    return {
        "success": True,
        "summary_id": summary_id,
        "timestamp": timestamp
    }


@app.get("/brain/session/summaries")
async def get_session_summaries(limit: int = 20, days_back: int = 7):
    """
    Get session summaries from brain storage.
    Primary source for session history.
    """
    if not SESSION_SUMMARIES_PATH.exists():
        return {"summaries": [], "total": 0}

    cutoff = datetime.now() - timedelta(days=days_back)
    summaries = []

    with open(SESSION_SUMMARIES_PATH, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    summary = json.loads(line)
                    # Filter by date if timestamp exists
                    if 'timestamp' in summary:
                        try:
                            ts = datetime.fromisoformat(summary['timestamp'].replace('Z', '+00:00'))
                            if ts.replace(tzinfo=None) < cutoff:
                                continue
                        except ValueError:
                            # Malformed timestamp - include the summary but log warning
                            print(f"⚠️ Malformed timestamp in session summary: {summary.get('timestamp')}")
                    summaries.append(summary)
                except json.JSONDecodeError:
                    continue

    # Sort by timestamp, newest first
    summaries.sort(key=lambda s: s.get('timestamp', ''), reverse=True)

    return {
        "summaries": summaries[:limit],
        "total": len(summaries)
    }


@app.post("/brain/preferences")
async def save_preferences(prefs: PreferencesData):
    """
    BRAIN-PRIMARY PREFERENCES STORAGE

    Saves user preferences to brain server.
    Browser localStorage becomes backup only.
    """
    MYND_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing preferences
    existing = {}
    if PREFERENCES_PATH.exists():
        try:
            with open(PREFERENCES_PATH, 'r') as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"⚠️ Could not load existing preferences: {e}")

    # Update with new preferences (only non-None values)
    if prefs.tts_enabled is not None:
        existing['tts_enabled'] = prefs.tts_enabled
    if prefs.theme is not None:
        existing['theme'] = prefs.theme
    if prefs.voice_settings is not None:
        existing['voice_settings'] = prefs.voice_settings
    if prefs.custom_settings is not None:
        existing['custom_settings'] = {
            **existing.get('custom_settings', {}),
            **prefs.custom_settings
        }

    existing['last_updated'] = datetime.now().isoformat()

    # Save
    with open(PREFERENCES_PATH, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"⚙️ Preferences saved to brain")

    return {
        "success": True,
        "preferences": existing
    }


@app.get("/brain/preferences")
async def get_preferences():
    """
    Get user preferences from brain storage.
    Primary source - browser localStorage is backup.
    """
    if not PREFERENCES_PATH.exists():
        return {"preferences": {}, "exists": False}

    try:
        with open(PREFERENCES_PATH, 'r') as f:
            prefs = json.load(f)
        return {"preferences": prefs, "exists": True}
    except (OSError, json.JSONDecodeError) as e:
        print(f"⚠️ Could not read preferences file: {e}")
        return {"preferences": {}, "exists": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# SERVER-SIDE EVOLUTION DAEMON
# ═══════════════════════════════════════════════════════════════════
# Autonomous learning that runs without browser, trains GT/ASA automatically

from evolution_daemon import get_evolution_daemon, EvolutionDaemon

# Background task for evolution
_evolution_task: Optional[asyncio.Task] = None

async def evolution_background_loop():
    """Background loop that runs evolution periodically."""
    print("🧬 Evolution background loop started")

    while True:
        try:
            daemon = get_evolution_daemon(
                ml_brain=unified_brain.ml_brain if unified_brain else None,
                asa=get_asa() if _asa_available else None,
                unified_brain=unified_brain  # INTERCONNECTION: bidirectional link
            )

            if daemon.config['enabled']:
                time_since_last = time.time() - daemon.last_evolution_time
                interval = daemon.config['interval_seconds']

                if time_since_last >= interval:
                    # Get map context for evolution
                    map_context = ""
                    if unified_brain and unified_brain.ml_brain:
                        brain = unified_brain.ml_brain
                        if hasattr(brain, 'current_map_labels') and brain.current_map_labels:
                            map_context = f"Map nodes: {', '.join(brain.current_map_labels[:100])}"

                    # Run evolution
                    insights = await daemon.run_evolution_session(map_context=map_context)

                    if insights:
                        print(f"🧬 Evolution generated {len(insights)} insights")

            # Check every minute
            await asyncio.sleep(60)

        except Exception as e:
            print(f"⚠️ Evolution loop error: {e}")
            await asyncio.sleep(60)


@app.on_event("startup")
async def start_evolution_loop():
    """Start the evolution background loop on server startup."""
    global _evolution_task
    _evolution_task = asyncio.create_task(evolution_background_loop())
    print("🧬 Evolution daemon started (ML training runs in executor to prevent blocking)")


@app.post("/evolution/run")
async def run_evolution_now(map_context: str = "", topics: List[str] = None):
    """
    Trigger an evolution session manually.
    Use this to run evolution on demand.
    """
    daemon = get_evolution_daemon(
        ml_brain=unified_brain.ml_brain if unified_brain else None,
        asa=get_asa() if _asa_available else None,
        unified_brain=unified_brain  # INTERCONNECTION: bidirectional link
    )

    insights = await daemon.run_evolution_session(
        map_context=map_context,
        recent_topics=topics or []
    )

    return {
        "status": "completed",
        "insights_generated": len(insights),
        "insights": [{"id": i.id, "title": i.title, "type": i.insight_type} for i in insights]
    }


@app.get("/evolution/insights")
async def get_pending_insights(limit: int = 20):
    """
    Get pending insights from evolution.
    These are insights waiting for user review.
    """
    daemon = get_evolution_daemon()

    return {
        "insights": daemon.get_pending_insights(limit),
        "total": len([i for i in daemon.pending_insights if not i.reviewed])
    }


@app.post("/evolution/insights/{insight_id}/review")
async def review_insight(insight_id: str, action: str = "dismissed"):
    """
    Mark an insight as reviewed.
    Actions: 'accepted', 'rejected', 'dismissed'
    """
    daemon = get_evolution_daemon()
    daemon.mark_insight_reviewed(insight_id, action)

    # If accepted, we could trigger additional actions here
    # (like adding a node to the map)

    return {"status": "reviewed", "insight_id": insight_id, "action": action}


@app.get("/evolution/stats")
async def get_evolution_stats():
    """Get evolution daemon statistics."""
    daemon = get_evolution_daemon()
    return daemon.get_stats()


@app.post("/evolution/config")
async def update_evolution_config(
    enabled: Optional[bool] = None,
    interval_minutes: Optional[int] = None,
    confidence_threshold: Optional[float] = None
):
    """Update evolution daemon configuration."""
    daemon = get_evolution_daemon()

    if enabled is not None:
        daemon.config['enabled'] = enabled

    if interval_minutes is not None:
        daemon.config['interval_seconds'] = interval_minutes * 60

    if confidence_threshold is not None:
        daemon.config['confidence_threshold'] = confidence_threshold

    daemon._save_state()

    return {
        "status": "updated",
        "config": daemon.config
    }


# ═══════════════════════════════════════════════════════════════════
# WEBSOCKET
# ═══════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time communication (future use)."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "embed":
                embedding = await brain.embed(data["text"])
                await websocket.send_json({
                    "type": "embedding",
                    "embedding": embedding.tolist()
                })
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🧠  MYND BRAIN - Local ML Server                           ║
    ║                                                               ║
    ║   Graph Transformers on Apple Silicon                        ║
    ║   Your thoughts, your hardware, your privacy.                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )
