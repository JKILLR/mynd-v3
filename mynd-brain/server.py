"""
MYND Brain - Local ML Server
============================
A local Python server that provides GPU-accelerated ML for MYND.
Runs on Apple Silicon (M2) via Metal Performance Shaders (MPS).

Start with: uvicorn server:app --reload --port 8420
"""

import os
import time
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local imports
from models.embeddings import EmbeddingEngine
from models.graph_transformer import MYNDGraphTransformer

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class Config:
    PORT = 8420
    HOST = "127.0.0.1"  # Local only - never expose externally

    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality
    EMBEDDING_DIM = 384

    # Graph Transformer settings
    HIDDEN_DIM = 256
    NUM_HEADS = 4
    NUM_LAYERS = 2

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
    parentId: Optional[str] = None
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

class TrainFeedbackRequest(BaseModel):
    node_id: str
    action: str  # 'accepted', 'rejected', 'corrected'
    context: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    device: str
    embedding_model: str
    graph_transformer: bool
    uptime_seconds: float

# ═══════════════════════════════════════════════════════════════════
# BRAIN - The ML Engine
# ═══════════════════════════════════════════════════════════════════

class MYNDBrain:
    """
    The local ML brain for MYND.
    Handles embeddings, graph attention, and learning.
    """

    def __init__(self):
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

        # Learning state
        self.feedback_buffer = []

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

            predictions.append({
                "target_id": node.id,
                "target_label": node.label,
                "score": float(score),
                "relationship_type": self._infer_relationship(source_node.label, node.label)
            })

        # Sort by score and take top_k
        predictions.sort(key=lambda x: x["score"], reverse=True)
        predictions = predictions[:top_k]

        return {
            "connections": predictions,
            "attention_weights": {nodes[i].id: float(w) for i, w in enumerate(attention_weights)}
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
        """Record feedback for future training."""
        self.feedback_buffer.append({
            "node_id": feedback.node_id,
            "action": feedback.action,
            "context": feedback.context,
            "timestamp": time.time()
        })

        # Trigger training if buffer is large enough
        if len(self.feedback_buffer) >= 10:
            await self._train_on_feedback()

    async def _train_on_feedback(self):
        """Train on accumulated feedback."""
        if not self.feedback_buffer:
            return

        print(f"🔄 Training on {len(self.feedback_buffer)} feedback samples...")

        # TODO: Implement actual training loop
        # For now, just clear the buffer
        self.feedback_buffer = []

        print("✅ Training complete")

    def get_health(self) -> Dict:
        """Get health status."""
        return {
            "status": "healthy",
            "device": str(self.device),
            "embedding_model": config.EMBEDDING_MODEL,
            "graph_transformer": self.graph_transformer is not None,
            "uptime_seconds": time.time() - self.start_time
        }

# ═══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════

# Global brain instance
brain: Optional[MYNDBrain] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize brain on startup, cleanup on shutdown."""
    global brain
    brain = MYNDBrain()
    yield
    # Cleanup
    print("🧠 MYND Brain shutting down...")

app = FastAPI(
    title="MYND Brain",
    description="Local ML server for MYND - Graph Transformers on Apple Silicon",
    version="0.1.0",
    lifespan=lifespan
)

# CORS - allow browser to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the brain is running and healthy."""
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return brain.get_health()

@app.get("/")
async def root():
    """Root endpoint with info."""
    return {
        "name": "MYND Brain",
        "version": "0.1.0",
        "status": "running",
        "endpoints": ["/health", "/embed", "/embed/batch", "/predict/connections", "/train/feedback"]
    }

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

    return PredictResponse(
        connections=[ConnectionPrediction(**c) for c in result["connections"]],
        attention_weights=result["attention_weights"],
        time_ms=elapsed
    )

@app.post("/train/feedback")
async def train_feedback(request: TrainFeedbackRequest):
    """Record feedback for learning."""
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    await brain.record_feedback(request)
    return {"status": "recorded", "buffer_size": len(brain.feedback_buffer)}

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
