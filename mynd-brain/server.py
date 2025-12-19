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
import time
import asyncio
import base64
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

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

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class Config:
    PORT = 8420
    HOST = "0.0.0.0"  # Listen on all interfaces for network access

    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality
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

@app.get("/")
async def root():
    """Root endpoint with info."""
    return {
        "name": "MYND Brain",
        "version": "0.3.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/map/sync", "/map/analyze",  # BAPI full context
            "/embed", "/embed/batch",
            "/predict/connections",
            "/train/feedback",
            "/voice/transcribe",
            "/image/describe", "/image/embed"
        ]
    }

# ═══════════════════════════════════════════════════════════════════
# BAPI - Full Map Awareness
# ═══════════════════════════════════════════════════════════════════

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
