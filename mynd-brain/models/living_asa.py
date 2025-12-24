"""
Hybrid ASE + Living ASA for MYND
=================================
Combines Atomic Semantic Embeddings (vector-based) with Living ASA (graph-based).

ASE Components (learned vectors):
- Nuclear vector: Stable identity ("what IS this concept")
- Shell vector: Contextual variation ("how is it used")
- Semantic charge: Polarity (-1 to +1) with magnitude
- Charge propagation: Negation flipping, compositional logic

Living ASA Components (structural):
- Bond shells: Proximity layers (1=core to 4=peripheral)
- Energy: Working memory activation (decays over time)
- Mass: Knowledge stability (grows with age/use)
- Typed bonds: IS_A, CAUSES, SUPPORTS, etc.
- Metabolism: Continuous decay, migration, strengthening

Integration points:
- /map/sync → convert_map_to_asa()
- /brain/context → learns from every message
- /brain/receive-from-claude → learns from Axel
- Background → metabolism heartbeat (5s tick)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
import time
import math
import threading
import logging
import re

logger = logging.getLogger(__name__)


# =============================================================================
# ASE: ATOMIC SEMANTIC ENCODER (Neural Component)
# =============================================================================

class AtomicEncoder(nn.Module):
    """
    Minimal Atomic Semantic Encoder.

    Encodes text into:
    - Nuclear vector (stable identity)
    - Shell vector (contextual)
    - Charge (polarity + magnitude)

    ~500K parameters, runs on CPU.
    """

    def __init__(self, input_dim: int = 384, nuclear_dim: int = 64,
                 shell_dim: int = 128, device: str = None):
        super().__init__()

        self.input_dim = input_dim
        self.nuclear_dim = nuclear_dim
        self.shell_dim = shell_dim

        # Determine device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Nuclear projection - stable identity
        self.nuclear_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, nuclear_dim),
        )

        # Shell projection - contextual variation
        self.shell_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, shell_dim),
        )

        # Charge head - polarity and magnitude
        self.charge_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [polarity, magnitude]
        )

        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode an embedding into atomic components.

        Args:
            embedding: Input embedding (batch, input_dim) or (input_dim,)

        Returns:
            Dict with nuclear, shell, polarity, magnitude, effective_charge
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        embedding = embedding.to(self.device)

        # Project to nuclear (identity) and shell (context)
        nuclear = self.nuclear_proj(embedding)
        shell = self.shell_proj(embedding)

        # Compute charge
        charge_raw = self.charge_head(embedding)
        polarity = torch.tanh(charge_raw[..., 0])      # -1 to 1
        magnitude = torch.sigmoid(charge_raw[..., 1])  # 0 to 1
        effective_charge = polarity * magnitude

        return {
            'nuclear': nuclear,
            'shell': shell,
            'polarity': polarity,
            'magnitude': magnitude,
            'effective_charge': effective_charge,
        }

    def encode_with_context(self, embedding: torch.Tensor,
                            context_embeddings: List[torch.Tensor] = None) -> Dict:
        """
        Encode with optional context for shell modulation.
        """
        base = self.forward(embedding)

        if context_embeddings and len(context_embeddings) > 0:
            # Modulate shell based on context
            context_stack = torch.stack([
                self.forward(c)['shell'] for c in context_embeddings
            ])
            context_mean = context_stack.mean(dim=0)

            # Shell becomes a blend of self and context
            base['shell'] = 0.7 * base['shell'] + 0.3 * context_mean

        return base

    def setup_training(self, lr: float = 1e-4):
        """Initialize optimizer for training."""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.training_stats = {
            'steps': 0,
            'total_loss': 0.0,
            'charge_loss': 0.0,
            'similarity_loss': 0.0,
        }
        self.train()  # Set to training mode

    def train_step(self, embeddings: List[torch.Tensor],
                   target_charges: List[float],
                   co_occurring_pairs: List[Tuple[int, int]] = None) -> Dict:
        """
        Train the encoder on conversation data.

        Args:
            embeddings: List of embeddings for mentioned atoms
            target_charges: Target charge values (-1 to 1) based on context
            co_occurring_pairs: Indices of atoms that appeared together (for similarity)

        Returns:
            Training stats dict
        """
        if not hasattr(self, 'optimizer'):
            self.setup_training()

        if len(embeddings) == 0:
            return {'loss': 0.0, 'trained': False}

        self.optimizer.zero_grad()

        # Stack embeddings
        emb_tensor = torch.stack([
            torch.tensor(e, dtype=torch.float32) if not isinstance(e, torch.Tensor) else e
            for e in embeddings
        ]).to(self.device)

        # Forward pass
        output = self.forward(emb_tensor)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # === CHARGE PREDICTION LOSS ===
        if target_charges:
            target_tensor = torch.tensor(target_charges, dtype=torch.float32, device=self.device)
            predicted_charge = output['effective_charge'].squeeze()

            # MSE loss on charge prediction
            charge_loss = F.mse_loss(predicted_charge, target_tensor)
            total_loss = total_loss + charge_loss
            self.training_stats['charge_loss'] += charge_loss.item()

        # === CO-OCCURRENCE SIMILARITY LOSS ===
        if co_occurring_pairs and len(co_occurring_pairs) > 0:
            shell_vectors = output['shell']
            similarity_loss = torch.tensor(0.0, device=self.device)

            for i, j in co_occurring_pairs:
                if i < len(shell_vectors) and j < len(shell_vectors):
                    # Atoms mentioned together should have similar shells
                    sim = F.cosine_similarity(
                        shell_vectors[i].unsqueeze(0),
                        shell_vectors[j].unsqueeze(0)
                    )
                    # Loss: 1 - similarity (want similarity close to 1)
                    similarity_loss = similarity_loss + (1 - sim.mean())

            if len(co_occurring_pairs) > 0:
                similarity_loss = similarity_loss / len(co_occurring_pairs)
                total_loss = total_loss + 0.5 * similarity_loss
                self.training_stats['similarity_loss'] += similarity_loss.item()

        # Backprop
        if total_loss.requires_grad and total_loss.item() > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.training_stats['steps'] += 1
        self.training_stats['total_loss'] += total_loss.item()

        return {
            'loss': total_loss.item(),
            'trained': True,
            'steps': self.training_stats['steps'],
        }


class ChargePropagator:
    """
    Handles charge propagation through semantic structures.

    Key operations:
    - Negation flipping: "not X" → -charge(X)
    - Composition: sentence charge from word charges
    - Inheritance: charge flows through IS_A bonds
    """

    # Negation words that flip charge
    NEGATIONS = {
        'not', 'no', 'never', 'none', 'nothing', 'neither', 'nobody',
        "n't", 'dont', "don't", 'doesnt', "doesn't", 'didnt', "didn't",
        'wont', "won't", 'cant', "can't", 'isnt', "isn't", 'arent', "aren't",
        'wasnt', "wasn't", 'werent', "weren't", 'without', 'lack', 'lacks',
        'lacking', 'absent', 'fail', 'fails', 'failed', 'failing',
    }

    # Intensifiers that boost magnitude
    INTENSIFIERS = {
        'very': 1.3, 'extremely': 1.5, 'incredibly': 1.5, 'absolutely': 1.4,
        'really': 1.2, 'truly': 1.2, 'highly': 1.3, 'deeply': 1.3,
        'completely': 1.4, 'totally': 1.3, 'utterly': 1.4,
    }

    # Diminishers that reduce magnitude
    DIMINISHERS = {
        'slightly': 0.5, 'somewhat': 0.6, 'a bit': 0.5, 'a little': 0.5,
        'fairly': 0.7, 'rather': 0.7, 'kind of': 0.5, 'sort of': 0.5,
        'barely': 0.3, 'hardly': 0.3, 'scarcely': 0.3,
    }

    @classmethod
    def detect_negation(cls, text: str) -> Tuple[bool, int]:
        """
        Detect if text contains negation and count negations.

        Returns:
            (is_negated, negation_count)
        """
        text_lower = text.lower()
        words = re.findall(r"\b\w+(?:'\w+)?\b", text_lower)

        negation_count = sum(1 for w in words if w in cls.NEGATIONS)

        # Odd number of negations = negated, even = positive
        is_negated = negation_count % 2 == 1

        return is_negated, negation_count

    @classmethod
    def get_intensity_modifier(cls, text: str) -> float:
        """Get intensity modifier from intensifiers/diminishers."""
        text_lower = text.lower()

        modifier = 1.0

        for word, mult in cls.INTENSIFIERS.items():
            if word in text_lower:
                modifier = max(modifier, mult)

        for phrase, mult in cls.DIMINISHERS.items():
            if phrase in text_lower:
                modifier = min(modifier, mult)

        return modifier

    @classmethod
    def propagate_charge(cls, base_charge: float, text: str) -> float:
        """
        Propagate charge through text, handling negation and intensity.

        Args:
            base_charge: The base semantic charge (-1 to 1)
            text: The text context

        Returns:
            Modified charge after propagation
        """
        is_negated, _ = cls.detect_negation(text)
        intensity = cls.get_intensity_modifier(text)

        # Apply negation (flip sign)
        if is_negated:
            base_charge = -base_charge

        # Apply intensity (scale magnitude, preserve sign)
        sign = 1 if base_charge >= 0 else -1
        magnitude = abs(base_charge) * intensity

        # Clamp to [-1, 1]
        return sign * min(1.0, magnitude)

    @classmethod
    def compose_sentence_charge(cls, word_charges: List[Tuple[str, float]],
                                 weights: List[float] = None) -> float:
        """
        Compose sentence charge from word charges.

        Args:
            word_charges: List of (word, charge) tuples
            weights: Optional attention weights

        Returns:
            Composite sentence charge
        """
        if not word_charges:
            return 0.0

        if weights is None:
            weights = [1.0] * len(word_charges)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weights = [w / total_weight for w in weights]

        # Weighted sum of charges
        composite = sum(c * w for (_, c), w in zip(word_charges, weights))

        return max(-1.0, min(1.0, composite))


# =============================================================================
# RELATION TYPES - Maps to MYND's link types
# =============================================================================

class RelationType(Enum):
    # Hierarchical (from MYND tree structure)
    IS_A = 0              # Child is_a Parent
    HAS_PART = 1          # Parent has_part Child
    PART_OF = 2           # Inverse

    # Semantic (from MYND links)
    CAUSES = 3
    CAUSED_BY = 4
    SIMILAR_TO = 5
    OPPOSITE_OF = 6
    RELATED_TO = 7
    EXAMPLE_OF = 8
    COMPONENT_OF = 9
    CONTRADICTS = 10

    # MYND-specific
    SUPPORTS = 11         # Evidence/argument
    QUESTION = 12         # Open question about
    TODO = 13             # Action item for

    @classmethod
    def from_mynd_link(cls, link_type: str) -> 'RelationType':
        """Convert MYND link type string to RelationType."""
        mapping = {
            'related_to': cls.RELATED_TO,
            'causes': cls.CAUSES,
            'caused_by': cls.CAUSED_BY,
            'component_of': cls.COMPONENT_OF,
            'example_of': cls.EXAMPLE_OF,
            'similar_to': cls.SIMILAR_TO,
            'opposite_of': cls.OPPOSITE_OF,
            'contradicts': cls.CONTRADICTS,
            'supports': cls.SUPPORTS,
            'question': cls.QUESTION,
            'todo': cls.TODO,
        }
        return mapping.get(link_type.lower(), cls.RELATED_TO)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Bond:
    """A living bond between atoms."""
    target_id: str  # Using MYND's string IDs
    relation: RelationType
    strength: float = 1.0
    shell: int = 4
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    binding_energy: float = 0.5
    source: str = "explicit"  # "explicit", "inferred", "gt_predicted"
    confidence: float = 1.0   # From Graph Transformer


@dataclass
class ValenceSlot:
    """A slot for a specific relation type."""
    relation: RelationType
    capacity: int = 3
    filled: List[str] = field(default_factory=list)
    required: bool = False

    @property
    def gap(self) -> int:
        return max(0, self.capacity - len(self.filled))


@dataclass
class SemanticAtom:
    """
    A LIVING semantic atom - wraps a MYND node.

    Combines ASE vector properties with Living ASA structural properties.
    """
    id: str  # MYND node ID
    name: str  # Node label

    # Original MYND data reference
    mynd_node: Dict = field(default_factory=dict)

    # Structure (Living ASA)
    shells: Dict[int, List[Bond]] = field(default_factory=lambda: {1: [], 2: [], 3: [], 4: []})
    valence: Dict[RelationType, ValenceSlot] = field(default_factory=dict)

    # === ASE VECTOR COMPONENTS ===
    # Nuclear vector - stable identity ("what IS this")
    nuclear: Optional[np.ndarray] = None  # (64,)

    # Shell vector - contextual variation ("how it's used")
    context_shell: Optional[np.ndarray] = None  # (128,)

    # Semantic charge - learned polarity
    semantic_charge: float = 0.0      # -1 to 1 (polarity)
    charge_magnitude: float = 0.5     # 0 to 1 (confidence)

    # === LIVING ASA DYNAMIC STATE ===
    # Structural charge (valence satisfaction) - renamed for clarity
    valence_charge: float = 0.0       # From unfilled slots
    energy: float = 0.0               # Working memory activation
    mass: float = 1.0                 # Stability/age

    # Combined effective charge (ASE semantic + Living ASA structural)
    @property
    def effective_charge(self) -> float:
        """Combined charge: semantic polarity weighted by structural state."""
        # Semantic charge weighted by magnitude, plus structural charge
        semantic = self.semantic_charge * self.charge_magnitude
        structural = self.valence_charge * 0.3  # Structural contributes less
        return max(-1.0, min(1.0, semantic + structural))

    # Legacy alias for backward compatibility
    @property
    def charge(self) -> float:
        return self.effective_charge

    # Base embedding (from MYND's embeddings - input to ASE encoder)
    embedding: Optional[np.ndarray] = None

    # Temporal
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    contradiction_count: int = 0

    # MYND-specific
    importance: float = 0.5  # From MYND node
    heat: float = 0.0  # From HeatTracker
    depth: int = 0  # Tree depth

    def __post_init__(self):
        if not self.valence:
            self.valence = {
                RelationType.IS_A: ValenceSlot(RelationType.IS_A, capacity=1),
                RelationType.HAS_PART: ValenceSlot(RelationType.HAS_PART, capacity=10),
                RelationType.CAUSES: ValenceSlot(RelationType.CAUSES, capacity=5),
                RelationType.SIMILAR_TO: ValenceSlot(RelationType.SIMILAR_TO, capacity=5),
                RelationType.RELATED_TO: ValenceSlot(RelationType.RELATED_TO, capacity=20),
                RelationType.SUPPORTS: ValenceSlot(RelationType.SUPPORTS, capacity=10),
                RelationType.QUESTION: ValenceSlot(RelationType.QUESTION, capacity=5),
            }
        if self.created_at == 0.0:
            self.created_at = time.time()
            self.last_accessed = self.created_at

    def get_all_bonds(self) -> List[Bond]:
        return [b for shell in self.shells.values() for b in shell]

    def get_bonds_by_relation(self, relation: RelationType) -> List[Bond]:
        return [b for b in self.get_all_bonds() if b.relation == relation]

    def count_bonds(self) -> int:
        return sum(len(shell) for shell in self.shells.values())

    def count_unfilled_slots(self) -> int:
        return sum(slot.gap for slot in self.valence.values())

    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def to_dict(self) -> Dict:
        """Serialize for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            # ASE charge components
            'semantic_charge': round(self.semantic_charge, 3),
            'charge_magnitude': round(self.charge_magnitude, 3),
            'valence_charge': round(self.valence_charge, 3),
            'effective_charge': round(self.effective_charge, 3),
            # Living ASA state
            'energy': round(self.energy, 3),
            'mass': round(self.mass, 2),
            'total_bonds': self.count_bonds(),
            'unfilled_slots': self.count_unfilled_slots(),
            'shells': {
                f'shell_{i}': len(bonds) for i, bonds in self.shells.items()
            },
            # ASE vectors present?
            'has_nuclear': self.nuclear is not None,
            'has_context_shell': self.context_shell is not None,
            # MYND properties
            'importance': self.importance,
            'depth': self.depth,
        }


# =============================================================================
# LIVING ASA FOR MYND
# =============================================================================

class MYNDLivingASA:
    """
    Hybrid ASE + Living Atomic Semantic Architecture for MYND.

    Combines:
    - ASE: Neural encoder for nuclear/shell vectors and semantic charge
    - Living ASA: Structural bonds, energy, mass, metabolism

    Converts MYND's node tree into a metabolizing semantic graph that
    learns continuously from conversations.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

        # Atom storage
        self.atoms: Dict[str, SemanticAtom] = {}

        # === ASE NEURAL ENCODER ===
        self.atomic_encoder = AtomicEncoder(
            input_dim=embedding_dim,
            nuclear_dim=64,
            shell_dim=128,
        )
        self.charge_propagator = ChargePropagator()

        # ASE training state
        self._ase_optimizer = None
        self._ase_training_initialized = False

        # Shell configuration (Living ASA)
        self.shell_config = {
            1: {'binding_energy': 0.95, 'capacity': 3, 'name': 'core', 'decay_rate': 0.01},
            2: {'binding_energy': 0.70, 'capacity': 6, 'name': 'inner', 'decay_rate': 0.03},
            3: {'binding_energy': 0.40, 'capacity': 12, 'name': 'outer', 'decay_rate': 0.08},
            4: {'binding_energy': 0.15, 'capacity': float('inf'), 'name': 'contextual', 'decay_rate': 0.15},
        }

        # Metabolism parameters
        self.energy_half_life = 600.0  # 10 minutes
        self.migration_threshold_up = 0.25
        self.migration_threshold_down = 0.05
        self.bond_death_threshold = 0.05

        # State
        self._last_tick = time.time()
        self._running = False
        self._metabolism_thread = None

        # Learning stats
        self._learning_stats = {
            'texts_processed': 0,
            'atoms_activated': 0,
            'bonds_created': 0,
            'bonds_strengthened': 0,
            'charges_learned': 0,
        }

        logger.info("Hybrid ASE + Living ASA initialized")

    # =========================================================================
    # MAP CONVERSION - Convert MYND nodes to atoms
    # =========================================================================

    def convert_map_to_asa(self, map_data: Dict, embeddings: Dict[str, np.ndarray] = None) -> None:
        """
        Convert MYND map data to Living ASA atoms.

        Args:
            map_data: MYND map JSON (root node with children)
            embeddings: Optional dict of node_id -> embedding vector
        """
        logger.info("Converting MYND map to Living ASA...")

        # Clear existing atoms
        self.atoms.clear()

        # Recursive conversion
        self._convert_node(map_data, parent_id=None, depth=0, embeddings=embeddings or {})

        # Update all charges after conversion
        for atom in self.atoms.values():
            self._update_charge(atom)
            self._update_mass(atom)

        logger.info(f"Converted {len(self.atoms)} nodes to atoms")

    def _convert_node(self, node: Dict, parent_id: Optional[str], depth: int,
                      embeddings: Dict[str, np.ndarray]) -> None:
        """Recursively convert a MYND node and its children."""
        node_id = node.get('id', '')
        if not node_id:
            return

        # Create atom
        atom = SemanticAtom(
            id=node_id,
            name=node.get('label', 'Untitled'),
            mynd_node=node,
            importance=node.get('importance', 0.5),
            depth=depth,
            embedding=embeddings.get(node_id),
        )

        # Parse timestamps
        if 'createdAt' in node:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(node['createdAt'].replace('Z', '+00:00'))
                atom.created_at = dt.timestamp()
            except:
                atom.created_at = time.time()

        self.atoms[node_id] = atom

        # Create IS_A bond to parent (child IS_A parent in taxonomy sense)
        # Or HAS_PART (parent HAS_PART child)
        if parent_id and parent_id in self.atoms:
            # Child has parent as IS_A target
            self._create_bond(node_id, parent_id, RelationType.IS_A,
                            strength=1.0, source="hierarchy")
            # Parent has child as HAS_PART
            self._create_bond(parent_id, node_id, RelationType.HAS_PART,
                            strength=1.0, source="hierarchy")

        # Convert semantic links if present
        for link in node.get('links', []):
            target_id = link.get('targetId') or link.get('target_id')
            link_type = link.get('type', 'related_to')
            confidence = link.get('confidence', 1.0)

            if target_id:
                rel_type = RelationType.from_mynd_link(link_type)
                self._create_bond(node_id, target_id, rel_type,
                                strength=confidence, source="semantic_link")

        # Process children
        for child in node.get('children', []):
            self._convert_node(child, node_id, depth + 1, embeddings)

    def sync_heat_from_mynd(self, heat_data: Dict[str, float]) -> None:
        """Sync heat values from MYND's HeatTracker to atom energy."""
        for node_id, heat in heat_data.items():
            if node_id in self.atoms:
                atom = self.atoms[node_id]
                atom.heat = heat
                # Heat contributes to energy
                atom.energy = max(atom.energy, heat * 0.5)

    def sync_gt_predictions(self, predictions: List[Dict]) -> None:
        """
        Sync Graph Transformer predictions as potential bonds.

        predictions: List of {source_id, target_id, relation, confidence}
        """
        for pred in predictions:
            source_id = pred.get('source_id')
            target_id = pred.get('target_id')
            relation = pred.get('relation', 'related_to')
            confidence = pred.get('confidence', 0.5)

            if source_id in self.atoms and target_id in self.atoms:
                rel_type = RelationType.from_mynd_link(relation)
                self._create_bond(source_id, target_id, rel_type,
                                strength=confidence * 0.5,  # Lower initial strength
                                source="gt_predicted",
                                start_shell=4)  # Start in outer shell

    # =========================================================================
    # BOND MANAGEMENT
    # =========================================================================

    def _get_ancestors(self, atom_id: str, visited: Set[str] = None) -> Set[str]:
        """Get all is_a ancestors of an atom (follows the inheritance chain)."""
        if visited is None:
            visited = set()
        if atom_id in visited or atom_id not in self.atoms:
            return set()
        visited.add(atom_id)

        ancestors = set()
        atom = self.atoms[atom_id]
        for bond in atom.get_bonds_by_relation(RelationType.IS_A):
            ancestors.add(bond.target_id)
            ancestors |= self._get_ancestors(bond.target_id, visited.copy())
        return ancestors

    def _check_contradiction(self, atom1_id: str, atom2_id: str) -> bool:
        """
        Check if two atoms contradict, including through is_a inheritance.

        Example: If Dog CONTRADICTS Cat, and Poodle IS_A Dog,
        then Poodle trying to be IS_A Cat should be blocked.
        """
        if atom1_id not in self.atoms or atom2_id not in self.atoms:
            return False

        # Get all ancestors for both atoms
        atom1_line = {atom1_id} | self._get_ancestors(atom1_id)
        atom2_line = {atom2_id} | self._get_ancestors(atom2_id)

        # Check if any atom in line 1 contradicts any atom in line 2
        for a1_id in atom1_line:
            if a1_id not in self.atoms:
                continue
            a1 = self.atoms[a1_id]
            for bond in a1.get_all_bonds():
                if bond.relation == RelationType.CONTRADICTS and bond.target_id in atom2_line:
                    return True

        # Check reverse direction too
        for a2_id in atom2_line:
            if a2_id not in self.atoms:
                continue
            a2 = self.atoms[a2_id]
            for bond in a2.get_all_bonds():
                if bond.relation == RelationType.CONTRADICTS and bond.target_id in atom1_line:
                    return True

        return False

    def _create_bond(self, source_id: str, target_id: str, relation: RelationType,
                     strength: float = 1.0, source: str = "explicit",
                     start_shell: int = 4) -> bool:
        """Create a bond between atoms."""
        if source_id not in self.atoms:
            return False
        if target_id not in self.atoms:
            return False

        atom = self.atoms[source_id]
        target = self.atoms[target_id]
        now = time.time()

        # === PAULI EXCLUSION CHECK ===
        # For IS_A bonds, check if this would create a contradiction
        if relation == RelationType.IS_A:
            # Check if source already has is_a parents that contradict the new target
            existing_parents = atom.get_bonds_by_relation(RelationType.IS_A)
            for parent_bond in existing_parents:
                if self._check_contradiction(parent_bond.target_id, target_id):
                    logger.warning(f"Pauli violation: {atom.name} can't be both "
                                 f"{self.atoms[parent_bond.target_id].name} and {target.name}")
                    atom.contradiction_count += 1
                    self._update_charge(atom)
                    return False

            # Also check if target's lineage contradicts source's existing lineage
            if self._check_contradiction(source_id, target_id):
                logger.warning(f"Pauli violation: {atom.name} contradicts {target.name} through inheritance")
                atom.contradiction_count += 1
                self._update_charge(atom)
                return False

        # Check if bond already exists
        for bond in atom.get_all_bonds():
            if bond.target_id == target_id and bond.relation == relation:
                # Strengthen existing bond
                bond.strength = min(1.0, bond.strength + 0.1)
                bond.access_count += 1
                bond.last_accessed = now
                return True

        # Determine starting shell based on source
        if source == "hierarchy":
            start_shell = 2  # Hierarchical bonds start closer to core
        elif source == "gt_predicted":
            start_shell = 4  # Predictions start outer

        # Create new bond
        bond = Bond(
            target_id=target_id,
            relation=relation,
            strength=strength,
            shell=start_shell,
            created_at=now,
            last_accessed=now,
            binding_energy=self.shell_config[start_shell]['binding_energy'],
            source=source,
        )

        atom.shells[start_shell].append(bond)

        if relation in atom.valence:
            atom.valence[relation].filled.append(target_id)

        return True

    # =========================================================================
    # DYNAMIC STATE UPDATES
    # =========================================================================

    def _update_charge(self, atom: SemanticAtom):
        """Update valence charge based on structural satisfaction."""
        unfilled = atom.count_unfilled_slots()
        contradictions = atom.contradiction_count

        # Also factor in MYND importance
        importance_factor = (atom.importance - 0.5) * 0.2

        seeking_pull = -0.08 * unfilled
        repelling_push = 0.3 * contradictions

        # Update valence_charge (structural component of total charge)
        atom.valence_charge = max(-1.0, min(1.0, seeking_pull + repelling_push + importance_factor))

    def _update_mass(self, atom: SemanticAtom):
        """Update mass based on age, connectivity, and importance."""
        age_factor = math.log(1 + atom.age_days())
        connectivity = atom.count_bonds()
        confirmations = atom.access_count

        # MYND importance boosts mass
        importance_boost = atom.importance * 2

        atom.mass = 1.0 + (age_factor * 0.3) + (connectivity * 0.2) + \
                   (confirmations * 0.1) + importance_boost

    def _decay_energy(self, atom: SemanticAtom, dt: float):
        """Decay energy with half-life."""
        decay_rate = 0.693 / self.energy_half_life
        atom.energy *= math.exp(-decay_rate * dt)

    def _decay_bonds(self, atom: SemanticAtom, dt: float):
        """Decay weak, unused bonds."""
        now = time.time()

        for shell_idx in list(atom.shells.keys()):
            surviving = []

            for bond in atom.shells[shell_idx]:
                # Don't decay hierarchy bonds as fast
                if bond.source == "hierarchy":
                    bond.strength -= 0.001 * dt / 100
                else:
                    shell_decay = self.shell_config[shell_idx]['decay_rate']
                    strength_decay = 0.1 * (1 - bond.strength)
                    time_decay = 0.005 * ((now - bond.last_accessed) / 86400)

                    total_decay = (shell_decay + strength_decay + time_decay) * dt / 100
                    bond.strength -= total_decay

                if bond.strength > self.bond_death_threshold:
                    surviving.append(bond)
                else:
                    # Bond dies - update valence
                    if bond.relation in atom.valence:
                        filled = atom.valence[bond.relation].filled
                        if bond.target_id in filled:
                            filled.remove(bond.target_id)
                    logger.debug(f"Bond died: {atom.name} → {bond.target_id}")

            atom.shells[shell_idx] = surviving

    def _migrate_bonds(self, atom: SemanticAtom):
        """Migrate bonds based on access patterns."""
        if atom.access_count == 0:
            return

        for shell_idx in [4, 3, 2]:
            for bond in atom.shells[shell_idx][:]:
                access_ratio = bond.access_count / max(atom.access_count, 1)

                # Migrate inward
                if access_ratio > self.migration_threshold_up:
                    target_shell = shell_idx - 1
                    if len(atom.shells[target_shell]) < self.shell_config[target_shell]['capacity']:
                        atom.shells[shell_idx].remove(bond)
                        bond.shell = target_shell
                        bond.binding_energy = self.shell_config[target_shell]['binding_energy']
                        atom.shells[target_shell].append(bond)
                        logger.debug(f"Bond migrated inward: {atom.name} → {bond.target_id} to shell {target_shell}")

    # =========================================================================
    # ACCESS AND ACTIVATION
    # =========================================================================

    def access_atom(self, atom_id: str, energy_boost: float = 0.3) -> None:
        """Access an atom - bring into working memory."""
        if atom_id not in self.atoms:
            return

        atom = self.atoms[atom_id]
        now = time.time()

        atom.energy = min(1.0, atom.energy + energy_boost)
        atom.last_accessed = now
        atom.access_count += 1

        for bonds in atom.shells.values():
            for bond in bonds:
                bond.access_count += 1
                bond.last_accessed = now

    def spread_activation(self, source_ids: List[str], depth: int = 3) -> Dict[str, float]:
        """Spread activation energy through bonds."""
        activated = {}

        for source_id in source_ids:
            if source_id not in self.atoms:
                continue

            source = self.atoms[source_id]
            activated[source_id] = source.energy
            frontier = [(source_id, source.energy, 0)]

            while frontier:
                current_id, current_energy, current_depth = frontier.pop(0)

                if current_depth >= depth or current_energy < 0.05:
                    continue

                if current_id not in self.atoms:
                    continue

                current_atom = self.atoms[current_id]

                for shell_idx, bonds in current_atom.shells.items():
                    decay_factor = 0.8 ** (shell_idx - 1)

                    for bond in bonds:
                        spread_amount = current_energy * decay_factor * bond.strength * 0.5

                        if spread_amount > 0.05 and bond.target_id in self.atoms:
                            target = self.atoms[bond.target_id]
                            new_energy = min(1.0, target.energy + spread_amount)

                            if bond.target_id not in activated or activated[bond.target_id] < new_energy:
                                target.energy = new_energy
                                activated[bond.target_id] = new_energy
                                frontier.append((bond.target_id, spread_amount, current_depth + 1))

        return activated

    # =========================================================================
    # METABOLISM
    # =========================================================================

    def tick(self, dt: float = None) -> None:
        """One metabolism tick."""
        now = time.time()
        if dt is None:
            dt = now - self._last_tick
        self._last_tick = now

        for atom in self.atoms.values():
            self._decay_energy(atom, dt)
            self._decay_bonds(atom, dt)
            self._migrate_bonds(atom)
            self._update_mass(atom)
            self._update_charge(atom)

    def start_metabolism(self, tick_interval: float = 5.0) -> None:
        """Start background metabolism thread."""
        if self._running:
            return

        self._running = True

        def metabolism_loop():
            while self._running:
                try:
                    self.tick(tick_interval)
                except Exception as e:
                    logger.error(f"Metabolism error: {e}")
                time.sleep(tick_interval)

        self._metabolism_thread = threading.Thread(target=metabolism_loop, daemon=True)
        self._metabolism_thread.start()
        logger.info("Metabolism started")

    def stop_metabolism(self) -> None:
        """Stop background metabolism."""
        self._running = False
        if self._metabolism_thread:
            self._metabolism_thread.join(timeout=2.0)
        logger.info("Metabolism stopped")

    # =========================================================================
    # QUERIES - Insights for MYND
    # =========================================================================

    def get_attention_needed(self, limit: int = 10) -> List[Dict]:
        """What nodes need attention? (high charge)"""
        results = []
        for atom in self.atoms.values():
            if abs(atom.charge) > 0.3:
                results.append({
                    'id': atom.id,
                    'name': atom.name,
                    'charge': atom.charge,
                    'status': 'seeking_connections' if atom.charge < 0 else 'unstable',
                    'unfilled_slots': atom.count_unfilled_slots(),
                    'reason': self._explain_charge(atom),
                })
        return sorted(results, key=lambda x: -abs(x['charge']))[:limit]

    def _explain_charge(self, atom: SemanticAtom) -> str:
        """Explain why an atom has its charge."""
        if atom.charge < -0.5:
            gaps = [f"{r.name}: {s.gap}" for r, s in atom.valence.items() if s.gap > 2]
            return f"Missing many connections: {', '.join(gaps[:3])}"
        elif atom.charge < 0:
            return "Could use more connections"
        elif atom.charge > 0.5:
            return "Has contradictions or conflicts"
        else:
            return "Stable"

    def get_working_memory(self, threshold: float = 0.2) -> List[Dict]:
        """What's currently active? (high energy)"""
        results = []
        for atom in self.atoms.values():
            if atom.energy > threshold:
                results.append({
                    'id': atom.id,
                    'name': atom.name,
                    'energy': atom.energy,
                    'last_accessed_ago': time.time() - atom.last_accessed,
                })
        return sorted(results, key=lambda x: -x['energy'])

    def get_neglected(self, days_threshold: float = 7.0) -> List[Dict]:
        """What's been neglected? (low energy, old)"""
        results = []
        now = time.time()
        for atom in self.atoms.values():
            days_cold = (now - atom.last_accessed) / 86400
            if atom.energy < 0.1 and days_cold > days_threshold:
                results.append({
                    'id': atom.id,
                    'name': atom.name,
                    'days_neglected': days_cold,
                    'importance': atom.importance,
                })
        return sorted(results, key=lambda x: -x['days_neglected'])

    def find_related_atoms(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Find atoms related to a query (for GT integration).
        Returns atoms that match the query or are bonded to matching atoms.
        """
        query_lower = query.lower()
        results = []

        # First, find atoms that match the query
        matching_atoms = []
        for atom in self.atoms.values():
            if query_lower in atom.name.lower() or atom.name.lower() in query_lower:
                matching_atoms.append(atom)

        # If no direct match, try partial word matching
        if not matching_atoms:
            query_words = set(query_lower.split())
            for atom in self.atoms.values():
                atom_words = set(atom.name.lower().split())
                if query_words & atom_words:  # Intersection
                    matching_atoms.append(atom)

        # Get bonded atoms from matching atoms
        for atom in matching_atoms:
            # Get all bonds (from all shells)
            for shell_idx, shell in enumerate(atom.shells):
                for bonded_id, bond_strength in shell.items():
                    bonded_atom = self.atoms.get(bonded_id)
                    if bonded_atom:
                        results.append({
                            'id': bonded_atom.id,
                            'name': bonded_atom.name,
                            'bond_strength': bond_strength,
                            'shell': shell_idx,
                            'via_atom': atom.name,
                            'energy': bonded_atom.energy,
                        })

        # Deduplicate by id, keeping highest bond strength
        seen = {}
        for r in results:
            if r['id'] not in seen or r['bond_strength'] > seen[r['id']]['bond_strength']:
                seen[r['id']] = r

        return sorted(seen.values(), key=lambda x: -x['bond_strength'])[:limit]

    def get_core_knowledge(self, limit: int = 20) -> List[Dict]:
        """What's core/stable? (high mass)"""
        results = []
        for atom in self.atoms.values():
            results.append({
                'id': atom.id,
                'name': atom.name,
                'mass': atom.mass,
                'bonds': atom.count_bonds(),
                'core_bonds': len(atom.shells[1]) + len(atom.shells[2]),
            })
        return sorted(results, key=lambda x: -x['mass'])[:limit]

    def get_incomplete_thinking(self, limit: int = 20) -> List[Dict]:
        """What's incomplete? (gaps in valence)"""
        gaps = []
        for atom in self.atoms.values():
            atom_gaps = []
            for rel, slot in atom.valence.items():
                if slot.gap > 0:
                    atom_gaps.append({
                        'relation': rel.name,
                        'wanted': slot.capacity,
                        'filled': len(slot.filled),
                        'gap': slot.gap,
                    })
            if atom_gaps:
                gaps.append({
                    'id': atom.id,
                    'name': atom.name,
                    'total_gap': sum(g['gap'] for g in atom_gaps),
                    'gaps': atom_gaps[:3],  # Top 3 gaps
                    'suggestion': self._suggest_for_gaps(atom, atom_gaps),
                })
        return sorted(gaps, key=lambda x: -x['total_gap'])[:limit]

    def _suggest_for_gaps(self, atom: SemanticAtom, gaps: List[Dict]) -> str:
        """Generate a suggestion for filling gaps."""
        biggest_gap = max(gaps, key=lambda g: g['gap'])
        rel = biggest_gap['relation']

        if rel == 'CAUSES':
            return f"What does '{atom.name}' cause or lead to?"
        elif rel == 'HAS_PART':
            return f"What are the components or parts of '{atom.name}'?"
        elif rel == 'SIMILAR_TO':
            return f"What is '{atom.name}' similar to?"
        elif rel == 'RELATED_TO':
            return f"What else relates to '{atom.name}'?"
        elif rel == 'SUPPORTS':
            return f"What evidence supports '{atom.name}'?"
        elif rel == 'QUESTION':
            return f"What questions do you have about '{atom.name}'?"
        else:
            return f"Consider adding more {rel.lower().replace('_', ' ')} connections"

    def activation_preview(self, node_ids: List[str], depth: int = 3) -> List[Dict]:
        """If user focuses on these nodes, what lights up?"""
        # Save current state
        original_energies = {aid: a.energy for aid, a in self.atoms.items()}

        # Activate seeds
        for nid in node_ids:
            if nid in self.atoms:
                self.atoms[nid].energy = 1.0

        # Spread
        activated = self.spread_activation(node_ids, depth)

        # Build results
        results = []
        for aid, energy in activated.items():
            if aid not in node_ids and energy > 0.1:
                results.append({
                    'id': aid,
                    'name': self.atoms[aid].name,
                    'activation': energy,
                })

        # Restore
        for aid, energy in original_energies.items():
            self.atoms[aid].energy = energy

        return sorted(results, key=lambda x: -x['activation'])

    def get_decaying_connections(self, threshold: float = 0.3) -> List[Dict]:
        """What connections are about to be lost?"""
        results = []
        for atom in self.atoms.values():
            for shell_idx, bonds in atom.shells.items():
                for bond in bonds:
                    if bond.strength < threshold and bond.target_id in self.atoms:
                        results.append({
                            'source_id': atom.id,
                            'source_name': atom.name,
                            'target_id': bond.target_id,
                            'target_name': self.atoms[bond.target_id].name,
                            'relation': bond.relation.name,
                            'strength': bond.strength,
                            'shell': shell_idx,
                            'source_type': bond.source,
                        })
        return sorted(results, key=lambda x: x['strength'])

    # =========================================================================
    # CONTEXT FOR CLAUDE
    # =========================================================================

    def get_context_for_claude(self, focus_node_ids: List[str] = None,
                                max_tokens: int = 2000) -> str:
        """
        Generate context block for Claude's system prompt.

        This is the key integration point with /brain/context
        """
        lines = ["## Semantic Analysis (Living ASA)"]
        lines.append("")

        # Working memory
        active = self.get_working_memory(threshold=0.3)
        if active:
            lines.append("**Currently Active Concepts:**")
            for a in active[:5]:
                lines.append(f"- {a['name']} (energy: {a['energy']:.0%})")
            lines.append("")

        # Attention needed
        attention = self.get_attention_needed(limit=5)
        if attention:
            lines.append("**Needs Attention:**")
            for a in attention:
                lines.append(f"- {a['name']}: {a['reason']}")
            lines.append("")

        # If focus nodes specified, show what activates
        if focus_node_ids:
            preview = self.activation_preview(focus_node_ids, depth=2)
            if preview:
                lines.append("**Related Concepts (by activation):**")
                for p in preview[:8]:
                    lines.append(f"- {p['name']} ({p['activation']:.0%})")
                lines.append("")

        # Incomplete thinking
        incomplete = self.get_incomplete_thinking(limit=3)
        if incomplete:
            lines.append("**Incomplete Thinking (gaps):**")
            for i in incomplete:
                lines.append(f"- {i['name']}: {i['suggestion']}")
            lines.append("")

        # Decaying connections (user might lose these)
        decaying = self.get_decaying_connections(threshold=0.25)
        if decaying:
            lines.append("**Fading Connections (reinforce or lose):**")
            for d in decaying[:3]:
                lines.append(f"- {d['source_name']} → {d['target_name']} ({d['relation']})")
            lines.append("")

        # Core knowledge
        core = self.get_core_knowledge(limit=5)
        if core:
            lines.append("**Core Concepts (high mass):**")
            for c in core:
                lines.append(f"- {c['name']} (mass: {c['mass']:.1f}, bonds: {c['bonds']})")

        context = "\n".join(lines)

        # Truncate if needed (rough token estimate)
        if len(context) > max_tokens * 4:
            context = context[:max_tokens * 4] + "\n...(truncated)"

        return context

    # =========================================================================
    # STATS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get overall ASA statistics including ASE components."""
        total_bonds = sum(a.count_bonds() for a in self.atoms.values())
        total_gaps = sum(a.count_unfilled_slots() for a in self.atoms.values())

        shell_counts = defaultdict(int)
        for atom in self.atoms.values():
            for shell_idx, bonds in atom.shells.items():
                shell_counts[shell_idx] += len(bonds)

        # ASE stats
        atoms_with_nuclear = sum(1 for a in self.atoms.values() if a.nuclear is not None)
        atoms_with_charge = sum(1 for a in self.atoms.values() if a.semantic_charge != 0)
        positive_atoms = sum(1 for a in self.atoms.values() if a.effective_charge > 0.2)
        negative_atoms = sum(1 for a in self.atoms.values() if a.effective_charge < -0.2)

        return {
            # Living ASA stats
            'atom_count': len(self.atoms),
            'total_bonds': total_bonds,
            'total_gaps': total_gaps,
            'avg_energy': sum(a.energy for a in self.atoms.values()) / max(len(self.atoms), 1),
            'avg_mass': sum(a.mass for a in self.atoms.values()) / max(len(self.atoms), 1),
            'hot_atoms': len([a for a in self.atoms.values() if a.energy > 0.3]),
            'seeking_atoms': len([a for a in self.atoms.values() if a.valence_charge < -0.3]),
            'bonds_by_shell': dict(shell_counts),
            'metabolism_running': self._running,
            # ASE stats
            'ase': {
                'atoms_encoded': atoms_with_nuclear,
                'atoms_with_charge': atoms_with_charge,
                'positive_atoms': positive_atoms,
                'negative_atoms': negative_atoms,
                'encoder_device': str(self.atomic_encoder.device),
            },
            # Learning stats
            'learning': self._learning_stats,
        }

    # =========================================================================
    # LEARN FROM TEXT - The key to continuous learning
    # =========================================================================

    def learn_from_text(self, text: str, source: str = "conversation",
                        energy_boost: float = 0.3) -> Dict:
        """
        Learn from any text input - conversations, thoughts, user messages.

        This is the CORE of the hybrid ASE + Living ASA system:
        1. Find which atoms are mentioned in the text
        2. Boost their energy (bring to working memory)
        3. Apply ASE charge propagation (negation, intensity)
        4. Strengthen bonds between co-occurring atoms
        5. Update learning statistics

        Args:
            text: The text to learn from
            source: Where this came from ("user", "axel", "system")
            energy_boost: How much energy to add to mentioned atoms

        Returns:
            Dict with learning stats
        """
        if not text or not self.atoms:
            return {'atoms_activated': 0, 'bonds_strengthened': 0, 'charge_updates': 0}

        text_lower = text.lower()

        # Find mentioned atoms
        mentioned = self._find_mentioned_atoms(text_lower)

        if not mentioned:
            return {'atoms_activated': 0, 'bonds_strengthened': 0, 'charge_updates': 0}

        # === ASE: Detect negation and intensity in context ===
        is_negated, negation_count = self.charge_propagator.detect_negation(text)
        intensity_modifier = self.charge_propagator.get_intensity_modifier(text)

        # Boost energy and update charges on mentioned atoms
        charge_updates = 0
        for atom_id in mentioned:
            self.access_atom(atom_id, energy_boost)

            # Apply ASE charge propagation
            atom = self.atoms[atom_id]
            if is_negated:
                # Flip semantic charge if text is negated
                atom.semantic_charge = -atom.semantic_charge
                charge_updates += 1

            # Intensity affects magnitude
            if intensity_modifier != 1.0:
                atom.charge_magnitude = min(1.0, atom.charge_magnitude * intensity_modifier)
                charge_updates += 1

        # Strengthen bonds between co-occurring atoms
        bonds_strengthened = self._strengthen_cooccurrence(mentioned, source)

        # Update learning stats
        self._learning_stats['texts_processed'] += 1
        self._learning_stats['atoms_activated'] += len(mentioned)
        self._learning_stats['bonds_strengthened'] += bonds_strengthened
        self._learning_stats['charges_learned'] += charge_updates

        # === NEURAL TRAINING: Train AtomicEncoder on this conversation ===
        encoder_trained = False
        encoder_loss = 0.0

        if self.atomic_encoder is not None and len(mentioned) >= 2:
            try:
                # Collect embeddings for mentioned atoms
                embeddings = []
                target_charges = []

                for atom_id in mentioned:
                    atom = self.atoms.get(atom_id)
                    if atom and atom.embedding is not None:
                        embeddings.append(atom.embedding)
                        # Target charge: based on context (negated = flip, intensity = scale)
                        base_charge = atom.semantic_charge if atom.semantic_charge != 0 else 0.1
                        if is_negated:
                            base_charge = -base_charge
                        target_charges.append(base_charge * intensity_modifier)

                # Generate co-occurring pairs (all atoms mentioned together)
                co_occurring_pairs = []
                mentioned_list = list(mentioned)
                for i in range(len(mentioned_list)):
                    for j in range(i + 1, len(mentioned_list)):
                        co_occurring_pairs.append((i, j))

                # Train the encoder!
                if len(embeddings) >= 2:
                    train_result = self.atomic_encoder.train_step(
                        embeddings=embeddings,
                        target_charges=target_charges,
                        co_occurring_pairs=co_occurring_pairs[:10]  # Limit pairs
                    )
                    encoder_trained = train_result.get('trained', False)
                    encoder_loss = train_result.get('loss', 0.0)

                    if encoder_trained:
                        self._learning_stats['encoder_steps'] = self._learning_stats.get('encoder_steps', 0) + 1
                        logger.info(f"🧠 ASE trained: loss={encoder_loss:.4f}, step={self._learning_stats.get('encoder_steps', 0)}")

            except Exception as e:
                logger.warning(f"ASE training error: {e}")

        # Log learning
        negation_str = " (NEGATED)" if is_negated else ""
        train_str = f", encoder_loss={encoder_loss:.4f}" if encoder_trained else ""
        logger.info(f"ASA learned from {source}{negation_str}: {len(mentioned)} atoms, {bonds_strengthened} bonds{train_str}")

        return {
            'atoms_activated': len(mentioned),
            'bonds_strengthened': bonds_strengthened,
            'charge_updates': charge_updates,
            'negated': is_negated,
            'intensity': intensity_modifier,
            'activated_names': [self.atoms[aid].name for aid in mentioned if aid in self.atoms][:10],
            'source': source,
            'encoder_trained': encoder_trained,
            'encoder_loss': encoder_loss
        }

    def _find_mentioned_atoms(self, text_lower: str) -> List[str]:
        """
        Find atoms that are mentioned in the text.
        Uses fuzzy matching on atom names.
        """
        mentioned = []

        for atom_id, atom in self.atoms.items():
            # Check if atom name (or significant part) appears in text
            name_lower = atom.name.lower()

            # Direct mention
            if name_lower in text_lower:
                mentioned.append(atom_id)
                continue

            # Word-based matching for multi-word names
            words = name_lower.split()
            if len(words) > 1:
                # If most words match, consider it mentioned
                matches = sum(1 for w in words if w in text_lower and len(w) > 2)
                if matches >= len(words) * 0.7:
                    mentioned.append(atom_id)
                    continue

            # Single significant word (must be longer to avoid false positives)
            if len(name_lower) >= 5 and name_lower in text_lower:
                mentioned.append(atom_id)

        return mentioned

    def _strengthen_cooccurrence(self, atom_ids: List[str], source: str) -> int:
        """
        Strengthen bonds between atoms that co-occur in the same text.
        If no bond exists, create a weak RELATED_TO bond.
        """
        bonds_strengthened = 0

        for i, aid1 in enumerate(atom_ids):
            for aid2 in atom_ids[i+1:]:
                if aid1 not in self.atoms or aid2 not in self.atoms:
                    continue

                atom1 = self.atoms[aid1]
                atom2 = self.atoms[aid2]

                # Check for existing bond in either direction
                existing_bond = None
                for bond in atom1.get_all_bonds():
                    if bond.target_id == aid2:
                        existing_bond = bond
                        break

                if existing_bond:
                    # Strengthen existing bond
                    existing_bond.strength = min(1.0, existing_bond.strength + 0.05)
                    existing_bond.access_count += 1
                    existing_bond.last_accessed = time.time()
                    bonds_strengthened += 1
                else:
                    # Create weak RELATED_TO bond (starts in outer shell)
                    self._create_bond(
                        aid1, aid2,
                        RelationType.RELATED_TO,
                        strength=0.2,  # Start weak
                        source=f"cooccur_{source}"
                    )
                    bonds_strengthened += 1

        return bonds_strengthened

    def learn_from_insight(self, insight: Dict) -> bool:
        """
        Learn from a structured insight (from Claude/Axel).

        insight format:
        {
            'subject': 'concept name',
            'relation': 'causes/has_part/etc',
            'object': 'another concept',
            'confidence': 0.8
        }
        """
        subject = insight.get('subject', '').lower()
        relation = insight.get('relation', 'related_to')
        obj = insight.get('object', '').lower()
        confidence = insight.get('confidence', 0.5)

        # Find matching atoms
        subject_id = None
        object_id = None

        for atom_id, atom in self.atoms.items():
            if atom.name.lower() == subject:
                subject_id = atom_id
            if atom.name.lower() == obj:
                object_id = atom_id

        if subject_id and object_id:
            rel_type = RelationType.from_mynd_link(relation)
            self._create_bond(subject_id, object_id, rel_type,
                            strength=confidence, source="insight")
            return True

        return False

    # =========================================================================
    # ASE ENCODING - Generate nuclear/shell vectors
    # =========================================================================

    def encode_atom(self, atom_id: str) -> bool:
        """
        Encode an atom's embedding into ASE nuclear/shell vectors.

        Requires the atom to have a base embedding (from MYND).

        Returns:
            True if encoding succeeded
        """
        if atom_id not in self.atoms:
            return False

        atom = self.atoms[atom_id]
        if atom.embedding is None:
            return False

        try:
            # Convert embedding to tensor
            embedding_tensor = torch.tensor(atom.embedding, dtype=torch.float32)

            # Encode through ASE
            with torch.no_grad():
                result = self.atomic_encoder.forward(embedding_tensor)

            # Store results in atom
            atom.nuclear = result['nuclear'].cpu().numpy().squeeze()
            atom.context_shell = result['shell'].cpu().numpy().squeeze()
            atom.semantic_charge = float(result['polarity'].cpu().item())
            atom.charge_magnitude = float(result['magnitude'].cpu().item())

            return True
        except Exception as e:
            logger.warning(f"Failed to encode atom {atom_id}: {e}")
            return False

    def encode_all_atoms(self) -> Dict[str, int]:
        """
        Encode all atoms that have embeddings.

        Returns:
            Dict with encoding stats
        """
        encoded = 0
        skipped = 0
        failed = 0

        for atom_id, atom in self.atoms.items():
            if atom.embedding is None:
                skipped += 1
                continue

            if self.encode_atom(atom_id):
                encoded += 1
            else:
                failed += 1

        logger.info(f"ASE encoding: {encoded} encoded, {skipped} skipped (no embedding), {failed} failed")

        return {
            'encoded': encoded,
            'skipped': skipped,
            'failed': failed
        }

    def compute_semantic_similarity(self, atom_id1: str, atom_id2: str,
                                     use_nuclear: bool = True) -> float:
        """
        Compute semantic similarity between two atoms using ASE vectors.

        Args:
            atom_id1, atom_id2: Atom IDs to compare
            use_nuclear: If True, use nuclear (identity), else use shell (context)

        Returns:
            Cosine similarity (-1 to 1)
        """
        if atom_id1 not in self.atoms or atom_id2 not in self.atoms:
            return 0.0

        atom1 = self.atoms[atom_id1]
        atom2 = self.atoms[atom_id2]

        if use_nuclear:
            v1, v2 = atom1.nuclear, atom2.nuclear
        else:
            v1, v2 = atom1.context_shell, atom2.context_shell

        if v1 is None or v2 is None:
            return 0.0

        # Cosine similarity
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    def get_learning_stats(self) -> Dict:
        """Get ASE learning statistics."""
        atoms_with_nuclear = sum(1 for a in self.atoms.values() if a.nuclear is not None)
        atoms_with_charge = sum(1 for a in self.atoms.values() if a.semantic_charge != 0)

        return {
            **self._learning_stats,
            'total_atoms': len(self.atoms),
            'atoms_with_nuclear': atoms_with_nuclear,
            'atoms_with_semantic_charge': atoms_with_charge,
            'encoder_device': str(self.atomic_encoder.device),
        }

    def compute_sentence_charge(self, text: str) -> Dict:
        """
        Compute the overall semantic charge of a sentence.

        Uses ASE charge propagation with negation and intensity detection.

        Returns:
            Dict with charge analysis
        """
        # Find mentioned atoms and their charges
        text_lower = text.lower()
        mentioned = self._find_mentioned_atoms(text_lower)

        if not mentioned:
            return {
                'sentence_charge': 0.0,
                'confidence': 0.0,
                'atoms_found': 0,
                'negated': False
            }

        # Gather atom charges
        word_charges = []
        for atom_id in mentioned:
            atom = self.atoms[atom_id]
            word_charges.append((atom.name, atom.effective_charge))

        # Detect negation
        is_negated, _ = self.charge_propagator.detect_negation(text)
        intensity = self.charge_propagator.get_intensity_modifier(text)

        # Compose sentence charge
        base_charge = self.charge_propagator.compose_sentence_charge(word_charges)

        # Apply negation and intensity
        final_charge = self.charge_propagator.propagate_charge(base_charge, text)

        return {
            'sentence_charge': final_charge,
            'base_charge': base_charge,
            'negated': is_negated,
            'intensity': intensity,
            'atoms_found': len(mentioned),
            'atom_charges': word_charges[:5],  # Top 5
            'confidence': sum(self.atoms[aid].charge_magnitude for aid in mentioned) / len(mentioned)
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

# Global instance for the brain server
_asa_instance: Optional[MYNDLivingASA] = None


def get_asa() -> MYNDLivingASA:
    """Get or create the global ASA instance."""
    global _asa_instance
    if _asa_instance is None:
        _asa_instance = MYNDLivingASA()
    return _asa_instance


def initialize_asa(map_data: Dict, embeddings: Dict = None) -> MYNDLivingASA:
    """Initialize ASA with map data."""
    asa = get_asa()
    asa.convert_map_to_asa(map_data, embeddings)
    return asa
