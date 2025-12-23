"""
Living ASA Integration for MYND
================================
Semantic backbone with metabolism.

This module converts MYND's node tree into a living semantic graph
with shells, bonds, charge, energy, and continuous metabolism.

Integration points:
- /map/sync → convert_map_to_asa()
- /brain/context → get_asa_insights()
- Background → metabolism heartbeat
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import time
import math
import threading
import logging

logger = logging.getLogger(__name__)


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
    """A LIVING semantic atom - wraps a MYND node."""
    id: str  # MYND node ID
    name: str  # Node label

    # Original MYND data reference
    mynd_node: Dict = field(default_factory=dict)

    # Structure
    shells: Dict[int, List[Bond]] = field(default_factory=lambda: {1: [], 2: [], 3: [], 4: []})
    valence: Dict[RelationType, ValenceSlot] = field(default_factory=dict)

    # Dynamic state
    charge: float = 0.0
    energy: float = 0.0
    mass: float = 1.0

    # Embedding (from MYND's embeddings)
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
            'charge': round(self.charge, 3),
            'energy': round(self.energy, 3),
            'mass': round(self.mass, 2),
            'total_bonds': self.count_bonds(),
            'unfilled_slots': self.count_unfilled_slots(),
            'shells': {
                f'shell_{i}': len(bonds) for i, bonds in self.shells.items()
            },
            'importance': self.importance,
            'depth': self.depth,
        }


# =============================================================================
# LIVING ASA FOR MYND
# =============================================================================

class MYNDLivingASA:
    """
    Living Atomic Semantic Architecture for MYND.

    Converts MYND's node tree into a metabolizing semantic graph.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

        # Atom storage
        self.atoms: Dict[str, SemanticAtom] = {}

        # Shell configuration
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

        logger.info("MYNDLivingASA initialized")

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
        """Update charge based on valence satisfaction."""
        unfilled = atom.count_unfilled_slots()
        contradictions = atom.contradiction_count

        # Also factor in MYND importance
        importance_factor = (atom.importance - 0.5) * 0.2

        seeking_pull = -0.08 * unfilled
        repelling_push = 0.3 * contradictions

        atom.charge = max(-1.0, min(1.0, seeking_pull + repelling_push + importance_factor))

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
        """Get overall ASA statistics."""
        total_bonds = sum(a.count_bonds() for a in self.atoms.values())
        total_gaps = sum(a.count_unfilled_slots() for a in self.atoms.values())

        shell_counts = defaultdict(int)
        for atom in self.atoms.values():
            for shell_idx, bonds in atom.shells.items():
                shell_counts[shell_idx] += len(bonds)

        return {
            'atom_count': len(self.atoms),
            'total_bonds': total_bonds,
            'total_gaps': total_gaps,
            'avg_energy': sum(a.energy for a in self.atoms.values()) / max(len(self.atoms), 1),
            'avg_mass': sum(a.mass for a in self.atoms.values()) / max(len(self.atoms), 1),
            'hot_atoms': len([a for a in self.atoms.values() if a.energy > 0.3]),
            'seeking_atoms': len([a for a in self.atoms.values() if a.charge < -0.3]),
            'bonds_by_shell': dict(shell_counts),
            'metabolism_running': self._running,
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
