"""
graph_setup.py

Pre-simulation setup (one-time initialization):
- Load frozen knowledge graph (read-only)
- Build inhibition matrix from competition
- Create node index mappings
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import networkx as nx

import sys
import os
# Add kg folder to path so pickle can find knowledge_graph module
kg_path = os.path.join(os.path.dirname(__file__), '..', 'kg')
sys.path.insert(0, os.path.abspath(kg_path))


class FrozenGraphLoader:
    """Loads and freezes a trained knowledge graph for simulation."""
    
    def __init__(self, checkpoint_path: str):
        """
        Load graph from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file (e.g., "checkpoint_83.kg")
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        print(f"Loading checkpoint from {self.checkpoint_path}...")
        with open(self.checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Extract knowledge graph (read-only)
        self.kg = checkpoint['kg']
        self.graph = self.kg.G  # NetworkX DiGraph
        
        # Freeze the graph - mark as read-only
        self._freeze_graph()
        
        # Extract node information
        self.user_node = self._get_user_node()
        self.category_nodes = self._get_category_nodes()
        self.entity_nodes = self._get_entity_nodes()
        self.all_nodes = [self.user_node] + self.category_nodes + self.entity_nodes
        
        # Create node mappings
        self.node_to_idx, self.idx_to_node = self._build_node_mappings()
        self.num_nodes = len(self.all_nodes)
        
        print(f"Loaded graph: {self.num_nodes} nodes "
              f"({len(self.category_nodes)} categories, {len(self.entity_nodes)} entities)")
        
        # Extract baseline edge weights
        self.baseline_weights = self._extract_baseline_weights()
        
        # Build inhibition matrix from competition
        self.inhibition_matrix = self._build_inhibition_matrix()
        
        print("Graph setup complete. Graph is now frozen (read-only).")
    
    def _freeze_graph(self):
        """Mark graph as read-only to prevent modifications."""
        # Add a flag to prevent accidental modifications
        self.graph.graph['frozen'] = True
        self.graph.graph['frozen_timestamp'] = str(Path(self.checkpoint_path).stat().st_mtime)
    
    def _get_user_node(self) -> str:
        """Find the USER node."""
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'user':
                return node
        raise ValueError("No USER node found in graph!")
    
    def _get_category_nodes(self) -> list:
        """Get all category nodes."""
        categories = []
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'category':
                categories.append(node)
        return sorted(categories)  # Consistent ordering
    
    def _get_entity_nodes(self) -> list:
        """Get all entity nodes."""
        entities = []
        for node, data in self.graph.nodes(data=True):
            if node.startswith('entity_'):
                entities.append(node)
        return sorted(entities)  # Consistent ordering
    
    def _build_node_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Create bidirectional node ↔ index mappings."""
        node_to_idx = {node: idx for idx, node in enumerate(self.all_nodes)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        return node_to_idx, idx_to_node
    
    def _extract_baseline_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Extract all baseline edge weights from frozen graph.
        
        Returns:
            Dict mapping (source_idx, target_idx) → baseline_weight
        """
        baseline_weights = {}
        
        for source, target, data in self.graph.edges(data=True):
            if source in self.node_to_idx and target in self.node_to_idx:
                source_idx = self.node_to_idx[source]
                target_idx = self.node_to_idx[target]
                weight = data.get('weight', 0.0)
                baseline_weights[(source_idx, target_idx)] = weight
        
        print(f"Extracted {len(baseline_weights)} baseline edge weights")
        return baseline_weights
    
    def _build_inhibition_matrix(self) -> np.ndarray:
        """
        Build inhibition matrix from category competition.
        
        For categories: I_ik = competition(C_i, C_k)
        where competition(A, B) = 1.0 - (times_A_and_B_together / times_A_appeared)
        
        Returns:
            n x n inhibition matrix (sparse - only categories compete)
        """
        n = self.num_nodes
        I = np.zeros((n, n), dtype=np.float32)
        
        # Compute category co-occurrence from search history
        category_counts = {cat: 0 for cat in self.category_nodes}
        cooccurrence_counts = {cat: {other: 0 for other in self.category_nodes} 
                               for cat in self.category_nodes}
        
        for search in self.kg.search_history:
            search_categories = list(search.get('categories', {}).keys())
            
            # Count individual appearances
            for cat in search_categories:
                if cat in category_counts:
                    category_counts[cat] += 1
            
            # Count co-occurrences (exclude self-pairs)
            for cat_a in search_categories:
                for cat_b in search_categories:
                    if cat_a != cat_b:  # Don't count self co-occurrence
                        if cat_a in cooccurrence_counts and cat_b in cooccurrence_counts:
                            cooccurrence_counts[cat_a][cat_b] += 1
        
        # Build competition matrix for categories
        for cat_a in self.category_nodes:
            for cat_b in self.category_nodes:
                if cat_a == cat_b:
                    continue  # No self-inhibition
                
                if category_counts[cat_a] == 0:
                    competition = 1.0  # Default: full competition if no data
                else:
                    co_occur = cooccurrence_counts[cat_a][cat_b]
                    total_a = category_counts[cat_a]
                    competition = 1.0 - (co_occur / total_a)
                
                # Map to indices
                idx_a = self.node_to_idx[cat_a]
                idx_b = self.node_to_idx[cat_b]
                I[idx_a, idx_b] = competition
        
        print(f"Built inhibition matrix: {n}x{n} (category competition only)")
        return I
    
    def get_baseline_weight(self, source_idx: int, target_idx: int) -> float:
        """Get baseline weight for an edge (returns 0.0 if edge doesn't exist)."""
        return self.baseline_weights.get((source_idx, target_idx), 0.0)
    
    def get_neighbors(self, node_idx: int) -> list:
        """Get list of neighbor indices (nodes with incoming edges to node_idx)."""
        node_id = self.idx_to_node[node_idx]
        neighbors = []
        
        for pred in self.graph.predecessors(node_id):
            if pred in self.node_to_idx:
                neighbors.append(self.node_to_idx[pred])
        
        return neighbors
    
    def verify_integrity(self):
        """Verify graph hasn't been modified during simulation."""
        if not self.graph.graph.get('frozen'):
            raise RuntimeError("Graph has been unfrozen - integrity violated!")
        
        # Check node count hasn't changed
        current_nodes = len(list(self.graph.nodes()))
        if current_nodes != self.num_nodes:
            raise RuntimeError(f"Node count changed: {self.num_nodes} → {current_nodes}")
        
        # Check edge count hasn't changed
        current_edges = len(list(self.graph.edges()))
        if current_edges != len(self.baseline_weights):
            raise RuntimeError(f"Edge count changed: {len(self.baseline_weights)} → {current_edges}")
        
        return True


# Example usage
if __name__ == "__main__":
    # Test loading
    loader = FrozenGraphLoader("checkpoint_83.kg")
    
    print(f"\nGraph Summary:")
    print(f"  Total nodes: {loader.num_nodes}")
    print(f"  USER: {loader.user_node}")
    print(f"  Categories: {loader.category_nodes}")
    print(f"  Entities: {len(loader.entity_nodes)} (first 5: {loader.entity_nodes[:5]})")
    print(f"  Baseline edges: {len(loader.baseline_weights)}")
    print(f"  Inhibition matrix shape: {loader.inhibition_matrix.shape}")
    
    # Verify integrity
    loader.verify_integrity()
    print("\n✓ Graph integrity verified")