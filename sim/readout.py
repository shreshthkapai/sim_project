"""
readout.py

Probabilistic readout from simulation results:
- Category probabilities (softmax)
- Entity probabilities given category (multiplicative)
- Full entity probabilities (marginalization)
"""

import numpy as np
from typing import Dict, List, Tuple

from sim_utils import softmax, get_top_k


class ProbabilisticReadout:
    """Computes probability distributions from final simulation state."""
    
    def __init__(self, graph_loader, state):
        """
        Initialize readout.
        
        Args:
            graph_loader: FrozenGraphLoader instance
            state: SimulationState instance (after dynamics)
        """
        self.graph = graph_loader
        self.state = state
        
        # Extract final activations
        self.x_final = state.x.copy()
        
        # Compute category probabilities
        self.category_probs = self._compute_category_probabilities()
        
        # Compute entity probabilities
        self.entity_probs_given_category = self._compute_entity_probabilities_given_category()
        self.entity_probs_marginal = self._compute_marginal_entity_probabilities()
        
        print("Readout computed")
    
    def _compute_category_probabilities(self) -> Dict[str, float]:
        """
        Compute category probabilities via softmax over activations.
        
        P(C_j | scenario) = exp(x_C_j) / Σ_k exp(x_C_k)
        
        Returns:
            Dict mapping category_name → probability
        """
        # Extract category activations
        category_activations = np.array([
            self.x_final[self.graph.node_to_idx[cat]]
            for cat in self.graph.category_nodes
        ])
        
        # Softmax
        probs = softmax(category_activations)
        
        # Map to category names
        category_probs = {
            cat: float(probs[i])
            for i, cat in enumerate(self.graph.category_nodes)
        }
        
        return category_probs
    
    def _compute_entity_probabilities_given_category(self) -> Dict[str, Dict[str, float]]:
        """
        Compute P(E_i | C_j) for all entities and categories.
        
        P(E_i | C_j) = (w_E→C * w_C→E * x_E_normalized) / Z_j
        
        Returns:
            Dict mapping category_name → {entity_name: probability}
        """
        entity_probs = {}
        x_max = self.state.params.x_max
        
        for category in self.graph.category_nodes:
            cat_idx = self.graph.node_to_idx[category]
            
            # Find all entities connected to this category
            entities_in_category = []
            numerators = []
            
            for entity in self.graph.entity_nodes:
                ent_idx = self.graph.node_to_idx[entity]
                
                # Get edge weights (baseline only, frozen graph)
                w_belongs = self.graph.get_baseline_weight(ent_idx, cat_idx)  # E→C
                w_suggests = self.graph.get_baseline_weight(cat_idx, ent_idx)  # C→E
                
                # Skip if either edge missing
                if w_belongs == 0.0 or w_suggests == 0.0:
                    continue
                
                # Get activation (normalized to [0,1])
                x_norm = self.x_final[ent_idx] / x_max
                
                # Compute numerator
                numerator = w_belongs * w_suggests * x_norm
                
                if numerator > 0:
                    entities_in_category.append(entity)
                    numerators.append(numerator)
            
            # Normalize to probabilities
            if len(numerators) > 0:
                total = sum(numerators)
                probs = [n / total for n in numerators]
                
                entity_probs[category] = {
                    entities_in_category[i]: probs[i]
                    for i in range(len(entities_in_category))
                }
            else:
                entity_probs[category] = {}
        
        return entity_probs
    
    def _compute_marginal_entity_probabilities(self) -> Dict[str, float]:
        """
        Compute P(E_i) via marginalization over categories.
        
        P(E_i) = Σ_j P(C_j) * P(E_i | C_j)
        
        Returns:
            Dict mapping entity_name → probability
        """
        marginal_probs = {}
        
        for entity in self.graph.entity_nodes:
            prob = 0.0
            
            # Sum over all categories
            for category in self.graph.category_nodes:
                p_cat = self.category_probs[category]
                
                if category in self.entity_probs_given_category:
                    p_ent_given_cat = self.entity_probs_given_category[category].get(entity, 0.0)
                    prob += p_cat * p_ent_given_cat
            
            if prob > 0:
                marginal_probs[entity] = prob
        
        return marginal_probs
    
    def get_category_distribution(self, top_k: int = None) -> Dict[str, float]:
        """
        Get category probability distribution.
        
        Args:
            top_k: Return only top-k categories (None = all)
            
        Returns:
            Dict mapping category → probability
        """
        if top_k is None:
            return self.category_probs
        
        return dict(get_top_k(self.category_probs, k=top_k))
    
    def get_entities_for_category(self, category: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top entities for a category.
        
        Args:
            category: Category name
            top_k: Number of top entities
            
        Returns:
            List of (entity_name, probability) tuples
        """
        if category not in self.entity_probs_given_category:
            return []
        
        entity_probs = self.entity_probs_given_category[category]
        return get_top_k(entity_probs, k=top_k)
    
    def get_top_entities(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top entities by marginal probability.
        
        Args:
            top_k: Number of top entities
            
        Returns:
            List of (entity_name, probability) tuples
        """
        return get_top_k(self.marginal_probs, k=top_k)
    
    def get_summary(self) -> Dict:
        """Get summary statistics of readout."""
        return {
            'num_categories': len(self.category_probs),
            'num_entities_total': len(self.marginal_probs),
            'top_category': max(self.category_probs.items(), key=lambda x: x[1]),
            'entropy_categories': self._compute_entropy(list(self.category_probs.values())),
            'total_probability_mass': sum(self.marginal_probs.values())
        }
    
    def _compute_entropy(self, probs: List[float]) -> float:
        """Compute Shannon entropy of probability distribution."""
        probs = np.array(probs)
        probs = probs[probs > 0]  # Filter zeros
        return -np.sum(probs * np.log(probs))


# Example usage
if __name__ == "__main__":
    from graph_setup import FrozenGraphLoader
    from state import SimulationState
    from injection import QueryInjector
    from dynamics import DynamicsEngine
    
    # Load graph
    loader = FrozenGraphLoader("checkpoint_83.kg")
    
    # Create state
    state = SimulationState(loader.num_nodes)
    
    # Inject query
    injector = QueryInjector(loader)
    injector.inject_query("What should I learn about machine learning?", state)
    
    # Run dynamics
    engine = DynamicsEngine(loader, state)
    results = engine.run_simulation()
    
    # Readout
    readout = ProbabilisticReadout(loader, state)
    
    print("\n" + "="*60)
    print("READOUT RESULTS")
    print("="*60)
    
    # Category distribution
    print("\nCategory Probabilities:")
    for cat, prob in readout.get_category_distribution(top_k=5):
        print(f"  {cat}: {prob:.3f}")
    
    # Top entities per category
    print("\nTop Entities by Category:")
    for cat, prob in readout.get_category_distribution(top_k=3):
        print(f"\n{cat} (P={prob:.3f}):")
        entities = readout.get_entities_for_category(cat, top_k=5)
        for ent, ent_prob in entities:
            print(f"  - {ent}: {ent_prob:.3f}")
    
    # Summary
    print("\nSummary:")
    summary = readout.get_summary()
    for key, val in summary.items():
        print(f"  {key}: {val}")