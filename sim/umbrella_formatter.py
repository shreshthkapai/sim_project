"""
umbrella_formatter.py

Format readout results as hierarchical umbrella output:
- Categories sorted by probability
- Top-k entities nested under each category
- Clean text and dict representations
"""

from typing import Dict, List, Tuple, Optional


class UmbrellaFormatter:
    """Formats probabilistic readout as hierarchical umbrella structure."""
    
    def __init__(self, readout):
        """
        Initialize formatter.
        
        Args:
            readout: ProbabilisticReadout instance
        """
        self.readout = readout
    
    def format_as_dict(self, 
                      top_categories: int = 5,
                      top_entities_per_category: int = 5,
                      min_category_prob: float = 0.01) -> Dict:
        """
        Format as nested dictionary.
        
        Args:
            top_categories: Number of top categories to include
            top_entities_per_category: Number of entities per category
            min_category_prob: Minimum probability threshold for categories
            
        Returns:
            Dict with hierarchical structure
        """
        output = {
            'categories': [],
            'metadata': {
                'total_categories': len(self.readout.category_probs),
                'total_entities': len(self.readout.marginal_probs),
                'top_category': self.readout.get_summary()['top_category'][0]
            }
        }
        
        # Get top categories
        top_cats = self.readout.get_category_distribution(top_k=top_categories)
        
        for category, cat_prob in top_cats:
            # Skip if below threshold
            if cat_prob < min_category_prob:
                continue
            
            # Get top entities for this category
            entities = self.readout.get_entities_for_category(
                category, 
                top_k=top_entities_per_category
            )
            
            # Format entity list
            entity_list = [
                {
                    'name': ent_name.replace('entity_', '').replace('_', ' ').title(),
                    'probability': float(ent_prob)
                }
                for ent_name, ent_prob in entities
            ]
            
            # Add category to output
            output['categories'].append({
                'name': category,
                'probability': float(cat_prob),
                'entities': entity_list
            })
        
        return output
    
    def format_as_text(self,
                      top_categories: int = 5,
                      top_entities_per_category: int = 5,
                      min_category_prob: float = 0.01) -> str:
        """
        Format as human-readable text.
        
        Args:
            top_categories: Number of top categories to include
            top_entities_per_category: Number of entities per category
            min_category_prob: Minimum probability threshold for categories
            
        Returns:
            Formatted text string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("PREDICTION RESULTS (UMBRELLA FORMAT)")
        lines.append("=" * 60)
        lines.append("")
        
        # Get top categories
        top_cats = self.readout.get_category_distribution(top_k=top_categories)
        
        for category, cat_prob in top_cats:
            # Skip if below threshold
            if cat_prob < min_category_prob:
                continue
            
            # Category header
            lines.append(f"Category: {category} (P = {cat_prob:.3f})")
            
            # Get top entities for this category
            entities = self.readout.get_entities_for_category(
                category,
                top_k=top_entities_per_category
            )
            
            # Format entities
            if entities:
                for ent_name, ent_prob in entities:
                    # Clean entity name (remove entity_ prefix, replace underscores)
                    clean_name = ent_name.replace('entity_', '').replace('_', ' ').title()
                    lines.append(f"  - {clean_name} (P = {ent_prob:.3f})")
            else:
                lines.append("  (no entities)")
            
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def format_compact(self, top_categories: int = 3) -> str:
        """
        Compact one-line summary.
        
        Args:
            top_categories: Number of categories to show
            
        Returns:
            Compact string
        """
        top_cats = self.readout.get_category_distribution(top_k=top_categories)
        
        parts = [f"{cat}({prob:.2f})" for cat, prob in top_cats]
        return " | ".join(parts)


# Example usage
if __name__ == "__main__":
    from graph_setup import FrozenGraphLoader
    from state import SimulationState
    from injection import QueryInjector
    from dynamics import DynamicsEngine
    from readout import ProbabilisticReadout
    
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
    
    # Format
    formatter = UmbrellaFormatter(readout)
    
    # Text format
    print(formatter.format_as_text(top_categories=5, top_entities_per_category=5))
    
    # Compact format
    print("\nCompact:", formatter.format_compact())
    
    # Dict format (for API)
    import json
    output_dict = formatter.format_as_dict(top_categories=3, top_entities_per_category=3)
    print("\nJSON:")
    print(json.dumps(output_dict, indent=2))