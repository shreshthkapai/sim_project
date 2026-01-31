"""
simulator.py

Main simulator orchestrator:
- Loads frozen graph (one-time)
- Runs simulations on natural language queries
- Manages state lifecycle
- Returns formatted predictions
"""

import time
from typing import Dict, List, Optional, Union

from graph_setup import FrozenGraphLoader
from state import SimulationState
from injection import QueryInjector
from dynamics import DynamicsEngine
from readout import ProbabilisticReadout
from umbrella_formatter import UmbrellaFormatter
from sim_config import SimulationParameters


class Simulator:
    """Main interface for running simulations on temporal knowledge graphs."""
    
    def __init__(self, 
                 checkpoint_path: str,
                 params: Optional[SimulationParameters] = None,
                 verbose: bool = True):
        """
        Initialize simulator with frozen graph.
        
        Args:
            checkpoint_path: Path to trained KG checkpoint
            params: Simulation parameters (uses defaults if None)
            verbose: Print progress messages
        """
        self.checkpoint_path = checkpoint_path
        self.params = params if params is not None else SimulationParameters()
        self.verbose = verbose
        
        # Load frozen graph (one-time, reused across simulations)
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"INITIALIZING SIMULATOR")
            print(f"{'='*60}")
        
        self.graph_loader = FrozenGraphLoader(checkpoint_path)
        self.injector = QueryInjector(self.graph_loader)
        
        # Statistics
        self.total_simulations = 0
        self.total_time = 0.0
        
        if self.verbose:
            print(f"Simulator ready.")
            print(f"{'='*60}\n")
    
    def run_query(self,
                  query: str,
                  format: str = 'text',
                  top_categories: int = 5,
                  top_entities_per_category: int = 5,
                  macrostep_size: float = 0.1) -> Union[str, Dict]:
        """
        Run simulation for a natural language query.
        
        Args:
            query: Natural language query
            format: Output format ('text', 'dict', 'compact')
            top_categories: Number of top categories in output
            top_entities_per_category: Number of entities per category
            macrostep_size: Macrostep size for dynamics (seconds)
            
        Returns:
            Formatted prediction results (string or dict based on format)
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RUNNING SIMULATION #{self.total_simulations + 1}")
            print(f"Query: '{query}'")
            print(f"{'='*60}")
        
        # Step 1: Initialize state
        state = SimulationState(self.graph_loader.num_nodes, self.params)
        
        # Step 2: Inject query
        injection_result = self.injector.inject_query(query, state)
        
        if injection_result['total_injected'] == 0:
            if self.verbose:
                print("⚠️  No entities/categories injected - query may not match graph")
        
        # Step 3: Run dynamics
        engine = DynamicsEngine(self.graph_loader, state, self.params)
        dynamics_result = engine.run_simulation(macrostep_size=macrostep_size)
        
        # Step 4: Readout probabilities
        readout = ProbabilisticReadout(self.graph_loader, state)
        
        # Step 5: Format output
        formatter = UmbrellaFormatter(readout)
        
        if format == 'text':
            output = formatter.format_as_text(
                top_categories=top_categories,
                top_entities_per_category=top_entities_per_category
            )
        elif format == 'dict':
            output = formatter.format_as_dict(
                top_categories=top_categories,
                top_entities_per_category=top_entities_per_category
            )
        elif format == 'compact':
            output = formatter.format_compact(top_categories=top_categories)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Step 6: Cleanup
        state.cleanup()
        
        # Step 7: Verify graph integrity
        self.graph_loader.verify_integrity()
        
        # Update statistics
        elapsed = time.time() - start_time
        self.total_simulations += 1
        self.total_time += elapsed
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SIMULATION COMPLETE")
            print(f"{'='*60}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Converged: {dynamics_result['converged']}")
            print(f"{'='*60}\n")
        
        return output
    
    def run_batch(self,
                  queries: List[str],
                  format: str = 'dict',
                  **kwargs) -> List[Union[str, Dict]]:
        """
        Run simulations for multiple queries.
        
        Args:
            queries: List of natural language queries
            format: Output format for each result
            **kwargs: Additional arguments passed to run_query
            
        Returns:
            List of prediction results
        """
        results = []
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"BATCH SIMULATION: {len(queries)} queries")
            print(f"{'='*60}\n")
        
        for i, query in enumerate(queries):
            if self.verbose:
                print(f"\nQuery {i+1}/{len(queries)}: {query}")
            
            result = self.run_query(query, format=format, **kwargs)
            results.append(result)
        
        if self.verbose:
            avg_time = self.total_time / self.total_simulations
            print(f"\n{'='*60}")
            print(f"BATCH COMPLETE")
            print(f"{'='*60}")
            print(f"Total queries: {len(queries)}")
            print(f"Total time: {self.total_time:.2f}s")
            print(f"Average time: {avg_time:.2f}s per query")
            print(f"{'='*60}\n")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get simulator usage statistics."""
        avg_time = self.total_time / self.total_simulations if self.total_simulations > 0 else 0.0
        
        return {
            'total_simulations': self.total_simulations,
            'total_time_seconds': self.total_time,
            'average_time_seconds': avg_time,
            'graph_nodes': self.graph_loader.num_nodes,
            'graph_edges': len(self.graph_loader.baseline_weights),
            'checkpoint': str(self.checkpoint_path)
        }
    
    def reset_statistics(self):
        """Reset usage statistics."""
        self.total_simulations = 0
        self.total_time = 0.0


# Example usage
if __name__ == "__main__":
    # Initialize simulator
    sim = Simulator("checkpoint_83.kg", verbose=True)
    
    # Single query
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Query")
    print("="*60)
    
    result = sim.run_query(
        query="What should I learn about machine learning?",
        format='text',
        top_categories=5,
        top_entities_per_category=5
    )
    print(result)
    
    # Batch queries
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Queries")
    print("="*60)
    
    queries = [
        "Best tech stack for web development",
        "How to plan a trip to Paris?",
        "What are emerging trends in AI?"
    ]
    
    results = sim.run_batch(queries, format='compact')
    
    for i, (query, result) in enumerate(zip(queries, results)):
        print(f"\nQuery {i+1}: {query}")
        print(f"Result: {result}")
    
    # Statistics
    print("\n" + "="*60)
    print("SIMULATOR STATISTICS")
    print("="*60)
    
    stats = sim.get_statistics()
    for key, val in stats.items():
        print(f"  {key}: {val}")