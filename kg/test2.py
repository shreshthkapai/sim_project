"""
Empirical Check: Entity→Category vs Category→Entity Weight Correlation
Answers the question: Are these edges redundant or independent?
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def load_checkpoint(filename='checkpoint_FINAL.pkl'):
    """Load the final checkpoint"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['kg']

def analyze_edge_correlation(kg):
    """
    Check correlation between E→C and C→E weights
    """
    print("="*70)
    print("EMPIRICAL EDGE WEIGHT CORRELATION ANALYSIS")
    print("="*70)
    print()
    
    # Collect all entity-category edge pairs
    entity_to_cat_weights = []
    cat_to_entity_weights = []
    entity_cat_pairs = []
    
    entities = [n for n, d in kg.G.nodes(data=True) if d.get('node_type') == 'entity']
    
    print(f"Analyzing {len(entities):,} entities across {len(kg.categories)} categories...")
    print()
    
    for entity in entities:
        for category in kg.categories:
            # Check if both edges exist
            has_e_to_c = kg.G.has_edge(entity, category)
            has_c_to_e = kg.G.has_edge(category, entity)
            
            if has_e_to_c and has_c_to_e:
                w_e_to_c = kg.G[entity][category]['weight']
                w_c_to_e = kg.G[category][entity]['weight']
                
                entity_to_cat_weights.append(w_e_to_c)
                cat_to_entity_weights.append(w_c_to_e)
                entity_cat_pairs.append((entity, category, w_e_to_c, w_c_to_e))
    
    print(f"Found {len(entity_cat_pairs):,} entity-category pairs with both edge directions")
    print()
    
    if len(entity_cat_pairs) < 10:
        print("ERROR: Not enough paired edges to analyze!")
        return
    
    # Convert to numpy arrays
    e_to_c = np.array(entity_to_cat_weights)
    c_to_e = np.array(cat_to_entity_weights)
    
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(e_to_c, c_to_e)
    spearman_corr, spearman_p = spearmanr(e_to_c, c_to_e)
    
    print("CORRELATION ANALYSIS:")
    print(f"  Pearson correlation:  r = {pearson_corr:.4f} (p = {pearson_p:.2e})")
    print(f"  Spearman correlation: ρ = {spearman_corr:.4f} (p = {spearman_p:.2e})")
    print()
    
    # Interpret results
    print("INTERPRETATION:")
    print("-"*70)
    
    if abs(pearson_corr) > 0.9:
        print("✓ HIGHLY CORRELATED (r > 0.9)")
        print("  → E→C and C→E weights are basically the same")
        print("  → RECOMMENDATION: Use suggests only (w_C→E)")
        print("  → Prediction should use: w_USER→C * w_C→E")
        print("  → The belongs_to (E→C) edge is redundant for prediction")
    
    elif abs(pearson_corr) > 0.7:
        print("✓ STRONGLY CORRELATED (0.7 < r < 0.9)")
        print("  → E→C and C→E weights are related but not identical")
        print("  → RECOMMENDATION: Use weighted average")
        print("  → Prediction could use: w_USER→C * (0.7*w_C→E + 0.3*w_E→C)")
    
    elif abs(pearson_corr) > 0.5:
        print("✓ MODERATELY CORRELATED (0.5 < r < 0.7)")
        print("  → E→C and C→E weights are somewhat related")
        print("  → RECOMMENDATION: Use both (multiplicative)")
        print("  → Prediction should use: w_USER→C * w_C→E * w_E→C")
    
    elif abs(pearson_corr) > 0.3:
        print("✓ WEAKLY CORRELATED (0.3 < r < 0.5)")
        print("  → E→C and C→E weights are mostly independent")
        print("  → RECOMMENDATION: Use both (multiplicative)")
        print("  → Prediction should use: w_USER→C * w_C→E * w_E→C")
        print("  → These edges capture different information!")
    
    else:
        print("✓ UNCORRELATED (r < 0.3)")
        print("  → E→C and C→E weights are independent")
        print("  → RECOMMENDATION: Use both (multiplicative)")
        print("  → Prediction should use: w_USER→C * w_C→E * w_E→C")
        print("  → These edges capture completely different information!")
    
    print("-"*70)
    print()
    
    # Summary statistics
    print("WEIGHT DISTRIBUTIONS:")
    print(f"  E→C (belongs_to):")
    print(f"    Mean: {np.mean(e_to_c):.4f}, Std: {np.std(e_to_c):.4f}")
    print(f"    Min: {np.min(e_to_c):.4f}, Max: {np.max(e_to_c):.4f}")
    print(f"    Median: {np.median(e_to_c):.4f}")
    print()
    print(f"  C→E (suggests):")
    print(f"    Mean: {np.mean(c_to_e):.4f}, Std: {np.std(c_to_e):.4f}")
    print(f"    Min: {np.min(c_to_e):.4f}, Max: {np.max(c_to_e):.4f}")
    print(f"    Median: {np.median(c_to_e):.4f}")
    print()
    
    # Show examples of discrepancies
    print("EXAMPLES OF EDGE WEIGHT DISCREPANCIES:")
    print()
    
    # Find pairs where weights differ significantly
    weight_diffs = np.abs(e_to_c - c_to_e)
    sorted_indices = np.argsort(weight_diffs)[::-1]
    
    print("Top 10 pairs with LARGEST weight differences:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        entity, category, w_ec, w_ce = entity_cat_pairs[idx]
        entity_label = kg.G.nodes[entity].get('label', entity)
        diff = abs(w_ec - w_ce)
        
        print(f"  {i+1}. {entity_label[:30]:30s} ↔ {category:15s}")
        print(f"     E→C: {w_ec:.4f}, C→E: {w_ce:.4f}, Diff: {diff:.4f}")
    
    print()
    print("Top 10 pairs with MOST SIMILAR weights:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[-(i+1)]
        entity, category, w_ec, w_ce = entity_cat_pairs[idx]
        entity_label = kg.G.nodes[entity].get('label', entity)
        diff = abs(w_ec - w_ce)
        
        print(f"  {i+1}. {entity_label[:30]:30s} ↔ {category:15s}")
        print(f"     E→C: {w_ec:.4f}, C→E: {w_ce:.4f}, Diff: {diff:.4f}")
    
    print()
    
    # Create visualization
    print("Creating visualization...")
    create_correlation_plot(e_to_c, c_to_e, pearson_corr)
    print("  Saved to: edge_weight_correlation.png")
    print()
    
    # Additional analysis: Check if one systematically higher
    mean_diff = np.mean(e_to_c - c_to_e)
    print("SYSTEMATIC BIAS:")
    if abs(mean_diff) > 0.05:
        if mean_diff > 0:
            print(f"  ✓ E→C weights are systematically HIGHER by {mean_diff:.4f}")
            print("    → belongs_to edges are stronger than suggests edges")
        else:
            print(f"  ✓ C→E weights are systematically HIGHER by {abs(mean_diff):.4f}")
            print("    → suggests edges are stronger than belongs_to edges")
    else:
        print(f"  ✓ No systematic bias (mean diff = {mean_diff:.4f})")
        print("    → Both edge directions have similar magnitudes")
    
    print()
    print("="*70)
    
    return pearson_corr, entity_cat_pairs


def create_correlation_plot(e_to_c, c_to_e, correlation):
    """Create scatter plot of edge weight correlation"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(e_to_c, c_to_e, alpha=0.3, s=10)
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect correlation')
    axes[0].set_xlabel('E→C Weight (belongs_to)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('C→E Weight (suggests)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Edge Weight Correlation\nr = {correlation:.4f}', 
                     fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)
    
    # Difference distribution
    diffs = e_to_c - c_to_e
    axes[1].hist(diffs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[1].axvline(np.mean(diffs), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean diff = {np.mean(diffs):.4f}')
    axes[1].set_xlabel('E→C - C→E (Weight Difference)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Weight Differences', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('edge_weight_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import sys
    
    checkpoint_file = 'checkpoint_FINAL.pkl'
    if len(sys.argv) > 1:
        checkpoint_file = sys.argv[1]
    
    print(f"Loading checkpoint: {checkpoint_file}")
    print()
    
    kg = load_checkpoint(checkpoint_file)
    print(f"Loaded graph: {kg.G.number_of_nodes():,} nodes, {kg.G.number_of_edges():,} edges")
    print()
    
    correlation, pairs = analyze_edge_correlation(kg)