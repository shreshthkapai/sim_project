"""
utils.py

Utility functions for simulation layer.
"""

import numpy as np
from typing import Dict, List, Tuple


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Numerically stable softmax with temperature scaling.
    
    Args:
        x: Input array
        temperature: Temperature parameter (lower = more peaked distribution)
        
    Returns:
        Softmax probabilities (sum to 1.0)
    """
    # Scale by temperature
    x_scaled = x / temperature
    
    # Numerical stability: subtract max
    x_shifted = x_scaled - np.max(x_scaled)
    
    # Compute exp
    exp_x = np.exp(x_shifted)
    
    # Normalize
    return exp_x / np.sum(exp_x)


def log_sum_exp(x: np.ndarray) -> float:
    """
    Numerically stable log-sum-exp.
    
    Computes log(sum(exp(x))) without overflow.
    """
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


def validate_activations(x: np.ndarray, node_names: List[str] = None) -> bool:
    """
    Validate activation vector for issues.
    
    Args:
        x: Activation vector
        node_names: Optional node names for debugging
        
    Returns:
        True if valid, raises exception if invalid
    """
    # Check for NaN
    if np.any(np.isnan(x)):
        nan_indices = np.where(np.isnan(x))[0]
        if node_names:
            nan_nodes = [node_names[i] for i in nan_indices[:5]]  # First 5
            raise ValueError(f"NaN detected in activations at nodes: {nan_nodes}")
        else:
            raise ValueError(f"NaN detected in activations at indices: {nan_indices[:5]}")
    
    # Check for Inf
    if np.any(np.isinf(x)):
        inf_indices = np.where(np.isinf(x))[0]
        if node_names:
            inf_nodes = [node_names[i] for i in inf_indices[:5]]
            raise ValueError(f"Inf detected in activations at nodes: {inf_nodes}")
        else:
            raise ValueError(f"Inf detected in activations at indices: {inf_indices[:5]}")
    
    # Check for negative values
    if np.any(x < 0):
        neg_indices = np.where(x < 0)[0]
        if node_names:
            neg_nodes = [node_names[i] for i in neg_indices[:5]]
            raise ValueError(f"Negative activations at nodes: {neg_nodes}")
        else:
            raise ValueError(f"Negative activations at indices: {neg_indices[:5]}")
    
    return True


def normalize_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a probability distribution (make sure it sums to 1.0).
    
    Args:
        probs: Dictionary of category/entity → probability
        
    Returns:
        Normalized probabilities
    """
    total = sum(probs.values())
    
    if total == 0:
        # Uniform distribution if all zeros
        n = len(probs)
        return {k: 1.0/n for k in probs.keys()}
    
    return {k: v/total for k, v in probs.items()}


def sparse_dict_to_matrix(sparse_dict: Dict[Tuple[int, int], float], 
                         shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert sparse dict representation to dense matrix.
    
    Args:
        sparse_dict: Dict mapping (i, j) → value
        shape: (n_rows, n_cols)
        
    Returns:
        Dense matrix
    """
    matrix = np.zeros(shape, dtype=np.float32)
    
    for (i, j), value in sparse_dict.items():
        matrix[i, j] = value
    
    return matrix


def get_top_k(items: Dict[str, float], k: int = 5, 
              min_value: float = 0.0) -> List[Tuple[str, float]]:
    """
    Get top-k items from a dict sorted by value.
    
    Args:
        items: Dict mapping item → score
        k: Number of top items to return
        min_value: Minimum value threshold
        
    Returns:
        List of (item, score) tuples sorted descending
    """
    # Filter by minimum value
    filtered = {k: v for k, v in items.items() if v >= min_value}
    
    # Sort descending
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    
    # Take top-k
    return sorted_items[:k]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero
        
    Returns:
        numerator / denominator, or default if denominator == 0
    """
    if denominator == 0:
        return default
    return numerator / denominator