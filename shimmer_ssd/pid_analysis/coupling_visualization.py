"""
Professional visualization utilities for Sinkhorn-Knopp coupling matrices.

This module provides functions to visualize and log coupling matrices to wandb
during training, helping to understand how the coupling evolves over iterations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Dict, Any
import warnings

# Try to import wandb, but make it optional
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def log_coupling_matrix_to_wandb(
    coupling_matrix: torch.Tensor,
    step: Optional[int] = None,
    prefix: str = "sinkhorn",
    cluster_names: Optional[list] = None,
    max_clusters: int = 20,
    log_stats: bool = True,
    log_heatmap: bool = True,
    colormap: str = "viridis",
    title_suffix: str = ""
) -> None:
    """
    Log Sinkhorn-Knopp coupling matrix visualization to wandb.
    
    This function creates professional visualizations of coupling matrices,
    including heatmaps and summary statistics, and logs them to wandb.
    
    Args:
        coupling_matrix: The coupling matrix tensor, shape [m, n] or [batch, m, n]
        step: Training step/iteration for logging (optional)
        prefix: Prefix for wandb log keys (default: "sinkhorn")
        cluster_names: Optional list of cluster names for axis labels
        max_clusters: Maximum number of clusters to visualize (default: 20)
        log_stats: Whether to log coupling matrix statistics (default: True)
        log_heatmap: Whether to log heatmap visualization (default: True)
        colormap: Matplotlib colormap for heatmap (default: "viridis")
        title_suffix: Additional suffix for plot title
        
    Returns:
        None
        
    Raises:
        Warning: If wandb is not available or coupling matrix is too large
    """
    if not HAS_WANDB:
        warnings.warn("wandb not available. Skipping coupling matrix logging.")
        return
    
    if wandb.run is None:
        warnings.warn("No active wandb run. Skipping coupling matrix logging.")
        return
    
    # Convert to numpy and handle batched inputs
    if isinstance(coupling_matrix, torch.Tensor):
        if coupling_matrix.dim() == 3:
            # Take mean over batch dimension
            matrix_np = coupling_matrix.mean(dim=0).detach().cpu().numpy()
        else:
            matrix_np = coupling_matrix.detach().cpu().numpy()
    else:
        matrix_np = np.array(coupling_matrix)
    
    # Ensure matrix is 2D
    if matrix_np.ndim != 2:
        warnings.warn(f"Expected 2D coupling matrix, got {matrix_np.ndim}D. Skipping visualization.")
        return
    
    m, n = matrix_np.shape
    
    # Limit visualization size for performance
    if m > max_clusters or n > max_clusters:
        warnings.warn(f"Coupling matrix size ({m}×{n}) exceeds max_clusters ({max_clusters}). "
                     f"Truncating to first {max_clusters}×{max_clusters}.")
        matrix_np = matrix_np[:max_clusters, :max_clusters]
        m, n = matrix_np.shape
    
    # Create log dictionary
    log_dict = {}
    
    # Log summary statistics
    if log_stats:
        stats = _compute_coupling_stats(matrix_np)
        for key, value in stats.items():
            log_dict[f"{prefix}/{key}"] = value
    
    # Create and log heatmap
    if log_heatmap:
        heatmap_fig = _create_coupling_heatmap(
            matrix_np, 
            cluster_names=cluster_names,
            colormap=colormap,
            title_suffix=title_suffix
        )
        log_dict[f"{prefix}/coupling_heatmap"] = wandb.Image(heatmap_fig)
        plt.close(heatmap_fig)  # Clean up memory
        
        # Create row and column marginal plots
        marginal_fig = _create_marginal_plots(matrix_np, cluster_names=cluster_names)
        log_dict[f"{prefix}/marginals"] = wandb.Image(marginal_fig)
        plt.close(marginal_fig)  # Clean up memory
    
    # Log to wandb
    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)


def _compute_coupling_stats(matrix_np: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for the coupling matrix."""
    stats = {
        "coupling_mean": float(np.mean(matrix_np)),
        "coupling_std": float(np.std(matrix_np)),
        "coupling_max": float(np.max(matrix_np)),
        "coupling_min": float(np.min(matrix_np)),
        "coupling_entropy": float(-np.sum(matrix_np * np.log(matrix_np + 1e-10))),
        "coupling_sparsity": float(np.mean(matrix_np < 1e-6)),
        "row_marginal_uniformity": float(np.std(np.sum(matrix_np, axis=1))),
        "col_marginal_uniformity": float(np.std(np.sum(matrix_np, axis=0))),
        "coupling_frobenius_norm": float(np.linalg.norm(matrix_np, 'fro')),
    }
    
    # Add diagonal dominance measure if square matrix
    if matrix_np.shape[0] == matrix_np.shape[1]:
        diag_sum = np.sum(np.diag(matrix_np))
        total_sum = np.sum(matrix_np)
        stats["diagonal_ratio"] = float(diag_sum / (total_sum + 1e-10))
    
    return stats


def _create_coupling_heatmap(
    matrix_np: np.ndarray, 
    cluster_names: Optional[list] = None,
    colormap: str = "viridis",
    title_suffix: str = ""
) -> plt.Figure:
    """Create a professional heatmap of the coupling matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(matrix_np, cmap=colormap, aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Coupling Strength', rotation=270, labelpad=20)
    
    # Set labels
    if cluster_names and len(cluster_names) >= max(matrix_np.shape):
        ax.set_xticks(range(matrix_np.shape[1]))
        ax.set_yticks(range(matrix_np.shape[0]))
        ax.set_xticklabels(cluster_names[:matrix_np.shape[1]], rotation=45, ha='right')
        ax.set_yticklabels(cluster_names[:matrix_np.shape[0]])
    else:
        ax.set_xlabel('Target Clusters')
        ax.set_ylabel('Source Clusters')
    
    # Add title
    title = f'Sinkhorn-Knopp Coupling Matrix'
    if title_suffix:
        title += f' {title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add text annotations for small matrices
    if matrix_np.shape[0] <= 10 and matrix_np.shape[1] <= 10:
        for i in range(matrix_np.shape[0]):
            for j in range(matrix_np.shape[1]):
                text = ax.text(j, i, f'{matrix_np[i, j]:.3f}',
                             ha="center", va="center", color="white" if matrix_np[i, j] < np.max(matrix_np)/2 else "black")
    
    plt.tight_layout()
    return fig


def _create_marginal_plots(matrix_np: np.ndarray, cluster_names: Optional[list] = None) -> plt.Figure:
    """Create plots showing row and column marginals of the coupling matrix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Row marginals (sum over columns)
    row_marginals = np.sum(matrix_np, axis=1)
    ax1.bar(range(len(row_marginals)), row_marginals, alpha=0.7, color='skyblue')
    ax1.set_title('Row Marginals (Source Distribution)', fontweight='bold')
    ax1.set_xlabel('Source Cluster')
    ax1.set_ylabel('Marginal Probability')
    if cluster_names and len(cluster_names) >= matrix_np.shape[0]:
        ax1.set_xticks(range(len(row_marginals)))
        ax1.set_xticklabels(cluster_names[:len(row_marginals)], rotation=45, ha='right')
    
    # Column marginals (sum over rows)
    col_marginals = np.sum(matrix_np, axis=0)
    ax2.bar(range(len(col_marginals)), col_marginals, alpha=0.7, color='lightcoral')
    ax2.set_title('Column Marginals (Target Distribution)', fontweight='bold')
    ax2.set_xlabel('Target Cluster')
    ax2.set_ylabel('Marginal Probability')
    if cluster_names and len(cluster_names) >= matrix_np.shape[1]:
        ax2.set_xticks(range(len(col_marginals)))
        ax2.set_xticklabels(cluster_names[:len(col_marginals)], rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


# Convenience function for one-liner integration
def log_sinkhorn_coupling(coupling_matrix: torch.Tensor, **kwargs) -> None:
    """
    One-liner function to log Sinkhorn coupling matrix to wandb.
    
    This is the main function to call from sinkhorn analysis code.
    
    Args:
        coupling_matrix: The coupling matrix tensor
        **kwargs: Additional arguments passed to log_coupling_matrix_to_wandb
        
    Example:
        from .coupling_visualization import log_sinkhorn_coupling
        log_sinkhorn_coupling(coupling_matrix, step=iteration, prefix="training")
    """
    log_coupling_matrix_to_wandb(coupling_matrix, **kwargs) 