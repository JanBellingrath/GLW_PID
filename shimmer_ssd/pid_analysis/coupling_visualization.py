"""
Professional visualization utilities for Sinkhorn-Knopp coupling matrices.

This module provides functions to visualize and log coupling matrices to wandb
during training, helping to understand how the coupling evolves over iterations.

🎯 **WANDB OUT-OF-ORDER LOGGING SOLUTION**:
This module implements a solution for the wandb step counter conflicts that occur
when the Sinkhorn algorithm logs its internal iteration steps while the main 
training process (especially LR finder) has already advanced to higher step numbers.

The solution uses wandb's `define_metric` feature to create custom step metrics
that allow the Sinkhorn coupling matrices to be logged independently of the main
training step counter, preventing "step must be monotonically increasing" errors.

📖 Reference: https://wandb.me/define-metric

🔧 **Usage**:
1. Call `initialize_sinkhorn_wandb_metrics()` early in your training script after `wandb.init()`
2. Use `log_sinkhorn_coupling()` to log coupling matrices with custom step metrics
3. The system automatically handles out-of-order logging without step conflicts

✅ **Benefits**:
- No more wandb step counter conflicts during LR finding
- Sinkhorn iterations can be logged independently 
- Maintains full coupling matrix visualization capabilities
- Compatible with existing wandb workflows
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

# Global set to track which Sinkhorn prefixes have been defined
_DEFINED_SINKHORN_PREFIXES = set()

def _setup_sinkhorn_wandb_metrics(prefix: str = "sinkhorn") -> None:
    """
    Set up custom wandb metrics for out-of-order Sinkhorn logging.
    
    This function defines custom step metrics that allow the Sinkhorn algorithm
    to log its internal iteration steps independently of the main training steps,
    preventing the "step must be monotonically increasing" errors.
    
    Args:
        prefix: Prefix for the Sinkhorn metrics
    """
    global _DEFINED_SINKHORN_PREFIXES
    
    if not HAS_WANDB or wandb.run is None or prefix in _DEFINED_SINKHORN_PREFIXES:
        return
    
    try:
        # Define a custom step metric for Sinkhorn iterations
        step_metric_name = f"{prefix}/sinkhorn_step"
        
        # Define the custom step metric
        wandb.define_metric(step_metric_name)
        
        # Define all Sinkhorn metrics to use the custom step
        # Use glob patterns to catch all Sinkhorn metrics with this prefix
        wandb.define_metric(f"{prefix}/*", step_metric=step_metric_name)
        
        print(f"🎯 Defining custom wandb metrics for Sinkhorn prefix '{prefix}' to allow out-of-order logging...")
        print(f"   ✅ Defined glob pattern '{prefix}/*' to use custom step metric '{step_metric_name}'")
        print(f"   📖 Reference: https://wandb.me/define-metric")
        
        _DEFINED_SINKHORN_PREFIXES.add(prefix)
        
    except Exception as e:
        warnings.warn(f"Failed to define custom Sinkhorn metrics for wandb prefix '{prefix}': {e}")


def log_coupling_matrix_to_wandb(
    coupling_matrix: torch.Tensor,
    step: Optional[int] = None,
    prefix: str = "sinkhorn",
    cluster_names: Optional[list] = None,
    max_clusters: int = 256,
    log_stats: bool = True,
    log_heatmap: bool = True,
    colormap: str = "viridis",
    title_suffix: str = "",
    lr_finding_mode: bool = False
) -> None:
    """
    Log Sinkhorn-Knopp coupling matrix visualization to wandb.
    
    This function creates professional visualizations of coupling matrices,
    including heatmaps and summary statistics, and logs them to wandb with
    custom step metrics to avoid step ordering conflicts.
    
    Args:
        coupling_matrix: The coupling matrix tensor, shape [m, n] or [batch, m, n]
        step: Training step/iteration for logging (optional)
        prefix: Prefix for wandb log keys (default: "sinkhorn")
        cluster_names: Optional list of cluster names for axis labels
        max_clusters: Maximum number of clusters to visualize (default: 256)
        log_stats: Whether to log coupling matrix statistics (default: True)
        log_heatmap: Whether to log heatmap visualization (default: True)
        colormap: Matplotlib colormap for heatmap (default: "viridis")
        title_suffix: Additional suffix for plot title
        lr_finding_mode: If True, skips expensive visualizations during LR finding (default: False)
        
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
    
    # Set up custom metrics for out-of-order logging
    _setup_sinkhorn_wandb_metrics(prefix)
    
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
    
    # Add the custom step metric to the log dictionary
    if step is not None:
        log_dict[f"{prefix}/sinkhorn_step"] = step
    
    # Log summary statistics (always logged, lightweight)
    if log_stats:
        stats = _compute_coupling_stats(matrix_np)
        for key, value in stats.items():
            log_dict[f"{prefix}/{key}"] = value
    
    # Skip expensive visualizations during LR finding for performance
    if lr_finding_mode:
        # During LR finding, only log basic coupling strength for monitoring
        log_dict[f"{prefix}/coupling_mean_lr_finding"] = float(np.mean(matrix_np))
        if step is not None:
            log_dict[f"{prefix}/lr_finding_step"] = step
    else:
        # Full visualization during normal training/evaluation
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
    
    # Log to wandb - now using custom step metrics, no step conflicts!
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


def log_discriminator_marginals_to_wandb(
    p_y_x1: torch.Tensor,
    p_y_x2: torch.Tensor,
    p_y_x1x2: torch.Tensor,
    p_y_marginal: Optional[torch.Tensor] = None,
    step: Optional[int] = None,
    prefix: str = "discriminator_marginals",
    cluster_names: Optional[list] = None,
    lr_finding_mode: bool = False
) -> None:
    """
    Log marginal distributions from the three discriminator networks to wandb.
    
    This function visualizes and logs the probability distributions produced by:
    - discrim_1: p(y|x1) - marginal distribution over labels given first domain
    - discrim_2: p(y|x2) - marginal distribution over labels given second domain  
    - discrim_12: p(y|x1,x2) - joint distribution over labels given both domains
    - p_y: p(y) - prior marginal distribution over labels (optional)
    
    Args:
        p_y_x1: Probability distribution p(y|x1) from first discriminator, shape [batch, num_labels]
        p_y_x2: Probability distribution p(y|x2) from second discriminator, shape [batch, num_labels]
        p_y_x1x2: Joint probability distribution p(y|x1,x2) from joint discriminator, shape [batch, num_labels]
        p_y_marginal: Prior marginal distribution p(y), shape [num_labels] (optional)
        step: Training step/iteration for logging (optional)
        prefix: Prefix for wandb log keys (default: "discriminator_marginals")
        cluster_names: Optional list of cluster/label names for axis labels
        lr_finding_mode: If True, skips expensive visualizations during LR finding (default: False)
        
    Returns:
        None
        
    Raises:
        Warning: If wandb is not available or tensor shapes are inconsistent
    """
    if not HAS_WANDB:
        warnings.warn("wandb not available. Skipping discriminator marginal logging.")
        return
    
    if wandb.run is None:
        warnings.warn("No active wandb run. Skipping discriminator marginal logging.")
        return
    
    # Set up custom metrics for out-of-order logging
    _setup_sinkhorn_wandb_metrics(prefix)
    
    # Convert tensors to numpy and compute batch averages
    try:
        p_y_x1_np = p_y_x1.mean(dim=0).detach().cpu().numpy()  # Average over batch -> [num_labels]
        p_y_x2_np = p_y_x2.mean(dim=0).detach().cpu().numpy()  # Average over batch -> [num_labels] 
        p_y_x1x2_np = p_y_x1x2.mean(dim=0).detach().cpu().numpy()  # Average over batch -> [num_labels]
        
        if p_y_marginal is not None:
            p_y_np = p_y_marginal.detach().cpu().numpy()  # [num_labels]
        else:
            p_y_np = None
            
    except Exception as e:
        warnings.warn(f"Error converting discriminator outputs to numpy: {e}")
        return
    
    # Validate shapes
    num_labels = p_y_x1_np.shape[0]
    if p_y_x2_np.shape[0] != num_labels or p_y_x1x2_np.shape[0] != num_labels:
        warnings.warn(f"Inconsistent number of labels: p_y_x1={p_y_x1_np.shape[0]}, "
                     f"p_y_x2={p_y_x2_np.shape[0]}, p_y_x1x2={p_y_x1x2_np.shape[0]}")
        return
    
    if p_y_np is not None and p_y_np.shape[0] != num_labels:
        warnings.warn(f"Inconsistent marginal shape: p_y={p_y_np.shape[0]}, expected={num_labels}")
        p_y_np = None
    
    # Create log dictionary
    log_dict = {}
    
    # Add the custom step metric to the log dictionary
    if step is not None:
        log_dict[f"{prefix}/sinkhorn_step"] = step
    
    # Always log lightweight statistics
    stats = _compute_discriminator_marginal_stats(p_y_x1_np, p_y_x2_np, p_y_x1x2_np, p_y_np)
    for key, value in stats.items():
        log_dict[f"{prefix}/{key}"] = value
    
    # Skip expensive visualizations during LR finding for performance
    if lr_finding_mode:
        # During LR finding, only log basic entropy and uniformity for monitoring
        log_dict[f"{prefix}/p_y_x1_entropy_lr_finding"] = float(-np.sum(p_y_x1_np * np.log(p_y_x1_np + 1e-10)))
        log_dict[f"{prefix}/p_y_x2_entropy_lr_finding"] = float(-np.sum(p_y_x2_np * np.log(p_y_x2_np + 1e-10)))
        log_dict[f"{prefix}/p_y_x1x2_entropy_lr_finding"] = float(-np.sum(p_y_x1x2_np * np.log(p_y_x1x2_np + 1e-10)))
        if step is not None:
            log_dict[f"{prefix}/lr_finding_step"] = step
    else:
        # Full visualization during normal training/evaluation
        # Create and log discriminator marginal distributions plot
        marginal_dist_fig = _create_discriminator_marginal_plots(
            p_y_x1_np, p_y_x2_np, p_y_x1x2_np, p_y_np, cluster_names=cluster_names
        )
        log_dict[f"{prefix}/marginal_distributions"] = wandb.Image(marginal_dist_fig)
        plt.close(marginal_dist_fig)  # Clean up memory
        
        # Create comparison plots showing differences between distributions
        comparison_fig = _create_discriminator_comparison_plots(
            p_y_x1_np, p_y_x2_np, p_y_x1x2_np, p_y_np, cluster_names=cluster_names
        )
        log_dict[f"{prefix}/distribution_comparisons"] = wandb.Image(comparison_fig)
        plt.close(comparison_fig)  # Clean up memory
    
    # Log to wandb - now using custom step metrics, no step conflicts!
    wandb.log(log_dict)


def _compute_discriminator_marginal_stats(
    p_y_x1_np: np.ndarray,
    p_y_x2_np: np.ndarray, 
    p_y_x1x2_np: np.ndarray,
    p_y_np: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute comprehensive statistics for discriminator marginal distributions."""
    stats = {}
    
    # Individual distribution statistics
    for name, dist in [("p_y_x1", p_y_x1_np), ("p_y_x2", p_y_x2_np), ("p_y_x1x2", p_y_x1x2_np)]:
        stats[f"{name}_entropy"] = float(-np.sum(dist * np.log(dist + 1e-10)))
        stats[f"{name}_max_prob"] = float(np.max(dist))
        stats[f"{name}_min_prob"] = float(np.min(dist))
        stats[f"{name}_std"] = float(np.std(dist))
        stats[f"{name}_uniformity"] = float(1.0 - np.sum((dist - 1.0/len(dist))**2))  # 1 - sum of squared deviations from uniform
        stats[f"{name}_dominant_cluster"] = int(np.argmax(dist))
        stats[f"{name}_concentration"] = float(np.sum(dist**2))  # Concentration measure (inverse Simpson diversity)
    
    # Cross-distribution comparisons
    stats["kl_divergence_x1_vs_x2"] = float(np.sum(p_y_x1_np * np.log((p_y_x1_np + 1e-10) / (p_y_x2_np + 1e-10))))
    stats["kl_divergence_x1_vs_x1x2"] = float(np.sum(p_y_x1_np * np.log((p_y_x1_np + 1e-10) / (p_y_x1x2_np + 1e-10))))
    stats["kl_divergence_x2_vs_x1x2"] = float(np.sum(p_y_x2_np * np.log((p_y_x2_np + 1e-10) / (p_y_x1x2_np + 1e-10))))
    
    # L2 distances between distributions
    stats["l2_distance_x1_vs_x2"] = float(np.linalg.norm(p_y_x1_np - p_y_x2_np))
    stats["l2_distance_x1_vs_x1x2"] = float(np.linalg.norm(p_y_x1_np - p_y_x1x2_np))
    stats["l2_distance_x2_vs_x1x2"] = float(np.linalg.norm(p_y_x2_np - p_y_x1x2_np))
    
    # Cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    
    stats["cosine_sim_x1_vs_x2"] = float(cosine_similarity(p_y_x1_np, p_y_x2_np))
    stats["cosine_sim_x1_vs_x1x2"] = float(cosine_similarity(p_y_x1_np, p_y_x1x2_np))
    stats["cosine_sim_x2_vs_x1x2"] = float(cosine_similarity(p_y_x2_np, p_y_x1x2_np))
    
    # If prior p(y) is available, compute comparisons
    if p_y_np is not None:
        for name, dist in [("p_y_x1", p_y_x1_np), ("p_y_x2", p_y_x2_np), ("p_y_x1x2", p_y_x1x2_np)]:
            stats[f"kl_divergence_{name}_vs_prior"] = float(np.sum(dist * np.log((dist + 1e-10) / (p_y_np + 1e-10))))
            stats[f"l2_distance_{name}_vs_prior"] = float(np.linalg.norm(dist - p_y_np))
            stats[f"cosine_sim_{name}_vs_prior"] = float(cosine_similarity(dist, p_y_np))
    
    return stats


def _create_discriminator_marginal_plots(
    p_y_x1_np: np.ndarray,
    p_y_x2_np: np.ndarray,
    p_y_x1x2_np: np.ndarray,
    p_y_np: Optional[np.ndarray] = None,
    cluster_names: Optional[list] = None
) -> plt.Figure:
    """Create bar plots showing all discriminator marginal distributions."""
    # Determine number of subplots
    num_plots = 4 if p_y_np is not None else 3
    
    if num_plots == 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
    else:
        fig, axes_raw = plt.subplots(2, 2, figsize=(15, 10))
        axes = [axes_raw[0, 0], axes_raw[0, 1], axes_raw[1, 0], axes_raw[1, 1]]
        # We'll remove the unused subplot later
    
    num_labels = len(p_y_x1_np)
    x_positions = np.arange(num_labels)
    
    # Set up cluster names
    if cluster_names and len(cluster_names) >= num_labels:
        labels = cluster_names[:num_labels]
    else:
        labels = [f'C{i}' for i in range(num_labels)]
    
    # Plot p(y|x1) from discriminator 1
    axes[0].bar(x_positions, p_y_x1_np, alpha=0.8, color='skyblue', edgecolor='darkblue')
    axes[0].set_title('p(y|x₁) - First Domain Discriminator', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Cluster/Label')
    axes[0].set_ylabel('Probability')
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot p(y|x2) from discriminator 2
    axes[1].bar(x_positions, p_y_x2_np, alpha=0.8, color='lightcoral', edgecolor='darkred')
    axes[1].set_title('p(y|x₂) - Second Domain Discriminator', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Cluster/Label')
    axes[1].set_ylabel('Probability')
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot p(y|x1,x2) from joint discriminator
    axes[2].bar(x_positions, p_y_x1x2_np, alpha=0.8, color='lightgreen', edgecolor='darkgreen')
    axes[2].set_title('p(y|x₁,x₂) - Joint Discriminator', fontweight='bold', fontsize=12)
    axes[2].set_xlabel('Cluster/Label')
    axes[2].set_ylabel('Probability')
    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    
    # Plot p(y) prior if available
    if p_y_np is not None and num_plots == 4:
        axes[3].bar(x_positions, p_y_np, alpha=0.8, color='orange', edgecolor='darkorange')
        axes[3].set_title('p(y) - Prior Marginal Distribution', fontweight='bold', fontsize=12)
        axes[3].set_xlabel('Cluster/Label')
        axes[3].set_ylabel('Probability')
        axes[3].set_xticks(x_positions)
        axes[3].set_xticklabels(labels, rotation=45, ha='right')
        axes[3].grid(axis='y', alpha=0.3)
    else:
        # Remove the unused subplot for 3-plot case
        fig.delaxes(axes[3])
    
    plt.suptitle('Discriminator Marginal Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def _create_discriminator_comparison_plots(
    p_y_x1_np: np.ndarray,
    p_y_x2_np: np.ndarray,
    p_y_x1x2_np: np.ndarray,
    p_y_np: Optional[np.ndarray] = None,
    cluster_names: Optional[list] = None
) -> plt.Figure:
    """Create comparison plots showing overlays and differences between discriminator distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    num_labels = len(p_y_x1_np)
    x_positions = np.arange(num_labels)
    
    # Set up cluster names
    if cluster_names and len(cluster_names) >= num_labels:
        labels = cluster_names[:num_labels]
    else:
        labels = [f'C{i}' for i in range(num_labels)]
    
    # Overlay plot: p(y|x1) vs p(y|x2)
    width = 0.35
    axes[0, 0].bar(x_positions - width/2, p_y_x1_np, width, alpha=0.7, color='skyblue', label='p(y|x₁)', edgecolor='darkblue')
    axes[0, 0].bar(x_positions + width/2, p_y_x2_np, width, alpha=0.7, color='lightcoral', label='p(y|x₂)', edgecolor='darkred')
    axes[0, 0].set_title('Domain Comparison: p(y|x₁) vs p(y|x₂)', fontweight='bold')
    axes[0, 0].set_xlabel('Cluster/Label')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_xticks(x_positions)
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Overlay plot: All three distributions
    width = 0.25
    axes[0, 1].bar(x_positions - width, p_y_x1_np, width, alpha=0.7, color='skyblue', label='p(y|x₁)', edgecolor='darkblue')
    axes[0, 1].bar(x_positions, p_y_x2_np, width, alpha=0.7, color='lightcoral', label='p(y|x₂)', edgecolor='darkred')
    axes[0, 1].bar(x_positions + width, p_y_x1x2_np, width, alpha=0.7, color='lightgreen', label='p(y|x₁,x₂)', edgecolor='darkgreen')
    axes[0, 1].set_title('All Discriminators: p(y|x₁) vs p(y|x₂) vs p(y|x₁,x₂)', fontweight='bold')
    axes[0, 1].set_xlabel('Cluster/Label')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_xticks(x_positions)
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Difference plot: p(y|x1,x2) - p(y|x1)
    diff_x1 = p_y_x1x2_np - p_y_x1_np
    colors_x1 = ['red' if d < 0 else 'green' for d in diff_x1]
    axes[1, 0].bar(x_positions, diff_x1, alpha=0.7, color=colors_x1)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_title('Joint vs First Domain: p(y|x₁,x₂) - p(y|x₁)', fontweight='bold')
    axes[1, 0].set_xlabel('Cluster/Label')
    axes[1, 0].set_ylabel('Probability Difference')
    axes[1, 0].set_xticks(x_positions)
    axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Difference plot: p(y|x1,x2) - p(y|x2)
    diff_x2 = p_y_x1x2_np - p_y_x2_np
    colors_x2 = ['red' if d < 0 else 'green' for d in diff_x2]
    axes[1, 1].bar(x_positions, diff_x2, alpha=0.7, color=colors_x2)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].set_title('Joint vs Second Domain: p(y|x₁,x₂) - p(y|x₂)', fontweight='bold')
    axes[1, 1].set_xlabel('Cluster/Label')
    axes[1, 1].set_ylabel('Probability Difference')
    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Discriminator Distribution Comparisons', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


# Convenience functions for one-liner integration
def log_discriminator_marginals(
    p_y_x1: torch.Tensor,
    p_y_x2: torch.Tensor,
    p_y_x1x2: torch.Tensor,
    **kwargs
) -> None:
    """
    One-liner function to log discriminator marginal distributions to wandb.
    
    This is the main function to call from PID analysis code to visualize discriminator outputs.
    
    Args:
        p_y_x1: Probability distribution p(y|x1) from first discriminator, shape [batch, num_labels]
        p_y_x2: Probability distribution p(y|x2) from second discriminator, shape [batch, num_labels]
        p_y_x1x2: Joint probability distribution p(y|x1,x2) from joint discriminator, shape [batch, num_labels]
        **kwargs: Additional arguments passed to log_discriminator_marginals_to_wandb
        
    Example:
        # During normal training
        log_discriminator_marginals(p_y_x1, p_y_x2, p_y_x1x2, step=iteration, prefix="training")
        
        # During LR finding (lightweight logging)
        log_discriminator_marginals(p_y_x1, p_y_x2, p_y_x1x2, lr_finding_mode=True)
    """
    log_discriminator_marginals_to_wandb(p_y_x1, p_y_x2, p_y_x1x2, **kwargs)


def log_sinkhorn_coupling(coupling_matrix: torch.Tensor, **kwargs) -> None:
    """
    One-liner function to log Sinkhorn coupling matrix to wandb.
    
    This is the main function to call from sinkhorn analysis code.
    
    Args:
        coupling_matrix: The coupling matrix tensor
        **kwargs: Additional arguments passed to log_coupling_matrix_to_wandb
        
    Example:
        # During normal training
        log_sinkhorn_coupling(coupling_matrix, step=iteration, prefix="training")
        
        # During LR finding (lightweight logging)
        log_sinkhorn_coupling(coupling_matrix, step=iteration, lr_finding_mode=True)
    """
    log_coupling_matrix_to_wandb(coupling_matrix, **kwargs)


def initialize_sinkhorn_wandb_metrics(prefixes: list = None) -> None:
    """
    Initialize custom wandb metrics for Sinkhorn coupling visualization.
    
    Call this function early in your training script (after wandb.init()) to
    set up custom step metrics that allow out-of-order logging from Sinkhorn
    iterations. This prevents step counter conflicts during LR finding and training.
    
    Args:
        prefixes: List of prefixes to set up metrics for. If None, uses common defaults.
        
    Example:
        import wandb
        
        wandb.init(project="my-project")
        initialize_sinkhorn_wandb_metrics(["sinkhorn", "coupling_cluster_0", "coupling_cluster_1"])
        # ... start training ...
    """
    if not HAS_WANDB or wandb.run is None:
        return
    
    if prefixes is None:
        # Default prefixes commonly used in the codebase
        prefixes = ["sinkhorn", "coupling_cluster_0", "coupling_cluster_1", "coupling_cluster_2", 
                   "coupling_cluster_3", "coupling_cluster_4", "coupling_cluster_5",
                   "discriminator_marginals", "gradient_magnitudes"]
    
    for prefix in prefixes:
        _setup_sinkhorn_wandb_metrics(prefix) 


def log_gradient_magnitudes_to_wandb(
    parameters_before: dict,
    parameters_after: dict,
    step: Optional[int] = None,
    prefix: str = "gradient_magnitudes",
    lr_finding_mode: bool = False
) -> None:
    """
    Log gradient magnitudes before and after Sinkhorn operations to wandb.
    
    This function visualizes and logs gradient magnitude changes across Sinkhorn
    operations to help understand gradient flow and training dynamics.
    
    Args:
        parameters_before: Dict of parameter names to gradient magnitudes before Sinkhorn
        parameters_after: Dict of parameter names to gradient magnitudes after Sinkhorn
        step: Training step/iteration for logging (optional)
        prefix: Prefix for wandb log keys (default: "gradient_magnitudes")
        lr_finding_mode: If True, skips expensive visualizations during LR finding (default: False)
        
    Returns:
        None
        
    Raises:
        Warning: If wandb is not available or parameter sets are inconsistent
    """
    if not HAS_WANDB:
        warnings.warn("wandb not available. Skipping gradient magnitude logging.")
        return
    
    if wandb.run is None:
        warnings.warn("No active wandb run. Skipping gradient magnitude logging.")
        return
    
    # Set up custom metrics for out-of-order logging
    _setup_sinkhorn_wandb_metrics(prefix)
    
    # Create log dictionary
    log_dict = {}
    
    # Add the custom step metric to the log dictionary
    if step is not None:
        log_dict[f"{prefix}/sinkhorn_step"] = step
    
    # Always log lightweight statistics
    stats = _compute_gradient_magnitude_stats(parameters_before, parameters_after)
    for key, value in stats.items():
        log_dict[f"{prefix}/{key}"] = value
    
    # Skip expensive visualizations during LR finding for performance
    if lr_finding_mode:
        # During LR finding, only log basic gradient magnitude summary for monitoring
        total_grad_mag_before = sum(parameters_before.values())
        total_grad_mag_after = sum(parameters_after.values())
        log_dict[f"{prefix}/total_grad_magnitude_before_lr_finding"] = float(total_grad_mag_before)
        log_dict[f"{prefix}/total_grad_magnitude_after_lr_finding"] = float(total_grad_mag_after)
        log_dict[f"{prefix}/grad_magnitude_ratio_lr_finding"] = float(total_grad_mag_after / (total_grad_mag_before + 1e-10))
        if step is not None:
            log_dict[f"{prefix}/lr_finding_step"] = step
    else:
        # Full visualization during normal training/evaluation
        # Create and log gradient magnitude comparison plots
        gradient_fig = _create_gradient_magnitude_plots(parameters_before, parameters_after)
        log_dict[f"{prefix}/gradient_magnitude_comparison"] = wandb.Image(gradient_fig)
        plt.close(gradient_fig)  # Clean up memory
        
        # Create gradient flow visualization
        flow_fig = _create_gradient_flow_plots(parameters_before, parameters_after)
        log_dict[f"{prefix}/gradient_flow"] = wandb.Image(flow_fig)
        plt.close(flow_fig)  # Clean up memory
    
    # Log to wandb - now using custom step metrics, no step conflicts!
    wandb.log(log_dict)


def _compute_gradient_magnitude_stats(
    parameters_before: dict,
    parameters_after: dict
) -> Dict[str, float]:
    """Compute comprehensive statistics for gradient magnitude changes."""
    stats = {}
    
    # Overall statistics
    total_grad_before = sum(parameters_before.values())
    total_grad_after = sum(parameters_after.values())
    
    stats["total_grad_magnitude_before"] = float(total_grad_before)
    stats["total_grad_magnitude_after"] = float(total_grad_after)
    stats["total_grad_magnitude_ratio"] = float(total_grad_after / (total_grad_before + 1e-10))
    stats["total_grad_magnitude_change"] = float(total_grad_after - total_grad_before)
    stats["total_grad_magnitude_change_percent"] = float(100 * (total_grad_after - total_grad_before) / (total_grad_before + 1e-10))
    
    # Per-parameter statistics (if parameters match)
    common_params = set(parameters_before.keys()) & set(parameters_after.keys())
    if common_params:
        param_ratios = []
        param_changes = []
        param_changes_percent = []
        
        for param_name in common_params:
            before_val = parameters_before[param_name]
            after_val = parameters_after[param_name]
            
            ratio = after_val / (before_val + 1e-10)
            change = after_val - before_val
            change_percent = 100 * change / (before_val + 1e-10)
            
            param_ratios.append(ratio)
            param_changes.append(change)
            param_changes_percent.append(change_percent)
            
            # Individual parameter stats
            stats[f"grad_ratio_{param_name}"] = float(ratio)
            stats[f"grad_change_{param_name}"] = float(change)
            stats[f"grad_change_percent_{param_name}"] = float(change_percent)
        
        # Aggregate per-parameter stats
        stats["mean_grad_ratio"] = float(np.mean(param_ratios))
        stats["std_grad_ratio"] = float(np.std(param_ratios))
        stats["max_grad_ratio"] = float(np.max(param_ratios))
        stats["min_grad_ratio"] = float(np.min(param_ratios))
        
        stats["mean_grad_change"] = float(np.mean(param_changes))
        stats["std_grad_change"] = float(np.std(param_changes))
        stats["max_grad_change"] = float(np.max(param_changes))
        stats["min_grad_change"] = float(np.min(param_changes))
        
        stats["mean_grad_change_percent"] = float(np.mean(param_changes_percent))
        stats["std_grad_change_percent"] = float(np.std(param_changes_percent))
        stats["max_grad_change_percent"] = float(np.max(param_changes_percent))
        stats["min_grad_change_percent"] = float(np.min(param_changes_percent))
        
        # Count parameters that increased/decreased/stayed similar
        increased = sum(1 for r in param_ratios if r > 1.1)  # More than 10% increase
        decreased = sum(1 for r in param_ratios if r < 0.9)  # More than 10% decrease
        similar = len(param_ratios) - increased - decreased
        
        stats["params_gradient_increased"] = increased
        stats["params_gradient_decreased"] = decreased
        stats["params_gradient_similar"] = similar
        stats["params_gradient_increased_percent"] = float(100 * increased / len(param_ratios))
        stats["params_gradient_decreased_percent"] = float(100 * decreased / len(param_ratios))
    
    return stats


def _create_gradient_magnitude_plots(
    parameters_before: dict,
    parameters_after: dict
) -> plt.Figure:
    """Create plots showing gradient magnitude changes before and after Sinkhorn."""
    # Find common parameters
    common_params = list(set(parameters_before.keys()) & set(parameters_after.keys()))
    
    if not common_params:
        # Create empty plot if no common parameters
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No common parameters found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Gradient Magnitude Comparison', fontweight='bold')
        return fig
    
    # Sort parameters by name for consistent ordering
    common_params.sort()
    
    before_values = [parameters_before[p] for p in common_params]
    after_values = [parameters_after[p] for p in common_params]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before vs After bar plot
    x_positions = np.arange(len(common_params))
    width = 0.35
    
    ax1.bar(x_positions - width/2, before_values, width, alpha=0.8, color='skyblue', 
            label='Before Sinkhorn', edgecolor='darkblue')
    ax1.bar(x_positions + width/2, after_values, width, alpha=0.8, color='lightcoral', 
            label='After Sinkhorn', edgecolor='darkred')
    
    ax1.set_title('Gradient Magnitudes: Before vs After Sinkhorn', fontweight='bold')
    ax1.set_xlabel('Parameter')
    ax1.set_ylabel('Gradient Magnitude')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([p.replace('_', '\n') for p in common_params], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_yscale('log')  # Log scale for gradient magnitudes
    
    # Ratio plot
    ratios = [after_values[i] / (before_values[i] + 1e-10) for i in range(len(common_params))]
    colors = ['red' if r < 1 else 'green' for r in ratios]
    
    ax2.bar(x_positions, ratios, alpha=0.8, color=colors)
    ax2.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='No Change')
    ax2.set_title('Gradient Magnitude Ratios (After/Before)', fontweight='bold')
    ax2.set_xlabel('Parameter')
    ax2.set_ylabel('Ratio (After/Before)')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([p.replace('_', '\n') for p in common_params], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')  # Log scale for ratios
    
    plt.tight_layout()
    return fig


def _create_gradient_flow_plots(
    parameters_before: dict,
    parameters_after: dict
) -> plt.Figure:
    """Create plots showing gradient flow patterns and changes."""
    # Find common parameters
    common_params = list(set(parameters_before.keys()) & set(parameters_after.keys()))
    
    if not common_params:
        # Create empty plot if no common parameters
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No common parameters found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Gradient Flow Analysis', fontweight='bold')
        return fig
    
    common_params.sort()
    before_values = [parameters_before[p] for p in common_params]
    after_values = [parameters_after[p] for p in common_params]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot: Before vs After
    ax1.scatter(before_values, after_values, alpha=0.7, s=50)
    
    # Add diagonal line for reference (no change)
    min_val = min(min(before_values), min(after_values))
    max_val = max(max(before_values), max(after_values))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='No Change')
    
    ax1.set_xlabel('Gradient Magnitude Before Sinkhorn')
    ax1.set_ylabel('Gradient Magnitude After Sinkhorn')
    ax1.set_title('Gradient Magnitude: Before vs After', fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add parameter labels for outliers
    for i, param in enumerate(common_params):
        if abs(np.log10(after_values[i]) - np.log10(before_values[i])) > 0.5:  # Significant change
            ax1.annotate(param.split('.')[-1], (before_values[i], after_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    # Change magnitude plot
    changes = [after_values[i] - before_values[i] for i in range(len(common_params))]
    x_positions = np.arange(len(common_params))
    
    colors = ['red' if c < 0 else 'green' for c in changes]
    ax2.bar(x_positions, changes, alpha=0.8, color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Gradient Magnitude Changes', fontweight='bold')
    ax2.set_xlabel('Parameter')
    ax2.set_ylabel('Change in Gradient Magnitude')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([p.split('.')[-1] for p in common_params], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def log_gradient_magnitudes(
    parameters_before: dict,
    parameters_after: dict,
    **kwargs
) -> None:
    """
    One-liner function to log gradient magnitudes to wandb.
    
    This is the main function to call from training code to visualize gradient changes.
    
    Args:
        parameters_before: Dict of parameter names to gradient magnitudes before operation
        parameters_after: Dict of parameter names to gradient magnitudes after operation
        **kwargs: Additional arguments passed to log_gradient_magnitudes_to_wandb
        
    Example:
        # During normal training
        log_gradient_magnitudes(grads_before, grads_after, step=iteration, prefix="sinkhorn")
        
        # During LR finding (lightweight logging)
        log_gradient_magnitudes(grads_before, grads_after, lr_finding_mode=True)
    """
    log_gradient_magnitudes_to_wandb(parameters_before, parameters_after, **kwargs)


def capture_gradient_magnitudes(model: torch.nn.Module, prefix: str = "") -> Dict[str, float]:
    """
    Capture current gradient magnitudes for all parameters in a model.
    
    Args:
        model: PyTorch model to capture gradients from
        prefix: Optional prefix for parameter names
        
    Returns:
        Dictionary mapping parameter names to gradient magnitudes
    """
    grad_magnitudes = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_magnitude = param.grad.norm().item()
            param_name = f"{prefix}_{name}" if prefix else name
            grad_magnitudes[param_name] = grad_magnitude
    
    return grad_magnitudes 