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

# Professional color scheme and styling constants
PROFESSIONAL_COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Deep rose
    'tertiary': '#F18F01',     # Warm orange  
    'quaternary': '#C73E1D',   # Deep red
    'accent': '#4CAF50',       # Success green
    'background': '#F8F9FA',   # Light background
    'text': '#2C3E50',         # Dark text
    'grid': '#BDC3C7',         # Light grid
    'highlight': '#FFC107',    # Warning yellow
    'neutral': '#6C757D'       # Neutral gray
}

def _setup_professional_style():
    """Setup consistent professional matplotlib styling."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('default')
    
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.framealpha': 0.95,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white'
    })

# Global set to track which Sinkhorn prefixes have been defined
_DEFINED_SINKHORN_PREFIXES = set()

def _setup_sinkhorn_wandb_metrics(prefix: str = "sinkhorn") -> None:
    """Setup custom wandb metrics for a given prefix to allow out-of-order logging."""
    global _DEFINED_SINKHORN_PREFIXES
    
    if not HAS_WANDB or wandb.run is None:
        return
    
    # Only define once per prefix to avoid wandb warnings
    if prefix in _DEFINED_SINKHORN_PREFIXES:
        return
    
    try:
        # Define custom step metric for this prefix
        wandb.define_metric(f"{prefix}/sinkhorn_step")
        wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/sinkhorn_step")
        _DEFINED_SINKHORN_PREFIXES.add(prefix)
        # Only print the first few metric definitions to avoid spam
        if len(_DEFINED_SINKHORN_PREFIXES) <= 3:
            print(f"🎯 Defining custom wandb metrics for Sinkhorn prefix '{prefix}'")
        elif len(_DEFINED_SINKHORN_PREFIXES) == 4:
            print(f"🎯 ... (continuing to define wandb metrics for additional prefixes silently)")
    except Exception as e:
        print(f"⚠️  Warning: Could not define wandb metrics for prefix '{prefix}': {e}")


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
    _setup_professional_style()
    
    # Choose optimal figure size based on matrix dimensions
    base_size = max(8, min(16, max(matrix_np.shape) * 0.5))
    fig, ax = plt.subplots(figsize=(base_size, base_size * 0.85))
    
    # Enhanced colormap selection for better visual appeal
    if colormap == "viridis":
        cmap = plt.cm.viridis
    elif colormap == "plasma":
        cmap = plt.cm.plasma
    elif colormap == "inferno":
        cmap = plt.cm.inferno
    else:
        cmap = colormap
    
    # Create heatmap with professional styling
    im = ax.imshow(matrix_np, cmap=cmap, aspect='auto', interpolation='bilinear')
    
    # Professional colorbar with better positioning
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Coupling Strength', rotation=270, labelpad=25, 
                   fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Enhanced axis labels and ticks
    if cluster_names and len(cluster_names) >= max(matrix_np.shape):
        # Use cluster names with better formatting
        x_labels = cluster_names[:matrix_np.shape[1]]
        y_labels = cluster_names[:matrix_np.shape[0]]
        
        ax.set_xticks(range(matrix_np.shape[1]))
        ax.set_yticks(range(matrix_np.shape[0]))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(y_labels, fontsize=10)
        
        ax.set_xlabel('Target Clusters', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Source Clusters', fontsize=12, fontweight='bold', labelpad=10)
    else:
        # Generic labels with improved styling
        ax.set_xlabel('Target Clusters', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Source Clusters', fontsize=12, fontweight='bold', labelpad=10)
        
        # Set fewer ticks for large matrices
        if matrix_np.shape[0] > 20:
            tick_step = max(1, matrix_np.shape[0] // 10)
            ax.set_yticks(range(0, matrix_np.shape[0], tick_step))
        if matrix_np.shape[1] > 20:
            tick_step = max(1, matrix_np.shape[1] // 10)
            ax.set_xticks(range(0, matrix_np.shape[1], tick_step))
    
    # Enhanced title with better formatting
    title = 'Sinkhorn-Knopp Coupling Matrix'
    if title_suffix:
        title += f' {title_suffix}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, 
                 color=PROFESSIONAL_COLORS['text'])
    
    # Add value annotations for small matrices with better contrast
    if matrix_np.shape[0] <= 10 and matrix_np.shape[1] <= 10:
        threshold = np.median(matrix_np)  # Better threshold for text color
        for i in range(matrix_np.shape[0]):
            for j in range(matrix_np.shape[1]):
                value = matrix_np[i, j]
                text_color = 'white' if value < threshold else 'black'
                ax.text(j, i, f'{value:.3f}', ha="center", va="center", 
                       color=text_color, fontweight='bold', fontsize=9)
    
    # Add subtle border around the heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color(PROFESSIONAL_COLORS['text'])
    
    plt.tight_layout()
    return fig


def _create_marginal_plots(matrix_np: np.ndarray, cluster_names: Optional[list] = None) -> plt.Figure:
    """Create professional plots showing row and column marginals of the coupling matrix."""
    _setup_professional_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Row marginals (sum over columns)
    row_marginals = np.sum(matrix_np, axis=1)
    x_pos1 = np.arange(len(row_marginals))
    
    # Professional bar plot for row marginals
    bars1 = ax1.bar(x_pos1, row_marginals, alpha=0.8, 
                    color=PROFESSIONAL_COLORS['primary'],
                    edgecolor=PROFESSIONAL_COLORS['text'], linewidth=1.2)
    
    # Add value labels on top of bars for small numbers of clusters
    if len(row_marginals) <= 15:
        for i, (bar, value) in enumerate(zip(bars1, row_marginals)):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(row_marginals)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_title('Row Marginals (Source Distribution)', fontweight='bold', 
                  fontsize=14, color=PROFESSIONAL_COLORS['text'], pad=15)
    ax1.set_xlabel('Source Cluster', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Marginal Probability', fontsize=12, fontweight='bold', labelpad=10)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.8)
    
    # Set cluster names if available
    if cluster_names and len(cluster_names) >= matrix_np.shape[0]:
        ax1.set_xticks(x_pos1)
        ax1.set_xticklabels(cluster_names[:len(row_marginals)], 
                           rotation=45, ha='right', fontsize=10)
    else:
        ax1.set_xticks(x_pos1)
        if len(row_marginals) > 20:  # Reduce tick density for large matrices
            tick_step = max(1, len(row_marginals) // 10)
            ax1.set_xticks(x_pos1[::tick_step])
    
    # Column marginals (sum over rows)
    col_marginals = np.sum(matrix_np, axis=0)
    x_pos2 = np.arange(len(col_marginals))
    
    # Professional bar plot for column marginals
    bars2 = ax2.bar(x_pos2, col_marginals, alpha=0.8, 
                    color=PROFESSIONAL_COLORS['secondary'],
                    edgecolor=PROFESSIONAL_COLORS['text'], linewidth=1.2)
    
    # Add value labels on top of bars for small numbers of clusters
    if len(col_marginals) <= 15:
        for i, (bar, value) in enumerate(zip(bars2, col_marginals)):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(col_marginals)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_title('Column Marginals (Target Distribution)', fontweight='bold', 
                  fontsize=14, color=PROFESSIONAL_COLORS['text'], pad=15)
    ax2.set_xlabel('Target Cluster', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Marginal Probability', fontsize=12, fontweight='bold', labelpad=10)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.8)
    
    # Set cluster names if available
    if cluster_names and len(cluster_names) >= matrix_np.shape[1]:
        ax2.set_xticks(x_pos2)
        ax2.set_xticklabels(cluster_names[:len(col_marginals)], 
                           rotation=45, ha='right', fontsize=10)
    else:
        ax2.set_xticks(x_pos2)
        if len(col_marginals) > 20:  # Reduce tick density for large matrices
            tick_step = max(1, len(col_marginals) // 10)
            ax2.set_xticks(x_pos2[::tick_step])
    
    # Add subtle borders
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color(PROFESSIONAL_COLORS['text'])
    
    # Add overall title
    fig.suptitle('Coupling Matrix Marginal Distributions', fontsize=16, fontweight='bold',
                 color=PROFESSIONAL_COLORS['text'], y=0.98)
    
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
    Log discriminator marginal distributions to wandb with individual-sample metrics.
    
    This function computes metrics on each sample individually, then averages the metrics.
    This preserves individual sample behavior instead of destroying it with batch averaging.
    
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
    
    # Convert tensors to numpy - keep batch dimension for individual sample metrics
    try:
        p_y_x1_batch = p_y_x1.detach().cpu().numpy()    # [batch_size, num_labels]
        p_y_x2_batch = p_y_x2.detach().cpu().numpy()    # [batch_size, num_labels] 
        p_y_x1x2_batch = p_y_x1x2.detach().cpu().numpy()  # [batch_size, num_labels]
        
        # Compute batch averages for visualization only
        p_y_x1_avg = p_y_x1_batch.mean(axis=0)     # [num_labels] - averaged for visualization
        p_y_x2_avg = p_y_x2_batch.mean(axis=0)     # [num_labels] - averaged for visualization
        p_y_x1x2_avg = p_y_x1x2_batch.mean(axis=0) # [num_labels] - averaged for visualization
        
        if p_y_marginal is not None:
            p_y_np = p_y_marginal.detach().cpu().numpy()  # [num_labels]
        else:
            p_y_np = None
            
    except Exception as e:
        warnings.warn(f"Error converting discriminator outputs to numpy: {e}")
        return
    
    # Validate shapes
    batch_size, num_labels = p_y_x1_batch.shape
    if p_y_x2_batch.shape != (batch_size, num_labels) or p_y_x1x2_batch.shape != (batch_size, num_labels):
        warnings.warn(f"Inconsistent batch shapes: p_y_x1={p_y_x1_batch.shape}, "
                     f"p_y_x2={p_y_x2_batch.shape}, p_y_x1x2={p_y_x1x2_batch.shape}")
        return
    
    if p_y_np is not None and p_y_np.shape[0] != num_labels:
        warnings.warn(f"Inconsistent marginal shape: p_y={p_y_np.shape[0]}, expected={num_labels}")
        p_y_np = None
    
    # Create log dictionary
    log_dict = {}
    
    # Add the custom step metric to the log dictionary
    if step is not None:
        log_dict[f"{prefix}/sinkhorn_step"] = step
    
    # Compute metrics on individual samples, then average the metrics
    individual_stats = _compute_individual_sample_basic_stats(p_y_x1_batch, p_y_x2_batch, p_y_x1x2_batch)
    for key, value in individual_stats.items():
        log_dict[f"{prefix}/{key}"] = value
    
    # Skip expensive visualizations during LR finding for performance
    if lr_finding_mode:
        # During LR finding, only log basic entropy and uniformity for monitoring
        log_dict[f"{prefix}/p_y_x1_entropy_lr_finding"] = float(-np.sum(p_y_x1_avg * np.log(p_y_x1_avg + 1e-10)))
        log_dict[f"{prefix}/p_y_x2_entropy_lr_finding"] = float(-np.sum(p_y_x2_avg * np.log(p_y_x2_avg + 1e-10)))
        log_dict[f"{prefix}/p_y_x1x2_entropy_lr_finding"] = float(-np.sum(p_y_x1x2_avg * np.log(p_y_x1x2_avg + 1e-10)))
        if step is not None:
            log_dict[f"{prefix}/lr_finding_step"] = step
    else:
        # Full visualization during normal training/evaluation (use averaged distributions for visualization)
        # Create and log discriminator marginal distributions plot
        marginal_dist_fig = _create_discriminator_marginal_plots(
            p_y_x1_avg, p_y_x2_avg, p_y_x1x2_avg, p_y_np, cluster_names=cluster_names
        )
        log_dict[f"{prefix}/marginal_distributions"] = wandb.Image(marginal_dist_fig)
        plt.close(marginal_dist_fig)  # Clean up memory
    
    # Log to wandb - now using custom step metrics, no step conflicts!
    wandb.log(log_dict)


def _compute_individual_sample_basic_stats(
    p_y_x1_batch: np.ndarray,     # [batch_size, num_labels]
    p_y_x2_batch: np.ndarray,     # [batch_size, num_labels]
    p_y_x1x2_batch: np.ndarray   # [batch_size, num_labels]
) -> Dict[str, float]:
    """
    Compute basic statistics on individual samples.
    This preserves individual sample behavior instead of destroying it with batch averaging.
    """
    batch_size, num_labels = p_y_x1_batch.shape
    stats = {}
    
    # Compute metrics for each individual sample
    distributions = [("p_y_x1", p_y_x1_batch), ("p_y_x2", p_y_x2_batch), ("p_y_x1x2", p_y_x1x2_batch)]
    
    for name, batch_dist in distributions:
        # Individual sample metrics
        sample_entropies = []
        sample_max_probs = []
        sample_min_probs = []
        sample_stds = []
        sample_uniformities = []
        sample_concentrations = []
        sample_dominant_clusters = []
        
        for sample_idx in range(batch_size):
            sample_dist = batch_dist[sample_idx]  # [num_labels]
            
            # Individual sample metrics
            entropy = -np.sum(sample_dist * np.log(sample_dist + 1e-10))
            max_prob = np.max(sample_dist)
            min_prob = np.min(sample_dist)
            std = np.std(sample_dist)
            uniformity = 1.0 - np.sum((sample_dist - 1.0/num_labels)**2)  # 1 - sum of squared deviations from uniform
            concentration = np.sum(sample_dist**2)  # Concentration measure (inverse Simpson diversity)
            dominant_cluster = np.argmax(sample_dist)
            
            sample_entropies.append(entropy)
            sample_max_probs.append(max_prob)
            sample_min_probs.append(min_prob)
            sample_stds.append(std)
            sample_uniformities.append(uniformity)
            sample_concentrations.append(concentration)
            sample_dominant_clusters.append(dominant_cluster)
        
        # Average the individual sample metrics
        stats[f"{name}_entropy"] = float(np.mean(sample_entropies))
        stats[f"{name}_max_prob"] = float(np.mean(sample_max_probs))
        stats[f"{name}_min_prob"] = float(np.mean(sample_min_probs))
        stats[f"{name}_std"] = float(np.mean(sample_stds))
        stats[f"{name}_uniformity"] = float(np.mean(sample_uniformities))
        stats[f"{name}_concentration"] = float(np.mean(sample_concentrations))
        stats[f"{name}_dominant_cluster"] = float(np.mean(sample_dominant_clusters))  # Average dominant cluster index
        
        # Additional metrics: variance of individual sample metrics (measure of consistency within each distribution)
        stats[f"{name}_entropy_std"] = float(np.std(sample_entropies))
        stats[f"{name}_max_prob_std"] = float(np.std(sample_max_probs))
        stats[f"{name}_uniformity_std"] = float(np.std(sample_uniformities))
        stats[f"{name}_concentration_std"] = float(np.std(sample_concentrations))
    
    return stats


def _create_discriminator_marginal_plots(
    p_y_x1_np: np.ndarray,
    p_y_x2_np: np.ndarray,
    p_y_x1x2_np: np.ndarray,
    p_y_np: Optional[np.ndarray] = None,
    cluster_names: Optional[list] = None
) -> plt.Figure:
    """Create professional bar plots showing all discriminator marginal distributions."""
    _setup_professional_style()
    
    # Determine number of subplots
    num_plots = 4 if p_y_np is not None else 3
    
    if num_plots == 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    else:
        fig, axes_raw = plt.subplots(2, 2, figsize=(16, 12))
        axes = [axes_raw[0, 0], axes_raw[0, 1], axes_raw[1, 0], axes_raw[1, 1]]
    
    num_labels = len(p_y_x1_np)
    x_positions = np.arange(num_labels)
    
    # Set up cluster names
    if cluster_names and len(cluster_names) >= num_labels:
        labels = cluster_names[:num_labels]
    else:
        labels = [f'C{i}' for i in range(num_labels)]
    
    # Professional color scheme for each discriminator
    colors = [PROFESSIONAL_COLORS['primary'], PROFESSIONAL_COLORS['secondary'], 
              PROFESSIONAL_COLORS['tertiary'], PROFESSIONAL_COLORS['quaternary']]
    
    distributions = [
        (p_y_x1_np, 'p(y|x₁) - First Domain Discriminator', colors[0]),
        (p_y_x2_np, 'p(y|x₂) - Second Domain Discriminator', colors[1]),
        (p_y_x1x2_np, 'p(y|x₁,x₂) - Joint Discriminator', colors[2])
    ]
    
    if p_y_np is not None:
        distributions.append((p_y_np, 'p(y) - Prior Marginal Distribution', colors[3]))
    
    for idx, (dist, title, color) in enumerate(distributions):
        ax = axes[idx]
        
        # Create professional bar plot
        bars = ax.bar(x_positions, dist, alpha=0.85, color=color,
                     edgecolor=PROFESSIONAL_COLORS['text'], linewidth=1.2)
        
        # Add value labels for small number of clusters
        if num_labels <= 12:
            for i, (bar, value) in enumerate(zip(bars, dist)):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(dist)*0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Professional styling
        ax.set_title(title, fontweight='bold', fontsize=13, 
                    color=PROFESSIONAL_COLORS['text'], pad=15)
        ax.set_xlabel('Cluster/Label', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylabel('Probability', fontsize=11, fontweight='bold', labelpad=8)
        ax.grid(axis='y', alpha=0.3, linewidth=0.8)
        
        # Set labels and ticks
        ax.set_xticks(x_positions)
        if num_labels <= 20:
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        else:
            # Reduce tick density for many clusters
            tick_step = max(1, num_labels // 10)
            ax.set_xticks(x_positions[::tick_step])
            ax.set_xticklabels([labels[i] for i in range(0, num_labels, tick_step)], 
                              rotation=45, ha='right', fontsize=9)
        
        # Add subtle borders
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color(PROFESSIONAL_COLORS['text'])
        
        # Set y-axis to start from 0 for better comparison
        ax.set_ylim(bottom=0)
    
    # Remove unused subplot if only 3 distributions
    if num_plots == 3:
        fig.delaxes(axes[3])
    
    # Professional overall title
    plt.suptitle('Discriminator Marginal Distributions', fontsize=18, fontweight='bold',
                 color=PROFESSIONAL_COLORS['text'], y=0.95)
    
    plt.tight_layout()
    return fig


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