#!/usr/bin/env python3
"""
Predictability Metrics for Synergy Analysis

Component-wise predictability metrics for measuring how well different information 
components (synergy, unique, redundant) are preserved under bottleneck compression.

Provides:
- Synergy accuracy and normalized cross-entropy for discrete synergy features
- R² metrics for continuous base features, partitioned by unique/redundant subsets
- Validation-wide accumulation for stable metric computation
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

# Baseline constants for random prediction performance
BASELINES = {
    'non_synergy_mse': 0.21,      # Expected MSE for random predictions on features 0-9
    'synergy_source_mse': 0.083,   # Expected MSE for random predictions on uniform_random feature  
    'synergy_accuracy': 0.125,     # Expected accuracy for random 8-class classification (1/8)
    'synergy_cross_entropy': 2.08, # Expected CE for random 8-class classification (-log(1/8))
    'synergy_norm_ce': 0.0,        # Expected normalized CE for random predictions (worst case)
}


class PredictabilityAccumulator:
    """Accumulates metrics across validation batches for stable computation."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        # Synergy metrics
        self.synergy_correct_count = 0
        self.synergy_total_count = 0
        self.synergy_ce_sum = 0.0
        
        # Base feature metrics - per feature accumulators
        self.base_sum_sq_err = {}  # domain -> tensor of per-feature MSE sums
        self.base_sum_sq_tot = {}  # domain -> tensor of per-feature variance sums  
        self.base_sample_count = {}  # domain -> sample count
        self.base_means = {}  # domain -> tensor of per-feature means
        
    def update_synergy_metrics(
        self, 
        synergy_logits: torch.Tensor, 
        synergy_targets: torch.Tensor,
        n_bins: int = 8
    ):
        """Update synergy accuracy and cross-entropy accumulators."""
        if synergy_logits.numel() == 0 or synergy_targets.numel() == 0:
            return
            
        # Convert targets to class indices
        target_flat = synergy_targets.view(-1)
        target_classes = torch.round(target_flat * (n_bins - 1)).long()
        target_classes = torch.clamp(target_classes, 0, n_bins - 1)
        
        # Reshape logits for classification
        logits_flat = synergy_logits.view(-1, n_bins)
        
        # Accuracy
        predicted_classes = torch.argmax(logits_flat, dim=1)
        correct = (predicted_classes == target_classes).sum().item()
        
        self.synergy_correct_count += correct
        self.synergy_total_count += target_classes.numel()
        
        # Cross-entropy
        ce_loss = F.cross_entropy(logits_flat, target_classes, reduction='sum')
        self.synergy_ce_sum += ce_loss.item()
    
    def update_base_metrics(
        self,
        domain: str,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """Update base feature R² accumulators for a domain."""
        if predictions.numel() == 0 or targets.numel() == 0:
            return
            
        # Ensure same shape
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
        
        batch_size = predictions.shape[0]
        n_features = predictions.shape[1]
        
        # Initialize domain accumulators if needed
        if domain not in self.base_sum_sq_err:
            self.base_sum_sq_err[domain] = torch.zeros(n_features, device=self.device)
            self.base_sum_sq_tot[domain] = torch.zeros(n_features, device=self.device)
            self.base_sample_count[domain] = 0
            self.base_means[domain] = torch.zeros(n_features, device=self.device)
        
        # Accumulate for Welford's algorithm
        old_count = self.base_sample_count[domain]
        new_count = old_count + batch_size
        
        # Update means using Welford's method
        delta = targets.mean(dim=0) - self.base_means[domain]
        self.base_means[domain] += delta * batch_size / new_count
        
        # Accumulate squared errors
        sq_err = ((predictions - targets) ** 2).sum(dim=0)
        self.base_sum_sq_err[domain] += sq_err
        
        # Accumulate total variance numerator  
        # For each feature: sum of (x_i - overall_mean)^2
        for i in range(batch_size):
            delta_old = targets[i] - self.base_means[domain]
            delta_new = targets[i] - (self.base_means[domain] + delta * (i + 1) / new_count)
            self.base_sum_sq_tot[domain] += delta_old * delta_new
            
        self.base_sample_count[domain] = new_count
    
    def compute_final_metrics(self) -> Dict[str, float]:
        """Compute final accumulated metrics."""
        metrics = {}
        
        # Synergy metrics
        if self.synergy_total_count > 0:
            synergy_acc = self.synergy_correct_count / self.synergy_total_count
            avg_ce = self.synergy_ce_sum / self.synergy_total_count
            # Normalized CE: 1 - CE/log(K), clipped to [0,1]
            synergy_norm_ce = max(0.0, min(1.0, 1.0 - avg_ce / np.log(8)))
            
            metrics['predictability/synergy_acc'] = synergy_acc
            metrics['predictability/synergy_norm_ce'] = synergy_norm_ce
        else:
            metrics['predictability/synergy_acc'] = 0.0
            metrics['predictability/synergy_norm_ce'] = 0.0
            
        return metrics
    
    def compute_base_r2_by_partition(
        self,
        domain: str,
        unique_indices: List[int],
        redundant_indices: List[int]
    ) -> Dict[str, float]:
        """Compute R² for unique and redundant feature partitions."""
        if domain not in self.base_sum_sq_err or self.base_sample_count[domain] <= 1:
            return {
                f'predictability/{domain}_unique_r2': 0.0,
                f'predictability/{domain}_redundant_r2': 0.0,
                'predictability/unique_r2': 0.0,
                'predictability/redundant_r2': 0.0
            }
        
        # Compute per-feature R² = 1 - MSE/Var
        mse_per_feature = self.base_sum_sq_err[domain] / self.base_sample_count[domain]
        var_per_feature = self.base_sum_sq_tot[domain] / (self.base_sample_count[domain] - 1)
        
        # Avoid division by zero
        var_per_feature = torch.clamp(var_per_feature, min=1e-8)
        r2_per_feature = 1.0 - mse_per_feature / var_per_feature
        r2_per_feature = torch.clamp(r2_per_feature, 0.0, 1.0)  # Clip to [0,1]
        
        metrics = {}
        
        # Unique features R²
        if unique_indices:
            unique_mask = torch.tensor(unique_indices, device=self.device, dtype=torch.long)
            unique_r2 = r2_per_feature[unique_mask].mean().item()
            metrics[f'predictability/{domain}_unique_r2'] = unique_r2
            metrics['predictability/unique_r2'] = unique_r2  # Domain-agnostic
        else:
            metrics[f'predictability/{domain}_unique_r2'] = 0.0
            metrics['predictability/unique_r2'] = 0.0
            
        # Redundant features R²
        if redundant_indices:
            redundant_mask = torch.tensor(redundant_indices, device=self.device, dtype=torch.long)
            redundant_r2 = r2_per_feature[redundant_mask].mean().item()
            metrics[f'predictability/{domain}_redundant_r2'] = redundant_r2
            metrics['predictability/redundant_r2'] = redundant_r2  # Domain-agnostic
        else:
            metrics[f'predictability/{domain}_redundant_r2'] = 0.0
            metrics['predictability/redundant_r2'] = 0.0
            
        return metrics


def extract_base_features_by_domain(
    decoded: torch.Tensor, 
    domain: str, 
    synergy_config: Dict[str, Any]
) -> torch.Tensor:
    """Extract base features from decoded tensor for a domain."""
    # Based on synergy_losses.py logic
    if domain == 'attr':
        # For attr: base is first 12 dimensions (includes synergy source feature at index 11)
        base_features = decoded[:, :12]
    elif domain == 'v':
        # For v: base is first 13 dimensions (12D VAE latents + 1D size)
        base_features = decoded[:, :13]
    else:
        # For other domains: assume all features are base features
        # unless synergy features are specified
        if domain in synergy_config.get('feature_indices', {}):
            synergy_features = len(synergy_config['feature_indices'][domain])
            n_synergy_classes = 8
            synergy_logit_dims = synergy_features * n_synergy_classes
            base_dim = decoded.shape[1] - synergy_logit_dims
            base_features = decoded[:, :base_dim]
        else:
            base_features = decoded
    
    return base_features


def extract_synergy_logits_by_domain(
    decoded: torch.Tensor,
    domain: str, 
    synergy_config: Dict[str, Any]
) -> Optional[torch.Tensor]:
    """Extract synergy logits from decoded tensor for a domain."""
    if domain not in synergy_config.get('feature_indices', {}):
        return None
        
    synergy_features = len(synergy_config['feature_indices'][domain])
    if synergy_features == 0:
        return None
        
    n_synergy_classes = 8
    synergy_logit_dims = synergy_features * n_synergy_classes
    
    # Synergy logits are at the end of the decoded tensor
    synergy_logits = decoded[:, -synergy_logit_dims:]
    
    return synergy_logits


def compute_predictability(
    decoded: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor], 
    partitions: Dict[str, Dict[str, List[int]]],
    synergy_config: Dict[str, Any],
    device: torch.device,
    n_bins: int = 8
) -> Dict[str, float]:
    """
    Compute predictability metrics from decoded outputs and targets.
    
    Args:
        decoded: Dictionary of decoded outputs per domain
        targets: Dictionary of target values per domain  
        partitions: Dictionary of unique/redundant indices per domain
        synergy_config: Synergy configuration
        device: Device for computations
        n_bins: Number of synergy classes (default: 8)
        
    Returns:
        Dictionary of predictability metrics
    """
    accumulator = PredictabilityAccumulator(device)
    
    for domain in decoded.keys():
        if domain not in targets:
            continue
            
        decoded_tensor = decoded[domain]
        target_tensor = targets[domain]
        
        # Extract base features and compute R²
        base_decoded = extract_base_features_by_domain(
            decoded_tensor, domain, synergy_config
        )
        base_targets = extract_base_features_by_domain(
            target_tensor, domain, synergy_config
        )
        
        accumulator.update_base_metrics(domain, base_decoded, base_targets)
        
        # Extract synergy features and compute accuracy/CE
        synergy_logits = extract_synergy_logits_by_domain(
            decoded_tensor, domain, synergy_config
        )
        synergy_targets = extract_synergy_logits_by_domain(
            target_tensor, domain, synergy_config
        )
        
        if synergy_logits is not None and synergy_targets is not None:
            # For targets, synergy are stored as normalized values, not logits
            # Extract the synergy portion from the target tensor
            synergy_features = len(synergy_config['feature_indices'][domain])
            if domain == 'attr':
                # For attr: synergy target is at index 12 (after 12D base features including source)
                synergy_target_values = target_tensor[:, 12:12+synergy_features]
            else:
                # For other domains, extract based on config
                base_dim = base_targets.shape[1]
                synergy_target_values = target_tensor[:, base_dim:base_dim+synergy_features]
                
            accumulator.update_synergy_metrics(
                synergy_logits, synergy_target_values, n_bins
            )
    
    # Compute final metrics
    metrics = accumulator.compute_final_metrics()
    
    # Add domain-specific R² metrics for each domain
    for domain in decoded.keys():
        if domain in partitions:
            unique_indices = partitions[domain].get('unique', [])
            redundant_indices = partitions[domain].get('redundant', [])
            
            domain_metrics = accumulator.compute_base_r2_by_partition(
                domain, unique_indices, redundant_indices
            )
            metrics.update(domain_metrics)
    
    return metrics


def create_reconstruction_plot(
    output_dir: str,
    workspace_dim: int,
    plot_data: Dict[str, List[float]],
    save_to_wandb: bool = True
) -> Optional[str]:
    """
    Create a plot showing reconstruction MSE for non-synergy vs synergy source features.
    
    Args:
        output_dir: Directory to save the plot
        workspace_dim: Workspace dimension for labeling
        plot_data: Dictionary with keys 'epochs', 'non_synergy_mse', 'synergy_source_mse'
        save_to_wandb: Whether to log the plot to W&B
        
    Returns:
        Path to saved plot file
    """
    if not plot_data or len(plot_data.get('epochs', [])) == 0:
        logger.warning("No plot data available for reconstruction plot")
        return None
    
    plt.figure(figsize=(10, 6))
    
    epochs = plot_data['epochs']
    non_synergy_mse = plot_data.get('non_synergy_mse', [])
    synergy_source_mse = plot_data.get('synergy_source_mse', [])
    
    # Plot non-synergy features (solid line)
    if non_synergy_mse:
        plt.plot(epochs[:len(non_synergy_mse)], non_synergy_mse, 
                'b-', linewidth=2, label='Non-synergy features (0-9)')
    
    # Plot synergy source feature (dashed line)
    if synergy_source_mse:
        plt.plot(epochs[:len(synergy_source_mse)], synergy_source_mse, 
                'r--', linewidth=2, label='Synergy source feature (10)')
    
    # Add baseline reference lines
    if epochs:
        plt.axhline(y=BASELINES['non_synergy_mse'], color='blue', linestyle=':', alpha=0.7, 
                   linewidth=1.5, label=f'Random baseline (non-synergy): {BASELINES["non_synergy_mse"]:.3f}')
        plt.axhline(y=BASELINES['synergy_source_mse'], color='red', linestyle=':', alpha=0.7,
                   linewidth=1.5, label=f'Random baseline (synergy source): {BASELINES["synergy_source_mse"]:.3f}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction MSE')
    plt.title(f'Attribute Reconstruction MSE (Workspace Dim = {workspace_dim})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often better for MSE
    
    # Save plot
    plot_path = Path(output_dir) / f'reconstruction_mse_dim_{workspace_dim}.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Log to W&B if available and requested
    if save_to_wandb:
        try:
            import wandb
            wandb.log({f"reconstruction_plot_dim_{workspace_dim}": wandb.Image(str(plot_path))})
            logger.info(f"Logged reconstruction plot to W&B for dim {workspace_dim}")
        except ImportError:
            logger.warning("W&B not available, skipping plot logging")
        except Exception as e:
            logger.warning(f"Failed to log plot to W&B: {e}")
    
    plt.close()
    logger.info(f"Saved reconstruction plot to {plot_path}")
    
    return str(plot_path)


def create_synergy_metrics_plot(
    output_dir: str,
    workspace_dim: int,
    plot_data: Dict[str, List[float]],
    save_to_wandb: bool = True
) -> Optional[str]:
    """
    Create a plot showing synergy accuracy and normalized cross-entropy metrics.
    
    Args:
        output_dir: Directory to save the plot
        workspace_dim: Workspace dimension for labeling
        plot_data: Dictionary with keys 'epochs', 'synergy_acc', 'synergy_norm_ce'
        save_to_wandb: Whether to log the plot to W&B
        
    Returns:
        Path to saved plot file
    """
    if not plot_data or len(plot_data.get('epochs', [])) == 0:
        logger.warning("No plot data available for synergy metrics plot")
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    epochs = plot_data['epochs']
    synergy_acc = plot_data.get('synergy_acc', [])
    synergy_norm_ce = plot_data.get('synergy_norm_ce', [])
    
    # Plot synergy accuracy (top subplot)
    if synergy_acc:
        ax1.plot(epochs[:len(synergy_acc)], synergy_acc, 
                'g-', linewidth=2, label='Synergy accuracy')
        ax1.axhline(y=BASELINES['synergy_accuracy'], color='gray', linestyle=':', alpha=0.7,
                   linewidth=1.5, label=f'Random baseline: {BASELINES["synergy_accuracy"]:.3f}')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Synergy Classification Metrics (Workspace Dim = {workspace_dim})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot normalized cross-entropy (bottom subplot)
    if synergy_norm_ce:
        ax2.plot(epochs[:len(synergy_norm_ce)], synergy_norm_ce, 
                'purple', linewidth=2, label='Normalized cross-entropy')
        ax2.axhline(y=BASELINES['synergy_norm_ce'], color='gray', linestyle=':', alpha=0.7,
                   linewidth=1.5, label=f'Random baseline: {BASELINES["synergy_norm_ce"]:.3f}')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Normalized CE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f'synergy_metrics_dim_{workspace_dim}.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Log to W&B if available and requested
    if save_to_wandb:
        try:
            import wandb
            wandb.log({f"synergy_metrics_plot_dim_{workspace_dim}": wandb.Image(str(plot_path))})
            logger.info(f"Logged synergy metrics plot to W&B for dim {workspace_dim}")
        except ImportError:
            logger.warning("W&B not available, skipping plot logging")
        except Exception as e:
            logger.warning(f"Failed to log plot to W&B: {e}")
    
    plt.close()
    logger.info(f"Saved synergy metrics plot to {plot_path}")
    
    return str(plot_path)
