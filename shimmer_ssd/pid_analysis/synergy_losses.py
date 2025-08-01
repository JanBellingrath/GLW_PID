#!/usr/bin/env python3
"""
Synergy-Aware Loss Extensions for Global Workspace Training

Minimal extensions to existing loss framework to add synergy feature tracking:
- Reuses existing calculate_fusion_loss, calculate_demi_cycle_loss, calculate_cycle_loss
- Adds synergy-specific loss decomposition and logging
- Maintains full compatibility with existing training infrastructure
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
import logging

# Import existing loss functions instead of rewriting them
from losses_and_weights_GLW_training import (
    calculate_fusion_loss, 
    calculate_demi_cycle_loss, 
    calculate_cycle_loss
)

logger = logging.getLogger(__name__)


def extract_synergy_features(
    tensor: torch.Tensor, 
    domain: str,
    synergy_config: Dict[str, Any],
    feature_names: Optional[List[str]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract synergistic and non-synergistic features from tensor.
    
    Args:
        tensor: Input tensor (..., feature_dim)
        domain: Domain name
        synergy_config: Synergy configuration
        feature_names: Optional feature names for string resolution
        
    Returns:
        Tuple of (synergy_features, non_synergy_features)
    """
    feature_indices = synergy_config.get('feature_indices', {})
    
    if domain not in feature_indices:
        # No synergy features for this domain
        return torch.empty(tensor.shape[:-1] + (0,), device=tensor.device), tensor
    
    synergy_specs = feature_indices[domain]
    synergy_idx = []
    
    for spec in synergy_specs:
        if isinstance(spec, int):
            synergy_idx.append(spec)
        elif isinstance(spec, str) and feature_names:
            if spec in feature_names:
                synergy_idx.append(feature_names.index(spec))
            else:
                logger.warning(f"Feature name '{spec}' not found in {feature_names}")
        else:
            logger.warning(f"Could not resolve feature spec: {spec}")
    
    if not synergy_idx:
        # No valid synergy features found
        return torch.empty(tensor.shape[:-1] + (0,), device=tensor.device), tensor
    
    # Extract synergy features
    synergy_features = tensor[..., synergy_idx]
    
    # Extract non-synergy features (all others)
    total_features = tensor.shape[-1]
    non_synergy_idx = [i for i in range(total_features) if i not in synergy_idx]
    
    if not non_synergy_idx:
        # All features are synergistic
        non_synergy_features = torch.empty(tensor.shape[:-1] + (0,), device=tensor.device)
    else:
        non_synergy_features = tensor[..., non_synergy_idx]
    
    return synergy_features, non_synergy_features


def add_synergy_metrics_to_loss_details(
    loss_details: Dict[str, float],
    decoded: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    synergy_config: Dict[str, Any],
    criterion: torch.nn.Module,
    loss_prefix: str = "fusion"
) -> Dict[str, float]:
    """
    Add synergy-specific metrics to existing loss details.
    
    Args:
        loss_details: Existing loss details dict to extend
        decoded: Decoded tensors per domain
        targets: Target tensors per domain
        synergy_config: Synergy configuration
        criterion: Loss function
        loss_prefix: Prefix for loss names (fusion/demi_cycle/cycle)
        
    Returns:
        Extended loss details dict
    """
    for domain_name, target in targets.items():
        if domain_name not in decoded:
            continue
            
        reconstruction = decoded[domain_name]
        
        # Extract synergy-specific components
        synergy_target, non_synergy_target = extract_synergy_features(
            target, domain_name, synergy_config
        )
        synergy_recon, non_synergy_recon = extract_synergy_features(
            reconstruction, domain_name, synergy_config
        )
        
        # Calculate synergy-specific loss
        if synergy_target.numel() > 0:
            synergy_loss = criterion(synergy_recon, synergy_target)
            loss_details[f"{loss_prefix}_{domain_name}_synergy_loss"] = synergy_loss.item()
            
            # Individual feature losses if multiple synergy features
            if synergy_target.shape[-1] > 1:
                for i in range(synergy_target.shape[-1]):
                    feat_loss = criterion(
                        synergy_recon[..., i:i+1], 
                        synergy_target[..., i:i+1]
                    )
                    loss_details[f"{loss_prefix}_{domain_name}_synergy_feat_{i}_loss"] = feat_loss.item()
        
        # Calculate non-synergy loss
        if non_synergy_target.numel() > 0:
            non_synergy_loss = criterion(non_synergy_recon, non_synergy_target)
            loss_details[f"{loss_prefix}_{domain_name}_non_synergy_loss"] = non_synergy_loss.item()
    
    return loss_details


def create_synergy_loss_function(synergy_config: Dict[str, Any]):
    """
    Create a loss function that can be used as a drop-in replacement for calculate_losses_with_weights.
    
    Args:
        synergy_config: Synergy configuration
        
    Returns:
        Function with same signature as calculate_losses_with_weights but with synergy tracking
    """
    
    def synergy_calculate_losses_with_weights(
        model,
        batch,
        criterion,
        loss_weights,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Calculate losses with synergy tracking - drop-in replacement for original function."""
        
        # Extract synergy targets if they exist (avoid mutating original batch)
        synergy_targets = batch.get('_synergy_targets', None)
        if synergy_targets is not None:
            # Create a clean batch copy without synergy targets for standard processing
            batch = {k: v for k, v in batch.items() if k != '_synergy_targets'}
            # Move synergy targets to device
            synergy_targets = {
                domain: data.to(device) if hasattr(data, 'to') else data
                for domain, data in synergy_targets.items()
            }
        
        # Use the original function for the main loss calculation
        from losses_and_weights_GLW_training import calculate_losses_with_weights
        
        # CRITICAL FIX: For synergy training, we need to:
        # 1. Encode the INPUTS (without synergy features)
        # 2. Compare decoded outputs with TARGETS (with synergy features)
        
        if synergy_targets is not None:
            # This is synergy training - we need custom logic
            total_loss = None
            loss_details = {}
            
            # Process inputs for encoding
            processed_inputs = {}
            for domain_name, domain_input in batch.items():
                # Apply projector for text domain if it exists
                if domain_name == 't' and hasattr(model, 'domain_mods') and hasattr(model.domain_mods[domain_name], 'projector'):
                    projector = model.domain_mods[domain_name].projector
                    processed_inputs[domain_name] = projector(domain_input)
                else:
                    processed_inputs[domain_name] = domain_input
            
            # 1. Fusion Loss with synergy training
            if loss_weights.get('fusion', 0.0) > 0:
                # Encode inputs
                encoded = {}
                for domain_name, domain_input in processed_inputs.items():
                    if domain_name in model.gw_encoders:
                        encoded[domain_name] = model.gw_encoders[domain_name](domain_input)
                
                if encoded:
                    # Fuse and decode
                    gw_state = model.fuse(encoded, selection_scores={})
                    decoded = {}
                    for domain_name in synergy_targets.keys():
                        if domain_name in model.gw_decoders:
                            decoded[domain_name] = model.gw_decoders[domain_name](gw_state)
                    
                    # Calculate losses against TARGETS (with synergy features)
                    fusion_loss = None
                    num_domains = 0
                    
                    for domain_name, target in synergy_targets.items():
                        if domain_name in decoded:
                            domain_loss = criterion(decoded[domain_name], target)
                            loss_details[f"fusion_{domain_name}_loss"] = domain_loss.item()
                            
                            if fusion_loss is None:
                                fusion_loss = domain_loss
                            else:
                                fusion_loss = fusion_loss + domain_loss
                            num_domains += 1
                    
                    # Average loss
                    if fusion_loss is not None and num_domains > 1:
                        fusion_loss = fusion_loss / num_domains
                    
                    # Add synergy metrics
                    if fusion_loss is not None:
                        loss_details = add_synergy_metrics_to_loss_details(
                            loss_details, decoded, synergy_targets, synergy_config, criterion, "fusion"
                        )
                        
                        weighted_fusion_loss = loss_weights['fusion'] * fusion_loss
                        loss_details['weighted_fusion_loss'] = weighted_fusion_loss.item()
                        loss_details['fusion_loss'] = fusion_loss.item()
                        total_loss = weighted_fusion_loss
            
            # 2. Other losses (demi-cycle, cycle) - use standard approach with targets
            # Restore the batch format for other losses
            standard_batch = synergy_targets
            
            if loss_weights.get('demi_cycle', 0.0) > 0:
                from losses_and_weights_GLW_training import calculate_demi_cycle_loss
                demi_loss, demi_details = calculate_demi_cycle_loss(model, standard_batch, criterion)
                loss_details.update(demi_details)
                
                weighted_demi_loss = loss_weights['demi_cycle'] * demi_loss
                loss_details['weighted_demi_cycle_loss'] = weighted_demi_loss.item()
                
                if total_loss is None:
                    total_loss = weighted_demi_loss
                else:
                    total_loss = total_loss + weighted_demi_loss
            
            if loss_weights.get('cycle', 0.0) > 0:
                from losses_and_weights_GLW_training import calculate_cycle_loss
                cycle_loss, cycle_details = calculate_cycle_loss(model, standard_batch, criterion)
                loss_details.update(cycle_details)
                
                weighted_cycle_loss = loss_weights['cycle'] * cycle_loss
                loss_details['weighted_cycle_loss'] = weighted_cycle_loss.item()
                
                if total_loss is None:
                    total_loss = weighted_cycle_loss
                else:
                    total_loss = total_loss + weighted_cycle_loss
            
            if total_loss is None:
                total_loss = torch.tensor(0.0, device=device)
            
            loss_details['total_loss'] = total_loss.item()
            
        else:
            # Standard training - use original function
            total_loss, loss_details = calculate_losses_with_weights(
                model, batch, criterion, loss_weights, device
            )
        
        return total_loss, loss_details
    
    return synergy_calculate_losses_with_weights


def calculate_synergy_aware_losses(
    model,
    batch_inputs: Dict[str, torch.Tensor],
    batch_targets: Dict[str, torch.Tensor],
    synergy_config: Dict[str, Any],
    loss_weights: Dict[str, float],
    criterion: torch.nn.Module
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Calculate synergy-aware losses by extending existing loss functions.
    
    Args:
        model: The GW model
        batch_inputs: Input tensors per domain (synergy features excluded)
        batch_targets: Target tensors per domain (synergy features included)
        synergy_config: Synergy configuration
        loss_weights: Loss weights
        criterion: Loss function
        
    Returns:
        Tuple of (total_loss, detailed_loss_dict)
    """
    # Convert to standard batch format for existing loss functions
    # Use targets as the batch since existing functions expect input=target for reconstruction
    processed_batch = batch_targets  # Standard reconstruction expects this
    
    total_loss = None
    loss_details = {}
    
    # 1. Fusion Loss with synergy tracking
    if loss_weights.get('fusion', 0.0) > 0:
        # Use existing fusion loss calculation
        fusion_loss, fusion_details = calculate_fusion_loss(
            model, processed_batch, criterion, 
            model.fusion_weights.get('use_weights_for_loss', False)
        )
        loss_details.update(fusion_details)
        
        # Add synergy-specific metrics
        # We need to manually reconstruct to get decoded tensors for synergy analysis
        encoded = {}
        for domain_name, domain_input in batch_inputs.items():
            if domain_name in model.gw_encoders:
                encoded[domain_name] = model.gw_encoders[domain_name](domain_input)
        
        if encoded:
            gw_state = model.fuse(encoded, selection_scores={})
            decoded = {}
            for domain_name in batch_targets.keys():
                if domain_name in model.gw_decoders:
                    decoded[domain_name] = model.gw_decoders[domain_name](gw_state)
            
            loss_details = add_synergy_metrics_to_loss_details(
                loss_details, decoded, batch_targets, synergy_config, criterion, "fusion"
            )
        
        weighted_fusion_loss = loss_weights['fusion'] * fusion_loss
        loss_details['weighted_fusion_loss'] = weighted_fusion_loss.item()
        
        total_loss = weighted_fusion_loss
    
    # 2. Demi-Cycle Loss with synergy tracking
    if loss_weights.get('demi_cycle', 0.0) > 0:
        demi_loss, demi_details = calculate_demi_cycle_loss(model, processed_batch, criterion)
        loss_details.update(demi_details)
        
        # TODO: Add synergy tracking for demi-cycle if needed
        
        weighted_demi_loss = loss_weights['demi_cycle'] * demi_loss
        loss_details['weighted_demi_cycle_loss'] = weighted_demi_loss.item()
        
        if total_loss is None:
            total_loss = weighted_demi_loss
        else:
            total_loss = total_loss + weighted_demi_loss
    
    # 3. Cycle Loss with synergy tracking
    if loss_weights.get('cycle', 0.0) > 0:
        cycle_loss, cycle_details = calculate_cycle_loss(model, processed_batch, criterion)
        loss_details.update(cycle_details)
        
        # TODO: Add synergy tracking for cycle if needed
        
        weighted_cycle_loss = loss_weights['cycle'] * cycle_loss
        loss_details['weighted_cycle_loss'] = weighted_cycle_loss.item()
        
        if total_loss is None:
            total_loss = weighted_cycle_loss
        else:
            total_loss = total_loss + weighted_cycle_loss
    
    # Handle case where no losses were computed
    if total_loss is None:
        device = next(model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
    
    loss_details['total_loss'] = total_loss.item()
    return total_loss, loss_details


def process_synergy_batch(
    batch: Dict[str, torch.Tensor], 
    device: torch.device
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Process a synergy batch to separate inputs and targets.
    
    Args:
        batch: Batch dictionary from SynergyDataset
        device: Target device
        
    Returns:
        Tuple of (inputs_dict, targets_dict)
    """
    inputs = {}
    targets = {}
    
    for domain, data in batch['inputs'].items():
        inputs[domain] = data.to(device)
    
    for domain, data in batch['targets'].items():
        targets[domain] = data.to(device)
    
    return inputs, targets 