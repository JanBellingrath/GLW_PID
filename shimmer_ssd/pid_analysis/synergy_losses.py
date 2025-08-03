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


def pad_attr_input_for_encoder(domain_input: torch.Tensor, domain_name: str) -> torch.Tensor:
    """
    Pad attribute domain input from 11D to 12D for encoder compatibility.
    
    Args:
        domain_input: Input tensor for the domain
        domain_name: Name of the domain
        
    Returns:
        Padded input tensor (12D for attr domain, unchanged for others)
    """
    if domain_name == 'attr':
        # Pad 11D attribute input to 12D with zeros
        padding = torch.zeros_like(domain_input[:, :1])  # Create 1D padding of same shape as batch
        return torch.cat([domain_input, padding], dim=1)
    else:
        return domain_input


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
        
        # Extract synergy targets from flattened format (avoid mutating original batch)
        synergy_targets = {}
        clean_batch = {}
        
        for key, value in batch.items():
            if key.startswith('_target_'):
                # Extract target domain name from key like '_target_v' -> 'v'
                domain_name = key[8:]  # Remove '_target_' prefix
                synergy_targets[domain_name] = value.to(device) if hasattr(value, 'to') else value
            else:
                # Keep non-target keys for standard processing
                clean_batch[key] = value
        
        # Use clean batch without target keys
        batch = clean_batch
        
        # Convert to None if no targets found (for compatibility)
        if not synergy_targets:
            synergy_targets = None
        
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
                        # Pad attribute inputs for encoder compatibility
                        padded_input = pad_attr_input_for_encoder(domain_input, domain_name)
                        
                        # For attribute domain: skip domain module, use preprocessed data directly
                        if domain_name == 'attr':
                            # padded_input is now 12D (11D preprocessed attributes + 1D padding)
                            encoded[domain_name] = model.gw_encoders[domain_name](padded_input)
                        elif domain_name == 'v':
                            # domain_input is already VAE latents (12D), bypass domain module
                            encoded[domain_name] = model.gw_encoders[domain_name](padded_input)
                        else:
                            # For other domains: use domain module first, then GW encoder
                            if domain_name in model.domain_mods:
                                domain_latent = model.domain_mods[domain_name](padded_input)
                                encoded[domain_name] = model.gw_encoders[domain_name](domain_latent)
                            else:
                                encoded[domain_name] = model.gw_encoders[domain_name](padded_input)
                
                if encoded:
                    # Fuse and decode
                    gw_state = model.fuse(encoded, selection_scores={})
                    decoded = {}
                    for domain_name in synergy_targets.keys():
                        if domain_name in model.gw_decoders:
                            # Decode from workspace to domain latent space
                            decoded_latent = model.gw_decoders[domain_name](gw_state)
                            
                            # For domains without domain modules: decoded output is already in target space
                            if domain_name == 'attr':
                                # decoded_latent is already in 11D attribute space
                                decoded[domain_name] = decoded_latent
                            elif domain_name == 'v':
                                # decoded_latent is already in VAE latent space (12D)
                                decoded[domain_name] = decoded_latent
                            else:
                                # For other domains: use domain module decoder if available
                                if (domain_name in model.domain_mods and 
                                    hasattr(model.domain_mods[domain_name], 'decode')):
                                    decoded[domain_name] = model.domain_mods[domain_name].decode(decoded_latent)
                                else:
                                    decoded[domain_name] = decoded_latent
                    
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
            
            # 2. Demi-cycle and cycle losses with padding for encoder compatibility
            # Create a modified batch with padded attribute inputs for these losses
            padded_batch = {}
            for domain_name, domain_input in batch.items():
                padded_batch[domain_name] = pad_attr_input_for_encoder(domain_input, domain_name)
            
            if loss_weights.get('demi_cycle', 0.0) > 0:
                from losses_and_weights_GLW_training import calculate_demi_cycle_loss
                demi_loss, demi_details = calculate_demi_cycle_loss(model, padded_batch, criterion)
                loss_details.update(demi_details)
                
                weighted_demi_loss = loss_weights['demi_cycle'] * demi_loss
                loss_details['weighted_demi_cycle_loss'] = weighted_demi_loss.item()
                
                if total_loss is None:
                    total_loss = weighted_demi_loss
                else:
                    total_loss = total_loss + weighted_demi_loss
            
            if loss_weights.get('cycle', 0.0) > 0:
                from losses_and_weights_GLW_training import calculate_cycle_loss
                cycle_loss, cycle_details = calculate_cycle_loss(model, padded_batch, criterion)
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