#!/usr/bin/env python3
"""
Synergy-Aware Loss Extensions for Global Workspace Training

Clean synergy loss implementation following the principle that only fusion loss 
sees synergy features, while all other losses operate purely on base features:

- Fusion loss: Encodes 11D attr inputs, decodes to full target (11D + synergy)
  * Uses MSE loss for base features (11D continuous attributes)
  * Uses Cross-Entropy loss for synergy features (8-class discrete categories)
- Demi-cycle & cycle losses: Only compare first 11D of reconstructions vs 11D inputs
- Translation loss: Masks out synergy features, learns only non-synergistic mappings
- No padding logic - encoders and decoders handle exact dimensions needed
- Maintains full compatibility with existing training infrastructure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

# Import existing loss functions instead of rewriting them
from losses_and_weights_GLW_training import (
    calculate_fusion_loss, 
    calculate_demi_cycle_loss, 
    calculate_cycle_loss
)

logger = logging.getLogger(__name__)


def convert_synergy_targets_to_classes(synergy_targets: torch.Tensor, n_bins: int = 8) -> torch.Tensor:
    """
    Convert normalized synergy targets [0,1] to class indices [0, n_bins-1].
    
    Args:
        synergy_targets: Normalized synergy values in [0,1] range
        n_bins: Number of discrete bins (default: 8)
        
    Returns:
        Class indices as LongTensor for CrossEntropyLoss
    """
    # Convert normalized values back to class indices
    # Normalized values are bin_index / (n_bins - 1)
    # So class_index = round(normalized_value * (n_bins - 1))
    class_indices = torch.round(synergy_targets * (n_bins - 1)).long()
    
    # Clamp to valid range [0, n_bins-1] for safety
    class_indices = torch.clamp(class_indices, 0, n_bins - 1)
    
    return class_indices


def calculate_synergy_loss_with_crossentropy(
    synergy_recon: torch.Tensor, 
    synergy_target: torch.Tensor,
    n_bins: int = 8
) -> torch.Tensor:
    """
    Calculate synergy loss using a classification-appropriate loss for discrete categories.
    
    Since the model outputs single values, we use a "soft target" approach:
    - Convert target to class indices
    - Apply cross-entropy using the distance between prediction and target bins
    
    Args:
        synergy_recon: Model output for synergy features (single values) [..., 1] or [...]
        synergy_target: Target synergy values (normalized) [..., 1] or [...]
        n_bins: Number of discrete synergy bins
        
    Returns:
        Classification loss for synergy features
    """
    # Handle empty tensors
    if synergy_recon.numel() == 0 or synergy_target.numel() == 0:
        return torch.tensor(0.0, device=synergy_recon.device, requires_grad=True)
    
    # Flatten to 1D for processing while preserving gradients
    # Handle both [..., 1] and [...] shapes
    if synergy_target.shape[-1] == 1:
        target_flat = synergy_target.squeeze(-1)
    else:
        target_flat = synergy_target
        
    if synergy_recon.shape[-1] == 1:
        recon_flat = synergy_recon.squeeze(-1)
    else:
        recon_flat = synergy_recon
    
    # Ensure same batch dimension
    if target_flat.shape != recon_flat.shape:
        raise ValueError(f"Shape mismatch: synergy_recon {recon_flat.shape} vs synergy_target {target_flat.shape}")
    
    # Flatten to 1D for cross-entropy (which expects [N] for targets and [N, C] for inputs)
    target_1d = target_flat.flatten()
    recon_1d = recon_flat.flatten()
    
    # Convert normalized targets to class indices
    target_classes = convert_synergy_targets_to_classes(target_1d, n_bins)
    
    # Create soft targets: use softmax over distances to each bin center
    bin_centers = torch.linspace(0, 1, n_bins, device=recon_1d.device)  # [0, 1/7, 2/7, ..., 1]
    
    # Calculate distances from output to each bin center
    distances = torch.abs(recon_1d.unsqueeze(-1) - bin_centers.unsqueeze(0))  # [N, n_bins]
    
    # Convert distances to probabilities (closer bins get higher probability)
    # Use negative distances with temperature scaling for softmax
    temperature = 0.1  # Small temperature for sharper distributions
    logits = -distances / temperature  # [N, n_bins]
    
    # Use cross-entropy loss
    ce_loss = F.cross_entropy(logits, target_classes)
    
    return ce_loss





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
        
        # Calculate synergy-specific loss using classification-appropriate loss
        if synergy_target.numel() > 0:
            try:
                synergy_loss = calculate_synergy_loss_with_crossentropy(
                    synergy_recon, synergy_target, n_bins=8
                )
                loss_details[f"{loss_prefix}_{domain_name}_synergy_loss"] = synergy_loss.item()
                
                # Individual feature losses if multiple synergy features
                if synergy_target.shape[-1] > 1:
                    for i in range(synergy_target.shape[-1]):
                        try:
                            feat_loss = calculate_synergy_loss_with_crossentropy(
                                synergy_recon[..., i:i+1], synergy_target[..., i:i+1], n_bins=8
                            )
                            loss_details[f"{loss_prefix}_{domain_name}_synergy_feat_{i}_loss"] = feat_loss.item()
                        except Exception as e:
                            logger.warning(f"Cross-entropy failed for synergy feature {i}, using MSE: {e}")
                            feat_loss = criterion(
                                synergy_recon[..., i:i+1], 
                                synergy_target[..., i:i+1]
                            )
                            loss_details[f"{loss_prefix}_{domain_name}_synergy_feat_{i}_loss"] = feat_loss.item()
                            
            except Exception as e:
                logger.warning(f"Cross-entropy loss failed for synergy features in metrics, falling back to MSE: {e}")
                synergy_loss = criterion(synergy_recon, synergy_target)
                loss_details[f"{loss_prefix}_{domain_name}_synergy_loss"] = synergy_loss.item()
        
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
                        # For attribute domain: skip domain module, use preprocessed data directly
                        if domain_name == 'attr':
                            # domain_input is 11D preprocessed attributes (no padding needed)
                            encoded[domain_name] = model.gw_encoders[domain_name](domain_input)
                        elif domain_name == 'v':
                            # domain_input is already VAE latents (12D), bypass domain module
                            encoded[domain_name] = model.gw_encoders[domain_name](domain_input)
                        else:
                            # For other domains: use domain module first, then GW encoder
                            if domain_name in model.domain_mods:
                                domain_latent = model.domain_mods[domain_name](domain_input)
                                encoded[domain_name] = model.gw_encoders[domain_name](domain_latent)
                            else:
                                encoded[domain_name] = model.gw_encoders[domain_name](domain_input)
                
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
                                # decoded_latent includes base (11D) + synergy features
                                decoded[domain_name] = decoded_latent
                            elif domain_name == 'v':
                                # decoded_latent is already in VAE latent space
                                decoded[domain_name] = decoded_latent
                            else:
                                # For other domains: use domain module decoder if available
                                if (domain_name in model.domain_mods and 
                                    hasattr(model.domain_mods[domain_name], 'decode')):
                                    decoded[domain_name] = model.domain_mods[domain_name].decode(decoded_latent)
                                else:
                                    decoded[domain_name] = decoded_latent
                    
                    # Calculate losses against TARGETS (with synergy features) 
                    # Apply synergy loss scaling to give more weight to synergy features
                    fusion_loss = None
                    num_domains = 0
                    synergy_loss_scale = synergy_config.get('loss_scale', 1.0)
                    
                    for domain_name, target in synergy_targets.items():
                        if domain_name in decoded:
                            # Extract synergy and non-synergy components for scaled loss
                            synergy_target, non_synergy_target = extract_synergy_features(
                                target, domain_name, synergy_config
                            )
                            synergy_recon, non_synergy_recon = extract_synergy_features(
                                decoded[domain_name], domain_name, synergy_config
                            )
                            
                            # Calculate separate losses
                            domain_loss = 0.0
                            if non_synergy_target.numel() > 0:
                                non_synergy_loss = criterion(non_synergy_recon, non_synergy_target)
                                domain_loss += non_synergy_loss
                                loss_details[f"fusion_{domain_name}_non_synergy_loss"] = non_synergy_loss.item()
                            
                            if synergy_target.numel() > 0:
                                # Use classification-appropriate loss for discrete synergy features
                                try:
                                    synergy_loss = calculate_synergy_loss_with_crossentropy(
                                        synergy_recon, synergy_target, n_bins=8
                                    )
                                except Exception as e:
                                    logger.warning(f"Cross-entropy loss failed for synergy features, falling back to MSE: {e}")
                                    # Fallback to MSE if cross-entropy fails
                                    synergy_loss = criterion(synergy_recon, synergy_target)
                                
                                # Apply scaling to synergy component
                                scaled_synergy_loss = synergy_loss_scale * synergy_loss
                                domain_loss += scaled_synergy_loss
                                loss_details[f"fusion_{domain_name}_synergy_loss"] = synergy_loss.item()
                                loss_details[f"fusion_{domain_name}_synergy_loss_scaled"] = scaled_synergy_loss.item()
                            
                            # Convert to tensor if needed
                            if not isinstance(domain_loss, torch.Tensor):
                                domain_loss = torch.tensor(domain_loss, device=target.device)
                            
                            loss_details[f"fusion_{domain_name}_loss"] = domain_loss.item()
                            
                            if fusion_loss is None:
                                fusion_loss = domain_loss
                            else:
                                fusion_loss = fusion_loss + domain_loss
                            num_domains += 1
                    
                    # Average loss
                    if fusion_loss is not None and num_domains > 1:
                        fusion_loss = fusion_loss / num_domains
                    
                    # Synergy metrics already computed inline above
                    if fusion_loss is not None:
                        weighted_fusion_loss = loss_weights['fusion'] * fusion_loss
                        loss_details['weighted_fusion_loss'] = weighted_fusion_loss.item()
                        loss_details['fusion_loss'] = fusion_loss.item()
                        loss_details['synergy_loss_scale'] = synergy_loss_scale
                        total_loss = weighted_fusion_loss
            
            # 2. Demi-cycle and cycle losses - ignore synergy features entirely
            # Use original functions but only compare base dimensions for attr domain
            
            # Helper function for monkey-patching attr decoder
            def _monkey_patch_attr_decoder_for_base_only(model, synergy_config):
                """Temporarily replace attr decoder to only output base features"""
                original_attr_decoder = None
                if 'attr' in model.gw_decoders:
                    original_attr_decoder = model.gw_decoders['attr']
                    
                    # Create a proper Module wrapper
                    class AttrBaseDecoderWrapper(torch.nn.Module):
                        def __init__(self, original_decoder, synergy_config):
                            super().__init__()
                            self.original_decoder = original_decoder
                            self.synergy_config = synergy_config
                        
                        def forward(self, x):
                            full_output = self.original_decoder(x)
                            # Calculate base feature count from synergy config
                            feature_indices = self.synergy_config.get('feature_indices', {})
                            if 'attr' in feature_indices:
                                # Total features minus synergy features = base features
                                synergy_count = len(feature_indices['attr'])
                                base_count = full_output.shape[-1] - synergy_count
                            else:
                                base_count = 11  # Default for attr domain
                            return full_output[..., :base_count]
                    
                    model.gw_decoders['attr'] = AttrBaseDecoderWrapper(original_attr_decoder, synergy_config)
                
                return original_attr_decoder

            if loss_weights.get('demi_cycle', 0.0) > 0:
                from losses_and_weights_GLW_training import calculate_demi_cycle_loss
                original_attr_decoder = _monkey_patch_attr_decoder_for_base_only(model, synergy_config)
                
                try:
                    demi_loss, demi_details = calculate_demi_cycle_loss(model, batch, criterion)
                finally:
                    # Restore original decoder
                    if original_attr_decoder is not None:
                        model.gw_decoders['attr'] = original_attr_decoder
                
                loss_details.update(demi_details)
                
                weighted_demi_loss = loss_weights['demi_cycle'] * demi_loss
                loss_details['weighted_demi_cycle_loss'] = weighted_demi_loss.item()
                
                if total_loss is None:
                    total_loss = weighted_demi_loss
                else:
                    total_loss = total_loss + weighted_demi_loss
            
            if loss_weights.get('cycle', 0.0) > 0:
                from losses_and_weights_GLW_training import calculate_cycle_loss
                original_attr_decoder = _monkey_patch_attr_decoder_for_base_only(model, synergy_config)
                
                try:
                    cycle_loss, cycle_details = calculate_cycle_loss(model, batch, criterion)
                finally:
                    # Restore original decoder
                    if original_attr_decoder is not None:
                        model.gw_decoders['attr'] = original_attr_decoder
                
                loss_details.update(cycle_details)
                
                weighted_cycle_loss = loss_weights['cycle'] * cycle_loss
                loss_details['weighted_cycle_loss'] = weighted_cycle_loss.item()
                
                if total_loss is None:
                    total_loss = weighted_cycle_loss
                else:
                    total_loss = total_loss + weighted_cycle_loss
            
            # 3. Translation Loss (cross-modal supervised learning on paired data)
            # Only operates on base features - synergy features are masked out
            if loss_weights.get('translation', 0.0) > 0:
                from losses_and_weights_GLW_training import calculate_translation_loss, detect_paired_samples
                # Detect paired samples for translation loss
                paired_mask = detect_paired_samples(batch, synergy_targets=synergy_targets)
                # Use existing translation loss - it already handles synergy_config masking
                translation_loss, translation_details = calculate_translation_loss(
                    model, batch, criterion, synergy_targets=synergy_targets, 
                    paired_mask=paired_mask, synergy_config=synergy_config
                )
                loss_details.update(translation_details)
                
                weighted_translation_loss = loss_weights['translation'] * translation_loss
                loss_details['weighted_translation_loss'] = weighted_translation_loss.item()
                
                if total_loss is None:
                    total_loss = weighted_translation_loss
                else:
                    total_loss = total_loss + weighted_translation_loss
            
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