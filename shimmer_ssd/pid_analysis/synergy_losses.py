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

# Debug logging controls to reduce noise
_DEBUG_MAX_CALLS = 2
_debug_state = {'synergy_loss_calls': 0}


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
    synergy_logits: torch.Tensor, 
    synergy_target: torch.Tensor,
    n_bins: int = 8
) -> torch.Tensor:
    """
    Calculate synergy loss using direct classification with cross-entropy.
    
    Direct classification approach:
    p_i = softmax(W*h + b)  # synergy_logits are already W*h + b
    L = -∑ log(p_{i,t_i})   # standard cross-entropy
    
    Args:
        synergy_logits: Model logits for synergy features [..., n_bins] or [..., n_features*n_bins]
        synergy_target: Target synergy values (normalized) [..., n_features]
        n_bins: Number of discrete synergy bins (default: 8)
        
    Returns:
        Cross-entropy classification loss for synergy features
    """
    # Handle empty tensors
    if synergy_logits.numel() == 0 or synergy_target.numel() == 0:
        return torch.tensor(0.0, device=synergy_logits.device, requires_grad=True)
    
    # Convert normalized targets to class indices
    # synergy_target should be shape [..., n_features] with values in [0,1]
    target_flat = synergy_target.view(-1)  # Flatten all dimensions
    target_classes = convert_synergy_targets_to_classes(target_flat, n_bins)
    
    # Reshape logits for cross-entropy
    # synergy_logits should be [..., n_features*n_bins]
    batch_dims = synergy_logits.shape[:-1]  # All dimensions except last
    n_features = synergy_target.shape[-1]
    
    # Reshape logits to [batch_size*n_features, n_bins]
    logits_reshaped = synergy_logits.view(-1, n_bins)
    
    # Ensure we have the right number of targets
    if target_classes.shape[0] != logits_reshaped.shape[0]:
        raise ValueError(f"Mismatch: {target_classes.shape[0]} targets vs {logits_reshaped.shape[0]} logit groups")
    
    # Apply standard cross-entropy loss
    # F.cross_entropy expects: input=[N, C], target=[N] with class indices
    ce_loss = F.cross_entropy(logits_reshaped, target_classes)
    
    return ce_loss





def extract_synergy_features(
    tensor: torch.Tensor, 
    domain: str,
    synergy_config: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
    is_model_output: bool = True,
    n_synergy_classes: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract synergistic and non-synergistic features from tensor.
    
    Args:
        tensor: Input tensor (..., feature_dim)
        domain: Domain name
        synergy_config: Synergy configuration
        feature_names: Optional feature names for string resolution
        is_model_output: True for model outputs (with logits), False for targets (with single values)
        n_synergy_classes: Number of classes per synergy feature (default: 8)
        
    Returns:
        Tuple of (synergy_features, non_synergy_features)
    """
    feature_indices = synergy_config.get('feature_indices', {})
    
    if domain not in feature_indices:
        # No synergy features for this domain
        return torch.empty(tensor.shape[:-1] + (0,), device=tensor.device), tensor
    
    synergy_specs = feature_indices[domain]
    n_synergy_features = len(synergy_specs)
    
    if n_synergy_features == 0:
        return torch.empty(tensor.shape[:-1] + (0,), device=tensor.device), tensor
    
    # Determine base dimensions based on domain
    if domain == 'attr':
        base_dims = 11
    else:
        # For other domains, use domain module latent_dim (you'd need to pass this)
        # For now, assume it's passed or use a default
        base_dims = tensor.shape[-1] - n_synergy_features  # fallback
    
    # Extract base features (always first base_dims)
    non_synergy_features = tensor[..., :base_dims]
    
    if is_model_output:
        # Model outputs: synergy features are logits (n_synergy_features × n_synergy_classes)
        synergy_logit_dims = n_synergy_features * n_synergy_classes
        synergy_features = tensor[..., base_dims:base_dims + synergy_logit_dims]
    else:
        # Targets: synergy features are single values (n_synergy_features × 1)
        synergy_features = tensor[..., base_dims:base_dims + n_synergy_features]
    
    return synergy_features, non_synergy_features


def extract_base_features_only(
    tensor: torch.Tensor,
    domain: str,
    synergy_config: Dict[str, Any]
) -> torch.Tensor:
    """
    Extract only the base (non-synergy) features from tensor.
    Used for demi-cycle, cycle, and translation losses which ignore synergy features completely.
    
    Args:
        tensor: Input tensor (..., feature_dim) 
        domain: Domain name
        synergy_config: Synergy configuration
        
    Returns:
        Base features tensor with synergy dimensions removed
    """
    # Determine base dimensions based on domain
    if domain == 'attr':
        base_dims = 11
    else:
        # For other domains, if they have synergy features, subtract them
        feature_indices = synergy_config.get('feature_indices', {})
        if domain in feature_indices:
            n_synergy_features = len(feature_indices[domain])
            # For model outputs, synergy takes n_features * 8 dims
            # For targets, synergy takes n_features dims
            # We'll assume the smaller case for safety and let the loss handle dimension checks
            base_dims = tensor.shape[-1] - n_synergy_features
        else:
            base_dims = tensor.shape[-1]  # No synergy features
    
    return tensor[..., :base_dims]


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
            target, domain_name, synergy_config, is_model_output=False
        )
        synergy_recon, non_synergy_recon = extract_synergy_features(
            reconstruction, domain_name, synergy_config, is_model_output=True
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
                            # Extract logits for feature i (8 consecutive dimensions)
                            start_idx = i * 8
                            end_idx = (i + 1) * 8
                            feat_logits = synergy_recon[..., start_idx:end_idx]
                            feat_target = synergy_target[..., i:i+1]
                            
                            feat_loss = calculate_synergy_loss_with_crossentropy(
                                feat_logits, feat_target, n_bins=8
                            )
                            loss_details[f"{loss_prefix}_{domain_name}_synergy_feat_{i}_loss"] = feat_loss.item()
                        except Exception as e:
                            logger.warning(f"Cross-entropy failed for synergy feature {i}, using MSE: {e}")
                            # Fallback with proper reshaping
                            start_idx = i * 8
                            end_idx = (i + 1) * 8
                            feat_logits = synergy_recon[..., start_idx:end_idx]
                            feat_target = synergy_target[..., i:i+1]
                            
                            feat_logits_fallback = torch.softmax(feat_logits, dim=-1).mean(dim=-1, keepdim=True)
                            feat_loss = criterion(feat_logits_fallback, feat_target)
                            loss_details[f"{loss_prefix}_{domain_name}_synergy_feat_{i}_loss"] = feat_loss.item()
                            
            except Exception as e:
                logger.warning(f"Cross-entropy loss failed for synergy features in metrics, falling back to MSE: {e}")
                # Fallback for overall synergy loss
                if synergy_recon.shape[-1] % 8 == 0:
                    synergy_recon_fallback = torch.softmax(synergy_recon.view(*synergy_recon.shape[:-1], -1, 8), dim=-1)
                    synergy_recon_fallback = synergy_recon_fallback.mean(dim=-1)
                else:
                    synergy_recon_fallback = synergy_recon
                synergy_loss = criterion(synergy_recon_fallback, synergy_target)
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

        # Optional debug: log input/target keys and tensor shapes for first few calls
        _debug_state['synergy_loss_calls'] += 1
        if _debug_state['synergy_loss_calls'] <= _DEBUG_MAX_CALLS:
            try:
                input_keys = list(clean_batch.keys())
                target_keys = list(synergy_targets.keys())
                logger.info(
                    f"[SynergyLoss] call={_debug_state['synergy_loss_calls']} "
                    f"input_keys={input_keys} target_keys={target_keys} device={device}"
                )
                for d, t in clean_batch.items():
                    if hasattr(t, 'shape'):
                        logger.info(
                            f"  input[{d}]: shape={tuple(t.shape)} dtype={getattr(t,'dtype',None)} dev={getattr(t,'device',None)}"
                        )
                for d, t in synergy_targets.items():
                    if hasattr(t, 'shape'):
                        try:
                            tmin = t.min().item() if t.numel() else 'NA'
                            tmax = t.max().item() if t.numel() else 'NA'
                        except Exception:
                            tmin, tmax = 'NA', 'NA'
                        logger.info(
                            f"  target[{d}]: shape={tuple(t.shape)} dtype={getattr(t,'dtype',None)} dev={getattr(t,'device',None)} min={tmin} max={tmax}"
                        )
            except Exception as e:
                logger.warning(f"[SynergyLoss] debug batch summary failed: {e}")
        
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

            if _debug_state['synergy_loss_calls'] <= _DEBUG_MAX_CALLS:
                for d, tin in processed_inputs.items():
                    if hasattr(tin, 'shape'):
                        logger.info(f"  processed_input[{d}]: shape={tuple(tin.shape)}")
            
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
                    if _debug_state['synergy_loss_calls'] <= _DEBUG_MAX_CALLS and hasattr(gw_state, 'shape'):
                        logger.info(f"  gw_state: shape={tuple(gw_state.shape)}")
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
                    if _debug_state['synergy_loss_calls'] <= _DEBUG_MAX_CALLS:
                        logger.info(f"  synergy_loss_scale={synergy_loss_scale}")
                    
                    for domain_name, target in synergy_targets.items():
                        if domain_name in decoded:
                            # Extract synergy and non-synergy components for scaled loss
                            synergy_target, non_synergy_target = extract_synergy_features(
                                target, domain_name, synergy_config, is_model_output=False
                            )
                            synergy_recon, non_synergy_recon = extract_synergy_features(
                                decoded[domain_name], domain_name, synergy_config, is_model_output=True
                            )
                            if _debug_state['synergy_loss_calls'] <= _DEBUG_MAX_CALLS:
                                logger.info(
                                    f"  domain={domain_name} non_synergy_target.shape={tuple(non_synergy_target.shape)} non_synergy_recon.shape={tuple(non_synergy_recon.shape)}"
                                )
                                logger.info(
                                    f"  domain={domain_name} synergy_target.shape={tuple(synergy_target.shape)} synergy_recon.shape={tuple(synergy_recon.shape)}"
                                )
                                try:
                                    if synergy_target.numel() > 0:
                                        classes = convert_synergy_targets_to_classes(synergy_target.view(-1), n_bins=8)
                                        bincount = torch.bincount(classes, minlength=8).tolist()
                                        logger.info(f"    synergy_target class_counts={bincount}")
                                except Exception as e:
                                    logger.warning(f"    synergy target stats failed: {e}")
                            
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
                                    # Fallback to MSE if cross-entropy fails - reshape synergy logits to single values
                                    # Take mean across logit dimensions as fallback
                                    if synergy_recon.shape[-1] == 8:
                                        synergy_recon_fallback = torch.softmax(synergy_recon, dim=-1).mean(dim=-1, keepdim=True)
                                    else:
                                        synergy_recon_fallback = synergy_recon
                                    synergy_loss = criterion(synergy_recon_fallback, synergy_target)
                                
                                # Apply scaling to synergy component
                                scaled_synergy_loss = synergy_loss_scale * synergy_loss
                                domain_loss += scaled_synergy_loss
                                loss_details[f"fusion_{domain_name}_synergy_loss"] = synergy_loss.item()
                                loss_details[f"fusion_{domain_name}_synergy_loss_scaled"] = scaled_synergy_loss.item()
                                if _debug_state['synergy_loss_calls'] <= _DEBUG_MAX_CALLS:
                                    logger.info(
                                        f"    synergy_loss={synergy_loss.item():.6f} scaled={scaled_synergy_loss.item():.6f}"
                                    )
                            
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
                            # Extract only base features for cycle/demi-cycle losses
                            return extract_base_features_only(full_output, 'attr', self.synergy_config)
                    
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
            if _debug_state['synergy_loss_calls'] <= _DEBUG_MAX_CALLS:
                logger.warning("[SynergyLoss] synergy_targets empty, falling back to standard loss")
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