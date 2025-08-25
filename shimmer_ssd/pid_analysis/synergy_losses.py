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
import sys
from pathlib import Path

# Ensure we import from the local losses_and_weights_GLW_training.py
current_dir = Path(__file__).parent
# Remove any paths that might have the other losses_and_weights_GLW_training.py
parent_repo_path = str(current_dir.parent.parent.parent)
if parent_repo_path in sys.path:
    sys.path.remove(parent_repo_path)
# Add local directory with highest priority
sys.path.insert(0, str(current_dir))

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
        # For attr domain: base features are the original 11D processed attributes
        # Input and target both have these 11D features (uniform_random is the synergy source at index 10)
        base_dims = 11  # Original 11 processed attribute features
    else:
        # For other domains, use domain module latent_dim (you'd need to pass this)
        # For now, assume it's passed or use a default
        base_dims = tensor.shape[-1] - n_synergy_features  # fallback
    
    # Extract base features (always first base_dims)
    non_synergy_features = tensor[..., :base_dims]
    
    if is_model_output:
        # Model outputs: synergy features are logits (n_synergy_features × n_synergy_classes)
        synergy_logit_dims = n_synergy_features * n_synergy_classes
        # If attr synergy logits are disabled on the decoder, return empty synergy slice
        if domain == 'attr' and not synergy_config.get('attr_includes_synergy', True):
            synergy_features = torch.empty(tensor.shape[:-1] + (0,), device=tensor.device)
        else:
            if domain == 'attr':
                # For attr: synergy logits start after the 11 base features
                synergy_features = tensor[..., 11:11 + synergy_logit_dims]
            else:
                synergy_features = tensor[..., base_dims:base_dims + synergy_logit_dims]
    else:
        # Targets: synergy features are single values (n_synergy_features × 1)
        if domain == 'attr':
            # For attr: synergy targets start after the 11 base features
            synergy_features = tensor[..., 11:11 + n_synergy_features]
        else:
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
        # For attr domain: base features are the original 11D processed attributes
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
            
            # Weight scheduling for flexible curricula
            schedule_cfg = synergy_config.get('schedule', {})
            use_schedule = bool(schedule_cfg)
            frac = None
            if use_schedule:
                # Prefer epoch-based scheduling if available
                if hasattr(model, 'current_epoch') and hasattr(model, 'total_epochs') and getattr(model, 'total_epochs', 0):
                    try:
                        total_epochs = max(1, int(getattr(model, 'total_epochs', 1)))
                        cur_epoch = float(getattr(model, 'current_epoch', 0))
                        frac = min(1.0, max(0.0, cur_epoch / float(total_epochs)))
                    except Exception:
                        frac = None
                # Fallback to step-based scheduling if provided
                if frac is None and hasattr(model, 'global_step'):
                    total = max(1, int(schedule_cfg.get('total_steps', 1)))
                    frac = min(1.0, float(getattr(model, 'global_step', 0)) / float(total))
            # Helper: blend two weights w = (1-t)*w_a + t*w_b
            def schedule_weight(name: str, base_value: float) -> float:
                if not use_schedule or frac is None:
                    return base_value
                sched = schedule_cfg.get(name, None)
                if isinstance(sched, dict):
                    start = float(sched.get('start', base_value))
                    end = float(sched.get('end', base_value))
                    t0 = float(sched.get('t0', 0.0))
                    t1 = float(sched.get('t1', 1.0))
                    t = 0.0
                    if frac <= t0:
                        t = 0.0
                    elif frac >= t1:
                        t = 1.0
                    else:
                        t = (frac - t0) / max(1e-8, (t1 - t0))
                    return (1.0 - t) * start + t * end
                return base_value

            fusion_w = schedule_weight('fusion', loss_weights.get('fusion', 0.0))
            demi_w = schedule_weight('demi_cycle', loss_weights.get('demi_cycle', 0.0))
            cycle_w = schedule_weight('cycle', loss_weights.get('cycle', 0.0))
            transl_w = schedule_weight('translation', loss_weights.get('translation', 0.0))

            # 1. Fusion Loss with synergy training
            if fusion_w > 0:
                # Encode inputs
                encoded = {}
                for domain_name, domain_input in processed_inputs.items():
                    if domain_name in model.gw_encoders:
                        # For attribute domain: skip domain module, use preprocessed data directly
                        if domain_name == 'attr':
                            encoded[domain_name] = model.gw_encoders[domain_name](domain_input)
                        elif domain_name == 'v':
                            # domain_input is already VAE latents (12D), bypass domain module
                            encoded[domain_name] = model.gw_encoders[domain_name](domain_input)
                        else:
                            # For other domains: use domain module first, then GW encoder if available
                            if hasattr(model, 'domain_mods') and (domain_name in model.domain_mods):
                                domain_latent = model.domain_mods[domain_name](domain_input)
                                encoded[domain_name] = model.gw_encoders[domain_name](domain_latent)
                            else:
                                encoded[domain_name] = model.gw_encoders[domain_name](domain_input)
                
                if encoded:
                    # Fuse
                    gw_state = model.fuse(encoded, selection_scores={})
                    # Optional post-tanh + noise injection
                    try:
                        noise_cfg = synergy_config.get('noise', {})
                        site = noise_cfg.get('site', 'post_fusion_post_tanh')
                        if 'post_tanh' in str(site):
                            gw_state = torch.tanh(gw_state)
                        std = float(noise_cfg.get('train_std', 0.0) if model.training else noise_cfg.get('eval_std', noise_cfg.get('train_std', 0.0)))
                        if std and std > 0.0:
                            gw_state = gw_state + std * torch.randn_like(gw_state)
                            if _debug_state['synergy_loss_calls'] <= _DEBUG_MAX_CALLS:
                                logger.info(f"  noise injected at site={site} std={std}")
                    except Exception as e:
                        logger.warning(f"Noise injection skipped due to error: {e}")
                    # Decode
                    if _debug_state['synergy_loss_calls'] <= _DEBUG_MAX_CALLS and hasattr(gw_state, 'shape'):
                        logger.info(f"  gw_state: shape={tuple(gw_state.shape)}")
                    decoded = {}
                    for domain_name in synergy_targets.keys():
                        if domain_name in model.gw_decoders:
                            # Decode from workspace to domain latent space
                            decoded_latent = model.gw_decoders[domain_name](gw_state)
                            
                            # For domains without domain modules: decoded output is already in target space
                            if domain_name in ('attr', 'v'):
                                decoded[domain_name] = decoded_latent
                            else:
                                # For other domains: use domain module decoder if available
                                if (hasattr(model, 'domain_mods') and (domain_name in model.domain_mods) and 
                                    hasattr(model.domain_mods[domain_name], 'decode')):
                                    decoded[domain_name] = model.domain_mods[domain_name].decode(decoded_latent)
                                else:
                                    decoded[domain_name] = decoded_latent
                    # Optionally decode synergy head (direct path)
                    if synergy_config.get('enable_syn_head', False) and 'syn' in getattr(model, 'gw_decoders', {}):
                        try:
                            decoded['syn'] = model.gw_decoders['syn'](gw_state)
                        except Exception as e:
                            logger.warning(f"Failed to decode 'syn' head: {e}")
                    
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
                                
                                # For attr domain: track reconstruction of synergy source feature separately
                                if domain_name == 'attr':
                                    # Split into features 0-9 (non-synergy) and feature 10 (uniform_random = synergy source)
                                    non_synergy_features_only = non_synergy_recon[:, :10]  # features 0-9
                                    synergy_source_feature = non_synergy_recon[:, 10:11]   # feature 10 (uniform_random)
                                    
                                    non_synergy_targets_only = non_synergy_target[:, :10]  # features 0-9
                                    synergy_source_target = non_synergy_target[:, 10:11]   # feature 10 (uniform_random)
                                    
                                    # Calculate MSE for each group
                                    non_synergy_mse = criterion(non_synergy_features_only, non_synergy_targets_only)
                                    synergy_source_mse = criterion(synergy_source_feature, synergy_source_target)
                                    
                                    loss_details[f"fusion_{domain_name}_non_synergy_features_mse"] = non_synergy_mse.item()
                                    loss_details[f"fusion_{domain_name}_synergy_source_feature_mse"] = synergy_source_mse.item()
                            
                            # Only compute synergy CE here if this domain outputs synergy logits
                            if synergy_target.numel() > 0 and synergy_recon.numel() > 0:
                                # Use classification-appropriate loss for discrete synergy features
                                try:
                                    n_bins = int(synergy_config.get('n_bins', 8))
                                    synergy_loss = calculate_synergy_loss_with_crossentropy(
                                        synergy_recon, synergy_target, n_bins=n_bins
                                    )
                                except Exception as e:
                                    logger.warning(f"Cross-entropy loss failed for synergy features, falling back to MSE: {e}")
                                    # Fallback to MSE if cross-entropy fails - reshape synergy logits to single values
                                    # Take mean across logit dimensions as fallback
                                    if synergy_recon.shape[-1] == int(synergy_config.get('n_bins', 8)):
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

                    # Add external 'syn' head CE loss using attr target as label source (mixture of direct and broadcast-cycle paths)
                    if synergy_config.get('enable_syn_head', False) and 'attr' in synergy_targets:
                        try:
                            attr_target_full = synergy_targets['attr']
                            n_bins = int(synergy_config.get('n_bins', 8))
                            # Build ground-truth target (prefer one-hot tail if present)
                            if attr_target_full.shape[-1] > 11:
                                syn_target_raw = attr_target_full[..., 11:]
                            else:
                                syn_target_raw, _ = extract_synergy_features(
                                    attr_target_full, 'attr', synergy_config, is_model_output=False, n_synergy_classes=n_bins
                                )
                            # Helper to compute CE from logits vs target (one-hot or scalar)
                            def ce_from_logits(logits: torch.Tensor) -> torch.Tensor:
                                if syn_target_raw.shape[-1] == n_bins:
                                    target_classes = syn_target_raw.argmax(dim=-1).view(-1)
                                    logits_reshaped = logits.view(-1, n_bins)
                                    return torch.nn.functional.cross_entropy(logits_reshaped, target_classes)
                                else:
                                    return calculate_synergy_loss_with_crossentropy(logits, syn_target_raw, n_bins=n_bins)

                            # Direct path CE
                            syn_ce_direct = None
                            if 'syn' in decoded:
                                syn_ce_direct = ce_from_logits(decoded['syn'])

                            # Broadcast-cycle path: decode base domains -> re-encode -> re-fuse -> syn decode
                            syn_ce_cycle = None
                            try:
                                # Decode base domains from the same gw_state (NO additional noise here)
                                re_latents = {}
                                if 'attr' in model.gw_decoders and 'attr' in model.gw_encoders:
                                    attr_base = model.gw_decoders['attr'](gw_state)
                                    re_latents['attr'] = model.gw_encoders['attr'](attr_base)
                                if 'v' in model.gw_decoders and 'v' in model.gw_encoders:
                                    v_base = model.gw_decoders['v'](gw_state)
                                    re_latents['v'] = model.gw_encoders['v'](v_base)
                                if re_latents:
                                    gw_state_refused = model.fuse(re_latents, selection_scores={})
                                    syn_logits_cycle = model.gw_decoders['syn'](gw_state_refused)
                                    syn_ce_cycle = ce_from_logits(syn_logits_cycle)
                            except Exception as e:
                                logger.warning(f"Failed syn-cycle path: {e}")

                            # Mix according to syn_cycle_ratio
                            ratio = float(synergy_config.get('syn_cycle_ratio', 0.5))
                            ratio = max(0.0, min(1.0, ratio))
                            path_terms = []
                            if syn_ce_direct is not None and (1.0 - ratio) > 0.0:
                                path_terms.append((1.0 - ratio) * syn_ce_direct)
                            if syn_ce_cycle is not None and ratio > 0.0:
                                path_terms.append(ratio * syn_ce_cycle)
                            if path_terms:
                                syn_ce_mixed = sum(path_terms)
                                scaled_syn_ce = synergy_loss_scale * syn_ce_mixed
                                if fusion_loss is None:
                                    fusion_loss = scaled_syn_ce
                                else:
                                    fusion_loss = fusion_loss + scaled_syn_ce
                                num_domains = max(num_domains, 1)
                                # Logging
                                if syn_ce_direct is not None:
                                    loss_details["fusion_syn_direct_ce"] = syn_ce_direct.item()
                                if syn_ce_cycle is not None:
                                    loss_details["fusion_syn_cycle_ce"] = syn_ce_cycle.item()
                                loss_details["fusion_syn_mixed_ce_scaled"] = scaled_syn_ce.item()
                                loss_details["fusion_syn_cycle_ratio"] = ratio
                        except Exception as e:
                            logger.warning(f"Failed to compute 'syn' head mixed CE loss: {e}")
                    
                    # Average loss
                    if fusion_loss is not None and num_domains > 1:
                        fusion_loss = fusion_loss / num_domains
                    
                    # Synergy metrics already computed inline above
                    if fusion_loss is not None:
                        weighted_fusion_loss = fusion_w * fusion_loss
                        loss_details['weighted_fusion_loss'] = weighted_fusion_loss.item()
                        loss_details['fusion_loss'] = fusion_loss.item()
                        loss_details['synergy_loss_scale'] = synergy_loss_scale
                        total_loss = weighted_fusion_loss
            
            # Helper to temporarily monkey-patch model.fuse to inject noise post-fusion (single injection only)
            def _monkey_patch_fuse_with_noise(model, synergy_config):
                original_fuse = getattr(model, 'fuse', None)
                if original_fuse is None:
                    return None
                noise_cfg = synergy_config.get('noise', {})
                site = noise_cfg.get('site', 'post_fusion_post_tanh')
                train_std = float(noise_cfg.get('train_std', 0.0))

                def noisy_fuse(encoded, selection_scores=None):
                    gw_state_inner = original_fuse(encoded, selection_scores)
                    if 'post_tanh' in str(site):
                        gw_state_inner = torch.tanh(gw_state_inner)
                    if train_std and train_std > 0.0:
                        gw_state_inner = gw_state_inner + train_std * torch.randn_like(gw_state_inner)
                    return gw_state_inner

                model.fuse = noisy_fuse
                return original_fuse

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

            if demi_w > 0:
                demi_style = str(synergy_config.get('demi_cycle_style', 'encode_decode'))
                # Allow auto-switch: encode_decode before schedule.t0, decode_encode after
                if demi_style == 'auto':
                    try:
                        t0 = float(schedule_cfg.get('demi_cycle', {}).get('t0', 0.0)) if use_schedule else 0.5
                        if frac is not None and frac >= t0:
                            demi_style = 'decode_encode'
                        else:
                            demi_style = 'encode_decode'
                    except Exception:
                        demi_style = 'encode_decode'
                if demi_style == 'encode_decode':
                    from losses_and_weights_GLW_training import calculate_demi_cycle_loss
                    original_attr_decoder = _monkey_patch_attr_decoder_for_base_only(model, synergy_config)
                    original_fuse = _monkey_patch_fuse_with_noise(model, synergy_config)
                    try:
                        demi_loss, demi_details = calculate_demi_cycle_loss(model, batch, criterion)
                    finally:
                        if original_attr_decoder is not None:
                            model.gw_decoders['attr'] = original_attr_decoder
                        if original_fuse is not None:
                            model.fuse = original_fuse
                else:
                    # decode_encode denoising: compare clean fused state vs cycle after decoding/encoding with noise injected once at clean fused state
                    # 1) Build clean fused state (no noise)
                    processed_inputs = {}
                    for domain_name, domain_input in batch.items():
                        if domain_name == 't' and hasattr(model, 'domain_mods') and hasattr(model.domain_mods[domain_name], 'projector'):
                            processed_inputs[domain_name] = model.domain_mods[domain_name].projector(domain_input)
                        else:
                            processed_inputs[domain_name] = domain_input
                    encoded_clean = {}
                    for domain_name, domain_input in processed_inputs.items():
                        if domain_name in model.gw_encoders:
                            if domain_name in ('attr', 'v'):
                                encoded_clean[domain_name] = model.gw_encoders[domain_name](domain_input)
                            elif domain_name in model.domain_mods:
                                encoded_clean[domain_name] = model.gw_encoders[domain_name](model.domain_mods[domain_name](domain_input))
                    if encoded_clean:
                        gw_clean = model.fuse(encoded_clean, selection_scores={})
                        # Detach target to avoid gradients through target branch
                        gw_target = gw_clean.detach()
                        # 2) Inject training noise once to create noisy fused state
                        noise_cfg = synergy_config.get('noise', {})
                        site = noise_cfg.get('site', 'post_fusion_post_tanh')
                        if 'post_tanh' in str(site):
                            gw_noisy = torch.tanh(gw_clean)
                        else:
                            gw_noisy = gw_clean
                        std = float(noise_cfg.get('train_std', 0.0))
                        if std and std > 0.0:
                            gw_noisy = gw_noisy + std * torch.randn_like(gw_noisy)
                        # 3) Decode to base domains, re-encode, and re-fuse
                        re_latents = {}
                        if 'attr' in model.gw_decoders and 'attr' in model.gw_encoders:
                            attr_base = model.gw_decoders['attr'](gw_noisy)
                            re_latents['attr'] = model.gw_encoders['attr'](attr_base)
                        if 'v' in model.gw_decoders and 'v' in model.gw_encoders:
                            v_base = model.gw_decoders['v'](gw_noisy)
                            re_latents['v'] = model.gw_encoders['v'](v_base)
                        if re_latents:
                            gw_refused = model.fuse(re_latents, selection_scores={})
                            # 4) MSE between gw_refused and gw_target (denoising objective)
                            demi_loss = criterion(gw_refused, gw_target)
                            demi_details = {"demi_cycle_decode_encode_denoise": demi_loss.item()}
                        else:
                            demi_loss = torch.tensor(0.0, device=gw_clean.device)
                            demi_details = {"demi_cycle_decode_encode_denoise": 0.0}
                 
                loss_details.update(demi_details)
                 
                weighted_demi_loss = demi_w * demi_loss
                loss_details['weighted_demi_cycle_loss'] = weighted_demi_loss.item()
                 
                if total_loss is None:
                     total_loss = weighted_demi_loss
                else:
                     total_loss = total_loss + weighted_demi_loss
            
            if cycle_w > 0:
                from losses_and_weights_GLW_training import calculate_cycle_loss
                original_attr_decoder = _monkey_patch_attr_decoder_for_base_only(model, synergy_config)
                original_fuse = _monkey_patch_fuse_with_noise(model, synergy_config)
                
                try:
                    cycle_loss, cycle_details = calculate_cycle_loss(model, batch, criterion)
                finally:
                    # Restore original decoder
                    if original_attr_decoder is not None:
                        model.gw_decoders['attr'] = original_attr_decoder
                    # Restore original fuse
                    if original_fuse is not None:
                        model.fuse = original_fuse
                
                loss_details.update(cycle_details)
                
                weighted_cycle_loss = cycle_w * cycle_loss
                loss_details['weighted_cycle_loss'] = weighted_cycle_loss.item()
                
                if total_loss is None:
                    total_loss = weighted_cycle_loss
                else:
                    total_loss = total_loss + weighted_cycle_loss
            
            # 3. Translation Loss (cross-modal supervised learning on paired data)
            # Only operates on base features - synergy features are masked out
            if transl_w > 0:
                from losses_and_weights_GLW_training import calculate_translation_loss, detect_paired_samples
                # Detect paired samples for translation loss
                paired_mask = detect_paired_samples(batch, synergy_targets=synergy_targets)
                # Inject noise in fuse for translation computations
                original_fuse = _monkey_patch_fuse_with_noise(model, synergy_config)
                try:
                    # Use existing translation loss - it already handles synergy_config masking
                    translation_loss, translation_details = calculate_translation_loss(
                        model, batch, criterion, synergy_targets=synergy_targets, 
                        paired_mask=paired_mask, synergy_config=synergy_config
                    )
                finally:
                    if original_fuse is not None:
                        model.fuse = original_fuse
                loss_details.update(translation_details)
                
                weighted_translation_loss = transl_w * translation_loss
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