#!/usr/bin/env python3
"""
Synergy-Aware Loss Functions for Global Workspace Training

Extends the existing loss framework to handle synergistic targets:
- Computes standard reconstruction losses on full vectors
- Extracts and monitors synergy-specific losses separately
- Maintains compatibility with fusion/demi-cycle/cycle loss patterns
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SynergyLossCalculator:
    """
    Calculator for synergy-aware losses in Global Workspace training.
    
    Handles separation of synergistic vs non-synergistic features
    while maintaining compatibility with existing loss functions.
    """
    
    def __init__(
        self,
        synergy_config: Dict[str, Any],
        criterion: Optional[nn.Module] = None,
        synergy_weight: float = 1.0,
        log_individual_features: bool = True
    ):
        """
        Initialize synergy loss calculator.
        
        Args:
            synergy_config: Configuration specifying synergistic features per domain
            criterion: Loss function (defaults to MSE)
            synergy_weight: Weight for synergy loss in total loss
            log_individual_features: Whether to log losses for individual synergy features
        """
        self.synergy_config = synergy_config
        self.criterion = criterion or nn.MSELoss()
        self.synergy_weight = synergy_weight
        self.log_individual_features = log_individual_features
        
        # Pre-compute feature indices for efficiency
        self._prepare_feature_indices()
        
        logger.info(f"Initialized SynergyLossCalculator with weight {synergy_weight}")
        logger.info(f"Synergy domains: {list(self.synergy_config.get('domains', []))}")
    
    def _prepare_feature_indices(self):
        """Pre-compute feature indices for each domain."""
        self.synergy_indices = {}
        self.non_synergy_indices = {}
        
        domains = self.synergy_config.get('domains', [])
        feature_indices = self.synergy_config.get('feature_indices', {})
        
        for domain in domains:
            if domain in feature_indices:
                synergy_feats = feature_indices[domain]
                # Convert string indices to numeric if needed
                # (This will be resolved at runtime when we know the actual tensor shapes)
                self.synergy_indices[domain] = synergy_feats
            else:
                self.synergy_indices[domain] = []
    
    def _resolve_feature_indices(
        self, 
        domain: str, 
        tensor: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Resolve string feature indices to numeric indices.
        
        Args:
            domain: Domain name
            tensor: Target tensor to determine shape
            feature_names: Optional list of feature names for string resolution
            
        Returns:
            Tuple of (synergy_indices, non_synergy_indices)
        """
        if domain not in self.synergy_indices:
            # No synergy features for this domain
            return [], list(range(tensor.shape[-1]))
        
        synergy_specs = self.synergy_indices[domain]
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
        
        # Non-synergy indices are all others
        total_features = tensor.shape[-1]
        non_synergy_idx = [i for i in range(total_features) if i not in synergy_idx]
        
        return synergy_idx, non_synergy_idx
    
    def extract_synergy_features(
        self, 
        tensor: torch.Tensor, 
        domain: str,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract synergistic and non-synergistic features from tensor.
        
        Args:
            tensor: Input tensor (..., feature_dim)
            domain: Domain name
            feature_names: Optional feature names for string resolution
            
        Returns:
            Tuple of (synergy_features, non_synergy_features)
        """
        synergy_idx, non_synergy_idx = self._resolve_feature_indices(
            domain, tensor, feature_names
        )
        
        if not synergy_idx:
            # No synergy features
            return torch.empty(tensor.shape[:-1] + (0,), device=tensor.device), tensor
        
        synergy_features = tensor[..., synergy_idx]
        
        if not non_synergy_idx:
            # All features are synergistic
            non_synergy_features = torch.empty(tensor.shape[:-1] + (0,), device=tensor.device)
        else:
            non_synergy_features = tensor[..., non_synergy_idx]
        
        return synergy_features, non_synergy_features
    
    def calculate_synergy_fusion_loss(
        self, 
        model,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        use_weights_for_loss: bool = False,
        feature_names: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate fusion loss with synergy awareness.
        
        Args:
            model: The GW model
            inputs: Input tensors per domain
            targets: Target tensors per domain
            use_weights_for_loss: Whether to weight loss by fusion weights
            feature_names: Optional feature names per domain
            
        Returns:
            Tuple of (total_loss, loss_details)
        """
        loss_details = {}
        
        # Forward pass: encode inputs
        encoded = {}
        for domain_name, domain_input in inputs.items():
            if domain_name in model.gw_encoders:
                encoded[domain_name] = model.gw_encoders[domain_name](domain_input)
        
        if not encoded:
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device), loss_details
        
        # Fusion step
        gw_state = model.fuse(encoded, selection_scores={})
        
        # Decode to get reconstructions
        decoded = {}
        for domain_name in targets.keys():
            if domain_name in model.gw_decoders:
                decoded[domain_name] = model.gw_decoders[domain_name](gw_state)
        
        # Calculate losses
        total_loss = None
        num_domains = 0
        
        for domain_name, target in targets.items():
            if domain_name not in decoded:
                continue
                
            reconstruction = decoded[domain_name]
            domain_feature_names = feature_names.get(domain_name) if feature_names else None
            
            # Full reconstruction loss
            full_loss = self.criterion(reconstruction, target)
            loss_details[f"fusion_{domain_name}_full_loss"] = full_loss.item()
            
            # Extract synergy-specific loss
            synergy_target, non_synergy_target = self.extract_synergy_features(
                target, domain_name, domain_feature_names
            )
            synergy_recon, non_synergy_recon = self.extract_synergy_features(
                reconstruction, domain_name, domain_feature_names
            )
            
            # Calculate component losses
            if synergy_target.numel() > 0:
                synergy_loss = self.criterion(synergy_recon, synergy_target)
                loss_details[f"fusion_{domain_name}_synergy_loss"] = synergy_loss.item()
                
                if self.log_individual_features and synergy_target.shape[-1] > 1:
                    # Log individual synergy feature losses
                    for i in range(synergy_target.shape[-1]):
                        feat_loss = self.criterion(
                            synergy_recon[..., i:i+1], 
                            synergy_target[..., i:i+1]
                        )
                        loss_details[f"fusion_{domain_name}_synergy_feat_{i}_loss"] = feat_loss.item()
            
            if non_synergy_target.numel() > 0:
                non_synergy_loss = self.criterion(non_synergy_recon, non_synergy_target)
                loss_details[f"fusion_{domain_name}_non_synergy_loss"] = non_synergy_loss.item()
            
            # Weight the loss
            if use_weights_for_loss:
                weight = model.fusion_weights.get(domain_name, 1.0)
                weighted_loss = weight * full_loss
            else:
                weighted_loss = full_loss
                num_domains += 1
            
            # Accumulate total loss
            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss
        
        # Average if not using weights
        if not use_weights_for_loss and total_loss is not None and num_domains > 1:
            total_loss = total_loss / num_domains
        
        # Handle case where no domains were processed
        if total_loss is None:
            device = next(model.parameters()).device
            total_loss = torch.tensor(0.0, device=device)
        
        loss_details['fusion_loss'] = total_loss.item()
        return total_loss, loss_details
    
    def calculate_synergy_demi_cycle_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        feature_names: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate demi-cycle loss with synergy awareness.
        
        Args:
            model: The GW model
            inputs: Input tensors per domain
            targets: Target tensors per domain
            feature_names: Optional feature names per domain
            
        Returns:
            Tuple of (total_loss, loss_details)
        """
        loss_details = {}
        total_loss = None
        num_domains = 0
        
        # Store original fusion weights
        original_weights = model.fusion_weights.copy()
        
        try:
            for domain_name, domain_input in inputs.items():
                if domain_name not in model.gw_encoders or domain_name not in targets:
                    continue
                
                # Set domain-specific weights for demi-cycle
                demi_cycle_weights = {d: 0.0 for d in model.fusion_weights.keys()}
                demi_cycle_weights[domain_name] = 1.0
                model.fusion_weights.update(demi_cycle_weights)
                
                # Forward pass for this domain only
                encoded = model.gw_encoders[domain_name](domain_input)
                gw_state = model.fuse({domain_name: encoded}, selection_scores={})
                decoded = model.gw_decoders[domain_name](gw_state)
                
                # Get target
                target = targets[domain_name]
                domain_feature_names = feature_names.get(domain_name) if feature_names else None
                
                # Calculate losses
                domain_loss = self.criterion(decoded, target)
                loss_details[f"demi_cycle_{domain_name}_full_loss"] = domain_loss.item()
                
                # Extract synergy-specific loss
                synergy_target, _ = self.extract_synergy_features(
                    target, domain_name, domain_feature_names
                )
                synergy_recon, _ = self.extract_synergy_features(
                    decoded, domain_name, domain_feature_names
                )
                
                if synergy_target.numel() > 0:
                    synergy_loss = self.criterion(synergy_recon, synergy_target)
                    loss_details[f"demi_cycle_{domain_name}_synergy_loss"] = synergy_loss.item()
                
                # Accumulate total loss
                if total_loss is None:
                    total_loss = domain_loss
                else:
                    total_loss = total_loss + domain_loss
                
                num_domains += 1
        
        except Exception as e:
            logger.error(f"Error in demi-cycle calculation: {e}")
            # Return zero loss on error
            device = next(model.parameters()).device
            total_loss = torch.tensor(0.0, device=device)
        
        finally:
            # Restore original weights
            model.fusion_weights.update(original_weights)
        
        # Average across domains
        if total_loss is not None and num_domains > 1:
            total_loss = total_loss / num_domains
        elif total_loss is None:
            device = next(model.parameters()).device
            total_loss = torch.tensor(0.0, device=device)
        
        loss_details['demi_cycle_loss'] = total_loss.item()
        return total_loss, loss_details
    
    def calculate_synergy_cycle_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        feature_names: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate cycle loss with synergy awareness.
        
        Args:
            model: The GW model
            inputs: Input tensors per domain
            targets: Target tensors per domain
            feature_names: Optional feature names per domain
            
        Returns:
            Tuple of (total_loss, loss_details)
        """
        loss_details = {}
        total_loss = None
        num_cycles = 0
        
        # Store original fusion weights
        original_weights = model.fusion_weights.copy()
        
        domain_names = list(inputs.keys())
        
        # Need at least 2 domains for cycle loss
        if len(domain_names) < 2:
            device = next(model.parameters()).device
            loss_details['cycle_loss'] = 0.0
            return torch.tensor(0.0, device=device), loss_details
        
        try:
            # For each domain pair, calculate cycle loss in both directions
            for i, domain_x in enumerate(domain_names):
                for j, domain_y in enumerate(domain_names):
                    if i == j or domain_x not in targets or domain_y not in targets:
                        continue
                    
                    # Cycle: X -> GW via Y -> X
                    # Step 1: Set weights for domain Y encoding
                    cycle_weights = {d: 0.0 for d in model.fusion_weights.keys()}
                    cycle_weights[domain_y] = 1.0
                    model.fusion_weights.update(cycle_weights)
                    
                    # Step 2: Encode Y, decode to X
                    input_y = inputs[domain_y]
                    encoded_y = model.gw_encoders[domain_y](input_y)
                    gw_state = model.fuse({domain_y: encoded_y}, selection_scores={})
                    decoded_x = model.gw_decoders[domain_x](gw_state)
                    
                    # Step 3: Calculate cycle loss
                    target_x = targets[domain_x]
                    cycle_loss = self.criterion(decoded_x, target_x)
                    loss_details[f"cycle_{domain_x}_via_{domain_y}_full_loss"] = cycle_loss.item()
                    
                    # Extract synergy-specific cycle loss
                    domain_feature_names = feature_names.get(domain_x) if feature_names else None
                    synergy_target, _ = self.extract_synergy_features(
                        target_x, domain_x, domain_feature_names
                    )
                    synergy_decoded, _ = self.extract_synergy_features(
                        decoded_x, domain_x, domain_feature_names
                    )
                    
                    if synergy_target.numel() > 0:
                        synergy_cycle_loss = self.criterion(synergy_decoded, synergy_target)
                        loss_details[f"cycle_{domain_x}_via_{domain_y}_synergy_loss"] = synergy_cycle_loss.item()
                    
                    # Accumulate total loss
                    if total_loss is None:
                        total_loss = cycle_loss
                    else:
                        total_loss = total_loss + cycle_loss
                    
                    num_cycles += 1
        
        except Exception as e:
            logger.error(f"Error in cycle calculation: {e}")
            device = next(model.parameters()).device
            total_loss = torch.tensor(0.0, device=device)
        
        finally:
            # Restore original weights
            model.fusion_weights.update(original_weights)
        
        # Average across all cycles
        if total_loss is not None and num_cycles > 1:
            total_loss = total_loss / num_cycles
        elif total_loss is None:
            device = next(model.parameters()).device
            total_loss = torch.tensor(0.0, device=device)
        
        loss_details['cycle_loss'] = total_loss.item()
        return total_loss, loss_details


def calculate_synergy_losses(
    model,
    batch_inputs: Dict[str, torch.Tensor],
    batch_targets: Dict[str, torch.Tensor],
    synergy_config: Dict[str, Any],
    loss_weights: Dict[str, float],
    criterion: Optional[nn.Module] = None,
    feature_names: Optional[Dict[str, List[str]]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Calculate all synergy-aware losses for a batch.
    
    Args:
        model: The GW model
        batch_inputs: Input tensors per domain
        batch_targets: Target tensors per domain  
        synergy_config: Synergy configuration
        loss_weights: Weights for different loss types
        criterion: Loss function
        feature_names: Optional feature names per domain
        
    Returns:
        Tuple of (total_loss, detailed_loss_dict)
    """
    # Initialize loss calculator
    calculator = SynergyLossCalculator(
        synergy_config=synergy_config,
        criterion=criterion,
        synergy_weight=loss_weights.get('synergy', 1.0)
    )
    
    total_loss = None
    loss_details = {}
    
    # 1. Fusion Loss
    if loss_weights.get('fusion', 0.0) > 0:
        fusion_loss, fusion_details = calculator.calculate_synergy_fusion_loss(
            model, batch_inputs, batch_targets,
            use_weights_for_loss=model.fusion_weights.get('use_weights_for_loss', False),
            feature_names=feature_names
        )
        loss_details.update(fusion_details)
        
        weighted_fusion_loss = loss_weights['fusion'] * fusion_loss
        loss_details['weighted_fusion_loss'] = weighted_fusion_loss.item()
        
        if total_loss is None:
            total_loss = weighted_fusion_loss
        else:
            total_loss = total_loss + weighted_fusion_loss
    
    # 2. Demi-Cycle Loss
    if loss_weights.get('demi_cycle', 0.0) > 0:
        demi_loss, demi_details = calculator.calculate_synergy_demi_cycle_loss(
            model, batch_inputs, batch_targets, feature_names=feature_names
        )
        loss_details.update(demi_details)
        
        weighted_demi_loss = loss_weights['demi_cycle'] * demi_loss
        loss_details['weighted_demi_cycle_loss'] = weighted_demi_loss.item()
        
        if total_loss is None:
            total_loss = weighted_demi_loss
        else:
            total_loss = total_loss + weighted_demi_loss
    
    # 3. Cycle Loss
    if loss_weights.get('cycle', 0.0) > 0:
        cycle_loss, cycle_details = calculator.calculate_synergy_cycle_loss(
            model, batch_inputs, batch_targets, feature_names=feature_names
        )
        loss_details.update(cycle_details)
        
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


# Utility functions for batch processing
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