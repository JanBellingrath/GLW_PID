#!/usr/bin/env python3
"""
Synergy Training for Global Workspace Models

Main training script for learning synergistic combinations in Global Workspace models.
This script trains models to predict synergistic targets (e.g., XOR combinations) 
from input modalities while monitoring both overall and synergy-specific losses.

Key features:
- Supports fusion, demi-cycle, cycle, and translation loss combinations
- Separates inputs from synergistic targets properly  
- Cross-modal supervised translation learning on paired data
- Comprehensive WandB logging of synergy metrics
- Flexible experiment configuration
- Compatible with pretrained domain modules
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

# GPU memory limiting will be handled after argument parsing

# Add path setup for imports - prioritize local directory
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))  # Local directory has highest priority

# Add shimmer module path - look for shimmer-ssd in parent directories
current_dir = script_dir
while current_dir != current_dir.parent:
    shimmer_path = current_dir.parent / 'shimmer-ssd'
    if shimmer_path.exists():
        sys.path.insert(0, str(shimmer_path))
        break
    current_dir = current_dir.parent

# Import our synergy modules
from synergy_dataset import SynergyDataset, create_synergy_dataloaders
from synergy_losses import create_synergy_loss_function, process_synergy_batch
from metrics.predictability import compute_predictability

# Import from the original training script - reuse existing functions
from losses_and_weights_GLW_training import (
    create_gw_model, load_domain_modules, save_checkpoint, load_checkpoint,
    GWModuleConfigurableFusion, train_model, evaluate_model
)
import losses_and_weights_GLW_training

# Try importing wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Experiment tracking disabled.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SynergyExperimentConfig:
    """Configuration class for synergy experiments."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from configuration dictionary."""
        # Data configuration
        self.data_dir = config_dict['data']['dir']
        self.image_dir = config_dict['data'].get('image_dir')
        self.batch_size = config_dict['data'].get('batch_size', 32)
        self.num_workers = config_dict['data'].get('num_workers', 4)
        
        # Model configuration
        self.domain_configs = config_dict['model']['domains']
        self.workspace_dim = config_dict['model'].get('workspace_dim', 12)
        self.hidden_dim = config_dict['model'].get('hidden_dim', 32)
        # Allow separate decoder hidden width; fallback to encoder hidden_dim
        self.decoder_hidden_dim = config_dict['model'].get('decoder_hidden_dim', self.hidden_dim)
        self.n_layers = config_dict['model'].get('n_layers', 4)
        self.fusion_weights = config_dict['model'].get('fusion_weights', {'v': 0.5, 'attr': 0.5})
        
        # Synergy configuration
        self.synergy_config = config_dict['synergy']
        
        # Add synergy loss scale if not present
        if 'loss_scale' not in self.synergy_config:
            self.synergy_config['loss_scale'] = 1.0
        
        # Add unique/redundant indices for predictability metrics (optional)
        self.unique_indices = self.synergy_config.get('unique_indices', {})
        self.redundant_indices = self.synergy_config.get('redundant_indices', {})
        
        # Training configuration
        self.training = config_dict['training']
        self.loss_configs = config_dict['training']['loss_configs']
        
        # Experiment configuration
        self.experiment = config_dict.get('experiment', {})
        self.output_dir = self.experiment.get('output_dir', 'experiments/synergy')
        self.wandb_project = self.experiment.get('wandb_project', 'synergy-glw')
        self.wandb_entity = self.experiment.get('wandb_entity')
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration consistency."""
        # Check that synergy domains are subset of model domains
        model_domains = set(config['name'] for config in self.domain_configs)
        synergy_domains = set(self.synergy_config['domains'])
        
        if not synergy_domains.issubset(model_domains):
            raise ValueError(
                f"Synergy domains {synergy_domains} not subset of model domains {model_domains}"
            )
        
        # Check loss configurations
        for loss_config in self.loss_configs:
            required_keys = ['name', 'weights']
            for key in required_keys:
                if key not in loss_config:
                    raise ValueError(f"Loss config missing required key: {key}")
        
        logger.info("Configuration validation passed")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SynergyExperimentConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def save_to_file(self, output_path: str):
        """Save configuration to JSON file."""
        config_dict = {
            'data': {
                'dir': self.data_dir,
                'image_dir': self.image_dir,
                'batch_size': self.batch_size,
                'num_workers': self.num_workers
            },
            'model': {
                'domains': self.domain_configs,
                'workspace_dim': self.workspace_dim,
                'hidden_dim': self.hidden_dim,
                'decoder_hidden_dim': self.decoder_hidden_dim,
                'n_layers': self.n_layers,
                'fusion_weights': self.fusion_weights
            },
            'synergy': {
                **self.synergy_config,
                'unique_indices': self.unique_indices,
                'redundant_indices': self.redundant_indices
            },
            'training': self.training,
            'experiment': self.experiment
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


class SynergyTrainer:
    """Trainer for synergy-aware Global Workspace models."""
    
    def __init__(
        self,
        config: SynergyExperimentConfig,
        device: Optional[torch.device] = None
    ):
        """Initialize trainer."""
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.dataloaders = {}
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = defaultdict(list)
        
        logger.info(f"Initialized SynergyTrainer on device: {self.device}")
    
    def _create_synergy_gw_model(self, domain_modules):
        """Create GW model with synergy-aware decoder dimensions."""
        from shimmer.modules.gw_module import GWEncoder, GWDecoder
        
        # Create encoders and decoders with adjusted dimensions
        gw_encoders = {}
        gw_decoders = {}
        
        for domain_name, domain_module in domain_modules.items():
            # Get base latent dimension
            latent_dim = domain_module.latent_dim
            
            # Special handling for domains that bypass domain modules
            if domain_name == 'attr':
                input_dim = 11  # Input attributes are 11D (preprocessed)
                logger.info(f"Attribute domain: bypassing domain module, using 11D input (preprocessed attributes")
            elif domain_name == 'v':
                input_dim = 13  # VAE latents (12D) + size value (1D) = 13D
                logger.info(f"Visual domain: bypassing domain module, using 13D input (12D VAE latents + 1D size)")
            else:
                input_dim = latent_dim  # Other domains use domain module output
            
            # Calculate output dimension based on synergy config
            if domain_name == 'attr':
                # For attributes: base is 11D preprocessed attributes
                base_dim_output = 11
            elif domain_name == 'v':
                # For visual: base is 13D (12D VAE latents + 1D size)
                base_dim_output = 13
            else:
                # For other domains: use domain module latent_dim
                base_dim_output = latent_dim
                
            output_dim = base_dim_output 

            if (domain_name in self.config.synergy_config.get('feature_indices', {}) and 
                self.config.synergy_config['feature_indices'][domain_name]): #TODO what is feature_indices? does it not exist for both domains?
                # Add synergy feature dimensions as logits (8 classes per synergy feature)
                synergy_features = len(self.config.synergy_config['feature_indices'][domain_name])
                n_synergy_classes = 8  # XOR has 8 discrete classes
                synergy_logit_dims = synergy_features * n_synergy_classes #TODO WHY? this seems wrong
                output_dim += synergy_logit_dims
                logger.info(f"Expanding {domain_name} decoder output: {base_dim_output} -> {output_dim} (+{synergy_features} synergy features × {n_synergy_classes} classes = +{synergy_logit_dims} logits)")
            
            # Create encoder with appropriate input dimension
            gw_encoders[domain_name] = GWEncoder(
                in_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                out_dim=self.config.workspace_dim,
                n_layers=self.config.n_layers,
            )
            
            # Create decoder with expanded output dimension (no global activation)
            decoder = GWDecoder(
                in_dim=self.config.workspace_dim,
                hidden_dim=self.config.decoder_hidden_dim,
                out_dim=output_dim,
                n_layers=self.config.n_layers,
            )
            
            # Use decoder directly without global sigmoid activation
            # This allows outputs to match the full range of target values
            gw_decoders[domain_name] = decoder
        
        # Create fusion weights
        fusion_weights = self.config.fusion_weights.copy()
        
        # Create GW module
        gw_module = GWModuleConfigurableFusion(
            domain_modules=domain_modules,
            workspace_dim=self.config.workspace_dim,
            gw_encoders=gw_encoders,
            gw_decoders=gw_decoders,
            fusion_weights=fusion_weights,
        )
        
        # Store architecture parameters
        gw_module.hidden_dim = self.config.hidden_dim
        gw_module.n_layers = self.config.n_layers
        
        # Freeze domain modules
        for domain_name, domain_module in domain_modules.items():
            for param in domain_module.parameters():
                param.requires_grad = False
            logger.info(f"Frozen domain module: {domain_name}")
        
        return gw_module
    
    def setup_model(self):
        """Set up the Global Workspace model with pretrained domains."""
        logger.info("Setting up model...")
        
        # Load domain modules
        domain_modules = load_domain_modules(self.config.domain_configs)
        
        # Create GW model with synergy-aware decoder dimensions
        self.model = self._create_synergy_gw_model(domain_modules)
        
        self.model.to(self.device)
        
        # Set up optimizer
        optimizer_config = self.config.training.get('optimizer', {'type': 'Adam', 'lr': 1e-3})
        if optimizer_config['type'] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-3),
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
        
        logger.info(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Log model architecture info
        for domain_name, domain_module in domain_modules.items():
            logger.info(f"Domain {domain_name}: latent_dim={domain_module.latent_dim}")
    
    def setup_data(self):
        """Set up data loaders."""
        logger.info("Setting up data loaders...")
        
        self.dataloaders = create_synergy_dataloaders(
            data_dir=self.config.data_dir,
            synergy_config=self.config.synergy_config,
            batch_size=self.config.batch_size,
            image_dir=self.config.image_dir,
            device=None,  # Let DataLoader handle device transfer
            num_workers=self.config.num_workers,
            load_images=False,  # Use precomputed features for now
            cache_data=True
        )
        
        for split, dataloader in self.dataloaders.items():
            logger.info(f"{split}: {len(dataloader.dataset)} samples, {len(dataloader)} batches")
        
        # Check if we have the required train dataloader
        if 'train' not in self.dataloaders:
            raise RuntimeError("Failed to create train dataloader. Check the data directory and file paths.")
    
    def create_synergy_dataloader_wrapper(self, split: str):
        """Create a wrapper that converts synergy batches to standard format."""
        original_loader = self.dataloaders[split]
        
        class SynergyDataLoaderWrapper:
            def __init__(self, loader, device, synergy_config):
                self.loader = loader
                self.device = device 
                self.synergy_config = synergy_config
                self._debug_batches = 0
                
            def __iter__(self):
                for batch in self.loader:
                    # Convert synergy batch to standard format that process_batch expects
                    # CRITICAL: Use INPUTS for encoding, TARGETS for loss computation
                    standard_batch = {}
                    
                    # Extract inputs from nested structure and create flat batch
                    if isinstance(batch, dict) and 'inputs' in batch:
                        # Our synergy dataset format
                        for domain, data in batch['inputs'].items():
                            # Don't call .to(device) here - let process_batch handle it
                            standard_batch[domain] = data
                        
                        # Flatten targets with prefix so process_batch can handle them as tensors
                        for domain, data in batch['targets'].items():
                            standard_batch[f'_target_{domain}'] = data

                        # Debug logging for first few batches
                        if self._debug_batches < 2:
                            try:
                                logger.info("[LoaderWrapper] New batch")
                                in_keys = list(batch['inputs'].keys())
                                tgt_keys = list(batch['targets'].keys())
                                logger.info(f"  inputs={in_keys} targets={tgt_keys} device={self.device}")
                                for d, t in batch['inputs'].items():
                                    if hasattr(t, 'shape'):
                                        logger.info(f"  input[{d}]: shape={tuple(t.shape)} dtype={getattr(t,'dtype',None)}")
                                for d, t in batch['targets'].items():
                                    if hasattr(t, 'shape'):
                                        logger.info(f"  target[{d}]: shape={tuple(t.shape)} dtype={getattr(t,'dtype',None)}")
                                for k, v in standard_batch.items():
                                    if hasattr(v, 'shape'):
                                        logger.info(f"  std[{k}]: shape={tuple(v.shape)}")
                            except Exception as e:
                                logger.warning(f"[LoaderWrapper] debug log failed: {e}")
                            self._debug_batches += 1
                    else:
                        # Standard dataset format - pass through
                        standard_batch = batch
                    
                    yield standard_batch
            
            def __len__(self):
                return len(self.loader)
        
        return SynergyDataLoaderWrapper(original_loader, self.device, self.config.synergy_config)
    
    def evaluate_predictability(self) -> Dict[str, float]:
        """
        Evaluate predictability metrics on validation set.
        
        Returns:
            Dictionary of predictability metrics
        """
        if 'val' not in self.dataloaders:
            logger.warning("No validation dataloader available for predictability evaluation")
            return {}
            
        if self.model is None:
            logger.warning("No model available for predictability evaluation")
            return {}
        
        logger.info("Computing predictability metrics on validation set...")
        
        self.model.eval()
        val_loader = self.create_synergy_dataloader_wrapper('val')
        
        # Collect all decoded outputs and targets
        all_decoded = {}
        all_targets = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # Process batch to get inputs and targets
                    batch_inputs = {}
                    batch_targets = {}
                    
                    for key, value in batch.items():
                        if key.startswith('_target_'):
                            domain = key.replace('_target_', '')
                            batch_targets[domain] = value.to(self.device)
                        else:
                            batch_inputs[key] = value.to(self.device)
                    
                    # Only evaluate domains that have synergy targets
                    synergy_domains = set(self.config.synergy_config['domains'])
                    eval_domains = synergy_domains.intersection(batch_targets.keys())
                    
                    if not eval_domains:
                        continue
                    
                    # Encode inputs
                    latents = {}
                    for domain, data in batch_inputs.items():
                        if domain in eval_domains:
                            # Use the same bypassing logic as in training
                            if domain == 'attr' or domain == 'v':
                                # Bypass domain module - use preprocessed data directly
                                latents[domain] = self.model.gw_encoders[domain](data)
                            else:
                                # Use domain module if available
                                domain_latent = self.model.domain_modules[domain].encode(data)
                                latents[domain] = self.model.gw_encoders[domain](domain_latent)
                    
                    if not latents:
                        continue
                    
                    # Fuse latents
                    fused_latent = self.model.fuse(latents, None)
                    
                    # Decode to all target domains
                    decoded = {}
                    for domain in eval_domains:
                        decoded[domain] = self.model.gw_decoders[domain](fused_latent)
                    
                    # Accumulate results
                    for domain in decoded.keys():
                        if domain not in all_decoded:
                            all_decoded[domain] = []
                            all_targets[domain] = []
                        
                        all_decoded[domain].append(decoded[domain].cpu())
                        all_targets[domain].append(batch_targets[domain].cpu())
                        
                except Exception as e:
                    logger.warning(f"Error processing validation batch {batch_idx}: {e}")
                    continue
        
        if not all_decoded:
            logger.warning("No validation data processed for predictability evaluation")
            return {}
        
        # Concatenate all batches
        final_decoded = {}
        final_targets = {}
        
        for domain in all_decoded.keys():
            final_decoded[domain] = torch.cat(all_decoded[domain], dim=0).to(self.device)
            final_targets[domain] = torch.cat(all_targets[domain], dim=0).to(self.device)
        
        # Prepare partitions from config
        partitions = {}
        for domain in final_decoded.keys():
            partitions[domain] = {
                'unique': self.config.unique_indices.get(domain, []),
                'redundant': self.config.redundant_indices.get(domain, [])
            }
        
        # Compute predictability metrics
        metrics = compute_predictability(
            decoded=final_decoded,
            targets=final_targets,
            partitions=partitions,
            synergy_config=self.config.synergy_config,
            device=self.device,
            n_bins=8
        )
        
        logger.info("Predictability metrics computed:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def run_experiment(
        self,
        loss_config: Dict[str, Any],
        log_to_wandb: bool = True
    ) -> Dict[str, Any]:
        """Run a single experiment using existing train_model function."""
        experiment_name = loss_config['name']
        loss_weights = loss_config['weights']
        epochs = loss_config.get('epochs', self.config.training.get('epochs', 50))
        
        logger.info(f"Starting experiment: {experiment_name}")
        logger.info(f"Loss weights: {loss_weights}")
        logger.info(
            f"Config summary: workspace_dim={self.config.workspace_dim}, hidden_dim={self.config.hidden_dim}, "
            f"decoder_hidden_dim={self.config.decoder_hidden_dim}, n_layers={self.config.n_layers}, "
            f"synergy_loss_scale={self.config.synergy_config.get('loss_scale', 1.0)}, device={self.device}"
        )
        
        # Create wrapped data loaders that convert synergy format to standard format
        train_loader = self.create_synergy_dataloader_wrapper('train')
        val_loader = self.create_synergy_dataloader_wrapper('val') if 'val' in self.dataloaders else None
        
        # Set up checkpoint directory
        checkpoint_dir = Path(self.config.output_dir) / experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Monkey-patch the loss function to add synergy tracking
        original_calculate_losses = losses_and_weights_GLW_training.calculate_losses_with_weights
        synergy_loss_function = create_synergy_loss_function(self.config.synergy_config)
        losses_and_weights_GLW_training.calculate_losses_with_weights = synergy_loss_function
        
        try:
            # Use existing train_model function - this reuses all existing training infrastructure!
            trained_model, final_checkpoint = train_model(
                model=self.model,
                train_data_loader=train_loader,
                val_data_loader=val_loader,
                num_epochs=epochs,
                learning_rate=self.config.training.get('optimizer', {}).get('lr', 1e-3),
                device=str(self.device),
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_interval=10,
                run_name=experiment_name,
                log_to_wandb=log_to_wandb,
                wandb_project=self.config.wandb_project,
                wandb_entity=self.config.wandb_entity,
                short_circuit=False,
                use_weighted_loss=True,  # Must be True to invoke patched calculate_losses_with_weights
                loss_weights=loss_weights  # Pass our loss weights directly
            )
        finally:
            # Restore original loss function
            losses_and_weights_GLW_training.calculate_losses_with_weights = original_calculate_losses
        
        # Get final metrics by evaluating the model
        if val_loader:
            eval_result = evaluate_model(trained_model, val_loader, str(self.device))
            # evaluate_model might return a tuple or dict, extract the loss value
            if isinstance(eval_result, (tuple, list)):
                final_val_loss = float(eval_result[0]) if eval_result else 0.0
            elif isinstance(eval_result, dict):
                final_val_loss = float(eval_result.get('total_loss', 0.0))
            else:
                final_val_loss = float(eval_result) if eval_result is not None else 0.0
        else:
            final_val_loss = 0.0
        
        # Evaluate predictability metrics and log to W&B
        predictability_metrics = self.evaluate_predictability()
        
        # Log predictability metrics to W&B if available and enabled
        if HAS_WANDB and log_to_wandb and predictability_metrics:
            try:
                wandb.log(predictability_metrics)
                logger.info("Predictability metrics logged to W&B")
            except Exception as e:
                logger.warning(f"Failed to log predictability metrics to W&B: {e}")
        
        # Save predictability summary to file
        if predictability_metrics:
            summary_file = Path(self.config.output_dir) / experiment_name / 'predictability_summary.json'
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            
            summary_data = {
                'experiment_name': experiment_name,
                'workspace_dim': self.config.workspace_dim,
                'loss_weights': loss_weights,
                'synergy_loss_scale': self.config.synergy_config.get('loss_scale', 1.0),
                'metrics': predictability_metrics
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            logger.info(f"Predictability summary saved to {summary_file}")
        
        # Return experiment results
        results = {
            'experiment_name': experiment_name,
            'loss_weights': loss_weights,
            'best_epoch': epochs,  # train_model doesn't return best epoch, use final
            'best_val_loss': final_val_loss,
            'checkpoint_path': final_checkpoint
        }
        
        logger.info(f"Completed experiment: {experiment_name}")
        logger.info(f"Final validation loss: {final_val_loss:.6f}")
        
        return results
    
    def run_all_experiments(self) -> List[Dict[str, Any]]:
        """Run all configured experiments."""
        logger.info(f"Running {len(self.config.loss_configs)} experiments...")
        
        results = []
        
        for i, loss_config in enumerate(self.config.loss_configs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {i+1}/{len(self.config.loss_configs)}")
            logger.info(f"{'='*60}")
            
            # Reset model for each experiment if configured
            if self.config.training.get('reset_model_per_experiment', False):
                self.setup_model()
            
            # Run experiment
            result = self.run_experiment(
                loss_config=loss_config,
                log_to_wandb=self.config.experiment.get('log_to_wandb', True)
            )
            
            results.append(result)
        
        # Save overall results
        results_file = Path(self.config.output_dir) / 'experiment_results.json'
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nAll experiments completed. Results saved to {results_file}")
        
        return results


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration for synergy experiments."""
    return {
        "data": {
            "dir": "data/simple_shapes_xor_targets",
            "image_dir": None,
            "batch_size": 32,
            "num_workers": 4
        },
        "model": {
            "domains": [
                {
                    "domain_type": "v_latents",
                    "checkpoint_path": "path/to/pretrained_vae.ckpt",
                    "name": "v"
                },
                {
                    "domain_type": "attr",
                    "checkpoint_path": "path/to/pretrained_attr.ckpt",
                    "name": "attr"
                }
            ],
            "workspace_dim": 12,
            "hidden_dim": 128,
            "n_layers": 4,
            "fusion_weights": {
                "v": 0.5,
                "attr": 0.5
            }
        },
        "synergy": {
            "domains": ["attr", "v"],
            "feature_indices": {
                "attr": ["xor_target_normalized"]
            },
            "loss_scale": 1.0,
            "unique_indices": {
                "attr": [],  # Example: indices that are unique to attr domain  
                "v": []      # Example: indices that are unique to v domain
            },
            "redundant_indices": {
                "attr": [],  # Example: indices that are redundant across modalities
                "v": []      # Example: indices that are redundant across modalities
            }
        },
        "training": {
            "epochs": 50,
            "optimizer": {
                "type": "Adam",
                "lr": 1e-3,
                "weight_decay": 0.0
            },
            "grad_clip": 1.0,
            "reset_model_per_experiment": True,
            "loss_configs": [
                {
                    "name": "fusion_only",
                    "weights": {"fusion": 1.0, "demi_cycle": 0.0, "cycle": 0.0, "translation": 0.0}
                },
                {
                    "name": "fusion_demi",
                    "weights": {"fusion": 1.0, "demi_cycle": 1.0, "cycle": 0.0, "translation": 0.0}
                },
                {
                    "name": "fusion_cycle",
                    "weights": {"fusion": 1.0, "demi_cycle": 0.0, "cycle": 1.0, "translation": 0.0}
                },
                {
                    "name": "translation_only",
                    "weights": {"fusion": 0.0, "demi_cycle": 0.0, "cycle": 0.0, "translation": 1.0}
                },
                {
                    "name": "all_losses",
                    "weights": {"fusion": 0.5, "demi_cycle": 0.5, "cycle": 0.5, "translation": 0.5}
                }
            ]
        },
        "experiment": {
            "output_dir": "experiments/synergy",
            "wandb_project": "synergy-glw",
            "wandb_entity": None,
            "log_to_wandb": True
        }
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train synergy-aware Global Workspace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run with default configuration
  python train_synergy_glw.py --config experiments/default_config.json
  
  # Generate default config file
  python train_synergy_glw.py --generate-config default_config.json
  
  # Run single experiment with custom settings
  python train_synergy_glw.py --config config.json --experiment-name fusion_only
  
  # Run with custom batch size and GPU memory limit
  python train_synergy_glw.py --config config.json --batch-size 64 --gpu-memory-percent 50.0
  
  # Run with synergy loss scaling and memory control
  python train_synergy_glw.py --config config.json --synergy-loss-scale 20.0 --gpu-memory-percent 75.0
        """
    )
    
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--generate-config", type=str, help="Generate default config file and exit")
    parser.add_argument("--experiment-name", type=str, help="Run only specific experiment")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--synergy-loss-scale", type=float, default=1.0, help="Scale factor for synergy feature loss contribution (default: 1.0)")
    parser.add_argument("--batch-size", type=int, help="Override batch size from config")
    parser.add_argument("--gpu-memory-percent", type=float, default=100.0, help="GPU memory usage limit as percentage of total GPU memory (default: 25.0 for 10GB/40GB)")
    parser.add_argument("--workspace-dim", type=int, default=None, help="Override workspace dimension (GLW latent)")
    parser.add_argument("--decoder-hidden-dim", type=int, default=None, help="Override decoder hidden width")
    
    args = parser.parse_args()
    
    # Generate default config if requested
    if args.generate_config:
        config_dict = create_default_config()
        with open(args.generate_config, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Generated default configuration: {args.generate_config}")
        return
    
    # Load configuration
    if not args.config:
        print("Error: --config is required (or use --generate-config)")
        return
    
    # Set GPU memory limit based on CLI argument
    if torch.cuda.is_available():
        memory_fraction = args.gpu_memory_percent / 100.0
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        memory_gb = 40.0 * memory_fraction  # Assuming A100 40GB
        print(f"🔒 GPU memory limited to {memory_gb:.1f}GB ({args.gpu_memory_percent}% of 40GB GPU)")
    
    config = SynergyExperimentConfig.from_file(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_wandb:
        config.experiment['log_to_wandb'] = False
    if args.workspace_dim:
        config.workspace_dim = int(args.workspace_dim)
        logger.info(f"Override workspace_dim: {config.workspace_dim}")
    if args.decoder_hidden_dim:
        config.decoder_hidden_dim = int(args.decoder_hidden_dim)
        logger.info(f"Override decoder_hidden_dim: {config.decoder_hidden_dim}")
    if args.synergy_loss_scale:
        config.synergy_config['loss_scale'] = args.synergy_loss_scale
        logger.info(f"Override synergy loss scale: {args.synergy_loss_scale}")
    if args.batch_size:
        config.batch_size = args.batch_size
        logger.info(f"Override batch size: {args.batch_size}")
    
    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Filter experiments if requested
    if args.experiment_name:
        config.loss_configs = [
            cfg for cfg in config.loss_configs 
            if cfg['name'] == args.experiment_name
        ]
        if not config.loss_configs:
            print(f"Error: Experiment '{args.experiment_name}' not found")
            return
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = Path(config.output_dir) / 'config.json'
    config.save_to_file(str(config_save_path))
    
    # Initialize trainer
    trainer = SynergyTrainer(config, device=device)
    
    # Setup model and data
    trainer.setup_model()
    trainer.setup_data()
    
    # Run experiments
    results = trainer.run_all_experiments()
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for result in results:
        print(f"{result['experiment_name']:20s}: "
              f"Best Val Loss = {result['best_val_loss']:.6f} "
              f"(epoch {result['best_epoch']})")
    
    print(f"\nResults saved to: {config.output_dir}")


if __name__ == "__main__":
    main() 