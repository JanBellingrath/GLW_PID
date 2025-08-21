#!/usr/bin/env python3
"""
Low Dimensionality Sweep for Synergy Prior Analysis

Systematic sweep of workspace_dim to test the hypothesis that decreasing bottleneck
dimensionality preferentially preserves synergistic information over unique and
redundant information.

Usage:
    python experiments/lowdim_sweep.py --config base.json --dims 2,4,8,12,16,24,32,48,64
    
Example quick test:
    python experiments/lowdim_sweep.py --config config.json --dims 8,12 --quick --no-wandb
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add path setup for imports
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

# Add shimmer module path - look for shimmer-ssd in parent directories
current_dir = script_dir
while current_dir != current_dir.parent:
    shimmer_path = current_dir.parent / 'shimmer-ssd'
    if shimmer_path.exists():
        sys.path.insert(0, str(shimmer_path))
        break
    current_dir = current_dir.parent

# Import our training modules
from train_synergy_glw import SynergyExperimentConfig, SynergyTrainer
from metrics.predictability import create_reconstruction_plot, create_synergy_metrics_plot

# Try importing wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def create_derived_config(
    base_config: SynergyExperimentConfig,
    workspace_dim: int,
    experiment_name: str,
    epochs: int,
    seed: int,
    batch_size: Optional[int] = None
) -> SynergyExperimentConfig:
    """Create a derived config with modified workspace_dim."""
    
    # Get the base config as dict
    config_dict = {
        'data': {
            'dir': base_config.data_dir,
            'image_dir': base_config.image_dir,
            'batch_size': batch_size if batch_size is not None else base_config.batch_size,
            'num_workers': base_config.num_workers
        },
        'model': {
            'domains': base_config.domain_configs,
            'workspace_dim': workspace_dim,  # Override this
            'hidden_dim': base_config.hidden_dim,
            'decoder_hidden_dim': base_config.decoder_hidden_dim,
            'n_layers': base_config.n_layers,
            'fusion_weights': base_config.fusion_weights
        },
        'synergy': {
            **base_config.synergy_config, # TODO: check if this is correct, is odd way to set it up semantically
            'unique_indices': base_config.unique_indices,
            'redundant_indices': base_config.redundant_indices
        },
        'training': {
            **base_config.training,
            'epochs': epochs,
            'reset_model_per_experiment': True  # Ensure clean init per run
        },
        'experiment': {
            **base_config.experiment,
            'output_dir': str(Path(base_config.output_dir) / f'dim_{workspace_dim}'),
            'wandb_project': base_config.wandb_project
        }
    }
    
    return SynergyExperimentConfig(config_dict)


def run_single_dimension(
    base_config: SynergyExperimentConfig,
    workspace_dim: int,
    experiment_type: str,
    epochs: int,
    seed: int,
    device: torch.device,
    log_to_wandb: bool = True,
    batch_size: Optional[int] = None
) -> Dict[str, Any]:
    """Run experiment for a single workspace dimension."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Running workspace_dim = {workspace_dim}")
    logger.info(f"{'='*80}")
    
    # Set seeds for reproducibility
    set_seeds(seed)
    
    # Create derived config
    config = create_derived_config(
        base_config=base_config,
        workspace_dim=workspace_dim,
        experiment_name=f'lowdim_sweep_dim_{workspace_dim}',
        epochs=epochs,
        seed=seed,
        batch_size=batch_size
    )
    
    # Save derived config for provenance
    config_file = Path(config.output_dir) / 'config.json'
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config.save_to_file(str(config_file))
    logger.info(f"Saved derived config to {config_file}")
    
    # Initialize trainer
    trainer = SynergyTrainer(config, device=device)
    trainer.setup_model()
    trainer.setup_data()
    
    matching_configs = [lc for lc in config.loss_configs if lc['name'] == experiment_type]
    if len(matching_configs) > 1:
        raise ValueError(f"Multiple configs found with name '{experiment_type}': {matching_configs}")
    elif len(matching_configs) == 0:
        raise ValueError(f"No config found with name '{experiment_type}'")
    else:
        loss_config = matching_configs[0]
    
    # Initialize W&B for this run
    if HAS_WANDB and log_to_wandb:
        try:
            wandb.init(
                project="synergy-bottleneck-sweep-2025",
                group="lowdim-synergy-prior",
                job_type="train",
                name=f"dim_syn_{workspace_dim}",
                tags=[f"dim_syn={workspace_dim}", experiment_type],
                config={
                    'workspace_dim': workspace_dim,
                    'experiment_type': experiment_type,
                    'epochs': epochs,
                    'seed': seed,
                    'hidden_dim': config.hidden_dim,
                    'decoder_hidden_dim': config.decoder_hidden_dim,
                    'n_layers': config.n_layers,
                    'synergy_loss_scale': config.synergy_config.get('loss_scale', 1.0),
                    'loss_weights': loss_config['weights']
                }
            )
            logger.info("Initialized W&B run")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            log_to_wandb = False
    
    try:
        # Run the experiment
        result = trainer.run_experiment(
            loss_config=loss_config,
            log_to_wandb=log_to_wandb
        )
        
        # Add dimension info to result
        result['workspace_dim'] = workspace_dim
        result['experiment_type'] = experiment_type
        result['seed'] = seed
        
        # Create reconstruction plot if W&B data is available
        if HAS_WANDB and log_to_wandb:
            try:
                # Get the current run's history
                run = wandb.run
                if run is not None:
                    history = run.history()
                    
                    # Extract reconstruction and synergy metrics
                    epochs = []
                    non_synergy_mse = []
                    synergy_source_mse = []
                    synergy_acc = []
                    synergy_norm_ce = []
                    
                    for _, row in history.iterrows():
                        if 'epoch' in row and not pd.isna(row['epoch']):
                            epochs.append(int(row['epoch']))
                            
                            # Extract MSE metrics
                            if 'fusion_attr_non_synergy_features_mse' in row:
                                non_synergy_mse.append(float(row['fusion_attr_non_synergy_features_mse']))
                            if 'fusion_attr_synergy_source_feature_mse' in row:
                                synergy_source_mse.append(float(row['fusion_attr_synergy_source_feature_mse']))
                                
                            # Extract synergy classification metrics
                            if 'predictability/synergy_acc' in row:
                                synergy_acc.append(float(row['predictability/synergy_acc']))
                            if 'predictability/synergy_norm_ce' in row:
                                synergy_norm_ce.append(float(row['predictability/synergy_norm_ce']))
                    
                    # Create reconstruction MSE plot
                    if epochs and (non_synergy_mse or synergy_source_mse):
                        plot_data = {
                            'epochs': epochs,
                            'non_synergy_mse': non_synergy_mse,
                            'synergy_source_mse': synergy_source_mse
                        }
                        
                        create_reconstruction_plot(
                            output_dir=config.output_dir,
                            workspace_dim=workspace_dim,
                            plot_data=plot_data,
                            save_to_wandb=True
                        )
                    
                    # Create synergy classification metrics plot
                    if epochs and (synergy_acc or synergy_norm_ce):
                        synergy_plot_data = {
                            'epochs': epochs,
                            'synergy_acc': synergy_acc,
                            'synergy_norm_ce': synergy_norm_ce
                        }
                        
                        create_synergy_metrics_plot(
                            output_dir=config.output_dir,
                            workspace_dim=workspace_dim,
                            plot_data=synergy_plot_data,
                            save_to_wandb=True
                        )
                        
            except Exception as e:
                logger.warning(f"Failed to create reconstruction plot: {e}")
        
        logger.info(f"Completed workspace_dim = {workspace_dim}")
        logger.info(f"Final validation loss: {result.get('best_val_loss', 'N/A')}")
        
        return result
        
    finally:
        if HAS_WANDB and log_to_wandb:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")
                pass


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Sweep workspace dimensions for synergy prior analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Full sweep
  python experiments/lowdim_sweep.py \\
    --config experiments/synergy/config_base.json \\
    --dims 2,4,8,12,16,24,32,48,64 \\
    --experiment fusion_only \\
    --epochs 50
  
  # Quick test
  python experiments/lowdim_sweep.py \\
    --config config.json \\
    --dims 8,12 \\
    --quick \\
    --no-wandb
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to base configuration JSON file"
    )
    parser.add_argument(
        "--dims", 
        type=str, 
        default="2,4,8,12,16,24,32,48,64",
        help="Comma-separated list of workspace dimensions to test"
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="fusion_only",
        help="Experiment type to run (default: fusion_only)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Number of epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Batch size (default: from config)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--gpu-memory-percent", 
        type=float, 
        default=75.0,
        help="GPU memory limit as percentage (default: 75.0)"
    )
    parser.add_argument(
        "--no-wandb", 
        action="store_true",
        help="Disable W&B logging"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Quick test mode (2 epochs, limited dims)"
    )
    parser.add_argument(
        "--synergy-loss-scale", 
        type=float, 
        default=None,
        help="Override synergy loss scale from config"
    )
    
    args = parser.parse_args()
    
    # Set GPU memory fraction if specified
    if args.gpu_memory_percent and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_percent / 100.0)
        logger.info(f"Set GPU memory limit to {args.gpu_memory_percent}%")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Load base configuration
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    base_config = SynergyExperimentConfig.from_file(args.config)
    logger.info(f"Loaded base config from {args.config}")
    
    # Override synergy loss scale if provided
    if args.synergy_loss_scale is not None:
        base_config.synergy_config['loss_scale'] = args.synergy_loss_scale
        logger.info(f"Override synergy loss scale: {args.synergy_loss_scale}")
    
    # Debug: Show what loss scale will be used
    current_loss_scale = base_config.synergy_config.get('loss_scale', 1.0)
    logger.info(f"Current synergy loss scale in config: {current_loss_scale}")
    
    # Parse dimensions
    if args.quick:
        dims = [8, 12]  # Quick test dimensions
        epochs = 2
        logger.info("Quick mode: using dims=[8,12] and epochs=2")
    else:
        dims = [int(d.strip()) for d in args.dims.split(',')]
        epochs = args.epochs if args.epochs else base_config.training.get('epochs', 50)
    
    logger.info(f"Testing dimensions: {dims}")
    logger.info(f"Using {epochs} epochs per run")
    logger.info(f"Experiment type: {args.experiment}")
    
    # Check W&B availability
    log_to_wandb = HAS_WANDB and not args.no_wandb
    if not HAS_WANDB and not args.no_wandb:
        logger.warning("W&B not available, disabling logging")
    
    # Run sweep
    results = []
    
    for i, workspace_dim in enumerate(dims):
        logger.info(f"\n--- Run {i+1}/{len(dims)} ---")
        
        try:
            result = run_single_dimension(
                base_config=base_config,
                workspace_dim=workspace_dim,
                experiment_type=args.experiment,
                epochs=epochs,
                seed=args.seed,
                device=device,
                log_to_wandb=log_to_wandb
            )
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed workspace_dim={workspace_dim}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall results
    output_dir = Path(base_config.output_dir) / 'lowdim_sweep_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f'sweep_results_{args.experiment}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Sweep completed! Results saved to {results_file}")
    logger.info(f"Successfully completed {len(results)}/{len(dims)} runs")
    
    # Print summary
    logger.info("\nSummary:")
    for result in results:
        logger.info(f"  dim {result['workspace_dim']:2d}: val_loss={result.get('best_val_loss', 'N/A')}")


if __name__ == "__main__":
    main()
