"""
Hyperparameter search for pretrained encoder models with individual discriminators only.

This script performs a systematic hyperparameter search over:
- num_clusters: Number of clusters for k-means clustering
- num_layers: Number of layers in individual discriminator networks  
- hidden_dim: Hidden dimension of individual discriminator networks

Focus: Only individual discriminators (d1, d2) with cross-entropy loss on k-means clusters.
No PID, no CE alignment, no joint discriminator.

The search uses Weights & Biases sweeps with ASHA early termination and
Bayesian optimization for efficient exploration of the hyperparameter space.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import wandb
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Add the root directory to the path for imports
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Add the shimmer-ssd directory to the path for shimmer_ssd imports
shimmer_ssd_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shimmer-ssd'))
if shimmer_ssd_root not in sys.path:
    sys.path.insert(0, shimmer_ssd_root)

# Import from our modules
from .data_interface import ModelDataProvider, GeneralizedDataInterface
from .utils import load_domain_modules
from .train import create_pretrained_discriminators
from .coupling_visualization import log_discriminator_marginals_to_wandb, _create_marginal_plots

# Import plotting functions
try:
    from .coupling_visualization import (
        log_discriminator_marginals_to_wandb,
        _create_marginal_plots, 
        _create_discriminator_marginal_plots
    )
    HAS_PLOTTING = True
except ImportError as e:
    HAS_PLOTTING = False
    print(f"⚠️  Warning: Plotting functions not available: {e}")

# Set device
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_json_or_file(value: str) -> Dict[str, Any]:
    """Parse a JSON string or load from a JSON file."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if os.path.exists(value):
            with open(value, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Could not parse as JSON string or find file: {value}")


def train_individual_discriminator(model, train_loader, val_loader, num_epochs, device, domain_name):
    """Train a single discriminator with cross-entropy loss and extensive logging."""
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_acc = train_correct / train_total
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                epoch_val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_acc = val_correct / val_total
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # Log to wandb
        wandb.log({
            f"{domain_name}_epoch": epoch,
            f"{domain_name}_train_loss": avg_train_loss,
            f"{domain_name}_train_accuracy": train_acc,
            f"{domain_name}_val_loss": avg_val_loss,
            f"{domain_name}_val_accuracy": val_acc,
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return model, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_acc,
        'final_train_accuracy': train_accuracies[-1],
        'final_val_accuracy': val_accuracies[-1]
    }


def create_training_plots(domain1_metrics, domain2_metrics, config):
    """Create comprehensive training visualization plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = range(len(domain1_metrics['train_losses']))
    
    # Loss plots
    axes[0, 0].plot(epochs, domain1_metrics['train_losses'], 'b-', label='Domain 1 Train')
    axes[0, 0].plot(epochs, domain1_metrics['val_losses'], 'b--', label='Domain 1 Val')
    axes[0, 0].plot(epochs, domain2_metrics['train_losses'], 'r-', label='Domain 2 Train')
    axes[0, 0].plot(epochs, domain2_metrics['val_losses'], 'r--', label='Domain 2 Val')
    axes[0, 0].set_title('Training Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plots
    axes[0, 1].plot(epochs, domain1_metrics['train_accuracies'], 'b-', label='Domain 1 Train')
    axes[0, 1].plot(epochs, domain1_metrics['val_accuracies'], 'b--', label='Domain 1 Val')
    axes[0, 1].plot(epochs, domain2_metrics['train_accuracies'], 'r-', label='Domain 2 Train')
    axes[0, 1].plot(epochs, domain2_metrics['val_accuracies'], 'r--', label='Domain 2 Val')
    axes[0, 1].set_title('Training Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hyperparameter influence bar plot
    hyperparams = ['Clusters', 'Hidden Dim', 'Layers']
    values = [config.num_clusters, config.discrim_hidden_dim, config.discrim_layers]
    axes[0, 2].bar(hyperparams, values, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 2].set_title('Hyperparameter Configuration')
    axes[0, 2].set_ylabel('Values')
    
    # Final performance comparison
    categories = ['Train Acc', 'Val Acc', 'Best Val Acc']
    domain1_perf = [domain1_metrics['final_train_accuracy'], 
                   domain1_metrics['final_val_accuracy'],
                   domain1_metrics['best_val_accuracy']]
    domain2_perf = [domain2_metrics['final_train_accuracy'],
                   domain2_metrics['final_val_accuracy'], 
                   domain2_metrics['best_val_accuracy']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, domain1_perf, width, label='Domain 1', alpha=0.8)
    axes[1, 0].bar(x + width/2, domain2_perf, width, label='Domain 2', alpha=0.8)
    axes[1, 0].set_title('Final Performance Comparison')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting analysis
    domain1_overfit = [train - val for train, val in zip(domain1_metrics['train_accuracies'], 
                                                        domain1_metrics['val_accuracies'])]
    domain2_overfit = [train - val for train, val in zip(domain2_metrics['train_accuracies'],
                                                        domain2_metrics['val_accuracies'])]
    
    axes[1, 1].plot(epochs, domain1_overfit, 'b-', label='Domain 1 Overfit Gap')
    axes[1, 1].plot(epochs, domain2_overfit, 'r-', label='Domain 2 Overfit Gap')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Overfitting Analysis (Train - Val Accuracy)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary text
    axes[1, 2].axis('off')
    summary_text = f"""
Experiment Summary:
• Clusters: {config.num_clusters}
• Hidden Dim: {config.discrim_hidden_dim}
• Layers: {config.discrim_layers}
• Epochs: {len(epochs)}

Domain 1 Best Val Acc: {domain1_metrics['best_val_accuracy']:.3f}
Domain 2 Best Val Acc: {domain2_metrics['best_val_accuracy']:.3f}
Average Best Val Acc: {(domain1_metrics['best_val_accuracy'] + domain2_metrics['best_val_accuracy'])/2:.3f}
"""
    axes[1, 2].text(0.1, 0.7, summary_text, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig


def run_single_experiment():
    """Run a single experiment focusing only on individual discriminators."""
    
    # Initialize wandb run
    run = wandb.init()
    config = wandb.config
    
    print(f"\n🔬 STARTING INDIVIDUAL DISCRIMINATOR EXPERIMENT")
    print("="*60)
    print(f"🎯 Run Name: {run.name}")
    print(f"📊 Configuration:")
    print(f"   ├── num_clusters: {config.num_clusters}")
    print(f"   ├── discrim_layers: {config.discrim_layers}")
    print(f"   └── discrim_hidden_dim: {config.discrim_hidden_dim}")
    print("="*60)
    
    try:
        device = global_device
        print(f"🖥️  Device: {device}")
        
        # Parse configurations from wandb config
        model_path = config.model_path
        domain_configs = config.domain_configs
        
        # Load domain modules
        print("🔧 Loading domain modules...")
        domain_modules = load_domain_modules(domain_configs)
        print(f"✅ Loaded domain modules: {', '.join(list(domain_modules.keys()))}")
        
        # Create output directory for this run
        output_dir = f"{config.base_output_dir}/{run.name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create ModelDataProvider
        provider = ModelDataProvider(
            model_path=model_path,
            domain_modules=domain_modules,
            data_module=None,
            dataset_split="test",
            use_gw_encoded=config.get('use_gw_encoded', False),
            device=device
        )
        
        data_interface = GeneralizedDataInterface(provider)
        
        print("🚀 Generating data...")
        # Generate data from model
        generated_data = data_interface.generate_data(config.get('n_samples', 8000))
        
        # Get domain names
        domain_names = list(domain_modules.keys())
        if len(domain_names) < 2:
            raise ValueError(f"Need at least 2 domains, got {len(domain_names)}")
        
        domain1_name, domain2_name = domain_names[0], domain_names[1]
        
        # Extract features
        x1 = generated_data[domain1_name].to(device)
        x2 = generated_data[domain2_name].to(device)
        
        print(f"📊 Data shapes: {domain1_name}={x1.shape}, {domain2_name}={x2.shape}")
        
        # Perform k-means clustering
        print(f"🎯 Running k-means clustering with {config.num_clusters} clusters...")
        
        # Use the target config data for clustering
        target_data = generated_data[config.target_config].cpu().numpy()
        kmeans = KMeans(n_clusters=config.num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(target_data)
        cluster_labels = torch.tensor(cluster_labels, dtype=torch.long)
        
        print(f"✅ K-means clustering complete. Cluster distribution:")
        unique, counts = torch.unique(cluster_labels, return_counts=True)
        for i, (cluster, count) in enumerate(zip(unique, counts)):
            print(f"   Cluster {cluster}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
        
        # Create train/val split
        n_samples = len(cluster_labels)
        indices = torch.randperm(n_samples)
        train_size = int(0.8 * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create data loaders
        train_dataset1 = TensorDataset(x1[train_indices], cluster_labels[train_indices])
        val_dataset1 = TensorDataset(x1[val_indices], cluster_labels[val_indices])
        train_dataset2 = TensorDataset(x2[train_indices], cluster_labels[train_indices])
        val_dataset2 = TensorDataset(x2[val_indices], cluster_labels[val_indices])
        
        train_loader1 = DataLoader(train_dataset1, batch_size=config.get('batch_size', 128), shuffle=True)
        val_loader1 = DataLoader(val_dataset1, batch_size=config.get('batch_size', 128), shuffle=False)
        train_loader2 = DataLoader(train_dataset2, batch_size=config.get('batch_size', 128), shuffle=True)
        val_loader2 = DataLoader(val_dataset2, batch_size=config.get('batch_size', 128), shuffle=False)
        
        # Load the model for creating discriminators
        print("🔧 Loading model for pretrained encoders...")
        from .utils import load_model
        model = load_model(model_path, domain_modules, device)
        
        # Create pretrained discriminators
        print("🧠 Creating pretrained discriminators...")
        d1, d2, _ = create_pretrained_discriminators(
            x1=x1[:100], x2=x2[:100], labels=cluster_labels[:100],
            num_labels=config.num_clusters,
            model=model,
            domain_names=domain_names,
            discrim_hidden_dim=config.discrim_hidden_dim,
            discrim_layers=config.discrim_layers,
            activation='relu'
        )
        
        # Train discriminators
        print(f"🏋️ Training discriminator 1 ({domain1_name})...")
        d1, domain1_metrics = train_individual_discriminator(
            d1, train_loader1, val_loader1, 
            config.get('discrim_epochs', 30), device, domain1_name
        )
        
        print(f"🏋️ Training discriminator 2 ({domain2_name})...")
        d2, domain2_metrics = train_individual_discriminator(
            d2, train_loader2, val_loader2,
            config.get('discrim_epochs', 30), device, domain2_name
        )
        
        # Create and log comprehensive visualizations
        print("📊 Creating training visualizations...")
        training_fig = create_training_plots(domain1_metrics, domain2_metrics, config)
        wandb.log({"training_curves": wandb.Image(training_fig)})
        plt.close(training_fig)
        
        # Evaluate discriminators and log marginal distributions
        print("📈 Evaluating discriminators and logging marginals...")
        d1.eval()
        d2.eval()
        
        with torch.no_grad():
            # Get predictions on validation set
            val_x1 = x1[val_indices]
            val_x2 = x2[val_indices]
            val_labels = cluster_labels[val_indices]
            
            d1_outputs = d1(val_x1)
            d2_outputs = d2(val_x2)
            
            d1_probs = F.softmax(d1_outputs, dim=1)
            d2_probs = F.softmax(d2_outputs, dim=1)
            
            # Log marginal distributions using our plotting functions
            if HAS_PLOTTING:
                # Create dummy joint for compatibility (not used)
                dummy_joint = torch.zeros_like(d1_probs)
                log_discriminator_marginals_to_wandb(
                    p_y_x1=d1_probs,
                    p_y_x2=d2_probs, 
                    p_y_x1x2=dummy_joint,
                    prefix="discriminator_outputs",
                    cluster_names=[f"Cluster_{i}" for i in range(config.num_clusters)]
                )
        
        # Calculate comprehensive metrics
        final_metrics = {
            'avg_best_val_accuracy': (domain1_metrics['best_val_accuracy'] + domain2_metrics['best_val_accuracy']) / 2,
            'domain1_best_val_accuracy': domain1_metrics['best_val_accuracy'],
            'domain2_best_val_accuracy': domain2_metrics['best_val_accuracy'],
            'avg_final_val_accuracy': (domain1_metrics['final_val_accuracy'] + domain2_metrics['final_val_accuracy']) / 2,
            'performance_gap': abs(domain1_metrics['best_val_accuracy'] - domain2_metrics['best_val_accuracy']),
            'optimization_objective': (domain1_metrics['best_val_accuracy'] + domain2_metrics['best_val_accuracy']) / 2
        }
        
        # Log all metrics
        wandb.log(final_metrics)
        
        print(f"\n✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"📊 Key Results:")
        print(f"   ├── Domain 1 Best Val Acc: {domain1_metrics['best_val_accuracy']:.3f}")
        print(f"   ├── Domain 2 Best Val Acc: {domain2_metrics['best_val_accuracy']:.3f}")
        print(f"   ├── Average Best Val Acc: {final_metrics['avg_best_val_accuracy']:.3f}")
        print(f"   └── Performance Gap: {final_metrics['performance_gap']:.3f}")
        print(f"📁 Results saved to: {output_dir}")
        print("="*60)
        

        
    except Exception as e:
        print(f"❌ EXPERIMENT FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Log failure to wandb
        wandb.log({
            'experiment_status': 'failed',
            'error_message': str(e)
        })
        
        # Mark run as failed
        wandb.run.summary['status'] = 'failed'
        raise
    
    finally:
        # Finish wandb run
        wandb.finish()


def create_sweep_config(args):
    """Create wandb sweep configuration for individual discriminator optimization."""
    
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'optimization_objective',  # Average of best validation accuracies
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            'eta': 2,
            'max_iter': 27
        },
        'parameters': {
            # Hyperparameters to optimize
            'num_clusters': {
                'distribution': 'int_uniform',
                'min': args.min_clusters,
                'max': args.max_clusters
            },
            'discrim_layers': {
                'distribution': 'int_uniform',
                'min': args.min_layers,
                'max': args.max_layers
            },
            'discrim_hidden_dim': {
                'values': args.hidden_dims
            },
            # Fixed parameters
            'model_path': {'value': args.model_path},
            'domain_configs': {'value': args.domain_configs},
            'target_config': {'value': args.target_config},
            'base_output_dir': {'value': args.output_dir},
            'n_samples': {'value': args.n_samples},
            'batch_size': {'value': args.batch_size},
            'discrim_epochs': {'value': args.discrim_epochs},
            'use_gw_encoded': {'value': args.use_gw_encoded}
        }
    }
    
    return sweep_config


def main():
    """Main entry point for individual discriminator hyperparameter search."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for individual discriminators on pretrained encoders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--domain-configs", type=str, nargs="+", required=True,
                       help="Domain configuration JSON strings or paths to JSON files")
    parser.add_argument("--target-config", type=str, default="gw_latent",
                       help="Target representation for clustering")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Base directory to save results")
    
    # Wandb configuration
    parser.add_argument("--wandb-project", type=str, required=True,
                       help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                       help="W&B entity name")
    parser.add_argument("--sweep-name", type=str, default="individual_discriminator_hyperparam_search",
                       help="Name for the wandb sweep")
    
    # Hyperparameter ranges
    parser.add_argument("--min-clusters", type=int, default=5,
                       help="Minimum number of clusters for k-means")
    parser.add_argument("--max-clusters", type=int, default=20,
                       help="Maximum number of clusters for k-means")
    parser.add_argument("--min-layers", type=int, default=2,
                       help="Minimum number of discriminator layers")
    parser.add_argument("--max-layers", type=int, default=8,
                       help="Maximum number of discriminator layers")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 64, 128, 256],
                       help="Hidden dimensions to explore")
    
    # Fixed experiment parameters
    parser.add_argument("--n-samples", type=int, default=8000,
                       help="Number of samples to generate for analysis")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for training")
    parser.add_argument("--discrim-epochs", type=int, default=30,
                       help="Number of epochs to train discriminators")
    parser.add_argument("--use-gw-encoded", action="store_true",
                       help="Use GW-encoded vectors instead of raw latents")
    
    # Sweep execution
    parser.add_argument("--count", type=int, default=50,
                       help="Number of experiments to run in the sweep")
    parser.add_argument("--agent-only", action="store_true",
                       help="Run agent only (sweep must already exist)")
    parser.add_argument("--sweep-id", type=str,
                       help="Existing sweep ID to join (for --agent-only)")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔍 INDIVIDUAL DISCRIMINATOR HYPERPARAMETER SEARCH")
    print("="*70)
    print(f"🤖 Model: {args.model_path}")
    print(f"🎯 Target: {args.target_config}")
    print(f"📊 Search Space:")
    print(f"   ├── Clusters: {args.min_clusters}-{args.max_clusters}")
    print(f"   ├── Layers: {args.min_layers}-{args.max_layers}")
    print(f"   └── Hidden Dims: {args.hidden_dims}")
    print(f"🔄 Experiments: {args.count}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🎯 Focus: Individual discriminators with k-means + cross-entropy")
    print("="*70 + "\n")
    
    # Parse configurations
    domain_configs = []
    for config_str in args.domain_configs:
        domain_configs.append(parse_json_or_file(config_str))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.agent_only:
        if not args.sweep_id:
            print("❌ --sweep-id is required when using --agent-only")
            return
        
        print(f"🔗 Joining existing sweep: {args.sweep_id}")
        wandb.agent(args.sweep_id, function=run_single_experiment, count=args.count)
    else:
        # Create new sweep
        print("🚀 Creating new wandb sweep...")
        
        # Update args with parsed configs for sweep config
        args.domain_configs = domain_configs
        
        sweep_config = create_sweep_config(args)
        
        print(f"📋 Sweep Configuration:")
        print(f"   ├── Method: {sweep_config['method']}")
        print(f"   ├── Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
        print(f"   ├── Early Termination: {sweep_config['early_terminate']['type']}")
        print(f"   ├── Focus: Individual discriminators only")
        print(f"   ├── Loss: Cross-entropy on k-means clusters")
        print(f"   └── Parameters: {len(sweep_config['parameters'])} total")
        
        # Initialize sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.sweep_name
        )
        
        print(f"✅ Created sweep: {sweep_id}")
        print(f"🌐 Sweep URL: https://wandb.ai/{args.wandb_entity or 'your-entity'}/{args.wandb_project}/sweeps/{sweep_id}")
        print(f"\n🏃 Starting {args.count} experiments...")
        
        # Run sweep
        wandb.agent(sweep_id, function=run_single_experiment, count=args.count)
    
    print(f"\n🎉 INDIVIDUAL DISCRIMINATOR SEARCH COMPLETE!")
    print(f"📊 Check results at: https://wandb.ai/{args.wandb_entity or 'your-entity'}/{args.wandb_project}")
    print(f"🎯 Focus was on individual discriminators with k-means clustering")
    print("="*70 + "\n")


if __name__ == "__main__":
    main() 