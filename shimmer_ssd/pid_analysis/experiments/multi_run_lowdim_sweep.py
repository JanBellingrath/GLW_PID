#!/usr/bin/env python3
"""
Multi-Run Low Dimensionality Sweep with Statistical Analysis

This script provides a comprehensive wrapper around lowdim_sweep.py to:
1. Run multiple repetitions per dimension for statistical reliability
2. Sweep across different synergy loss scale values
3. Aggregate results with proper statistical analysis (means, std devs, confidence intervals)
4. Generate comprehensive visualizations showing trends across dimensions and synergy scales

Features:
- Robust error handling and recovery mechanisms
- Comprehensive logging and progress monitoring
- Statistical analysis with confidence intervals
- Multiple visualization modes (per-dimension, per-synergy-scale)
- Efficient parallel execution options
- Checkpointing and resume functionality

Usage:
    python experiments/multi_run_lowdim_sweep.py \
        --config config.json \
        --dims 8,12,16,24,32 \
        --synergy-scales 0.1,0.5,1.0,2.0,5.0,10.0 \
        --runs-per-condition 5 \
        --output-dir experiments/multi_run_analysis
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import subprocess
import time
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from scipy import stats
import warnings

# Set up path for imports
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

# Import the original lowdim_sweep for configuration
from experiments.lowdim_sweep import SynergyExperimentConfig

# Try importing wandb for data extraction
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Set up logging with rich formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multi_run_sweep.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress excessive warnings
warnings.filterwarnings('ignore', category=UserWarning)
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')


class ExperimentCondition:
    """Represents a single experimental condition (dim + synergy_scale)."""
    
    def __init__(self, workspace_dim: int, synergy_scale: float, run_id: int):
        self.workspace_dim = workspace_dim
        self.synergy_scale = synergy_scale
        self.run_id = run_id
        self.seed = 42 + run_id  # Deterministic but different seeds
        
    def __str__(self):
        return f"dim_{self.workspace_dim}_scale_{self.synergy_scale:.1f}_run_{self.run_id}"
    
    def __repr__(self):
        return self.__str__()


class MetricsAggregator:
    """Handles aggregation and statistical analysis of metrics across runs."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        
    def aggregate_metrics(
        self, 
        metrics_list: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across multiple runs.
        
        Args:
            metrics_list: List of metric dictionaries from individual runs
            
        Returns:
            Dictionary with aggregated statistics for each metric
        """
        if not metrics_list:
            return {}
            
        # Collect all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        aggregated = {}
        
        for key in all_keys:
            values = []
            for metrics in metrics_list:
                if key in metrics and metrics[key] is not None:
                    try:
                        val = float(metrics[key])
                        if not (np.isnan(val) or np.isinf(val)):
                            values.append(val)
                    except (ValueError, TypeError):
                        continue
            
            if len(values) >= 2:  # Need at least 2 values for meaningful stats
                values = np.array(values)
                
                # Calculate statistics
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)  # Sample standard deviation
                n = len(values)
                
                # Calculate confidence interval using t-distribution
                t_critical = stats.t.ppf(1 - self.alpha/2, df=n-1)
                margin_error = t_critical * (std_val / np.sqrt(n))
                ci_lower = mean_val - margin_error
                ci_upper = mean_val + margin_error
                
                aggregated[key] = {
                    'mean': mean_val,
                    'std': std_val,
                    'n': n,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'sem': std_val / np.sqrt(n),  # Standard error of the mean
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
            elif len(values) == 1:
                # Single value - no statistics possible
                aggregated[key] = {
                    'mean': values[0],
                    'std': 0.0,
                    'n': 1,
                    'ci_lower': values[0],
                    'ci_upper': values[0],
                    'sem': 0.0,
                    'min': values[0],
                    'max': values[0],
                    'median': values[0]
                }
        
        return aggregated
    
    def aggregate_time_series(
        self,
        time_series_list: List[Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Aggregate time series data (e.g., training curves) across runs.
        
        Args:
            time_series_list: List of time series dictionaries
            
        Returns:
            Aggregated time series with mean, std, and confidence intervals
        """
        if not time_series_list:
            return {}
        
        # Find common keys and maximum length
        all_keys = set()
        for ts in time_series_list:
            all_keys.update(ts.keys())
        
        aggregated = {}
        
        for key in all_keys:
            # Collect all series for this key
            series_list = []
            max_length = 0
            
            for ts in time_series_list:
                if key in ts and ts[key]:
                    series = np.array(ts[key])
                    if len(series) > 0:
                        series_list.append(series)
                        max_length = max(max_length, len(series))
            
            if not series_list:
                continue
            
            # Pad series to same length (forward fill)
            padded_series = []
            for series in series_list:
                if len(series) < max_length:
                    # Forward fill the last value
                    padded = np.concatenate([series, np.full(max_length - len(series), series[-1])])
                else:
                    padded = series[:max_length]
                padded_series.append(padded)
            
            # Convert to numpy array
            data_matrix = np.array(padded_series)  # Shape: (n_runs, n_epochs)
            
            # Calculate statistics across runs (axis=0)
            mean_curve = np.mean(data_matrix, axis=0)
            std_curve = np.std(data_matrix, axis=0, ddof=1) if data_matrix.shape[0] > 1 else np.zeros_like(mean_curve)
            n_runs = data_matrix.shape[0]
            
            # Calculate confidence intervals
            if n_runs > 1:
                t_critical = stats.t.ppf(1 - self.alpha/2, df=n_runs-1)
                sem_curve = std_curve / np.sqrt(n_runs)
                margin_curve = t_critical * sem_curve
                ci_lower_curve = mean_curve - margin_curve
                ci_upper_curve = mean_curve + margin_curve
            else:
                ci_lower_curve = mean_curve
                ci_upper_curve = mean_curve
            
            aggregated[key] = {
                'mean': mean_curve,
                'std': std_curve,
                'ci_lower': ci_lower_curve,
                'ci_upper': ci_upper_curve,
                'n_runs': n_runs,
                'epochs': np.arange(len(mean_curve))
            }
        
        return aggregated


class MultiRunSweepRunner:
    """Main runner for multi-run dimensionality sweeps."""
    
    def __init__(
        self,
        config_path: str,
        output_dir: str,
        dims: List[int],
        synergy_scales: List[float],
        runs_per_condition: int,
        experiment_type: str = "fusion_only",
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        parallel_jobs: int = 1,
        resume: bool = True
    ):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.dims = sorted(dims)
        self.synergy_scales = sorted(synergy_scales)
        self.runs_per_condition = runs_per_condition
        self.experiment_type = experiment_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.parallel_jobs = min(parallel_jobs, mp.cpu_count())
        self.resume = resume
        
        # Initialize components
        self.aggregator = MetricsAggregator()
        self.conditions = self._generate_conditions()
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.output_dir / 'individual_results'
        self.results_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # State tracking
        self.state_file = self.output_dir / 'sweep_state.pkl'
        self.state = self._load_state()
        
        logger.info(f"Initialized MultiRunSweepRunner:")
        logger.info(f"  Dimensions: {self.dims}")
        logger.info(f"  Synergy scales: {self.synergy_scales}")
        logger.info(f"  Runs per condition: {self.runs_per_condition}")
        logger.info(f"  Total conditions: {len(self.conditions)}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Parallel jobs: {self.parallel_jobs}")
    
    def _generate_conditions(self) -> List[ExperimentCondition]:
        """Generate all experimental conditions."""
        conditions = []
        for dim in self.dims:
            for scale in self.synergy_scales:
                for run_id in range(self.runs_per_condition):
                    conditions.append(ExperimentCondition(dim, scale, run_id))
        return conditions
    
    def _load_state(self) -> Dict[str, Any]:
        """Load previous state for resuming."""
        if self.resume and self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                logger.info(f"Loaded previous state: {len(state.get('completed', []))} completed runs")
                return state
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        
        return {
            'completed': [],
            'failed': [],
            'results': {},
            'start_time': datetime.now().isoformat()
        }
    
    def _save_state(self):
        """Save current state."""
        try:
            with open(self.state_file, 'wb') as f:
                pickle.dump(self.state, f)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _run_single_condition(self, condition: ExperimentCondition) -> Tuple[ExperimentCondition, Optional[Dict[str, Any]]]:
        """
        Run a single experimental condition.
        
        Args:
            condition: The experimental condition to run
            
        Returns:
            Tuple of (condition, result_dict or None if failed)
        """
        condition_id = str(condition)
        
        # Check if already completed
        if condition_id in self.state['completed']:
            logger.info(f"Skipping already completed condition: {condition_id}")
            return condition, self.state['results'].get(condition_id)
        
        logger.info(f"Running condition: {condition_id}")
        
        try:
            # Set output directory for this specific run
            run_output_dir = self.results_dir / condition_id
            run_output_dir.mkdir(exist_ok=True)
            
            # Create a temporary config file with modified output directory
            temp_config_path = run_output_dir / 'temp_config.json'
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Modify the output directory to be unique for this run
            if 'experiment' not in config_data:
                config_data['experiment'] = {}
            config_data['experiment']['output_dir'] = str(run_output_dir / 'synergy_experiment')
            
            # Save temporary config
            with open(temp_config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Build command for lowdim_sweep.py
            cmd = [
                sys.executable, 
                str(script_dir / 'experiments' / 'lowdim_sweep.py'),
                '--config', str(temp_config_path),  # Use the temporary config
                '--dims', str(condition.workspace_dim),
                '--experiment', self.experiment_type,
                '--seed', str(condition.seed),
                '--synergy-loss-scale', str(condition.synergy_scale)
            ]
            
            if self.epochs:
                cmd.extend(['--epochs', str(self.epochs)])
            if self.batch_size:
                cmd.extend(['--batch-size', str(self.batch_size)])
            if self.device:
                cmd.extend(['--device', self.device])
            
            # Add no-wandb flag to avoid conflicts
            cmd.append('--no-wandb')
            
            # Set environment variables for the subprocess
            env = os.environ.copy()
            env['PYTHONPATH'] = str(script_dir) + ':' + env.get('PYTHONPATH', '')
            
            # Run the command
            logger.debug(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,

                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env=env,
                cwd=str(script_dir)
            )
            
            if result.returncode != 0:
                logger.error(f"Command failed for {condition_id}")
                logger.error(f"Stdout: {result.stdout}")
                logger.error(f"Stderr: {result.stderr}")
                return condition, None
            
            # Parse result from the lowdim_sweep output
            # Look for the results JSON file in the run-specific directory
            results_pattern = run_output_dir / 'synergy_experiment' / 'lowdim_sweep_results' / f"sweep_results_{self.experiment_type}.json"
            
            if results_pattern.exists():
                with open(results_pattern, 'r') as f:
                    sweep_results = json.load(f)
                
                # Extract the result for our specific dimension
                result_data = None
                for res in sweep_results:
                    if res.get('workspace_dim') == condition.workspace_dim:
                        result_data = res
                        break
                
                if result_data:
                    # Add condition metadata
                    result_data.update({
                        'workspace_dim': condition.workspace_dim,
                        'synergy_scale': condition.synergy_scale,
                        'run_id': condition.run_id,
                        'seed': condition.seed,
                        'condition_id': condition_id
                    })
                    
                    logger.info(f"Successfully completed {condition_id}")
                    return condition, result_data
                else:
                    logger.error(f"No result found for dimension {condition.workspace_dim} in {results_pattern}")
                    return condition, None
            else:
                logger.error(f"Results file not found: {results_pattern}")
                return condition, None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout for condition: {condition_id}")
            return condition, None
        except Exception as e:
            logger.error(f"Error running condition {condition_id}: {e}")
            return condition, None
    
    def run_all_conditions(self):
        """Run all experimental conditions."""
        logger.info(f"Starting sweep with {len(self.conditions)} total conditions")
        
        # Filter out already completed conditions
        pending_conditions = [
            c for c in self.conditions 
            if str(c) not in self.state['completed']
        ]
        
        if not pending_conditions:
            logger.info("All conditions already completed!")
            return
        
        logger.info(f"Running {len(pending_conditions)} pending conditions")
        
        # Run conditions
        if self.parallel_jobs > 1:
            self._run_parallel(pending_conditions)
        else:
            self._run_sequential(pending_conditions)
        
        # Save final state
        self._save_state()
        
        logger.info(f"Completed sweep. Successful: {len(self.state['completed'])}, Failed: {len(self.state['failed'])}")
    
    def _run_sequential(self, conditions: List[ExperimentCondition]):
        """Run conditions sequentially."""
        for i, condition in enumerate(conditions):
            logger.info(f"Running condition {i+1}/{len(conditions)}: {condition}")
            
            condition, result = self._run_single_condition(condition)
            self._update_state(condition, result)
            
            # Save state periodically
            if (i + 1) % 5 == 0:
                self._save_state()
    
    def _run_parallel(self, conditions: List[ExperimentCondition]):
        """Run conditions in parallel."""
        logger.info(f"Running {len(conditions)} conditions with {self.parallel_jobs} parallel jobs")
        
        with ProcessPoolExecutor(max_workers=self.parallel_jobs) as executor:
            # Submit all jobs
            future_to_condition = {
                executor.submit(self._run_single_condition, condition): condition
                for condition in conditions
            }
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_condition)):
                condition, result = future.result()
                self._update_state(condition, result)
                
                logger.info(f"Completed {i+1}/{len(conditions)} conditions")
                
                # Save state periodically
                if (i + 1) % 5 == 0:
                    self._save_state()
    
    def _update_state(self, condition: ExperimentCondition, result: Optional[Dict[str, Any]]):
        """Update state tracking."""
        condition_id = str(condition)
        
        if result is not None:
            self.state['completed'].append(condition_id)
            self.state['results'][condition_id] = result
        else:
            self.state['failed'].append(condition_id)
    
    def aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all completed runs."""
        logger.info("Aggregating results across all runs...")
        
        if not self.state['results']:
            logger.warning("No results to aggregate!")
            return {}
        
        # Group results by (dimension, synergy_scale)
        grouped_results = defaultdict(list)
        
        for condition_id, result in self.state['results'].items():
            dim = result['workspace_dim']
            scale = result['synergy_scale']
            key = (dim, scale)
            grouped_results[key].append(result)
        
        # Aggregate each group
        aggregated_results = {}
        
        for (dim, scale), results_list in grouped_results.items():
            logger.info(f"Aggregating {len(results_list)} runs for dim={dim}, scale={scale}")
            
            # Aggregate final metrics
            final_metrics = [r for r in results_list if r is not None]
            aggregated_metrics = self.aggregator.aggregate_metrics(final_metrics)
            
            aggregated_results[(dim, scale)] = {
                'workspace_dim': dim,
                'synergy_scale': scale,
                'n_runs': len(results_list),
                'aggregated_metrics': aggregated_metrics,
                'individual_results': results_list
            }
        
        # Save aggregated results
        output_file = self.output_dir / 'aggregated_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in aggregated_results.items():
            key_str = f"dim_{key[0]}_scale_{key[1]:.1f}"
            serializable_value = {}
            
            for k, v in value.items():
                if k == 'aggregated_metrics':
                    # Convert numpy values to Python types
                    serializable_metrics = {}
                    for metric_name, metric_stats in v.items():
                        serializable_stats = {}
                        for stat_name, stat_value in metric_stats.items():
                            if isinstance(stat_value, np.ndarray):
                                serializable_stats[stat_name] = stat_value.tolist()
                            elif isinstance(stat_value, (np.float32, np.float64)):
                                serializable_stats[stat_name] = float(stat_value)
                            elif isinstance(stat_value, (np.int32, np.int64)):
                                serializable_stats[stat_name] = int(stat_value)
                            else:
                                serializable_stats[stat_name] = stat_value
                        serializable_metrics[metric_name] = serializable_stats
                    serializable_value[k] = serializable_metrics
                else:
                    serializable_value[k] = v
            
            serializable_results[key_str] = serializable_value
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved aggregated results to {output_file}")
        
        return aggregated_results
    
    def create_visualizations(self, aggregated_results: Dict[str, Any]):
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        # Create different types of plots
        self._plot_metrics_vs_dimensions(aggregated_results)
        self._plot_metrics_vs_synergy_scale(aggregated_results)
        self._plot_heatmaps(aggregated_results)
        self._plot_statistical_summary(aggregated_results)
        
        logger.info(f"All plots saved to {self.plots_dir}")
    
    def _plot_metrics_vs_dimensions(self, aggregated_results: Dict[str, Any]):
        """Plot metrics as a function of workspace dimensions, grouped by synergy scale."""
        
        # Extract key metrics to plot
        key_metrics = [
            'best_val_loss',
            'predictability/synergy_acc',
            'predictability/synergy_norm_ce'
        ]
        
        for metric in key_metrics:
            # Check if metric exists in results
            has_metric = any(
                metric in result.get('aggregated_metrics', {})
                for result in aggregated_results.values()
            )
            
            if not has_metric:
                logger.warning(f"Metric {metric} not found in results, skipping plot")
                continue
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Group by synergy scale
            for scale in self.synergy_scales:
                dims_for_scale = []
                means = []
                stds = []
                cis_lower = []
                cis_upper = []
                
                for dim in self.dims:
                    key = (dim, scale)
                    if key in aggregated_results:
                        result = aggregated_results[key]
                        metrics = result.get('aggregated_metrics', {})
                        
                        if metric in metrics:
                            stats = metrics[metric]
                            dims_for_scale.append(dim)
                            means.append(stats['mean'])
                            stds.append(stats['std'])
                            cis_lower.append(stats['ci_lower'])
                            cis_upper.append(stats['ci_upper'])
                
                if dims_for_scale:
                    # Plot mean line
                    ax.plot(dims_for_scale, means, 'o-', label=f'Synergy Scale {scale:.1f}', linewidth=2, markersize=6)
                    
                    # Add confidence interval
                    ax.fill_between(dims_for_scale, cis_lower, cis_upper, alpha=0.2)
            
            ax.set_xlabel('Workspace Dimension', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Workspace Dimension', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.plots_dir / f'{metric.replace("/", "_")}_vs_dimensions.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved plot: {plot_path}")
    
    def _plot_metrics_vs_synergy_scale(self, aggregated_results: Dict[str, Any]):
        """Plot metrics as a function of synergy scale, grouped by dimension."""
        
        key_metrics = [
            'best_val_loss',
            'predictability/synergy_acc',
            'predictability/synergy_norm_ce'
        ]
        
        for metric in key_metrics:
            # Check if metric exists
            has_metric = any(
                metric in result.get('aggregated_metrics', {})
                for result in aggregated_results.values()
            )
            
            if not has_metric:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Group by dimension
            for dim in self.dims:
                scales_for_dim = []
                means = []
                stds = []
                cis_lower = []
                cis_upper = []
                
                for scale in self.synergy_scales:
                    key = (dim, scale)
                    if key in aggregated_results:
                        result = aggregated_results[key]
                        metrics = result.get('aggregated_metrics', {})
                        
                        if metric in metrics:
                            stats = metrics[metric]
                            scales_for_dim.append(scale)
                            means.append(stats['mean'])
                            stds.append(stats['std'])
                            cis_lower.append(stats['ci_lower'])
                            cis_upper.append(stats['ci_upper'])
                
                if scales_for_dim:
                    # Plot mean line
                    ax.plot(scales_for_dim, means, 'o-', label=f'Dim {dim}', linewidth=2, markersize=6)
                    
                    # Add confidence interval
                    ax.fill_between(scales_for_dim, cis_lower, cis_upper, alpha=0.2)
            
            ax.set_xlabel('Synergy Loss Scale', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Synergy Loss Scale', fontsize=14, fontweight='bold')
            ax.set_xscale('log')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.plots_dir / f'{metric.replace("/", "_")}_vs_synergy_scale.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved plot: {plot_path}")
    
    def _plot_heatmaps(self, aggregated_results: Dict[str, Any]):
        """Create heatmaps showing metrics across dimension and synergy scale space."""
        
        key_metrics = [
            'best_val_loss',
            'predictability/synergy_acc',
            'predictability/synergy_norm_ce'
        ]
        
        for metric in key_metrics:
            # Check if metric exists
            has_metric = any(
                metric in result.get('aggregated_metrics', {})
                for result in aggregated_results.values()
            )
            
            if not has_metric:
                continue
            
            # Create data matrix
            data_matrix = np.full((len(self.synergy_scales), len(self.dims)), np.nan)
            
            for i, scale in enumerate(self.synergy_scales):
                for j, dim in enumerate(self.dims):
                    key = (dim, scale)
                    if key in aggregated_results:
                        result = aggregated_results[key]
                        metrics = result.get('aggregated_metrics', {})
                        
                        if metric in metrics:
                            data_matrix[i, j] = metrics[metric]['mean']
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(data_matrix, aspect='auto', cmap='viridis', origin='lower')
            
            # Set ticks and labels
            ax.set_xticks(range(len(self.dims)))
            ax.set_xticklabels(self.dims)
            ax.set_yticks(range(len(self.synergy_scales)))
            ax.set_yticklabels([f'{s:.1f}' for s in self.synergy_scales])
            
            ax.set_xlabel('Workspace Dimension', fontsize=12)
            ax.set_ylabel('Synergy Loss Scale', fontsize=12)
            ax.set_title(f'Heatmap: {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric.replace('_', ' ').title(), fontsize=12)
            
            # Add value annotations
            for i in range(len(self.synergy_scales)):
                for j in range(len(self.dims)):
                    if not np.isnan(data_matrix[i, j]):
                        text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                                     ha="center", va="center", color="white", fontsize=8)
            
            plt.tight_layout()
            plot_path = self.plots_dir / f'heatmap_{metric.replace("/", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved heatmap: {plot_path}")
    
    def _plot_statistical_summary(self, aggregated_results: Dict[str, Any]):
        """Create summary plots showing statistical properties."""
        
        # Plot number of successful runs per condition
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Success rate heatmap
        success_matrix = np.zeros((len(self.synergy_scales), len(self.dims)))
        
        for i, scale in enumerate(self.synergy_scales):
            for j, dim in enumerate(self.dims):
                key = (dim, scale)
                if key in aggregated_results:
                    n_runs = aggregated_results[key]['n_runs']
                    success_rate = n_runs / self.runs_per_condition
                    success_matrix[i, j] = success_rate
        
        im1 = ax1.imshow(success_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
        ax1.set_xticks(range(len(self.dims)))
        ax1.set_xticklabels(self.dims)
        ax1.set_yticks(range(len(self.synergy_scales)))
        ax1.set_yticklabels([f'{s:.1f}' for s in self.synergy_scales])
        ax1.set_xlabel('Workspace Dimension')
        ax1.set_ylabel('Synergy Loss Scale')
        ax1.set_title('Success Rate (Fraction of Completed Runs)')
        
        # Add annotations
        for i in range(len(self.synergy_scales)):
            for j in range(len(self.dims)):
                ax1.text(j, i, f'{success_matrix[i, j]:.2f}',
                        ha="center", va="center", fontsize=8)
        
        plt.colorbar(im1, ax=ax1)
        
        # Overall statistics
        total_conditions = len(self.conditions)
        completed_conditions = len(self.state['completed'])
        failed_conditions = len(self.state['failed'])
        
        labels = ['Completed', 'Failed', 'Pending']
        sizes = [completed_conditions, failed_conditions, total_conditions - completed_conditions - failed_conditions]
        colors = ['lightgreen', 'lightcoral', 'lightgray']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Overall Completion Status')
        
        plt.tight_layout()
        plot_path = self.plots_dir / 'statistical_summary.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved statistical summary: {plot_path}")


def parse_number_list(s: str) -> List[float]:
    """Parse comma-separated list of numbers."""
    return [float(x.strip()) for x in s.split(',')]


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Multi-run dimensionality sweep with statistical analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic sweep with 3 runs per condition
  python experiments/multi_run_lowdim_sweep.py \\
    --config config.json \\
    --dims 8,12,16,24,32 \\
    --synergy-scales 0.1,0.5,1.0,2.0,5.0,10.0 \\
    --runs-per-condition 3 \\
    --output-dir experiments/multi_run_analysis
  
  # Quick test with parallel execution
  python experiments/multi_run_lowdim_sweep.py \\
    --config config.json \\
    --dims 8,12 \\
    --synergy-scales 1.0,2.0 \\
    --runs-per-condition 2 \\
    --parallel-jobs 4 \\
    --epochs 10
  
  # Resume interrupted sweep
  python experiments/multi_run_lowdim_sweep.py \\
    --config config.json \\
    --dims 8,12,16,24,32 \\
    --synergy-scales 0.1,0.5,1.0,2.0,5.0,10.0 \\
    --runs-per-condition 5 \\
    --output-dir experiments/multi_run_analysis \\
    --resume
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
        default="8,12,16,24,32",
        help="Comma-separated list of workspace dimensions to test"
    )
    parser.add_argument(
        "--synergy-scales",
        type=str,
        default="0.1,0.5,1.0,2.0,5.0,10.0",
        help="Comma-separated list of synergy loss scale values"
    )
    parser.add_argument(
        "--runs-per-condition",
        type=int,
        default=5,
        help="Number of runs per (dimension, synergy_scale) condition"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="fusion_only",
        help="Experiment type to run"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs per run"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1 for sequential)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from previous state"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip running experiments, only analyze existing results"
    )
    
    args = parser.parse_args()
    
    # Parse lists
    dims = [int(x) for x in parse_number_list(args.dims)]
    synergy_scales = parse_number_list(args.synergy_scales)
    
    # Validate inputs
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    if args.runs_per_condition < 1:
        logger.error("--runs-per-condition must be >= 1")
        return 1
    
    # Initialize runner
    runner = MultiRunSweepRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        dims=dims,
        synergy_scales=synergy_scales,
        runs_per_condition=args.runs_per_condition,
        experiment_type=args.experiment_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        parallel_jobs=args.parallel_jobs,
        resume=not args.no_resume
    )
    
    try:
        # Run experiments unless analyze-only mode
        if not args.analyze_only:
            runner.run_all_conditions()
        
        # Aggregate results and create visualizations
        aggregated_results = runner.aggregate_results()
        
        if aggregated_results:
            runner.create_visualizations(aggregated_results)
            
            logger.info("="*70)
            logger.info("MULTI-RUN SWEEP COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            logger.info(f"Total conditions: {len(runner.conditions)}")
            logger.info(f"Completed: {len(runner.state['completed'])}")
            logger.info(f"Failed: {len(runner.state['failed'])}")
            logger.info(f"Results saved to: {runner.output_dir}")
            logger.info(f"Plots saved to: {runner.plots_dir}")
        else:
            logger.warning("No results to analyze!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user. State has been saved.")
        return 1
    except Exception as e:
        logger.error(f"Error during sweep: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
