#!/usr/bin/env python3
"""
W&B Data Extractor for Multi-Run Analysis

This script extracts detailed training metrics from W&B runs for comprehensive analysis.
It can be used in conjunction with multi_run_lowdim_sweep.py to get time-series data
that might not be captured in the final results JSON.

Features:
- Extract training curves from W&B runs
- Process and aggregate time-series data across multiple runs
- Create detailed training curve visualizations
- Export data for external analysis

Usage:
    python experiments/wandb_data_extractor.py \
        --project synergy-bottleneck-sweep-2025 \
        --output-dir extracted_data \
        --group lowdim-synergy-prior
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import defaultdict
import warnings

# Try importing wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Cannot extract W&B data.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')


class WandBDataExtractor:
    """Extracts and processes data from W&B runs."""
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        output_dir: str = "extracted_wandb_data"
    ):
        self.project = project
        self.entity = entity
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_WANDB:
            raise ImportError("wandb is required for data extraction")
        
        # Initialize W&B API
        self.api = wandb.Api()
        
        logger.info(f"Initialized W&B data extractor for project: {project}")
    
    def extract_runs_data(
        self,
        group_filter: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
        state_filter: str = "finished"
    ) -> Dict[str, Any]:
        """
        Extract data from W&B runs.
        
        Args:
            group_filter: Filter runs by group name
            tag_filters: Filter runs by tags
            state_filter: Filter runs by state (finished, running, crashed)
            
        Returns:
            Dictionary containing extracted run data
        """
        logger.info("Extracting runs from W&B...")
        
        # Build filter string
        filters = []
        if group_filter:
            filters.append(f"group:{group_filter}")
        if tag_filters:
            for tag in tag_filters:
                filters.append(f"tags:{tag}")
        if state_filter:
            filters.append(f"state:{state_filter}")
        
        filter_string = " AND ".join(filters) if filters else None
        
        # Get runs
        if self.entity:
            runs_path = f"{self.entity}/{self.project}"
        else:
            runs_path = self.project
        
        runs = self.api.runs(runs_path, filters=filter_string)
        
        logger.info(f"Found {len(runs)} runs matching criteria")
        
        extracted_data = {
            'runs': [],
            'time_series': {},
            'metadata': {
                'project': self.project,
                'entity': self.entity,
                'filter_string': filter_string,
                'extraction_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        for i, run in enumerate(runs):
            logger.info(f"Processing run {i+1}/{len(runs)}: {run.name}")
            
            try:
                # Extract run metadata
                run_data = {
                    'id': run.id,
                    'name': run.name,
                    'state': run.state,
                    'group': run.group,
                    'tags': run.tags,
                    'config': dict(run.config),
                    'summary': dict(run.summary),
                    'created_at': run.created_at.isoformat() if run.created_at else None,
                    'updated_at': run.updated_at.isoformat() if run.updated_at else None
                }
                
                # Extract time series data
                history = run.history()
                if not history.empty:
                    # Convert to serializable format
                    time_series_data = {}
                    for column in history.columns:
                        if column != '_step':  # Skip internal step column
                            values = history[column].dropna().tolist()
                            if values:  # Only include non-empty series
                                time_series_data[column] = values
                    
                    run_data['time_series'] = time_series_data
                    extracted_data['time_series'][run.id] = time_series_data
                
                extracted_data['runs'].append(run_data)
                
            except Exception as e:
                logger.warning(f"Failed to process run {run.name}: {e}")
                continue
        
        logger.info(f"Successfully extracted data from {len(extracted_data['runs'])} runs")
        
        # Save raw extracted data
        output_file = self.output_dir / 'raw_wandb_data.json'
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=2)
        
        logger.info(f"Saved raw data to {output_file}")
        
        return extracted_data
    
    def process_for_multi_run_analysis(
        self,
        extracted_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process extracted data for multi-run analysis compatibility.
        
        Args:
            extracted_data: Data from extract_runs_data()
            
        Returns:
            Processed data structured for aggregation
        """
        logger.info("Processing data for multi-run analysis...")
        
        # Group runs by experimental conditions
        grouped_runs = defaultdict(list)
        
        for run_data in extracted_data['runs']:
            config = run_data.get('config', {})
            
            # Extract condition identifiers
            workspace_dim = config.get('workspace_dim')
            synergy_loss_scale = config.get('synergy_loss_scale', 1.0)
            experiment_type = config.get('experiment_type', 'unknown')
            
            if workspace_dim is not None:
                condition_key = (workspace_dim, synergy_loss_scale, experiment_type)
                grouped_runs[condition_key].append(run_data)
        
        # Process each group
        processed_data = {
            'conditions': {},
            'metadata': extracted_data['metadata']
        }
        
        for condition_key, runs in grouped_runs.items():
            workspace_dim, synergy_scale, experiment_type = condition_key
            
            logger.info(f"Processing {len(runs)} runs for condition: "
                       f"dim={workspace_dim}, scale={synergy_scale}, type={experiment_type}")
            
            # Aggregate time series across runs
            aggregated_series = self._aggregate_time_series([
                run['time_series'] for run in runs if 'time_series' in run
            ])
            
            # Aggregate final metrics
            final_metrics = []
            for run in runs:
                summary = run.get('summary', {})
                if summary:
                    final_metrics.append(summary)
            
            condition_data = {
                'workspace_dim': workspace_dim,
                'synergy_scale': synergy_scale,
                'experiment_type': experiment_type,
                'n_runs': len(runs),
                'aggregated_time_series': aggregated_series,
                'final_metrics': final_metrics,
                'run_ids': [run['id'] for run in runs]
            }
            
            condition_key_str = f"dim_{workspace_dim}_scale_{synergy_scale:.1f}_{experiment_type}"
            processed_data['conditions'][condition_key_str] = condition_data
        
        # Save processed data
        output_file = self.output_dir / 'processed_wandb_data.json'
        
        # Convert numpy arrays for JSON serialization
        serializable_data = self._make_json_serializable(processed_data)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Saved processed data to {output_file}")
        
        return processed_data
    
    def _aggregate_time_series(
        self,
        time_series_list: List[Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate time series data across multiple runs."""
        if not time_series_list:
            return {}
        
        # Find all metric names
        all_metrics = set()
        for ts in time_series_list:
            all_metrics.update(ts.keys())
        
        aggregated = {}
        
        for metric in all_metrics:
            # Collect series for this metric
            series_list = []
            max_length = 0
            
            for ts in time_series_list:
                if metric in ts and ts[metric]:
                    series = np.array(ts[metric])
                    if len(series) > 0:
                        series_list.append(series)
                        max_length = max(max_length, len(series))
            
            if not series_list:
                continue
            
            # Pad series to same length (forward fill)
            padded_series = []
            for series in series_list:
                if len(series) < max_length:
                    padded = np.concatenate([series, np.full(max_length - len(series), series[-1])])
                else:
                    padded = series[:max_length]
                padded_series.append(padded)
            
            # Convert to matrix and compute statistics
            data_matrix = np.array(padded_series)
            
            mean_curve = np.mean(data_matrix, axis=0)
            std_curve = np.std(data_matrix, axis=0, ddof=1) if data_matrix.shape[0] > 1 else np.zeros_like(mean_curve)
            
            # Confidence intervals (95% by default)
            from scipy import stats as scipy_stats
            alpha = 0.05
            n_runs = data_matrix.shape[0]
            
            if n_runs > 1:
                t_critical = scipy_stats.t.ppf(1 - alpha/2, df=n_runs-1)
                sem_curve = std_curve / np.sqrt(n_runs)
                margin_curve = t_critical * sem_curve
                ci_lower = mean_curve - margin_curve
                ci_upper = mean_curve + margin_curve
            else:
                ci_lower = mean_curve
                ci_upper = mean_curve
            
            aggregated[metric] = {
                'mean': mean_curve,
                'std': std_curve,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_runs': n_runs,
                'epochs': np.arange(len(mean_curve))
            }
        
        return aggregated
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-JSON types to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def create_training_curve_plots(
        self,
        processed_data: Dict[str, Any],
        metrics_to_plot: Optional[List[str]] = None
    ):
        """Create comprehensive training curve visualizations."""
        logger.info("Creating training curve plots...")
        
        if metrics_to_plot is None:
            # Default metrics to plot
            metrics_to_plot = [
                'train_loss',
                'val_loss',
                'fusion_attr_non_synergy_features_mse',
                'fusion_attr_synergy_source_feature_mse',
                'predictability/synergy_acc',
                'predictability/synergy_norm_ce'
            ]
        
        plots_dir = self.output_dir / 'training_curves'
        plots_dir.mkdir(exist_ok=True)
        
        conditions = processed_data.get('conditions', {})
        
        # Group conditions by dimension and synergy scale for different plot types
        dims = sorted(set(cond['workspace_dim'] for cond in conditions.values()))
        scales = sorted(set(cond['synergy_scale'] for cond in conditions.values()))
        
        for metric in metrics_to_plot:
            # Check if metric exists in any condition
            has_metric = any(
                metric in cond.get('aggregated_time_series', {})
                for cond in conditions.values()
            )
            
            if not has_metric:
                logger.warning(f"Metric {metric} not found in data, skipping")
                continue
            
            # Plot 1: All conditions on one plot
            self._plot_all_conditions(conditions, metric, plots_dir)
            
            # Plot 2: Grouped by dimension
            self._plot_grouped_by_dimension(conditions, metric, plots_dir, dims, scales)
            
            # Plot 3: Grouped by synergy scale
            self._plot_grouped_by_scale(conditions, metric, plots_dir, dims, scales)
    
    def _plot_all_conditions(
        self,
        conditions: Dict[str, Any],
        metric: str,
        plots_dir: Path
    ):
        """Plot all conditions for a metric on one figure."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for condition_name, condition_data in conditions.items():
            time_series = condition_data.get('aggregated_time_series', {})
            
            if metric not in time_series:
                continue
            
            data = time_series[metric]
            epochs = data['epochs']
            mean_curve = data['mean']
            ci_lower = data['ci_lower']
            ci_upper = data['ci_upper']
            
            # Create label
            dim = condition_data['workspace_dim']
            scale = condition_data['synergy_scale']
            label = f"Dim {dim}, Scale {scale:.1f}"
            
            # Plot mean line
            ax.plot(epochs, mean_curve, label=label, linewidth=2)
            
            # Add confidence interval
            ax.fill_between(epochs, ci_lower, ci_upper, alpha=0.2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Training Curves: {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = plots_dir / f'{metric.replace("/", "_")}_all_conditions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {plot_path}")
    
    def _plot_grouped_by_dimension(
        self,
        conditions: Dict[str, Any],
        metric: str,
        plots_dir: Path,
        dims: List[int],
        scales: List[float]
    ):
        """Create subplots grouped by dimension."""
        n_dims = len(dims)
        n_cols = min(3, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_dims == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, dim in enumerate(dims):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue
            
            for scale in scales:
                # Find condition matching this dim and scale
                matching_conditions = [
                    (name, data) for name, data in conditions.items()
                    if data['workspace_dim'] == dim and abs(data['synergy_scale'] - scale) < 1e-6
                ]
                
                if not matching_conditions:
                    continue
                
                condition_name, condition_data = matching_conditions[0]
                time_series = condition_data.get('aggregated_time_series', {})
                
                if metric not in time_series:
                    continue
                
                data = time_series[metric]
                epochs = data['epochs']
                mean_curve = data['mean']
                ci_lower = data['ci_lower']
                ci_upper = data['ci_upper']
                
                # Plot
                ax.plot(epochs, mean_curve, label=f'Scale {scale:.1f}', linewidth=2)
                ax.fill_between(epochs, ci_lower, ci_upper, alpha=0.2)
            
            ax.set_title(f'Dimension {dim}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(n_dims, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plot_path = plots_dir / f'{metric.replace("/", "_")}_by_dimension.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {plot_path}")
    
    def _plot_grouped_by_scale(
        self,
        conditions: Dict[str, Any],
        metric: str,
        plots_dir: Path,
        dims: List[int],
        scales: List[float]
    ):
        """Create subplots grouped by synergy scale."""
        n_scales = len(scales)
        n_cols = min(3, n_scales)
        n_rows = (n_scales + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_scales == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, scale in enumerate(scales):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue
            
            for dim in dims:
                # Find condition matching this scale and dim
                matching_conditions = [
                    (name, data) for name, data in conditions.items()
                    if abs(data['synergy_scale'] - scale) < 1e-6 and data['workspace_dim'] == dim
                ]
                
                if not matching_conditions:
                    continue
                
                condition_name, condition_data = matching_conditions[0]
                time_series = condition_data.get('aggregated_time_series', {})
                
                if metric not in time_series:
                    continue
                
                data = time_series[metric]
                epochs = data['epochs']
                mean_curve = data['mean']
                ci_lower = data['ci_lower']
                ci_upper = data['ci_upper']
                
                # Plot
                ax.plot(epochs, mean_curve, label=f'Dim {dim}', linewidth=2)
                ax.fill_between(epochs, ci_lower, ci_upper, alpha=0.2)
            
            ax.set_title(f'Synergy Scale {scale:.1f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(n_scales, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plot_path = plots_dir / f'{metric.replace("/", "_")}_by_synergy_scale.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {plot_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract and analyze W&B data for multi-run experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Extract data from specific project and group
  python experiments/wandb_data_extractor.py \\
    --project synergy-bottleneck-sweep-2025 \\
    --group lowdim-synergy-prior \\
    --output-dir extracted_data
  
  # Extract data with specific tags
  python experiments/wandb_data_extractor.py \\
    --project synergy-bottleneck-sweep-2025 \\
    --tags fusion_only dim_syn=8 \\
    --output-dir extracted_data
  
  # Process existing extracted data only
  python experiments/wandb_data_extractor.py \\
    --output-dir extracted_data \\
    --process-only
        """
    )
    
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="W&B project name"
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="W&B entity (username or team)"
    )
    parser.add_argument(
        "--group",
        type=str,
        help="Filter runs by group name"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Filter runs by tags"
    )
    parser.add_argument(
        "--state",
        type=str,
        default="finished",
        choices=["finished", "running", "crashed", "failed"],
        help="Filter runs by state"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted_wandb_data",
        help="Output directory for extracted data"
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only process existing raw data, don't extract new data"
    )
    
    args = parser.parse_args()
    
    if not HAS_WANDB:
        logger.error("wandb is required but not installed")
        return 1
    
    try:
        # Initialize extractor
        extractor = WandBDataExtractor(
            project=args.project,
            entity=args.entity,
            output_dir=args.output_dir
        )
        
        # Extract or load data
        if args.process_only:
            # Load existing raw data
            raw_data_file = Path(args.output_dir) / 'raw_wandb_data.json'
            if not raw_data_file.exists():
                logger.error(f"Raw data file not found: {raw_data_file}")
                return 1
            
            with open(raw_data_file, 'r') as f:
                extracted_data = json.load(f)
            
            logger.info(f"Loaded existing raw data from {raw_data_file}")
        else:
            # Extract new data
            extracted_data = extractor.extract_runs_data(
                group_filter=args.group,
                tag_filters=args.tags,
                state_filter=args.state
            )
        
        # Process for multi-run analysis
        processed_data = extractor.process_for_multi_run_analysis(extracted_data)
        
        # Create training curve plots
        extractor.create_training_curve_plots(processed_data)
        
        logger.info("="*70)
        logger.info("W&B DATA EXTRACTION COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Extracted data from {len(extracted_data['runs'])} runs")
        logger.info(f"Processed {len(processed_data['conditions'])} unique conditions")
        logger.info(f"Results saved to: {extractor.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during data extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
