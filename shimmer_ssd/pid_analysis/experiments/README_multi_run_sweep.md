# Multi-Run Dimensionality Sweep with Statistical Analysis

This package provides a comprehensive framework for conducting statistically rigorous experiments on the synergy prior hypothesis in Global Workspace models. It wraps the original `lowdim_sweep.py` to enable multiple runs per condition and systematic variation of synergy loss scales.

## Overview

The framework addresses the key challenge of experimental noise in synergy measurements by:

1. **Multiple Repetitions**: Run n experiments per (dimension, synergy_scale) condition
2. **Statistical Analysis**: Compute means, standard deviations, and confidence intervals
3. **Comprehensive Visualization**: Generate plots showing trends across dimensions and synergy scales
4. **Robust Infrastructure**: Error handling, progress tracking, and resumable execution

## Key Features

### 🔬 Experimental Design
- **Multi-factorial experiments**: Workspace dimension × Synergy loss scale × Multiple runs
- **Statistical reliability**: Configurable number of runs per condition (default: 5)
- **Systematic parameter sweeps**: Test synergy loss scales from 0.1 to 10.0
- **Controlled randomness**: Deterministic but different seeds per run

### 📊 Statistical Analysis
- **Confidence intervals**: 95% confidence intervals using t-distribution
- **Robust aggregation**: Handles missing data and failed runs gracefully
- **Time-series analysis**: Training curve aggregation with statistical bounds
- **Multiple metrics**: Validation loss, synergy accuracy, cross-entropy, MSE metrics

### 📈 Comprehensive Visualization
- **Dimension trends**: Metrics vs. workspace dimensions (grouped by synergy scale)
- **Scale sensitivity**: Metrics vs. synergy loss scale (grouped by dimension)
- **Heatmaps**: 2D visualization of metric landscape
- **Training curves**: Aggregated learning curves with confidence bands
- **Statistical summaries**: Completion rates and experimental overview

### 🛠 Production-Ready Infrastructure
- **Parallel execution**: Configurable parallel jobs for faster completion
- **Resume capability**: Automatically resume interrupted experiments
- **Error handling**: Robust error recovery and detailed logging
- **Progress monitoring**: Real-time progress tracking and estimation
- **Data provenance**: Complete experimental metadata and configuration logging

## Files Overview

```
experiments/
├── multi_run_lowdim_sweep.py      # Main wrapper script
├── wandb_data_extractor.py        # W&B data extraction for detailed analysis
├── config_multi_run_template.json # Configuration template
├── run_multi_sweep_example.sh     # Example execution script
└── README_multi_run_sweep.md      # This documentation
```

## Quick Start

### 1. Basic Usage

```bash
# Run a comprehensive sweep
python experiments/multi_run_lowdim_sweep.py \
    --config config.json \
    --dims 8,12,16,24,32 \
    --synergy-scales 0.1,0.5,1.0,2.0,5.0,10.0 \
    --runs-per-condition 5 \
    --output-dir experiments/my_sweep_results
```

### 2. Quick Test

```bash
# Quick test with fewer conditions
python experiments/multi_run_lowdim_sweep.py \
    --config config.json \
    --dims 8,12 \
    --synergy-scales 1.0,2.0 \
    --runs-per-condition 2 \
    --epochs 10 \
    --output-dir experiments/test_sweep
```

### 3. Parallel Execution

```bash
# Use parallel execution for faster completion
python experiments/multi_run_lowdim_sweep.py \
    --config config.json \
    --dims 8,12,16,24,32 \
    --synergy-scales 0.1,0.5,1.0,2.0,5.0,10.0 \
    --runs-per-condition 5 \
    --parallel-jobs 4 \
    --output-dir experiments/parallel_sweep
```

### 4. Resume Interrupted Sweep

```bash
# Resume from where you left off
python experiments/multi_run_lowdim_sweep.py \
    --config config.json \
    --dims 8,12,16,24,32 \
    --synergy-scales 0.1,0.5,1.0,2.0,5.0,10.0 \
    --runs-per-condition 5 \
    --output-dir experiments/my_sweep_results  # Same directory
```

## Configuration

### Base Configuration

Create a configuration file based on `config_multi_run_template.json`:

```json
{
  "data": {
    "dir": "data/simple_shapes_xor_targets",
    "batch_size": 32,
    "num_workers": 4
  },
  "model": {
    "domains": [...],
    "workspace_dim": 12,
    "hidden_dim": 128,
    "decoder_hidden_dim": 128,
    "n_layers": 4
  },
  "synergy": {
    "domains": ["attr", "v"],
    "feature_indices": {
      "attr": ["xor_target_normalized"]
    },
    "loss_scale": 1.0,
    "unique_indices": {...},
    "redundant_indices": {...}
  },
  "training": {
    "epochs": 50,
    "optimizer": {...},
    "loss_configs": [...]
  }
}
```

### Key Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--dims` | Workspace dimensions to test | "8,12,16,24,32" | 5-7 values |
| `--synergy-scales` | Synergy loss scale values | "0.1,0.5,1.0,2.0,5.0,10.0" | Log-spaced |
| `--runs-per-condition` | Runs per (dim, scale) pair | 5 | 3-10 |
| `--parallel-jobs` | Parallel execution | 1 | 2-4 |
| `--epochs` | Training epochs per run | From config | 50+ |

## Output Structure

```
output_dir/
├── aggregated_results.json        # Main aggregated results
├── sweep_state.pkl                # Resume state (binary)
├── individual_results/             # Individual run results
│   ├── dim_8_scale_1.0_run_0/
│   ├── dim_8_scale_1.0_run_1/
│   └── ...
└── plots/                          # Generated visualizations
    ├── best_val_loss_vs_dimensions.png
    ├── predictability_synergy_acc_vs_synergy_scale.png
    ├── heatmap_best_val_loss.png
    └── statistical_summary.png
```

## Advanced Usage

### W&B Data Extraction

Extract detailed training curves from W&B runs:

```bash
python experiments/wandb_data_extractor.py \
    --project synergy-bottleneck-sweep-2025 \
    --group lowdim-synergy-prior \
    --output-dir extracted_wandb_data
```

### Analysis Only Mode

Analyze existing results without re-running experiments:

```bash
python experiments/multi_run_lowdim_sweep.py \
    --config config.json \
    --dims 8,12,16,24,32 \
    --synergy-scales 0.1,0.5,1.0,2.0,5.0,10.0 \
    --runs-per-condition 5 \
    --output-dir experiments/existing_sweep \
    --analyze-only
```

## Statistical Analysis Details

### Metrics Computed

For each condition (dimension, synergy_scale), the system computes:

- **Mean**: Average across runs
- **Standard deviation**: Sample standard deviation (ddof=1)
- **Confidence interval**: 95% CI using t-distribution
- **Standard error**: SEM = std / sqrt(n)
- **Min/Max/Median**: Distribution summary

### Key Metrics Tracked

1. **Validation Loss** (`best_val_loss`): Overall model performance
2. **Synergy Accuracy** (`predictability/synergy_acc`): Discrete synergy prediction accuracy
3. **Synergy Cross-Entropy** (`predictability/synergy_norm_ce`): Normalized synergy prediction loss
4. **Reconstruction MSE**: Separate tracking for synergy vs. non-synergy features
5. **Training Curves**: Complete time-series for all metrics

### Statistical Robustness

- **Missing data handling**: Graceful handling of failed runs
- **Outlier detection**: Automatic exclusion of invalid values (NaN, inf)
- **Sample size adaptation**: Appropriate statistical tests based on sample size
- **Multiple comparisons**: Consider Bonferroni correction for multiple testing

## Visualization Gallery

### 1. Dimension Trends
Shows how metrics change with workspace dimension, with separate lines for each synergy scale. Includes confidence intervals to show statistical uncertainty.

### 2. Scale Sensitivity  
Demonstrates the effect of synergy loss scaling, with separate lines for each dimension. Log-scale x-axis to show wide range of scale values.

### 3. Heatmaps
2D visualization of the metric landscape across (dimension, synergy_scale) space. Useful for identifying optimal parameter regions.

### 4. Training Curves
Aggregated learning curves showing how metrics evolve during training, with confidence bands indicating uncertainty across runs.

### 5. Statistical Summary
Overview plots showing completion rates, success statistics, and experimental coverage.

## Performance Considerations

### Computational Requirements

- **Single run time**: 5-30 minutes depending on epochs and data size
- **Total experiment time**: (n_dims × n_scales × n_runs) × single_run_time
- **Memory**: GPU memory scales with batch size and model size
- **Storage**: ~10-100 MB per run depending on logging verbosity

### Scaling Guidelines

| Scale | Dims | Scales | Runs/Condition | Total Runs | Est. Time |
|-------|------|--------|----------------|------------|-----------|
| Quick test | 2 | 2 | 2 | 8 | 1-2 hours |
| Medium | 5 | 4 | 3 | 60 | 5-10 hours |
| Full | 7 | 6 | 5 | 210 | 1-2 days |
| Comprehensive | 10 | 10 | 10 | 1000 | 1 week |

### Optimization Tips

1. **Parallel execution**: Use `--parallel-jobs` based on available GPUs/CPUs
2. **Batch size tuning**: Larger batches for faster training (if memory allows)
3. **Early stopping**: Implement early stopping for faster convergence
4. **Resume capability**: Use resume feature for long experiments
5. **GPU memory management**: Monitor GPU memory with `nvidia-smi`

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce batch size or use gradient accumulation
2. **Failed runs**: Check individual run logs in `individual_results/`
3. **Missing metrics**: Verify metric names in original lowdim_sweep.py
4. **W&B conflicts**: Use `--no-wandb` flag if W&B causes issues
5. **Permission errors**: Check write permissions on output directory

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Inspection

Check individual run results:

```bash
# View individual run output
cat output_dir/individual_results/dim_8_scale_1.0_run_0/config.json
cat output_dir/individual_results/dim_8_scale_1.0_run_0/results.json
```

## Integration with Existing Workflow

### With Original lowdim_sweep.py

The wrapper is designed as a drop-in enhancement:

- **Same configuration format**: Uses existing config files
- **Same command-line interface**: Similar argument structure  
- **Same output format**: Compatible result structure
- **Same dependencies**: No additional requirements

### With Analysis Pipeline

Results integrate with downstream analysis:

```python
import json
import pandas as pd

# Load aggregated results
with open('output_dir/aggregated_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame for analysis
data = []
for condition_name, condition_data in results.items():
    row = {
        'workspace_dim': condition_data['workspace_dim'],
        'synergy_scale': condition_data['synergy_scale'],
        'n_runs': condition_data['n_runs']
    }
    
    # Add aggregated metrics
    for metric_name, metric_stats in condition_data['aggregated_metrics'].items():
        row[f'{metric_name}_mean'] = metric_stats['mean']
        row[f'{metric_name}_std'] = metric_stats['std']
        row[f'{metric_name}_ci_lower'] = metric_stats['ci_lower']
        row[f'{metric_name}_ci_upper'] = metric_stats['ci_upper']
    
    data.append(row)

df = pd.DataFrame(data)
print(df.describe())
```

## Example Complete Workflow

```bash
# 1. Prepare configuration
cp experiments/config_multi_run_template.json my_config.json
# Edit my_config.json with your data paths and model configurations

# 2. Run comprehensive sweep  
python experiments/multi_run_lowdim_sweep.py \
    --config my_config.json \
    --dims 8,12,16,24,32,48,64 \
    --synergy-scales 0.1,0.3,1.0,3.0,10.0 \
    --runs-per-condition 5 \
    --parallel-jobs 2 \
    --output-dir experiments/synergy_hypothesis_test

# 3. Extract additional W&B data (optional)
python experiments/wandb_data_extractor.py \
    --project synergy-bottleneck-sweep-2025 \
    --group lowdim-synergy-prior \
    --output-dir experiments/synergy_hypothesis_test/wandb_data

# 4. View results
ls experiments/synergy_hypothesis_test/plots/
open experiments/synergy_hypothesis_test/plots/best_val_loss_vs_dimensions.png

# 5. Analyze programmatically
python -c "
import json
with open('experiments/synergy_hypothesis_test/aggregated_results.json', 'r') as f:
    results = json.load(f)
print(f'Analyzed {len(results)} conditions')
print('Available metrics:', list(list(results.values())[0]['aggregated_metrics'].keys()))
"
```

## Contributing

### Adding New Metrics

To track additional metrics:

1. Ensure they're logged in the original training script
2. Add metric names to the visualization functions
3. Update the statistical analysis if needed

### Extending Visualizations

Create custom plot functions:

```python
def create_custom_plot(aggregated_results, output_dir):
    # Your custom visualization logic
    pass

# Add to MultiRunSweepRunner.create_visualizations()
```

### Performance Improvements

- **Caching**: Add result caching for expensive computations
- **Incremental updates**: Update only changed conditions
- **Database backend**: Replace JSON with database for large experiments
- **Distributed computing**: Extend to cluster/cloud execution

## References and Related Work

- Original lowdim_sweep.py: Core dimensionality sweep functionality
- Global Workspace Theory: Theoretical foundation
- Synergy Prior Hypothesis: Core experimental hypothesis
- Statistical Analysis: Best practices for experimental design

---

*For questions or issues, please check the troubleshooting section or examine the detailed logging output.*
