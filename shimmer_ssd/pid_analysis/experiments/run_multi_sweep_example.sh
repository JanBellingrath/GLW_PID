#!/bin/bash
# 
# Example script to run multi-run dimensionality sweep with statistical analysis
#
# This script demonstrates how to use the multi_run_lowdim_sweep.py wrapper
# to conduct comprehensive experiments with statistical reliability.
#

set -e  # Exit on any error

# Configuration
CONFIG_FILE="experiments/config_multi_run_template.json"
OUTPUT_BASE_DIR="experiments/multi_run_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE_DIR}/sweep_${TIMESTAMP}"

# Experimental parameters
DIMS="8,12,16,24,32"
SYNERGY_SCALES="0.1,0.5,1.0,2.0,5.0,10.0"
RUNS_PER_CONDITION=5
EPOCHS=50
BATCH_SIZE=32

# Computational parameters
PARALLEL_JOBS=2  # Adjust based on your system
GPU_MEMORY_PERCENT=75.0

echo "=========================================="
echo "Multi-Run Dimensionality Sweep"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Dimensions: ${DIMS}"
echo "Synergy scales: ${SYNERGY_SCALES}"
echo "Runs per condition: ${RUNS_PER_CONDITION}"
echo "Epochs per run: ${EPOCHS}"
echo "Parallel jobs: ${PARALLEL_JOBS}"
echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Copy configuration for provenance
cp "${CONFIG_FILE}" "${OUTPUT_DIR}/base_config.json"

# Log system information
echo "System Information:" > "${OUTPUT_DIR}/system_info.txt"
echo "Date: $(date)" >> "${OUTPUT_DIR}/system_info.txt"
echo "User: $(whoami)" >> "${OUTPUT_DIR}/system_info.txt"
echo "Host: $(hostname)" >> "${OUTPUT_DIR}/system_info.txt"
echo "Working directory: $(pwd)" >> "${OUTPUT_DIR}/system_info.txt"
echo "Python version: $(python --version)" >> "${OUTPUT_DIR}/system_info.txt"
echo "GPU info:" >> "${OUTPUT_DIR}/system_info.txt"
nvidia-smi >> "${OUTPUT_DIR}/system_info.txt" 2>/dev/null || echo "No NVIDIA GPU detected" >> "${OUTPUT_DIR}/system_info.txt"

# Run the multi-run sweep
echo "Starting multi-run sweep..."
python experiments/multi_run_lowdim_sweep.py \
    --config "${CONFIG_FILE}" \
    --dims "${DIMS}" \
    --synergy-scales "${SYNERGY_SCALES}" \
    --runs-per-condition ${RUNS_PER_CONDITION} \
    --output-dir "${OUTPUT_DIR}" \
    --experiment-type fusion_only \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --parallel-jobs ${PARALLEL_JOBS} \
    2>&1 | tee "${OUTPUT_DIR}/sweep_log.txt"

# Check if sweep completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Multi-run sweep completed successfully!"
    echo "=========================================="
    echo "Results saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Generated files:"
    find "${OUTPUT_DIR}" -type f -name "*.png" -o -name "*.json" | sort
    echo ""
    echo "To view results:"
    echo "  - Check plots in: ${OUTPUT_DIR}/plots/"
    echo "  - Check aggregated data: ${OUTPUT_DIR}/aggregated_results.json"
    echo "  - Check individual results: ${OUTPUT_DIR}/individual_results/"
    echo ""
    
    # Generate summary statistics
    echo "Generating summary report..."
    python -c "
import json
import numpy as np
from pathlib import Path

# Load aggregated results
results_file = Path('${OUTPUT_DIR}') / 'aggregated_results.json'
if results_file.exists():
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print('Summary Statistics:')
    print('==================')
    
    total_conditions = len(data)
    print(f'Total conditions: {total_conditions}')
    
    # Extract completion rates
    completion_rates = []
    for condition_name, condition_data in data.items():
        n_runs = condition_data.get('n_runs', 0)
        completion_rate = n_runs / ${RUNS_PER_CONDITION}
        completion_rates.append(completion_rate)
    
    avg_completion = np.mean(completion_rates)
    print(f'Average completion rate: {avg_completion:.2%}')
    
    # Best validation losses
    val_losses = []
    for condition_name, condition_data in data.items():
        metrics = condition_data.get('aggregated_metrics', {})
        if 'best_val_loss' in metrics:
            val_loss_mean = metrics['best_val_loss'].get('mean')
            if val_loss_mean is not None:
                val_losses.append(val_loss_mean)
    
    if val_losses:
        print(f'Validation loss range: {min(val_losses):.4f} - {max(val_losses):.4f}')
        print(f'Mean validation loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}')
    
    print('')
    print('Individual condition results:')
    print('============================')
    for condition_name, condition_data in sorted(data.items()):
        dim = condition_data.get('workspace_dim', 'Unknown')
        scale = condition_data.get('synergy_scale', 'Unknown')
        n_runs = condition_data.get('n_runs', 0)
        
        metrics = condition_data.get('aggregated_metrics', {})
        val_loss_stats = metrics.get('best_val_loss', {})
        val_loss_mean = val_loss_stats.get('mean', 'N/A')
        val_loss_std = val_loss_stats.get('std', 'N/A')
        
        print(f'  Dim {dim:2d}, Scale {scale:4.1f}: {n_runs}/{RUNS_PER_CONDITION} runs, Val Loss: {val_loss_mean} ± {val_loss_std}')
else:
    print('No aggregated results file found.')
"
    
else
    echo "=========================================="
    echo "Multi-run sweep failed!"
    echo "=========================================="
    echo "Check the log file for details: ${OUTPUT_DIR}/sweep_log.txt"
    exit 1
fi

# Optional: Extract W&B data if available
if command -v wandb &> /dev/null; then
    echo ""
    echo "W&B is available. You can extract additional training curves with:"
    echo "python experiments/wandb_data_extractor.py \\"
    echo "    --project synergy-bottleneck-sweep-2025 \\"
    echo "    --group lowdim-synergy-prior \\"
    echo "    --output-dir ${OUTPUT_DIR}/wandb_data"
fi

echo ""
echo "Sweep completed at: $(date)"
