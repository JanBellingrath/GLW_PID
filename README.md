# GLW_PID: Global Workspace Partial Information Decomposition Analysis

Framework for analyzing information flow in Global Workspace neural architectures using Partial Information Decomposition (PID) and Sinkhorn-Knopp optimal transport.

## Overview

Global Workspace models integrate information from multiple domains (visual, textual, etc.) through shared latent representations. This repository implements PID analysis to decompose mutual information between domains into four components:

- **Redundancy**: Information shared by both input domains about the target
- **Unique Information A**: Information exclusively from domain A
- **Unique Information B**: Information exclusively from domain B  
- **Synergy**: Information only available when both domains are combined

This framework builds on the theoretical foundations established by [Liang et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/575286a73f238b6516ce0467d67eadb2-Abstract-Conference.html), who proposed information-theoretic approaches for quantifying multimodal interactions using PID statistics.

## Key Features

- PID analysis with discriminator-based estimation
- Sinkhorn-Knopp coupling matrix visualization with wandb integration
- Support for synthetic Boolean functions and real multimodal data
- Automatic cluster generation and validation
- Real-time coupling matrix tracking during training

## Repository Structure

```
GLW_PID/
├── shimmer_ssd/
│   └── pid_analysis/           # Core PID analysis framework
│       ├── main.py            # Command-line interface and entry point
│       ├── models.py          # Neural network architectures (CEAlignment, Discriminators)
│       ├── eval.py            # Model evaluation and PID computation
│       ├── train.py           # Training procedures for discriminators
│       ├── sinkhorn.py        # Sinkhorn-Knopp algorithm with wandb visualization
│       ├── coupling_visualization.py  # Coupling matrix visualization functions
│       ├── data_interface.py  # Unified data handling interface
│       ├── synthetic_data.py  # Boolean function synthesis for validation
│       ├── cluster_visualization_validation.py  # Cluster validation tools
│       ├── utils.py           # Utility functions and configurations
│       ├── cluster_optimization_gpu_v2.py  # GPU-optimized clustering
│       ├── losses_and_weights_GLW_training.py  # Training loss implementations
│       ├── sweep_pid_discrim.py  # Hyperparameter sweeps for discriminators
│       ├── domain_v_config.json  # Visual domain configuration
│       └── domain_t_config.json  # Text domain configuration
├── analyze_pid_new.py         # Legacy PID analysis script
└── run_pid_analysis.py        # Quick analysis launcher
```

## Installation

Requirements:
- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)

```bash
git clone https://github.com/JanBellingrath/GLW_PID.git
cd GLW_PID

pip install torch torchvision torchaudio
pip install numpy matplotlib seaborn scikit-learn
pip install wandb lightning
```

## Usage

### Model Analysis
```bash
python -m shimmer_ssd.pid_analysis.main model \
  --single-model \
  --model-path path/to/model.ckpt \
  --domain-configs shimmer_ssd/pid_analysis/domain_v_config.json shimmer_ssd/pid_analysis/domain_t_config.json \
  --target-config "gw_rep" \
  --source-config '{"v_latents":"v_latents_latent","t":"t_latent"}' \
  --output-dir results \
  --n-samples 10000 \
  --num-clusters 20 \
  --wandb \
  --wandb-project "pid-analysis"
```

### Synthetic Validation
```bash
python -m shimmer_ssd.pid_analysis.main synthetic \
  --functions and xor or \
  --output-dir synthetic_results \
  --compare-theoretical \
  --wandb
```

### Sinkhorn Visualization
```python
from shimmer_ssd.pid_analysis.coupling_visualization import log_sinkhorn_coupling
log_sinkhorn_coupling(coupling_matrix, step=iteration, prefix="training")
```

## Analysis Modes

### Model-Based
- Analyze trained Global Workspace models
- Extract latent representations from multiple domains
- Compute PID components using discriminator networks
- Visualize coupling matrices and cluster evolution

### Synthetic 
- Test PID estimation on known Boolean functions
- Compare empirical results with theoretical values
- Validate methodology with ground-truth information

### File-Based
- Process pre-saved domain representations
- Flexible input for custom data formats
- Batch analysis of multiple datasets

## Configuration

Domain parameters are specified in JSON files:
```json
{
  "domain_name": "v_latents",
  "encoder_type": "pretrained",
  "hidden_dim": 512,
  "output_dim": 256
}
```

Wandb logging includes:
- PID component values over time
- Coupling matrix visualizations
- Training metrics and convergence plots
- Cluster validation results

## Implementation Details

- Mixed precision training with automatic FP16 support
- Gradient checkpointing for memory-efficient Sinkhorn computation
- Chunked processing for large datasets
- GPU memory management with cleanup options

## Validation

The framework includes validation against theoretical PID values for Boolean functions (AND, XOR, OR) and cluster meaningfulness validation with visual data.

## Citation

If you use this code, please cite both this repository and the foundational theoretical work:

```bibtex
@software{bellingrath2024glw_pid,
  title={GLW_PID: Global Workspace Partial Information Decomposition Analysis},
  author={Bellingrath, Jan},
  year={2024},
  url={https://github.com/JanBellingrath/GLW_PID},
  institution={CNRS, Cerco, Toulouse}
}

@inproceedings{liang2023quantifying,
  title={Quantifying \& Modeling Multimodal Interactions: An Information Decomposition Framework},
  author={Liang, Paul Pu and Cheng, Yun and Fan, Xiang and Ling, Chun Kai and Nie, Suzanne and Chen, Richard and Deng, Zihao and Allen, Nicholas and Auerbach, Randy and Mahmood, Faisal and Salakhutdinov, Ruslan and Morency, Louis-Philippe},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023},
  url={https://proceedings.neurips.cc/paper_files/paper/2023/hash/575286a73f238b6516ce0467d67eadb2-Abstract-Conference.html}
}
```

## Contact

Jan Bellingrath  
Computational Neuroscience × Artificial Intelligence  
Cerco (CNRS), Toulouse 