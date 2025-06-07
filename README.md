# GLW_PID: Global Workspace Partial Information Decomposition Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org)
[![Wandb](https://img.shields.io/badge/Wandb-Integration-yellow.svg)](https://wandb.ai)

A comprehensive framework for analyzing information flow in Global Workspace neural architectures using **Partial Information Decomposition (PID)** and **Sinkhorn-Knopp optimal transport**. This repository provides tools for understanding how information is redundantly shared, uniquely processed, and synergistically combined across different modalities in multimodal neural networks.

## 🎯 Overview

Global Workspace models integrate information from multiple domains (visual, textual, etc.) through shared latent representations. This repository implements PID analysis to decompose the mutual information between domains into interpretable components:

- **🔄 Redundancy**: Information shared by both input domains about the target
- **🅰️ Unique Information A**: Information exclusively from domain A
- **🅱️ Unique Information B**: Information exclusively from domain B  
- **⚡ Synergy**: Information only available when both domains are combined

## 🚀 Key Features

### 📊 **Professional Sinkhorn Coupling Visualization**
- Real-time coupling matrix visualization during training
- Professional heatmaps with statistical analysis
- Wandb integration for experiment tracking
- Memory-efficient visualization for large matrices

### 🧠 **Advanced PID Analysis**
- Complete PID framework with discriminator-based estimation
- Support for both synthetic Boolean functions and real multimodal data
- Automatic cluster generation and validation
- Extensible to arbitrary numbers of domains

### 🎨 **Comprehensive Visualization Suite**
- Cluster distribution validation with real data
- Training evolution tracking
- Marginal distribution analysis
- Professional publication-ready plots

## 📁 Repository Structure

```
GLW_PID/
├── shimmer_ssd/
│   └── pid_analysis/           # Core PID analysis framework
│       ├── main.py            # Command-line interface and entry point
│       ├── models.py          # Neural network architectures (CEAlignment, Discriminators)
│       ├── eval.py            # Model evaluation and PID computation
│       ├── train.py           # Training procedures for discriminators
│       ├── sinkhorn.py        # Sinkhorn-Knopp algorithm with wandb visualization
│       ├── coupling_visualization.py  # Professional coupling matrix visualization
│       ├── data_interface.py  # Unified data handling interface
│       ├── synthetic_data.py  # Boolean function synthesis for validation
│       ├── cluster_visualization_validation.py  # Cluster validation tools
│       ├── utils.py           # Utility functions and configurations
│       ├── domain_v_config.json  # Visual domain configuration
│       └── domain_t_config.json  # Text domain configuration
├── analyze_pid_new.py         # Legacy PID analysis script
├── cluster_optimization_gpu_v2.py  # GPU-optimized clustering
├── losses_and_weights_GLW_training.py  # Training loss implementations
├── sweep_pid_discrim.py       # Hyperparameter sweeps for discriminators
└── run_pid_analysis.py        # Quick analysis launcher
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/JanBellingrath/GLW_PID.git
cd GLW_PID

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib seaborn scikit-learn
pip install wandb  # For experiment tracking
pip install lightning  # For model training

# Optional: Install for development
pip install pytest black flake8
```

## 🎮 Quick Start

### 1. **Analyze a Trained Model**
```bash
python -m shimmer_ssd.pid_analysis.main model \
  --single-model \
  --model-path path/to/your/model.ckpt \
  --domain-configs shimmer_ssd/pid_analysis/domain_v_config.json shimmer_ssd/pid_analysis/domain_t_config.json \
  --target-config "gw_rep" \
  --source-config '{"v_latents":"v_latents_latent","t":"t_latent"}' \
  --output-dir results \
  --n-samples 10000 \
  --num-clusters 20 \
  --wandb \
  --wandb-project "pid-analysis"
```

### 2. **Validate with Boolean Functions**
```bash
python -m shimmer_ssd.pid_analysis.main synthetic \
  --functions and xor or \
  --output-dir synthetic_results \
  --compare-theoretical \
  --wandb \
  --wandb-project "pid-synthetic"
```

### 3. **One-Liner Sinkhorn Visualization**
```python
from shimmer_ssd.pid_analysis.coupling_visualization import log_sinkhorn_coupling

# In your training loop:
log_sinkhorn_coupling(coupling_matrix, step=iteration, prefix="training")
```

## 📊 Analysis Modes

### **Model-Based Analysis**
- Analyze trained Global Workspace models
- Extract latent representations from multiple domains
- Compute PID components using discriminator networks
- Visualize coupling matrices and cluster evolution

### **Synthetic Validation**
- Test PID estimation on known Boolean functions
- Compare empirical results with theoretical values
- Validate methodology with ground-truth information

### **File-Based Analysis**
- Process pre-saved domain representations
- Flexible input for custom data formats
- Batch analysis of multiple datasets

## 🎨 Visualization Features

### **Sinkhorn Coupling Matrices**
- **Real-time heatmaps** showing optimal transport plans
- **Statistical tracking**: entropy, sparsity, diagonal dominance
- **Evolution visualization** over training iterations
- **Memory-efficient** handling of large matrices

### **Cluster Validation**
- **Professional cluster visualization** with real data samples
- **Distribution analysis** across domains
- **Quality metrics** for cluster meaningfulness
- **Publication-ready plots** with customizable styling

## 🔧 Configuration

### **Domain Configuration**
Configure domain-specific parameters in JSON files:
```json
{
  "domain_name": "v_latents",
  "encoder_type": "pretrained",
  "hidden_dim": 512,
  "output_dim": 256
}
```

### **Wandb Integration**
Automatic experiment tracking includes:
- PID component values over time
- Coupling matrix visualizations
- Training metrics and convergence plots
- Cluster validation results

## 📈 Performance Optimizations

- **Mixed Precision Training**: Automatic FP16 support
- **Gradient Checkpointing**: Memory-efficient Sinkhorn computation
- **Chunked Processing**: Scalable to large datasets
- **GPU Memory Management**: Aggressive cleanup options

## 🧪 Testing & Validation

### **Theoretical Validation**
The framework includes validation against theoretical PID values for Boolean functions:
- **Perfect accuracy** on known cases (AND, XOR, OR)
- **Convergence testing** with different hyperparameters
- **Robustness analysis** across various configurations

### **Real Data Validation**
- **Cluster meaningfulness** validation with visual data
- **Cross-modal consistency** checks
- **Statistical significance** testing

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{bellingrath2024glw_pid,
  title={GLW_PID: Global Workspace Partial Information Decomposition Analysis},
  author={Bellingrath, Jan},
  year={2024},
  url={https://github.com/JanBellingrath/GLW_PID},
  institution={CNRS, Cerco, Toulouse}
}
```

## 📧 Contact

**Jan Bellingrath**  
Computational Neuroscience × Artificial Intelligence  
Cerco (CNRS), Toulouse  

- GitHub: [@JanBellingrath](https://github.com/JanBellingrath)
- Email: [Your Email]

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sinkhorn Algorithm**: Implementation based on optimal transport theory
- **PID Framework**: Inspired by Williams & Beer (2010) information decomposition
- **Global Workspace Theory**: Following Baars (1988) and Dehaene et al. (2017)
- **Wandb Integration**: Professional experiment tracking and visualization

---

*This repository is part of ongoing research into information processing in Global Workspace neural architectures. For questions about the theoretical background or implementation details, please open an issue or contact the authors.* 