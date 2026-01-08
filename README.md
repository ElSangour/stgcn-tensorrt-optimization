# GCN_Pytorch2TRT_Optimization
This Repo contains an unofficial Model Optimization for GCNs neural network ( Graph Convolutional Network) to TensorRT.
In this repo we will implement a high-performance ST-GCN (Spatial Temporal Graph Convolutional Networks) implementation optimized for NVIDIA Jetson NX using PyTorch-to-TensorRT conversion.

# ST-GCN PyTorch to TensorRT Optimization for NVIDIA Jetson

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.x-green.svg)](https://developer.nvidia.com/tensorrt)

Production-ready optimization pipeline for deploying Spatial Temporal Graph Convolutional Networks (ST-GCN) on NVIDIA Jetson edge devices. This repository provides comprehensive tools and scripts to convert PyTorch ST-GCN models to optimized TensorRT engines, achieving significant performance improvements for real-time skeleton-based action recognition.

## Project Overview

ST-GCN models are powerful for skeleton-based action recognition but can be computationally intensive for edge deployment. This project addresses deployment challenges on resource-constrained devices by:

- Converting PyTorch models to optimized TensorRT engines
- Supporting FP16 and INT8 quantization for improved inference speed
- Providing benchmarking and validation tools
- Maintaining model accuracy while maximizing throughput

**Target Hardware:** NVIDIA Jetson Xavier NX (16GB RAM)

## Key Features

- **End-to-End Optimization Pipeline**: Complete workflow from PyTorch model to TensorRT engine
- **Multiple Precision Support**: FP32, FP16, and INT8 quantization options
- **Comprehensive Benchmarking**: Performance, accuracy, and power consumption metrics
- **Real-Time Inference Examples**: Ready-to-use scripts for video processing
- **Detailed Documentation**: Step-by-step guides and best practices
- **Community-Driven**: Open for contributions and improvements

## Performance Benchmarks

Performance improvements on NVIDIA Jetson Xavier NX:

| Model Variant | Inference Time | FPS | Memory Usage | Power Draw |
|--------------|---------------|-----|--------------|------------|
| PyTorch FP32 | Coming soon   | -   | -            | -          |
| TensorRT FP16| Coming soon   | -   | -            | -          |
| TensorRT INT8| Coming soon   | -   | -            | -          |

*Benchmarks will be updated as optimization progresses. Tested on JetPack 5.x with batch size 1.*

## Hardware & Software Requirements

### Hardware
- NVIDIA Jetson Xavier NX (16GB RAM recommended)
- Minimum 32GB storage
- Active cooling recommended for sustained workloads

### Software
- JetPack 5.x or 6.x
- Python 3.8+
- PyTorch 2.x (for Jetson)
- TensorRT 8.x
- CUDA 11.x or 12.x
- cuDNN 8.x

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/GCN_Pytorch2TRT_Optimization.git
cd GCN_Pytorch2TRT_Optimization

# Install dependencies
pip3 install -r requirements.txt

# Verify TensorRT installation
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

### Basic Usage
```python
# Coming soon: Basic inference example
```

Detailed usage examples will be added as the project develops.

## Project Structure
```
GCN_Pytorch2TRT_Optimization/
├── models/                 # Model architectures and weights
├── optimization/           # Conversion and optimization scripts
├── utils/                  # Helper functions and tools
├── examples/              # Usage examples and demos
├── benchmarks/            # Performance testing scripts
├── docs/                  # Detailed documentation
└── tests/                 # Unit and integration tests
```

## Roadmap

- [x] Repository initialization
- [x] PyTorch ST-GCN model implementation
- [x] ONNX export pipeline
- [ ] TensorRT FP16 conversion
- [ ] TensorRT INT8 quantization with calibration
- [ ] Comprehensive benchmarking suite
- [ ] Real-time inference examples
- [ ] Documentation and tutorials
- [ ] Community examples and use cases

## Contributing

Contributions are welcome. This project aims to build a community resource for ST-GCN optimization on edge devices.

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original ST-GCN paper: [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455)
- NVIDIA for TensorRT and Jetson platform
- PyTorch community for framework support
- Contributors and community members

## Contact

For questions, suggestions, or collaboration opportunities:
- Open an issue in this repository
- Discussions tab for general questions

## Citation

If you use this project in your research or production, please cite:
```bibtex
@misc{gcn_pytorch2trt,
  author = {Your Name},
  title = {ST-GCN PyTorch to TensorRT Optimization for NVIDIA Jetson},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/GCN_Pytorch2TRT_Optimization}
}
```

---

**Status:** Active Development | **Last Updated:** January 2026
