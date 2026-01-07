#!/usr/bin/env python3
"""
Test script for ST-GCN model with Torch-TensorRT optimization.
This script compares the performance of the original PyTorch model 
with a TensorRT-optimized version.
"""

import torch
import torch_tensorrt
import numpy as np
import time
import argparse
import sys
import os
from typing import Tuple, Dict

# Add the project root to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wisevision-shoplifting-jetson'))

try:
    from wisevision-shoplifting-jetson.net.st_gcn_twostream import Model as STGCN_Model
    from wisevision-shoplifting-jetson.real_time_detection.config import load_config
except ImportError as e:
    print(f"[ERROR] Failed to import modules: {e}")
    print("Please ensure you're in the correct directory and the project structure is intact.")
    sys.exit(1)


class TensorRTOptimizer:
    """Handles TensorRT optimization for ST-GCN models."""
    
    def __init__(self, precision: str = "fp16"):
        """
        Initialize TensorRT optimizer.
        
        Args:
            precision: Precision mode ("fp16", "fp32", "int8")
        """
        self.precision = precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TensorRT optimization")
    
    def optimize_model(self, model: torch.nn.Module, input_shape: Tuple[int, ...]) -> torch.nn.Module:
        """
        Optimize a PyTorch model using TensorRT.
        
        Args:
            model: PyTorch model to optimize
            input_shape: Shape of input tensor (N, C, T, V, M)
            
        Returns:
            TensorRT-optimized model
        """
        model.eval()
        
        # Create sample input for tracing
        sample_input = torch.randn(input_shape, device=self.device, dtype=torch.float32)
        
        # Convert to TensorRT
        print(f"[INFO] Optimizing model with TensorRT ({self.precision} precision)...")
        
        try:
            # Compile with torch_tensorrt
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[sample_input],
                enabled_precisions={torch.float16 if self.precision == "fp16" else torch.float32},
                workspace_size=1 << 30,  # 1GB
                min_block_size=1,
                truncate_long_and_double=True,
            )
            
            print(f"[SUCCESS] Model optimized with TensorRT")
            return trt_model
            
        except Exception as e:
            print(f"[ERROR] TensorRT optimization failed: {e}")
            raise


class STGCNBenchmark:
    """Benchmarking class for ST-GCN models."""
    
    def __init__(self, model_path: str, input_shape: Tuple[int, ...] = (1, 3, 300, 33, 1)):
        """
        Initialize benchmark with model path and input shape.
        
        Args:
            model_path: Path to the ST-GCN model file
            input_shape: Input tensor shape (N, C, T, V, M)
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_pytorch_model(self) -> torch.nn.Module:
        """Load the original PyTorch ST-GCN model."""
        print(f"[INFO] Loading PyTorch model from {self.model_path}")
        
        try:
            # Initialize ST-GCN model
            model = STGCN_Model(
                in_channels=3, 
                num_class=2, 
                edge_importance_weighting=True,
                graph_args={'layout': 'mediapipe', 'strategy': 'spatial'}
            )
            
            # Load weights
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            
            print(f"[SUCCESS] PyTorch model loaded successfully")
            return model
            
        except Exception as e:
            print(f"[ERROR] Failed to load PyTorch model: {e}")
            raise
    
    def benchmark_model(self, model: torch.nn.Module, num_iterations: int = 100, 
                       warmup_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark model inference time.
        
        Args:
            model: Model to benchmark
            num_iterations: Number of inference iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with timing statistics
        """
        print(f"[INFO] Benchmarking model with {num_iterations} iterations...")
        
        # Create random input data
        input_data = torch.randn(self.input_shape, device=self.device, dtype=torch.float32)
        
        # Warmup
        print(f"[INFO] Running {warmup_iterations} warmup iterations...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_data)
                torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for i in range(num_iterations):
                start_time = time.perf_counter()
                output = model(input_data)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                
                if (i + 1) % 20 == 0:
                    print(f"[INFO] Completed {i + 1}/{num_iterations} iterations")
        
        # Calculate statistics
        times = np.array(times)
        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'fps': 1.0 / np.mean(times)
        }
        
        return stats, output
    
    def compare_outputs(self, output1: torch.Tensor, output2: torch.Tensor, 
                       tolerance: float = 1e-3) -> Dict[str, float]:
        """
        Compare outputs from two models.
        
        Args:
            output1: Output from first model
            output2: Output from second model
            tolerance: Tolerance for comparison
            
        Returns:
            Dictionary with comparison metrics
        """
        with torch.no_grad():
            # Convert to numpy for easier comparison
            out1_np = output1.cpu().numpy()
            out2_np = output2.cpu().numpy()
            
            # Calculate metrics
            mse = np.mean((out1_np - out2_np) ** 2)
            mae = np.mean(np.abs(out1_np - out2_np))
            max_diff = np.max(np.abs(out1_np - out2_np))
            
            # Check if outputs are close
            are_close = np.allclose(out1_np, out2_np, atol=tolerance, rtol=tolerance)
            
            return {
                'mse': mse,
                'mae': mae,
                'max_diff': max_diff,
                'are_close': are_close,
                'tolerance': tolerance
            }
    
    def run_complete_benchmark(self, precision: str = "fp16", num_iterations: int = 100):
        """
        Run complete benchmark comparing PyTorch and TensorRT models.
        
        Args:
            precision: TensorRT precision ("fp16" or "fp32")
            num_iterations: Number of benchmark iterations
        """
        print("="*80)
        print("ST-GCN TensorRT Optimization Benchmark")
        print("="*80)
        print(f"Model: {self.model_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Device: {self.device}")
        print(f"Precision: {precision}")
        print(f"Iterations: {num_iterations}")
        print("="*80)
        
        # Load original PyTorch model
        pytorch_model = self.load_pytorch_model()
        
        # Benchmark PyTorch model
        print("\n[BENCHMARK] PyTorch Model")
        pytorch_stats, pytorch_output = self.benchmark_model(pytorch_model, num_iterations)
        
        # Optimize with TensorRT
        try:
            optimizer = TensorRTOptimizer(precision=precision)
            tensorrt_model = optimizer.optimize_model(pytorch_model, self.input_shape)
            
            # Benchmark TensorRT model
            print("\n[BENCHMARK] TensorRT Model")
            tensorrt_stats, tensorrt_output = self.benchmark_model(tensorrt_model, num_iterations)
            
            # Compare outputs
            print("\n[COMPARISON] Output Accuracy")
            comparison = self.compare_outputs(pytorch_output, tensorrt_output)
            
            # Print results
            self.print_results(pytorch_stats, tensorrt_stats, comparison)
            
        except Exception as e:
            print(f"\n[ERROR] TensorRT optimization failed: {e}")
            print("Running PyTorch-only benchmark...")
            self.print_pytorch_only_results(pytorch_stats)
    
    def print_results(self, pytorch_stats: Dict, tensorrt_stats: Dict, comparison: Dict):
        """Print benchmark results."""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        print(f"\n📊 PyTorch Model Performance:")
        print(f"   Mean inference time: {pytorch_stats['mean_time']*1000:.2f} ms")
        print(f"   Standard deviation:  {pytorch_stats['std_time']*1000:.2f} ms")
        print(f"   Min/Max time:        {pytorch_stats['min_time']*1000:.2f} / {pytorch_stats['max_time']*1000:.2f} ms")
        print(f"   Median time:         {pytorch_stats['median_time']*1000:.2f} ms")
        print(f"   FPS:                 {pytorch_stats['fps']:.2f}")
        
        print(f"\n🚀 TensorRT Model Performance:")
        print(f"   Mean inference time: {tensorrt_stats['mean_time']*1000:.2f} ms")
        print(f"   Standard deviation:  {tensorrt_stats['std_time']*1000:.2f} ms")
        print(f"   Min/Max time:        {tensorrt_stats['min_time']*1000:.2f} / {tensorrt_stats['max_time']*1000:.2f} ms")
        print(f"   Median time:         {tensorrt_stats['median_time']*1000:.2f} ms")
        print(f"   FPS:                 {tensorrt_stats['fps']:.2f}")
        
        # Calculate speedup
        speedup = pytorch_stats['mean_time'] / tensorrt_stats['mean_time']
        fps_improvement = tensorrt_stats['fps'] / pytorch_stats['fps']
        
        print(f"\n⚡ Performance Improvement:")
        print(f"   Speedup:             {speedup:.2f}x")
        print(f"   FPS improvement:     {fps_improvement:.2f}x")
        print(f"   Time reduction:      {(1 - 1/speedup)*100:.1f}%")
        
        print(f"\n🎯 Output Accuracy:")
        print(f"   MSE:                 {comparison['mse']:.2e}")
        print(f"   MAE:                 {comparison['mae']:.2e}")
        print(f"   Max difference:      {comparison['max_diff']:.2e}")
        print(f"   Outputs close:       {'✅ Yes' if comparison['are_close'] else '❌ No'}")
        print(f"   Tolerance:           {comparison['tolerance']:.2e}")
        
        print("="*80)
    
    def print_pytorch_only_results(self, pytorch_stats: Dict):
        """Print PyTorch-only results when TensorRT fails."""
        print("\n" + "="*80)
        print("PYTORCH BENCHMARK RESULTS")
        print("="*80)
        
        print(f"\n📊 PyTorch Model Performance:")
        print(f"   Mean inference time: {pytorch_stats['mean_time']*1000:.2f} ms")
        print(f"   Standard deviation:  {pytorch_stats['std_time']*1000:.2f} ms")
        print(f"   Min/Max time:        {pytorch_stats['min_time']*1000:.2f} / {pytorch_stats['max_time']*1000:.2f} ms")
        print(f"   Median time:         {pytorch_stats['median_time']*1000:.2f} ms")
        print(f"   FPS:                 {pytorch_stats['fps']:.2f}")
        
        print("="*80)


def main():
    """Main function to run the TensorRT benchmark."""
    parser = argparse.ArgumentParser(description='ST-GCN TensorRT Optimization Benchmark')
    parser.add_argument('--model', type=str, 
                       default='models/epoch50_model.pt',
                       help='Path to ST-GCN model file')
    parser.add_argument('--precision', type=str, choices=['fp16', 'fp32'], 
                       default='fp16',
                       help='TensorRT precision mode')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--sequence-length', type=int, default=300,
                       help='Sequence length (temporal dimension)')
    
    args = parser.parse_args()
    
    # Validate CUDA availability
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. TensorRT requires CUDA.")
        return
    
    # Print system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    try:
        import torch_tensorrt
        print(f"Torch-TensorRT version: {torch_tensorrt.__version__}")
    except ImportError:
        print("[ERROR] torch_tensorrt is not installed. Please install it first.")
        return
    
    # Define input shape (N, C, T, V, M)
    # N: batch size, C: channels (3), T: temporal frames, V: vertices (33), M: persons (1)
    input_shape = (args.batch_size, 3, args.sequence_length, 33, 1)
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        print("Available models:")
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pt'):
                    print(f"  - {os.path.join('models', f)}")
        return
    
    # Run benchmark
    benchmark = STGCNBenchmark(args.model, input_shape)
    benchmark.run_complete_benchmark(
        precision=args.precision,
        num_iterations=args.iterations
    )


if __name__ == "__main__":
    main()
