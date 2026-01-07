#!/usr/bin/env python3
"""
Simple test script to verify torch_tensorrt installation and basic ST-GCN optimization.
This is a minimal version to test the basic functionality.
"""

import torch
import numpy as np
import time
import os
import sys

def test_torch_tensorrt_installation():
    """Test if torch_tensorrt is properly installed."""
    print("="*60)
    print("Testing Torch-TensorRT Installation")
    print("="*60)
    
    try:
        import torch_tensorrt
        print(f"✅ torch_tensorrt imported successfully")
        print(f"   Version: {torch_tensorrt.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import torch_tensorrt: {e}")
        return False

def test_cuda_availability():
    """Test CUDA availability."""
    print(f"\n🔍 CUDA Information:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("   ❌ CUDA not available - TensorRT requires CUDA")
        return False

def create_simple_test_model():
    """Create a simple neural network for testing."""
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(128, 2)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    return SimpleNet()

def test_simple_tensorrt_compilation():
    """Test basic TensorRT compilation with a simple model."""
    print(f"\n🧪 Testing Simple TensorRT Compilation")
    print("-" * 40)
    
    try:
        import torch_tensorrt
        
        # Create simple model
        model = create_simple_test_model()
        model.eval()
        model = model.cuda()
        
        # Create sample input
        sample_input = torch.randn(1, 3, 224, 224, device='cuda')
        
        print(f"   Model created: {type(model).__name__}")
        print(f"   Input shape: {sample_input.shape}")
        
        # Test regular PyTorch inference
        with torch.no_grad():
            pytorch_output = model(sample_input)
            print(f"   PyTorch output shape: {pytorch_output.shape}")
        
        # Compile with TensorRT
        print(f"   Compiling with TensorRT...")
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[sample_input],
            enabled_precisions={torch.float16},
            workspace_size=1 << 28,  # 256MB
        )
        
        # Test TensorRT inference
        with torch.no_grad():
            trt_output = trt_model(sample_input)
            print(f"   TensorRT output shape: {trt_output.shape}")
        
        # Compare outputs
        diff = torch.abs(pytorch_output - trt_output).max().item()
        print(f"   Max difference: {diff:.2e}")
        
        print(f"   ✅ Simple TensorRT compilation successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ TensorRT compilation failed: {e}")
        return False

def patch_stgcn_models():
    """Patch ST-GCN models to fix avgpool dimension issue and CUDA tensor warnings."""
    try:
        # Add project paths
        project_root = "/home/wisevision/projects/shoplifting-deploy/wisevision-shoplifting-jetson"
        sys.path.insert(0, project_root)
        
        from net import st_gcn
        from net import st_gcn_twostream
        import torch.nn.functional as F
        
        # Patch 1: Fix ST-GCN avgpool dimensions
        original_forward = st_gcn.Model.forward
        
        def patched_forward(self, x):
            # Original normalization code
            N, C, T, V, M = x.shape
            x = x.permute(0, 4, 3, 1, 2).contiguous()
            x = x.view(N * M, V * C, T)
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T)
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = x.view(N * M, C, T, V)

            # Forward through ST-GCN layers
            for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
                x, _ = gcn(x, self.A * importance)

            # FIXED: Use current tensor dimensions instead of original input dimensions
            current_T, current_V = x.shape[2], x.shape[3]
            x = F.avg_pool2d(x, kernel_size=(current_T, current_V))
            x = x.view(N, M, -1, 1, 1).mean(dim=1)

            # Prediction
            x = self.fcn(x)
            x = x.view(x.shape[0], -1)

            return x
        
        # Patch 2: Fix twostream CUDA tensor creation
        original_twostream_forward = st_gcn_twostream.Model.forward
        
        def patched_twostream_forward(self, x):
            N, C, T, V, M = x.size()
            # FIXED: Use modern tensor creation instead of deprecated CUDA constructors
            zeros_tensor = torch.zeros(N, C, 1, V, M, device=x.device, dtype=x.dtype)
            m = torch.cat((zeros_tensor, x[:, :, 1:] - x[:, :, :-1]), 2)
            
            res = self.origin_stream(x) + self.motion_stream(m)
            return res
        
        # Apply patches
        st_gcn.Model.forward = patched_forward
        st_gcn_twostream.Model.forward = patched_twostream_forward
        
        print(f"   ✅ ST-GCN patches applied successfully")
        print(f"      - Fixed avgpool dimension issue")
        print(f"      - Fixed deprecated CUDA tensor warnings")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to patch ST-GCN models: {e}")
        return False

def load_stgcn_model(model_path):
    """Load ST-GCN model for testing."""
    print(f"\n🏗️ Loading ST-GCN Model")
    print("-" * 40)
    
    try:
        # Apply patches first
        if not patch_stgcn_models():
            return None
        
        # Add project paths
        project_root = "/home/wisevision/projects/shoplifting-deploy/wisevision-shoplifting-jetson"
        sys.path.insert(0, project_root)
        
        from net.st_gcn_twostream import Model as STGCN_Model
        
        # Create model
        model = STGCN_Model(
            in_channels=3, 
            num_class=2, 
            edge_importance_weighting=True,
            graph_args={'layout': 'mediapipe', 'strategy': 'spatial'}
        )
        
        # Load weights if model file exists
        if os.path.exists(model_path):
            print(f"   Loading weights from: {model_path}")
            state_dict = torch.load(model_path, map_location='cuda', weights_only=True)
            model.load_state_dict(state_dict)
            print(f"   ✅ Weights loaded successfully")
        else:
            print(f"   ⚠️ Model file not found: {model_path}")
            print(f"   Using randomly initialized weights")
        
        model = model.cuda()
        model.eval()
        
        print(f"   Model type: {type(model).__name__}")
        return model
        
    except Exception as e:
        print(f"   ❌ Failed to load ST-GCN model: {e}")
        return None

def test_stgcn_tensorrt(model, input_shape=(1, 3, 300, 33, 1)):
    """Test ST-GCN model with TensorRT."""
    print(f"\n🚀 Testing ST-GCN with TensorRT")
    print("-" * 40)
    
    try:
        import torch_tensorrt
        
        # Create sample input (N, C, T, V, M)
        sample_input = torch.randn(input_shape, device='cuda', dtype=torch.float32)
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Note: Using patched ST-GCN model with fixed avgpool dimensions")
        
        # Test PyTorch inference
        print(f"   Testing PyTorch inference...")
        with torch.no_grad():
            start_time = time.perf_counter()
            pytorch_output = model(sample_input)
            torch.cuda.synchronize()
            pytorch_time = time.perf_counter() - start_time
            
        print(f"   PyTorch inference time: {pytorch_time*1000:.2f} ms")
        print(f"   Output shape: {pytorch_output.shape}")
        
        # Compile with TensorRT
        print(f"   Compiling ST-GCN with TensorRT...")
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[sample_input],
            enabled_precisions={torch.float16},
            workspace_size=1 << 30,  # 1GB
            min_block_size=1,
        )
        
        # Test TensorRT inference
        print(f"   Testing TensorRT inference...")
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = trt_model(sample_input)
                torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.perf_counter()
            trt_output = trt_model(sample_input)
            torch.cuda.synchronize()
            trt_time = time.perf_counter() - start_time
        
        print(f"   TensorRT inference time: {trt_time*1000:.2f} ms")
        
        # Compare outputs
        diff = torch.abs(pytorch_output - trt_output).max().item()
        mean_diff = torch.abs(pytorch_output - trt_output).mean().item()
        
        print(f"   Max difference: {diff:.2e}")
        print(f"   Mean difference: {mean_diff:.2e}")
        
        # Calculate speedup
        speedup = pytorch_time / trt_time
        print(f"   Speedup: {speedup:.2f}x")
        
        print(f"   ✅ ST-GCN TensorRT optimization successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ ST-GCN TensorRT optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_inference_speeds(model, input_shape=(1, 3, 300, 33, 1), num_iterations=50):
    """Benchmark inference speeds for detailed comparison."""
    print(f"\n📊 Detailed Performance Benchmark")
    print("-" * 40)
    
    try:
        import torch_tensorrt
        
        sample_input = torch.randn(input_shape, device='cuda', dtype=torch.float32)
        
        # Benchmark PyTorch
        print(f"   Benchmarking PyTorch model...")
        pytorch_times = []
        with torch.no_grad():
            for i in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(sample_input)
                torch.cuda.synchronize()
                pytorch_times.append(time.perf_counter() - start_time)
        
        pytorch_mean = np.mean(pytorch_times) * 1000
        pytorch_std = np.std(pytorch_times) * 1000
        
        # Compile and benchmark TensorRT
        print(f"   Compiling and benchmarking TensorRT model...")
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[sample_input],
            enabled_precisions={torch.float16},
            workspace_size=1 << 30,
        )
        
        trt_times = []
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = trt_model(sample_input)
                torch.cuda.synchronize()
            
            # Benchmark
            for i in range(num_iterations):
                start_time = time.perf_counter()
                _ = trt_model(sample_input)
                torch.cuda.synchronize()
                trt_times.append(time.perf_counter() - start_time)
        
        trt_mean = np.mean(trt_times) * 1000
        trt_std = np.std(trt_times) * 1000
        
        # Results
        print(f"\n   📈 Benchmark Results ({num_iterations} iterations):")
        print(f"   PyTorch:   {pytorch_mean:.2f} ± {pytorch_std:.2f} ms")
        print(f"   TensorRT:  {trt_mean:.2f} ± {trt_std:.2f} ms")
        print(f"   Speedup:   {pytorch_mean/trt_mean:.2f}x")
        print(f"   FPS Improvement: {trt_mean/pytorch_mean:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Benchmarking failed: {e}")
        return False

def main():
    """Main test function."""
    print("🔬 Torch-TensorRT ST-GCN Test Suite")
    print("="*60)
    
    # Test 1: Installation
    if not test_torch_tensorrt_installation():
        print("\n❌ torch_tensorrt not properly installed. Exiting.")
        return
    
    # Test 2: CUDA
    if not test_cuda_availability():
        print("\n❌ CUDA not available. Exiting.")
        return
    
    # Test 3: Simple TensorRT compilation
    if not test_simple_tensorrt_compilation():
        print("\n❌ Basic TensorRT compilation failed.")
        return
    
    # Test 4: Load ST-GCN model
    model_path = "/home/wisevision/projects/shoplifting-deploy/models/epoch50_model.pt"
    stgcn_model = load_stgcn_model(model_path)
    
    if stgcn_model is None:
        print("\n❌ Failed to load ST-GCN model. Exiting.")
        return
    
    # Test 5: ST-GCN TensorRT optimization
    if not test_stgcn_tensorrt(stgcn_model):
        print("\n❌ ST-GCN TensorRT optimization failed.")
        return
    
    # Test 6: Detailed benchmarking
    benchmark_inference_speeds(stgcn_model)
    
    print("\n" + "="*60)
    print("🎉 All tests completed successfully!")
    print("   Your torch_tensorrt installation is working properly")
    print("   ST-GCN model can be optimized with TensorRT")
    print("="*60)

if __name__ == "__main__":
    main()
