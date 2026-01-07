#!/usr/bin/env python3
"""
Advanced TensorRT test for ST-GCN with different optimization strategies
"""

import torch
import torch_tensorrt
import numpy as np
import time
import os
import sys

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
        def patched_forward(self, x):
            N, C, T, V, M = x.shape
            x = x.permute(0, 4, 3, 1, 2).contiguous()
            x = x.view(N * M, V * C, T)
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T)
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = x.view(N * M, C, T, V)

            for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
                x, _ = gcn(x, self.A * importance)

            current_T, current_V = x.shape[2], x.shape[3]
            x = F.avg_pool2d(x, kernel_size=(current_T, current_V))
            x = x.view(N, M, -1, 1, 1).mean(dim=1)
            x = self.fcn(x)
            x = x.view(x.shape[0], -1)
            return x
        
        # Patch 2: Fix twostream CUDA tensor creation
        def patched_twostream_forward(self, x):
            N, C, T, V, M = x.size()
            zeros_tensor = torch.zeros(N, C, 1, V, M, device=x.device, dtype=x.dtype)
            m = torch.cat((zeros_tensor, x[:, :, 1:] - x[:, :, :-1]), 2)
            res = self.origin_stream(x) + self.motion_stream(m)
            return res
        
        st_gcn.Model.forward = patched_forward
        st_gcn_twostream.Model.forward = patched_twostream_forward
        return True
        
    except Exception as e:
        print(f"Failed to patch: {e}")
        return False

def test_tensorrt_strategies():
    """Test different TensorRT compilation strategies for ST-GCN."""
    print("="*80)
    print("Advanced TensorRT Optimization Test for ST-GCN")
    print("="*80)
    
    # Patch models
    if not patch_stgcn_models():
        print("❌ Failed to patch ST-GCN models")
        return
    
    # Load model
    project_root = "/home/wisevision/projects/shoplifting-deploy/wisevision-shoplifting-jetson"
    sys.path.insert(0, project_root)
    
    from net.st_gcn_twostream import Model as STGCN_Model
    
    model = STGCN_Model(
        in_channels=3, 
        num_class=2, 
        edge_importance_weighting=True,
        graph_args={'layout': 'mediapipe', 'strategy': 'spatial'}
    )
    
    model_path = "/home/wisevision/projects/shoplifting-deploy/models/epoch50_model.pt"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cuda', weights_only=True)
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded")
    
    model = model.cuda().eval()
    
    # Test input
    input_shape = (1, 3, 300, 33, 1)
    sample_input = torch.randn(input_shape, device='cuda', dtype=torch.float32)
    
    print(f"\n📊 Model: ST-GCN TwoStream")
    print(f"📊 Input shape: {input_shape}")
    
    # Baseline PyTorch performance
    print(f"\n🔥 PyTorch Baseline Performance")
    pytorch_times = []
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(sample_input)
            torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(50):
            start = time.perf_counter()
            output = model(sample_input)
            torch.cuda.synchronize()
            pytorch_times.append(time.perf_counter() - start)
    
    pytorch_mean = np.mean(pytorch_times) * 1000
    pytorch_std = np.std(pytorch_times) * 1000
    print(f"   Average: {pytorch_mean:.2f} ± {pytorch_std:.2f} ms")
    print(f"   FPS: {1000/pytorch_mean:.2f}")
    
    # TensorRT strategies to test
    strategies = [
        {
            "name": "Conservative FP16",
            "settings": {
                "enabled_precisions": {torch.float16},
                "workspace_size": 1 << 28,  # 256MB
                "min_block_size": 7,
                "truncate_long_and_double": True,
                "require_full_compilation": False,
            }
        },
        {
            "name": "Aggressive FP16", 
            "settings": {
                "enabled_precisions": {torch.float16},
                "workspace_size": 1 << 30,  # 1GB
                "min_block_size": 3,
                "truncate_long_and_double": True,
                "require_full_compilation": False,
            }
        },
        {
            "name": "Conservative FP32",
            "settings": {
                "enabled_precisions": {torch.float32},
                "workspace_size": 1 << 29,  # 512MB
                "min_block_size": 5,
                "truncate_long_and_double": True,
                "require_full_compilation": False,
            }
        },
        {
            "name": "Fallback Mode",
            "settings": {
                "enabled_precisions": {torch.float16},
                "workspace_size": 1 << 27,  # 128MB
                "min_block_size": 10,
                "truncate_long_and_double": True,
                "require_full_compilation": False,
            }
        }
    ]
    
    successful_optimizations = 0
    
    for strategy in strategies:
        print(f"\n🚀 Testing {strategy['name']}")
        print("-" * 60)
        
        try:
            # Compile with TensorRT
            print(f"   Compiling...")
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[sample_input],
                **strategy['settings']
            )
            
            print(f"   ✅ Compilation successful!")
            
            # Benchmark TensorRT performance
            trt_times = []
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = trt_model(sample_input)
                    torch.cuda.synchronize()
                
                # Benchmark
                for _ in range(50):
                    start = time.perf_counter()
                    trt_output = trt_model(sample_input)
                    torch.cuda.synchronize()
                    trt_times.append(time.perf_counter() - start)
            
            trt_mean = np.mean(trt_times) * 1000
            trt_std = np.std(trt_times) * 1000
            speedup = pytorch_mean / trt_mean
            
            # Accuracy check
            diff = torch.abs(output - trt_output).max().item()
            
            print(f"   Performance: {trt_mean:.2f} ± {trt_std:.2f} ms")
            print(f"   FPS: {1000/trt_mean:.2f}")
            print(f"   Speedup: {speedup:.2f}x")
            print(f"   Max diff: {diff:.2e}")
            
            successful_optimizations += 1
            
            # Clean up
            del trt_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}...")
    
    print(f"\n" + "="*80)
    print(f"SUMMARY")
    print(f"="*80)
    print(f"✅ ST-GCN model fixed and working with PyTorch")
    print(f"⚡ PyTorch baseline: {pytorch_mean:.2f} ms ({1000/pytorch_mean:.2f} FPS)")
    print(f"🚀 Successful TensorRT optimizations: {successful_optimizations}/{len(strategies)}")
    
    if successful_optimizations == 0:
        print(f"\n💡 Recommendations:")
        print(f"   1. ST-GCN's graph convolution operations may be too complex for TensorRT")
        print(f"   2. Consider using PyTorch's built-in optimizations:")
        print(f"      - torch.jit.script() for JIT compilation")
        print(f"      - torch.compile() for PyTorch 2.x optimization")
        print(f"   3. Profile specific bottlenecks and optimize manually")
        print(f"   4. Consider model architecture simplification for deployment")
    else:
        print(f"\n🎉 TensorRT optimization successful!")
        print(f"   Your torch_tensorrt installation is working properly")

def test_pytorch_optimizations():
    """Test PyTorch's native optimization methods."""
    print(f"\n" + "="*80)
    print(f"PyTorch Native Optimization Test")
    print(f"="*80)
    
    if not patch_stgcn_models():
        return
    
    project_root = "/home/wisevision/projects/shoplifting-deploy/wisevision-shoplifting-jetson"
    sys.path.insert(0, project_root)
    
    from net.st_gcn_twostream import Model as STGCN_Model
    
    model = STGCN_Model(
        in_channels=3, 
        num_class=2, 
        edge_importance_weighting=True,
        graph_args={'layout': 'mediapipe', 'strategy': 'spatial'}
    )
    
    model_path = "/home/wisevision/projects/shoplifting-deploy/models/epoch50_model.pt"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cuda', weights_only=True)
        model.load_state_dict(state_dict)
    
    model = model.cuda().eval()
    
    input_shape = (1, 3, 300, 33, 1)
    sample_input = torch.randn(input_shape, device='cuda', dtype=torch.float32)
    
    optimizations = [
        ("Baseline", model),
    ]
    
    # Test torch.jit.script if available
    try:
        print("🔍 Testing TorchScript optimization...")
        scripted_model = torch.jit.script(model)
        optimizations.append(("TorchScript", scripted_model))
        print("   ✅ TorchScript compilation successful")
    except Exception as e:
        print(f"   ❌ TorchScript failed: {e}")
    
    # Test torch.compile if available (PyTorch 2.x)
    try:
        if hasattr(torch, 'compile'):
            print("🔍 Testing torch.compile optimization...")
            compiled_model = torch.compile(model, mode='reduce-overhead')
            optimizations.append(("torch.compile", compiled_model))
            print("   ✅ torch.compile successful")
    except Exception as e:
        print(f"   ❌ torch.compile failed: {e}")
    
    print(f"\n📊 Performance Comparison")
    print("-" * 60)
    
    for name, opt_model in optimizations:
        print(f"Testing {name}...")
        
        try:
            times = []
            with torch.no_grad():
                # Warmup
                for _ in range(5):
                    _ = opt_model(sample_input)
                    torch.cuda.synchronize()
                
                # Benchmark
                for _ in range(30):
                    start = time.perf_counter()
                    _ = opt_model(sample_input)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
            
            mean_time = np.mean(times) * 1000
            std_time = np.std(times) * 1000
            fps = 1000 / mean_time
            
            print(f"   {name}: {mean_time:.2f} ± {std_time:.2f} ms ({fps:.2f} FPS)")
            
        except Exception as e:
            print(f"   {name}: ❌ Failed - {e}")

if __name__ == "__main__":
    test_tensorrt_strategies()
    test_pytorch_optimizations()
