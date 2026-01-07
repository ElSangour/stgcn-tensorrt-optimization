#!/usr/bin/env python3
"""
Debug script to understand ST-GCN dimensional transformations
"""

import torch
import sys
import os

def debug_stgcn_dimensions():
    """Debug ST-GCN model to understand dimension changes."""
    print("="*60)
    print("Debugging ST-GCN Dimensional Transformations")  
    print("="*60)
    
    try:
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
        
        model = model.cuda()
        model.eval()
        
        print(f"Model created successfully")
        
        # Test with different input sizes to find the issue
        test_shapes = [
            (1, 3, 300, 33, 1),  # Original config
            (1, 3, 150, 33, 1),  # After 1 stride=2
            (1, 3, 75, 33, 1),   # After 2 stride=2
            (1, 3, 38, 33, 1),   # After 3 stride=2 (hypothetical)
        ]
        
        for shape in test_shapes:
            print(f"\n🔍 Testing input shape: {shape}")
            try:
                sample_input = torch.randn(shape, device='cuda', dtype=torch.float32)
                
                with torch.no_grad():
                    output = model(sample_input)
                    print(f"   ✅ Success! Output shape: {output.shape}")
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()

def analyze_stgcn_layers():
    """Analyze individual ST-GCN layers to understand stride effects."""
    print(f"\n" + "="*60)
    print("Analyzing ST-GCN Layer Configuration")  
    print("="*60)
    
    try:
        # Add project paths
        project_root = "/home/wisevision/projects/shoplifting-deploy/wisevision-shoplifting-jetson"
        sys.path.insert(0, project_root)
        
        from net.st_gcn import Model as ST_GCN
        
        # Create single stream model for analysis
        model = ST_GCN(
            in_channels=3, 
            num_class=2, 
            edge_importance_weighting=True,
            graph_args={'layout': 'mediapipe', 'strategy': 'spatial'}
        )
        
        model = model.cuda()
        model.eval()
        
        print(f"ST-GCN layers configuration:")
        for i, layer in enumerate(model.st_gcn_networks):
            if hasattr(layer, 'tcn') and hasattr(layer.tcn, '__getitem__'):
                # Find Conv2d layer in the sequential
                for sublayer in layer.tcn:
                    if isinstance(sublayer, torch.nn.Conv2d):
                        stride = sublayer.stride
                        print(f"   Layer {i}: stride={stride}")
                        break
            else:
                print(f"   Layer {i}: stride=unknown")
        
        # Calculate expected temporal dimension reduction
        initial_T = 300
        current_T = initial_T
        print(f"\nTemporal dimension progression (starting T={initial_T}):")
        
        stride_layers = [4, 7]  # Based on ST-GCN configuration (stride=2 layers)
        for i in range(10):  # 10 ST-GCN layers
            if i in stride_layers:
                current_T = current_T // 2
                print(f"   After layer {i} (stride=2): T={current_T}")
            else:
                print(f"   After layer {i} (stride=1): T={current_T}")
        
        print(f"\nFinal expected T: {current_T}")
        print(f"Avgpool kernel should be: ({current_T}, 33)")
        
    except Exception as e:
        print(f"❌ Failed to analyze layers: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_stgcn_dimensions()
    analyze_stgcn_layers()
