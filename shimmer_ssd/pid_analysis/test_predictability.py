#!/usr/bin/env python3
"""
Simple test script for predictability metrics without shimmer dependencies.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import our metrics module
from metrics.predictability import compute_predictability, PredictabilityAccumulator

def test_predictability_metrics():
    """Test the predictability metrics computation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create mock data
    batch_size = 100
    
    # Mock decoded outputs
    decoded = {
        'attr': torch.randn(batch_size, 19).to(device),  # 11 base + 8 synergy logits
        'v': torch.randn(batch_size, 13).to(device)      # 13 base features (no synergy)
    }
    
    # Mock targets
    targets = {
        'attr': torch.cat([
            torch.randn(batch_size, 11),  # 11 base features
            torch.rand(batch_size, 1)     # 1 synergy feature (normalized)
        ], dim=1).to(device),
        'v': torch.randn(batch_size, 13).to(device)  # 13 base features
    }
    
    # Mock partitions
    partitions = {
        'attr': {
            'unique': [0, 1, 2, 3, 4],
            'redundant': [5, 6, 7, 8, 9, 10]
        },
        'v': {
            'unique': [0, 1, 2, 3, 4, 5, 6],
            'redundant': [7, 8, 9, 10, 11, 12]
        }
    }
    
    # Mock synergy config
    synergy_config = {
        'domains': ['attr', 'v'],
        'feature_indices': {
            'attr': [11]  # One synergy feature at index 11
        }
    }
    
    # Compute metrics
    print("Computing predictability metrics...")
    
    try:
        metrics = compute_predictability(
            decoded=decoded,
            targets=targets,
            partitions=partitions,
            synergy_config=synergy_config,
            device=device,
            n_bins=8
        )
        
        print("Metrics computed successfully:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Verify expected keys exist
        expected_keys = [
            'predictability/synergy_acc',
            'predictability/synergy_norm_ce',
            'predictability/unique_r2',
            'predictability/redundant_r2'
        ]
        
        for key in expected_keys:
            if key in metrics:
                print(f"✓ Found expected metric: {key}")
            else:
                print(f"✗ Missing expected metric: {key}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during metrics computation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_accumulator():
    """Test the PredictabilityAccumulator class."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTesting PredictabilityAccumulator on {device}...")
    
    accumulator = PredictabilityAccumulator(device)
    
    # Test synergy metrics
    synergy_logits = torch.randn(50, 8).to(device)  # 50 samples, 8 classes
    synergy_targets = torch.rand(50, 1).to(device)  # 50 samples, 1 synergy feature
    
    accumulator.update_synergy_metrics(synergy_logits, synergy_targets, n_bins=8)
    
    # Test base metrics
    predictions = torch.randn(50, 11).to(device)
    targets = torch.randn(50, 11).to(device)
    
    accumulator.update_base_metrics('attr', predictions, targets)
    
    # Get final metrics
    final_metrics = accumulator.compute_final_metrics()
    print("Accumulator metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test R² by partition
    unique_indices = [0, 1, 2, 3, 4]
    redundant_indices = [5, 6, 7, 8, 9, 10]
    
    r2_metrics = accumulator.compute_base_r2_by_partition(
        'attr', unique_indices, redundant_indices
    )
    print("R² partition metrics:")
    for key, value in r2_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("Accumulator test completed successfully!")
    return True

if __name__ == "__main__":
    print("Testing predictability metrics module...")
    
    success1 = test_accumulator()
    success2 = test_predictability_metrics()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
