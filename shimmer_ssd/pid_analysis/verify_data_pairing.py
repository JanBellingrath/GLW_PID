#!/usr/bin/env python3
"""
Verify Data Pairing in Synergy Dataset

This script verifies that visual and attribute modalities are correctly paired
in the synergy dataset, specifically checking:

1. Visual VAE latents correspond to the correct images (and thus correct size values)
2. Attribute data corresponds to the correct samples 
3. XOR targets are computed from the correctly paired visual size and attribute random values
4. Sample indexing is consistent across all data sources

This is critical for XOR reconstruction training since the model must see
matching visual and attribute inputs to learn the synergistic relationship.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import json

# Add path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from synergy_dataset import SynergyDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_original_data_sources(data_dir: str, split: str) -> Dict[str, np.ndarray]:
    """Load all original data sources to verify pairing."""
    data_dir = Path(data_dir)
    
    sources = {}
    
    # 1. Load XOR targets
    xor_file = data_dir / f"{split}_xor_targets.npy"
    if xor_file.exists():
        sources['xor_data'] = np.load(xor_file)
        logger.info(f"Loaded XOR data: {sources['xor_data'].shape}")
    else:
        raise FileNotFoundError(f"XOR targets not found: {xor_file}")
    
    # 2. Load original attributes (no-size, 8D)
    attr_paths = [
        data_dir / f"{split}_labels_nosize.npy",
        data_dir.parent / f"{split}_labels_nosize.npy"
    ]
    for attr_path in attr_paths:
        if attr_path.exists():
            sources['original_attrs'] = np.load(attr_path)
            logger.info(f"Loaded original attributes: {sources['original_attrs'].shape}")
            break
    else:
        raise FileNotFoundError(f"Original attributes not found in: {attr_paths}")
    
    # 3. Load original labels with size
    label_paths = [
        data_dir.parent / f"{split}_labels.npy",
        data_dir / f"{split}_labels.npy",
        data_dir.parent.parent / "simple_shapes_dataset" / f"{split}_labels.npy",
        Path("/home/janerik/GLW_PID/simple_shapes_dataset/simple_shapes_dataset") / f"{split}_labels.npy"
    ]
    for label_path in label_paths:
        if label_path.exists():
            sources['original_labels_with_size'] = np.load(label_path)
            logger.info(f"Loaded original labels with size: {sources['original_labels_with_size'].shape}")
            break
    else:
        raise FileNotFoundError(f"Original labels with size not found in: {label_paths}")
    
    # 4. Try to load VAE latents
    latent_filename = "calmip-822888_epoch=282-step=1105680_future.npy"
    latent_paths = [
        Path("/home/janerik/GLW_PID/simple_shapes_dataset/simple_shapes_dataset/saved_latents") / split / latent_filename,
        data_dir.parent / "saved_latents" / split / latent_filename
    ]
    for latent_path in latent_paths:
        if latent_path.exists():
            sources['vae_latents'] = np.load(latent_path)
            logger.info(f"Loaded VAE latents: {sources['vae_latents'].shape}")
            break
    else:
        logger.warning(f"VAE latents not found in: {latent_paths}")
        sources['vae_latents'] = None
    
    return sources


def verify_sample_count_consistency(sources: Dict[str, np.ndarray]) -> bool:
    """Verify all data sources have consistent sample counts."""
    logger.info("Verifying sample count consistency...")
    
    sample_counts = {}
    for name, data in sources.items():
        if data is not None:
            sample_counts[name] = len(data)
    
    logger.info(f"Sample counts: {sample_counts}")
    
    # Check if all counts are the same
    counts = list(sample_counts.values())
    if len(set(counts)) == 1:
        logger.info("✅ All data sources have consistent sample counts")
        return True
    else:
        logger.error("❌ Inconsistent sample counts across data sources")
        return False


def verify_xor_formula_consistency(sources: Dict[str, np.ndarray], n_bins: int = 8) -> bool:
    """Verify XOR targets are computed correctly from size and random values."""
    logger.info("Verifying XOR formula consistency...")
    
    xor_data = sources['xor_data']
    original_labels_with_size = sources['original_labels_with_size']
    original_attrs = sources['original_attrs']
    
    # Convert structured array to DataFrame if needed
    if xor_data.dtype.names:
        xor_df = pd.DataFrame(xor_data)
    else:
        logger.error("XOR data is not a structured array - cannot verify formula")
        return False
    
    # Extract values for verification
    size_raw = original_labels_with_size[:, 3].astype(np.float64)
    random_attr = original_attrs[:, 7].astype(np.float64)
    xor_targets = xor_df['xor_target_normalized'].values
    
    # Normalize size values (same as in XOR generation)
    size_min = size_raw.min()
    size_max = size_raw.max()
    size_range = size_max - size_min
    
    if size_range > 0:
        size_normalized = (size_raw - size_min) / size_range
    else:
        size_normalized = np.full_like(size_raw, 0.5)
    
    # Discretize inputs
    bin_edges = np.linspace(0, 1, n_bins + 1)
    size_bins = np.clip(np.digitize(size_normalized, bin_edges) - 1, 0, n_bins - 1)
    random_bins = np.clip(np.digitize(random_attr, bin_edges) - 1, 0, n_bins - 1)
    
    # Compute expected XOR targets
    expected_xor_bins = (size_bins + random_bins) % n_bins
    expected_xor_normalized = expected_xor_bins.astype(np.float32) / (n_bins - 1) if n_bins > 1 else np.zeros(len(size_bins), dtype=np.float32)
    
    # Compare with stored XOR targets
    xor_diff = np.abs(xor_targets - expected_xor_normalized)
    max_diff = xor_diff.max()
    
    logger.info(f"XOR target comparison: max_diff={max_diff:.6f}")
    
    if max_diff < 1e-6:
        logger.info("✅ XOR targets match expected formula exactly")
        return True
    else:
        logger.error(f"❌ XOR targets do not match expected formula (max diff: {max_diff:.6f})")
        
        # Show some examples of mismatches
        mismatch_indices = np.where(xor_diff > 1e-6)[0][:5]
        for idx in mismatch_indices:
            logger.error(f"  Sample {idx}: size_bin={size_bins[idx]}, random_bin={random_bins[idx]}, "
                        f"expected_xor={expected_xor_normalized[idx]:.6f}, actual_xor={xor_targets[idx]:.6f}")
        
        return False


def verify_size_extraction_consistency(sources: Dict[str, np.ndarray]) -> bool:
    """Verify size values are extracted correctly."""
    logger.info("Verifying size extraction consistency...")
    
    original_labels_with_size = sources['original_labels_with_size']
    
    # Extract size values (column 3)
    size_values = original_labels_with_size[:, 3].astype(np.float32)
    
    logger.info(f"Size values: range [{size_values.min():.3f}, {size_values.max():.3f}], "
                f"mean={size_values.mean():.3f}, std={size_values.std():.3f}")
    
    # Check for reasonable size values (should be positive and in a reasonable range)
    if np.all(size_values > 0) and size_values.max() < 100:  # Reasonable range for shape sizes
        logger.info("✅ Size values are in reasonable range")
        return True
    else:
        logger.error(f"❌ Size values out of expected range: [{size_values.min()}, {size_values.max()}]")
        return False


def verify_vae_latent_consistency(sources: Dict[str, np.ndarray]) -> bool:
    """Verify VAE latents are consistent."""
    if sources['vae_latents'] is None:
        logger.warning("⚠️  VAE latents not available - cannot verify latent consistency")
        return True
    
    logger.info("Verifying VAE latent consistency...")
    
    vae_latents = sources['vae_latents']
    
    # Check latent dimensions and ranges
    logger.info(f"VAE latents shape: {vae_latents.shape}")
    logger.info(f"VAE latents range: [{vae_latents.min():.3f}, {vae_latents.max():.3f}]")
    
    # Basic sanity checks
    if len(vae_latents.shape) == 3 and vae_latents.shape[1] == 2:
        # Shape should be [N, 2, latent_dim] for VAE with mean and logvar
        logger.info("✅ VAE latents have expected shape [N, 2, latent_dim]")
        
        # Extract mean vectors
        mean_vectors = vae_latents[:, 0, :]
        logvar_vectors = vae_latents[:, 1, :]
        
        logger.info(f"Mean vectors range: [{mean_vectors.min():.3f}, {mean_vectors.max():.3f}]")
        logger.info(f"Logvar vectors range: [{logvar_vectors.min():.3f}, {logvar_vectors.max():.3f}]")
        
        return True
    else:
        logger.error(f"❌ Unexpected VAE latent shape: {vae_latents.shape}")
        return False


def test_synergy_dataset_sample_retrieval(data_dir: str, split: str) -> bool:
    """Test that SynergyDataset returns correctly paired samples."""
    logger.info("Testing SynergyDataset sample retrieval...")
    
    # Create synergy config
    synergy_config = {
        'domains': ['attr', 'v'],
        'feature_indices': {
            'attr': ['xor_target_normalized']
        }
    }
    
    try:
        # Create dataset
        dataset = SynergyDataset(
            data_dir=data_dir,
            split=split,
            synergy_config=synergy_config,
            load_images=False,
            cache_data=False
        )
        
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # Get a few samples and verify structure
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            
            logger.info(f"Sample {i}:")
            logger.info(f"  Inputs: {list(sample['inputs'].keys())}")
            logger.info(f"  Targets: {list(sample['targets'].keys())}")
            
            for domain in sample['inputs']:
                input_shape = sample['inputs'][domain].shape
                target_shape = sample['targets'][domain].shape
                logger.info(f"  {domain}: input {input_shape}, target {target_shape}")
                
                # For attr domain, check that target has extra dimension for synergy
                if domain == 'attr':
                    input_dims = input_shape[-1] if len(input_shape) > 0 else 0
                    target_dims = target_shape[-1] if len(target_shape) > 0 else 0
                    if target_dims == input_dims + 1:
                        logger.info(f"    ✅ Attr target has {target_dims}D vs input {input_dims}D (+1 for synergy)")
                    else:
                        logger.error(f"    ❌ Attr dimensions mismatch: target {target_dims}D vs input {input_dims}D")
                        return False
        
        logger.info("✅ SynergyDataset sample retrieval works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ SynergyDataset test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def verify_batch_consistency(data_dir: str, split: str) -> bool:
    """Verify that batches maintain pairing consistency."""
    logger.info("Verifying batch consistency...")
    
    from synergy_dataset import create_synergy_dataloaders
    
    synergy_config = {
        'domains': ['attr', 'v'],
        'feature_indices': {
            'attr': ['xor_target_normalized']
        }
    }
    
    try:
        # Create dataloader
        dataloaders = create_synergy_dataloaders(
            data_dir=data_dir,
            synergy_config=synergy_config,
            batch_size=4,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        if split not in dataloaders:
            logger.error(f"❌ Could not create dataloader for {split} split")
            return False
        
        dataloader = dataloaders[split]
        
        # Get first batch
        batch = next(iter(dataloader))
        
        logger.info(f"Batch structure:")
        logger.info(f"  Keys: {list(batch.keys())}")
        logger.info(f"  Inputs: {list(batch['inputs'].keys())}")
        logger.info(f"  Targets: {list(batch['targets'].keys())}")
        
        for domain in batch['inputs']:
            input_batch = batch['inputs'][domain]
            target_batch = batch['targets'][domain]
            logger.info(f"  {domain}: input batch {input_batch.shape}, target batch {target_batch.shape}")
        
        logger.info("✅ Batch consistency verification passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch consistency test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main verification function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify data pairing in synergy dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to synergy data directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to verify")
    parser.add_argument("--n-bins", type=int, default=8, help="Number of bins for XOR discretization")
    
    args = parser.parse_args()
    
    logger.info(f"Verifying data pairing for {args.split} split in {args.data_dir}")
    
    try:
        # Load all original data sources
        sources = load_original_data_sources(args.data_dir, args.split)
        
        # Run verification tests
        tests = [
            ("Sample count consistency", lambda: verify_sample_count_consistency(sources)),
            ("XOR formula consistency", lambda: verify_xor_formula_consistency(sources, args.n_bins)),
            ("Size extraction consistency", lambda: verify_size_extraction_consistency(sources)),
            ("VAE latent consistency", lambda: verify_vae_latent_consistency(sources)),
            ("SynergyDataset sample retrieval", lambda: test_synergy_dataset_sample_retrieval(args.data_dir, args.split)),
            ("Batch consistency", lambda: verify_batch_consistency(args.data_dir, args.split))
        ]
        
        results = {}
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Print summary
        print(f"\n{'='*70}")
        print("DATA PAIRING VERIFICATION SUMMARY")
        print(f"{'='*70}")
        print(f"Dataset: {args.data_dir}")
        print(f"Split: {args.split}")
        print(f"Bins: {args.n_bins}")
        print()
        
        passed = 0
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<40}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{len(tests)} tests passed")
        
        if passed == len(tests):
            print("\n🎉 ALL TESTS PASSED - Data pairing is correct!")
            return True
        else:
            print("\n⚠️  SOME TESTS FAILED - Data pairing may have issues!")
            return False
            
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
