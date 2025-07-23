#!/usr/bin/env python3
"""
Create Size-Random XOR Targets for No-Size SimpleShapes Dataset

This script creates XOR-style targets that require both visual and attribute information:
1. Visual size information (preserved in images, extracted from original labels)
2. Random attribute information (column 7 from no-size labels)
3. Combines them via discretization and modular addition to create targets

The goal is to create prediction targets that cannot be determined from either:
- Visual information (size) alone
- Attribute information (random variable) alone
But require both sources jointly, enabling PID analysis of information synergy.

Key workflow:
- Load original labels (for size values) and no-size labels (for random attribute)
- Normalize size values to [0,1] range
- Discretize both size and random attribute into n_bins
- Compute XOR target as (size_bin + random_bin) % n_bins
- Save results as structured data with metadata
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import Tuple, Dict, Any

# Set up logging with detailed formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SizeRandomXORGenerator:
    """
    Main class for generating XOR targets from visual size and random attributes.
    
    This generator:
    1. Loads original labels (22 columns) to extract size values
    2. Loads no-size labels (8 columns) to extract random attribute values
    3. Normalizes size values to [0,1] range
    4. Discretizes both inputs into bins
    5. Computes XOR targets via modular addition
    6. Saves structured results with comprehensive metadata
    """
    
    def __init__(self, n_bins: int = 8, seed: int = 42):
        """
        Initialize the XOR target generator.
        
        Args:
            n_bins: Number of bins for discretization (default: 8)
            seed: Random seed for reproducibility (used for validation)
        """
        self.n_bins = n_bins
        self.seed = seed
        np.random.seed(seed)
        logger.info(f"Initialized SizeRandomXORGenerator with n_bins={n_bins}, seed={seed}")
    
    def load_label_files(self, input_dir: str, split_name: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load both original and no-size label files for a split.
        
        Args:
            input_dir: Directory containing label files
            split_name: Name of the split (train/val/test)
            
        Returns:
            Tuple of (original_labels, nosize_labels, metadata_dict)
        """
        # Define file paths
        original_path = os.path.join(input_dir, f"{split_name}_labels.npy")
        nosize_path = os.path.join(input_dir, f"{split_name}_labels_nosize.npy")
        
        # Check file existence
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Original labels file not found: {original_path}")
        if not os.path.exists(nosize_path):
            raise FileNotFoundError(f"No-size labels file not found: {nosize_path}")
        
        # Load arrays
        original_labels = np.load(original_path)
        nosize_labels = np.load(nosize_path)
        
        logger.info(f"Loaded original labels: shape {original_labels.shape}")
        logger.info(f"Loaded no-size labels: shape {nosize_labels.shape}")
        
        # Validate dimensions
        if original_labels.shape[1] < 8:
            raise ValueError(f"Original labels need at least 8 columns, got {original_labels.shape[1]}")
        if nosize_labels.shape[1] != 8:
            raise ValueError(f"No-size labels should have 8 columns, got {nosize_labels.shape[1]}")
        if len(original_labels) != len(nosize_labels):
            raise ValueError(f"Sample count mismatch: original={len(original_labels)}, no-size={len(nosize_labels)}")
        
        # Create metadata
        metadata = {
            'n_samples': len(original_labels),
            'original_shape': original_labels.shape,
            'nosize_shape': nosize_labels.shape,
            'split_name': split_name,
            'size_column_index': 3,  # Size is column 3 in original labels
            'random_column_index': 7,  # Random attribute is column 7 in no-size labels
        }
        
        logger.info(f"Validated label files for {split_name}: {metadata['n_samples']} samples")
        
        return original_labels, nosize_labels, metadata
    
    def extract_and_normalize_inputs(self, original_labels: np.ndarray, 
                                   nosize_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Extract size and random attribute values, normalizing size to [0,1].
        
        Args:
            original_labels: Original label array (contains size in column 3)
            nosize_labels: No-size label array (contains random attr in column 7)
            
        Returns:
            Tuple of (size_normalized, random_attribute, extraction_metadata)
        """
        # Extract raw size values from original labels (column 3)
        size_raw = original_labels[:, 3].astype(np.float64)
        
        # Extract random attribute values from no-size labels (column 7)
        # These should already be in [0,1] range from the no-size generation
        random_attribute = nosize_labels[:, 7].astype(np.float64)
        
        # Normalize size values to [0,1] range
        size_min = size_raw.min()
        size_max = size_raw.max()
        size_range = size_max - size_min
        
        if size_range == 0:
            logger.warning("All size values are identical, setting normalized size to 0.5")
            size_normalized = np.full_like(size_raw, 0.5)
        else:
            size_normalized = (size_raw - size_min) / size_range
        
        # Validate ranges
        assert np.all((size_normalized >= 0) & (size_normalized <= 1)), "Size normalization failed"
        assert np.all((random_attribute >= 0) & (random_attribute <= 1)), "Random attribute out of [0,1] range"
        
        # Create extraction metadata
        extraction_metadata = {
            'size_raw_stats': {
                'min': float(size_min),
                'max': float(size_max),
                'mean': float(size_raw.mean()),
                'std': float(size_raw.std()),
                'range': float(size_range)
            },
            'size_normalized_stats': {
                'min': float(size_normalized.min()),
                'max': float(size_normalized.max()),
                'mean': float(size_normalized.mean()),
                'std': float(size_normalized.std())
            },
            'random_attribute_stats': {
                'min': float(random_attribute.min()),
                'max': float(random_attribute.max()),
                'mean': float(random_attribute.mean()),
                'std': float(random_attribute.std())
            }
        }
        
        logger.info(f"Size normalization: raw range [{size_min}, {size_max}] → [0, 1]")
        logger.info(f"Random attribute range: [{random_attribute.min():.3f}, {random_attribute.max():.3f}]")
        
        return size_normalized, random_attribute, extraction_metadata
    
    def discretize_and_compute_xor(self, size_values: np.ndarray, 
                                  random_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Discretize inputs into bins and compute XOR targets via modular addition.
        
        Args:
            size_values: Normalized size values in [0,1]
            random_values: Random attribute values in [0,1]
            
        Returns:
            Tuple of (size_bins, random_bins, xor_targets, discretization_metadata)
        """
        n_samples = len(size_values)
        
        # Create bin edges for discretization
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        
        # Discretize both inputs into bins
        # np.digitize returns 1-indexed bins, so subtract 1 to get 0-indexed
        size_bins = np.digitize(size_values, bin_edges) - 1
        random_bins = np.digitize(random_values, bin_edges) - 1
        
        # Clip to valid range [0, n_bins-1] to handle edge cases
        size_bins = np.clip(size_bins, 0, self.n_bins - 1)
        random_bins = np.clip(random_bins, 0, self.n_bins - 1)
        
        # Compute XOR targets using modular addition
        # This creates targets that require both inputs to predict
        xor_bins = (size_bins + random_bins) % self.n_bins
        
        # Optionally normalize XOR targets back to [0,1] range
        xor_targets_normalized = xor_bins.astype(np.float32) / (self.n_bins - 1) if self.n_bins > 1 else np.zeros(n_samples, dtype=np.float32)
        
        # Create discretization metadata
        discretization_metadata = {
            'n_bins': self.n_bins,
            'bin_edges': bin_edges.tolist(),
            'size_bin_distribution': np.bincount(size_bins, minlength=self.n_bins).tolist(),
            'random_bin_distribution': np.bincount(random_bins, minlength=self.n_bins).tolist(),
            'xor_bin_distribution': np.bincount(xor_bins, minlength=self.n_bins).tolist(),
            'xor_targets_stats': {
                'min': float(xor_targets_normalized.min()),
                'max': float(xor_targets_normalized.max()),
                'mean': float(xor_targets_normalized.mean()),
                'std': float(xor_targets_normalized.std())
            }
        }
        
        logger.info(f"Discretized inputs into {self.n_bins} bins")
        logger.info(f"Size bin distribution: {discretization_metadata['size_bin_distribution']}")
        logger.info(f"Random bin distribution: {discretization_metadata['random_bin_distribution']}")
        logger.info(f"XOR target distribution: {discretization_metadata['xor_bin_distribution']}")
        
        return size_bins, random_bins, xor_targets_normalized, discretization_metadata
    
    def create_results_dataframe(self, size_normalized: np.ndarray, random_attribute: np.ndarray,
                               size_bins: np.ndarray, random_bins: np.ndarray, 
                               xor_targets: np.ndarray) -> pd.DataFrame:
        """
        Create a structured DataFrame with all results.
        
        Args:
            size_normalized: Normalized size values [0,1]
            random_attribute: Random attribute values [0,1]
            size_bins: Discretized size bins [0, n_bins-1]
            random_bins: Discretized random bins [0, n_bins-1]
            xor_targets: XOR target values [0,1]
            
        Returns:
            DataFrame with columns for all inputs and outputs
        """
        n_samples = len(size_normalized)
        
        # Create the DataFrame
        results_df = pd.DataFrame({
            'sample_idx': np.arange(n_samples),
            'size_normalized': size_normalized,
            'random_attribute': random_attribute,
            'size_bin': size_bins,
            'random_bin': random_bins,
            'xor_target_normalized': xor_targets,
            'xor_target_bin': (xor_targets * (self.n_bins - 1)).round().astype(int) if self.n_bins > 1 else np.zeros(n_samples, dtype=int)
        })
        
        logger.info(f"Created results DataFrame with shape {results_df.shape}")
        logger.info(f"DataFrame columns: {list(results_df.columns)}")
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, output_dir: str, split_name: str) -> None:
        """
        Save results in multiple formats.
        
        Args:
            results_df: DataFrame containing all results
            output_dir: Directory to save results
            split_name: Name of the split
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output paths
        csv_path = os.path.join(output_dir, f"{split_name}_xor_targets.csv")
        npy_path = os.path.join(output_dir, f"{split_name}_xor_targets.npy")
        
        # Save as CSV (human-readable)
        results_df.to_csv(csv_path, index=False, float_format='%.6f')
        logger.info(f"Saved results as CSV: {csv_path}")
        
        # Save as numpy array (compact)
        # Convert DataFrame to structured numpy array for efficient storage
        results_array = results_df.to_records(index=False)
        np.save(npy_path, results_array)
        logger.info(f"Saved results as numpy array: {npy_path}")
        
        # Log file sizes for reference
        csv_size = os.path.getsize(csv_path) / 1024 / 1024  # MB
        npy_size = os.path.getsize(npy_path) / 1024 / 1024  # MB
        logger.info(f"File sizes: CSV={csv_size:.2f}MB, NPY={npy_size:.2f}MB")
    
    def save_metadata(self, output_path: str, file_metadata: dict, extraction_metadata: dict,
                     discretization_metadata: dict, split_name: str) -> None:
        """
        Save comprehensive metadata about the XOR target generation process.
        
        Args:
            output_path: Path to save metadata file
            file_metadata: Metadata from file loading
            extraction_metadata: Metadata from input extraction
            discretization_metadata: Metadata from discretization
            split_name: Name of the dataset split
        """
        # Create comprehensive metadata
        full_metadata = {
            'process': 'size_random_xor_target_generation',
            'split_name': split_name,
            'n_bins': self.n_bins,
            'seed': self.seed,
            'file_info': file_metadata,
            'extraction_info': extraction_metadata,
            'discretization_info': discretization_metadata,
            'formula': f"xor_target = (size_bin + random_bin) % {self.n_bins}",
            'key_properties': [
                'Targets require both visual size and attribute random variable',
                'Neither input alone is sufficient to predict targets',
                'XOR structure creates information synergy for PID analysis',
                'Discretization ensures bounded target space'
            ]
        }
        
        # Write metadata as human-readable text
        with open(output_path, 'w') as f:
            f.write(f"Size-Random XOR Targets - {split_name} split\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Process: {full_metadata['process']}\n")
            f.write(f"Number of bins: {full_metadata['n_bins']}\n")
            f.write(f"Random seed: {full_metadata['seed']}\n")
            f.write(f"Target formula: {full_metadata['formula']}\n")
            f.write(f"Processing date: {os.popen('date').read().strip()}\n\n")
            
            f.write("INPUT FILES:\n")
            f.write(f"  Samples: {file_metadata['n_samples']}\n")
            f.write(f"  Original labels shape: {file_metadata['original_shape']}\n")
            f.write(f"  No-size labels shape: {file_metadata['nosize_shape']}\n")
            f.write(f"  Size column: {file_metadata['size_column_index']}\n")
            f.write(f"  Random column: {file_metadata['random_column_index']}\n\n")
            
            f.write("INPUT PROCESSING:\n")
            f.write(f"  Size raw range: [{extraction_metadata['size_raw_stats']['min']}, {extraction_metadata['size_raw_stats']['max']}]\n")
            f.write(f"  Size normalized: [{extraction_metadata['size_normalized_stats']['min']}, {extraction_metadata['size_normalized_stats']['max']}]\n")
            f.write(f"  Random attribute: [{extraction_metadata['random_attribute_stats']['min']:.3f}, {extraction_metadata['random_attribute_stats']['max']:.3f}]\n\n")
            
            f.write("DISCRETIZATION:\n")
            f.write(f"  Bins: {discretization_metadata['n_bins']}\n")
            f.write(f"  Bin edges: {discretization_metadata['bin_edges']}\n")
            f.write(f"  Size distribution: {discretization_metadata['size_bin_distribution']}\n")
            f.write(f"  Random distribution: {discretization_metadata['random_bin_distribution']}\n")
            f.write(f"  XOR distribution: {discretization_metadata['xor_bin_distribution']}\n\n")
            
            f.write("TARGET PROPERTIES:\n")
            for prop in full_metadata['key_properties']:
                f.write(f"  • {prop}\n")
            f.write(f"\nTarget stats: min={discretization_metadata['xor_targets_stats']['min']:.3f}, ")
            f.write(f"max={discretization_metadata['xor_targets_stats']['max']:.3f}, ")
            f.write(f"mean={discretization_metadata['xor_targets_stats']['mean']:.3f}, ")
            f.write(f"std={discretization_metadata['xor_targets_stats']['std']:.3f}\n")
        
        logger.info(f"Saved comprehensive metadata to {output_path}")
    
    def process_split(self, input_dir: str, output_dir: str, split_name: str) -> bool:
        """
        Process a complete dataset split to generate XOR targets.
        
        Args:
            input_dir: Directory containing input label files
            output_dir: Directory for output files
            split_name: Name of the split to process
            
        Returns:
            True if processing succeeded, False otherwise
        """
        logger.info(f"Processing {split_name} split...")
        
        try:
            # Step 1: Load label files
            original_labels, nosize_labels, file_metadata = self.load_label_files(input_dir, split_name)
            
            # Step 2: Extract and normalize inputs
            size_normalized, random_attribute, extraction_metadata = self.extract_and_normalize_inputs(
                original_labels, nosize_labels
            )
            
            # Step 3: Discretize and compute XOR targets
            size_bins, random_bins, xor_targets, discretization_metadata = self.discretize_and_compute_xor(
                size_normalized, random_attribute
            )
            
            # Step 4: Create results DataFrame
            results_df = self.create_results_dataframe(
                size_normalized, random_attribute, size_bins, random_bins, xor_targets
            )
            
            # Step 5: Save results
            self.save_results(results_df, output_dir, split_name)
            
            # Step 6: Save metadata
            metadata_path = os.path.join(output_dir, f"{split_name}_metadata.txt")
            self.save_metadata(metadata_path, file_metadata, extraction_metadata, 
                             discretization_metadata, split_name)
            
            logger.info(f"Successfully completed processing {split_name} split")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {split_name} split: {e}")
            return False
    
    def validate_output(self, output_dir: str, split_name: str) -> bool:
        """
        Validate the generated output for consistency and correctness.
        
        Args:
            output_dir: Output directory to validate
            split_name: Split name to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info(f"Validating output for {split_name} split...")
        
        # Check if output files exist
        csv_path = os.path.join(output_dir, f"{split_name}_xor_targets.csv")
        npy_path = os.path.join(output_dir, f"{split_name}_xor_targets.npy")
        metadata_path = os.path.join(output_dir, f"{split_name}_metadata.txt")
        
        for path in [csv_path, npy_path, metadata_path]:
            if not os.path.exists(path):
                logger.error(f"Missing output file: {path}")
                return False
        
        try:
            # Load and validate results
            results_df = pd.read_csv(csv_path)
            results_array = np.load(npy_path)
            
            # Check DataFrame structure
            expected_columns = ['sample_idx', 'size_normalized', 'random_attribute', 
                              'size_bin', 'random_bin', 'xor_target_normalized', 'xor_target_bin']
            if list(results_df.columns) != expected_columns:
                logger.error(f"Unexpected columns: {list(results_df.columns)}")
                return False
            
            # Check value ranges
            if not np.all((results_df['size_normalized'] >= 0) & (results_df['size_normalized'] <= 1)):
                logger.error("Size normalized values out of [0,1] range")
                return False
            
            if not np.all((results_df['random_attribute'] >= 0) & (results_df['random_attribute'] <= 1)):
                logger.error("Random attribute values out of [0,1] range")
                return False
            
            if not np.all((results_df['xor_target_normalized'] >= 0) & (results_df['xor_target_normalized'] <= 1)):
                logger.error("XOR target values out of [0,1] range")
                return False
            
            # Check bin ranges
            if not np.all((results_df['size_bin'] >= 0) & (results_df['size_bin'] < self.n_bins)):
                logger.error(f"Size bin values out of [0,{self.n_bins}) range")
                return False
            
            if not np.all((results_df['random_bin'] >= 0) & (results_df['random_bin'] < self.n_bins)):
                logger.error(f"Random bin values out of [0,{self.n_bins}) range")
                return False
            
            if not np.all((results_df['xor_target_bin'] >= 0) & (results_df['xor_target_bin'] < self.n_bins)):
                logger.error(f"XOR target bin values out of [0,{self.n_bins}) range")
                return False
            
            # Verify XOR formula
            expected_xor_bins = (results_df['size_bin'] + results_df['random_bin']) % self.n_bins
            if not np.all(results_df['xor_target_bin'] == expected_xor_bins):
                logger.error("XOR target bins do not match expected formula")
                return False
            
            logger.info(f"Validation passed for {split_name}: {len(results_df)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed for {split_name}: {e}")
            return False


def main():
    """Main function to run the size-random XOR target generation process."""
    
    parser = argparse.ArgumentParser(
        description="Generate XOR targets from visual size and random attributes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script generates XOR-style targets that require both visual and attribute information:

1. Loads original labels (for size values) and no-size labels (for random attributes)
2. Normalizes size values to [0,1] range
3. Discretizes both inputs into n_bins 
4. Computes XOR targets as (size_bin + random_bin) % n_bins
5. Saves results as CSV and numpy arrays with comprehensive metadata

The resulting targets can only be predicted using both modalities jointly,
making them ideal for PID analysis of information synergy.

Example usage:
  python create_size_random_xor_targets.py \\
      --input-dir data/simple_shapes_nosize \\
      --output-dir data/simple_shapes_xor_targets \\
      --splits train val test \\
      --n-bins 8 \\
      --seed 42
        """
    )
    
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing no-size SimpleShapes data")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save XOR target results")
    parser.add_argument("--splits", type=str, nargs="+", 
                       default=["train", "val", "test"],
                       help="Dataset splits to process (default: train val test)")
    parser.add_argument("--n-bins", type=int, default=8,
                       help="Number of bins for discretization (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Validate n_bins
    if args.n_bins < 2:
        logger.error(f"n_bins must be at least 2, got {args.n_bins}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize generator
    generator = SizeRandomXORGenerator(n_bins=args.n_bins, seed=args.seed)
    
    # Process each split
    successful_splits = []
    failed_splits = []
    
    for split in args.splits:
        try:
            success = generator.process_split(args.input_dir, args.output_dir, split)
            
            if success and generator.validate_output(args.output_dir, split):
                successful_splits.append(split)
            else:
                failed_splits.append(split)
                
        except Exception as e:
            logger.error(f"Error processing {split} split: {e}")
            failed_splits.append(split)
    
    # Print final summary
    print("\n" + "="*70)
    print("SIZE-RANDOM XOR TARGET GENERATION COMPLETE")
    print("="*70)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of bins:   {args.n_bins}")
    print(f"Random seed:      {args.seed}")
    print(f"Successful splits: {successful_splits}")
    if failed_splits:
        print(f"Failed splits:     {failed_splits}")
    
    print("\nTarget generation:")
    print(f"• Formula: xor_target = (size_bin + random_bin) % {args.n_bins}")
    print("• Requires both visual size and attribute random variable")
    print("• Neither input alone is sufficient for prediction")
    print("• Creates information synergy for PID analysis")
    
    print("\nFiles generated per split:")
    print("• <split>_xor_targets.csv - Human-readable results")
    print("• <split>_xor_targets.npy - Compact numpy format")
    print("• <split>_metadata.txt    - Comprehensive processing metadata")
    
    if failed_splits:
        print(f"\nWARNING: {len(failed_splits)} splits failed. Check logs above.")
        sys.exit(1)
    else:
        print(f"\nSUCCESS: All {len(successful_splits)} splits processed successfully!")


if __name__ == "__main__":
    main() 