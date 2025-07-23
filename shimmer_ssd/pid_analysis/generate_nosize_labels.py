#!/usr/bin/env python3
"""
Generate SimpleShapes Dataset with Size Attribute Removed

This script creates a modified version of the SimpleShapes dataset where:
1. Visual images remain completely unchanged (preserving visual size information)
2. The "size" attribute (column 3) is removed from the attribute vectors
3. A new uniform random variable is added to maintain dimensionality
4. Original images are symlinked to keep everything together

The goal is to test PID analysis when size information is absent from the attribute
side but still present in the visual side, creating a specific information asymmetry.

Key differences from generate_modified_shapes.py:
- No visual modifications (no recoloring, no background changes)
- Much simpler: only label manipulation
- Removes size attribute instead of adding background scalar
- Uses symlinks for images instead of regenerating them
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from typing import Tuple, List

# Set up logging with detailed formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NoSizeDatasetModifier:
    """
    Main class for removing size attributes from SimpleShapes datasets.
    
    This modifier:
    1. Loads original label files (22 columns, uses first 8)
    2. Removes the size attribute (column 3 of the core 8)
    3. Adds a uniform random variable to maintain 8-dimensional attributes
    4. Symlinks original images to output directory
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the dataset modifier.
        
        Args:
            seed: Random seed for reproducible uniform random variable generation
        """
        self.seed = seed
        np.random.seed(seed)
        logger.info(f"Initialized NoSizeDatasetModifier with seed={seed}")
    
    def load_original_labels(self, labels_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load original SimpleShapes dataset labels.
        
        The original dataset has 22 columns, but we only use the first 8 core attributes:
        [0: category, 1: x, 2: y, 3: size, 4: rotation, 5: r, 6: g, 7: b]
        
        Args:
            labels_path: Path to .npy file with original labels
            
        Returns:
            Tuple of (core_labels_array, metadata_dict)
        """
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        # Load full label array (typically 22 columns)
        full_labels = np.load(labels_path)
        logger.info(f"Loaded dataset with {len(full_labels)} samples from {labels_path}")
        logger.info(f"Full label array shape: {full_labels.shape}")
        
        # Validate minimum required columns
        if full_labels.shape[1] < 8:
            raise ValueError(f"Dataset has {full_labels.shape[1]} columns, need at least 8")
        
        # Extract only the first 8 core columns
        core_labels = full_labels[:, :8]  # [category, x, y, size, rotation, r, g, b]
        
        # Create metadata for tracking
        metadata = {
            'n_samples': len(full_labels),
            'original_full_shape': full_labels.shape,
            'core_shape': core_labels.shape,
            'core_columns': ['category', 'x', 'y', 'size', 'rotation', 'r', 'g', 'b'],
            'size_column_index': 3,  # This is what we'll remove
        }
        
        logger.info(f"Extracted core labels shape: {core_labels.shape}")
        logger.info(f"Core columns: {metadata['core_columns']}")
        
        return core_labels, metadata
    
    def create_nosize_labels(self, core_labels: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Create new label array with size attribute removed and uniform random variable added.
        
        Original core layout (8 columns):
        [0: category, 1: x, 2: y, 3: size, 4: rotation, 5: r, 6: g, 7: b]
        
        New layout (8 columns):
        [0: category, 1: x, 2: y, 3: rotation, 4: r, 5: g, 6: b, 7: uniform_random]
        
        Args:
            core_labels: Original core label array (n_samples, 8)
            
        Returns:
            Tuple of (new_labels_array, generation_metadata)
        """
        n_samples = len(core_labels)
        logger.info(f"Creating no-size labels for {n_samples} samples")
        
        # Extract components (excluding size at index 3)
        category = core_labels[:, 0:1]        # Column 0: category
        location = core_labels[:, 1:3]        # Columns 1-2: x, y
        # Skip size at index 3 - this is the key modification!
        rotation = core_labels[:, 4:5]        # Column 4: rotation
        colors = core_labels[:, 5:8]          # Columns 5-7: r, g, b
        
        # Generate uniform random variable to replace size
        # Using U(0,1) distribution for consistency and interpretability
        uniform_random = np.random.rand(n_samples, 1).astype(np.float32)
        
        # Assemble new label array
        new_labels = np.concatenate([
            category,       # Column 0: category (unchanged)
            location,       # Columns 1-2: x, y (unchanged)  
            rotation,       # Column 3: rotation (shifted from index 4)
            colors,         # Columns 4-6: r, g, b (shifted from indices 5-7)
            uniform_random  # Column 7: new uniform random variable
        ], axis=1)
        
        # Verify shape consistency
        assert new_labels.shape == (n_samples, 8), f"Expected shape ({n_samples}, 8), got {new_labels.shape}"
        
        # Create metadata about the transformation
        generation_metadata = {
            'removed_attribute': 'size',
            'removed_from_index': 3,
            'added_attribute': 'uniform_random',
            'added_at_index': 7,
            'uniform_range': [0.0, 1.0],
            'uniform_stats': {
                'min': float(uniform_random.min()),
                'max': float(uniform_random.max()),
                'mean': float(uniform_random.mean()),
                'std': float(uniform_random.std())
            },
            'new_column_mapping': {
                0: 'category',
                1: 'x_location',
                2: 'y_location', 
                3: 'rotation',
                4: 'color_r',
                5: 'color_g',
                6: 'color_b',
                7: 'uniform_random'
            }
        }
        
        logger.info(f"Generated new labels shape: {new_labels.shape}")
        logger.info(f"Uniform random stats: min={generation_metadata['uniform_stats']['min']:.3f}, "
                   f"max={generation_metadata['uniform_stats']['max']:.3f}, "
                   f"mean={generation_metadata['uniform_stats']['mean']:.3f}, "
                   f"std={generation_metadata['uniform_stats']['std']:.3f}")
        
        return new_labels, generation_metadata
    
    def save_nosize_labels(self, new_labels: np.ndarray, output_path: str) -> None:
        """
        Save the modified labels to disk.
        
        Args:
            new_labels: Modified label array (n_samples, 8)
            output_path: Path to save the new labels
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as numpy array
        np.save(output_path, new_labels)
        logger.info(f"Saved no-size labels to {output_path}")
        logger.info(f"Saved array shape: {new_labels.shape}")
    
    def symlink_images(self, input_images_dir: str, output_images_dir: str) -> None:
        """
        Create symbolic links to original images in the output directory.
        
        This preserves the original images exactly while organizing them
        in the output directory structure.
        
        Args:
            input_images_dir: Source directory containing original images
            output_images_dir: Target directory for symlinked images
        """
        if not os.path.exists(input_images_dir):
            logger.warning(f"Input images directory not found: {input_images_dir}")
            return
        
        # Create output directory
        os.makedirs(output_images_dir, exist_ok=True)
        
        # Get list of image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(Path(input_images_dir).glob(ext))
        
        if not image_files:
            logger.warning(f"No image files found in {input_images_dir}")
            return
        
        logger.info(f"Creating symbolic links for {len(image_files)} images")
        
        # Create symbolic links for each image
        for image_file in tqdm(image_files, desc="Symlinking images"):
            source_path = image_file.resolve()  # Get absolute path
            target_path = Path(output_images_dir) / image_file.name
            
            # Remove existing link/file if it exists
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            
            # Create symbolic link
            try:
                target_path.symlink_to(source_path)
            except Exception as e:
                logger.error(f"Failed to create symlink for {image_file.name}: {e}")
        
        logger.info(f"Completed symlinking images to {output_images_dir}")
    
    def save_metadata(self, output_path: str, original_metadata: dict, 
                     generation_metadata: dict, split_name: str) -> None:
        """
        Save comprehensive metadata about the dataset modification.
        
        Args:
            output_path: Path to save metadata file
            original_metadata: Metadata from original dataset loading
            generation_metadata: Metadata from label generation
            split_name: Name of the dataset split (train/val/test)
        """
        # Combine all metadata
        full_metadata = {
            'dataset_modification': 'size_attribute_removal',
            'split_name': split_name,
            'seed': self.seed,
            'original_data': original_metadata,
            'label_generation': generation_metadata,
            'key_changes': [
                'Removed size attribute from attribute vectors',
                'Added uniform random variable U(0,1) as replacement',
                'Visual images unchanged (size still visible)',
                'Images symlinked (not copied) to preserve disk space'
            ]
        }
        
        # Write metadata as human-readable text
        with open(output_path, 'w') as f:
            f.write(f"No-Size SimpleShapes Dataset - {split_name} split\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Modification: {full_metadata['dataset_modification']}\n")
            f.write(f"Random seed: {full_metadata['seed']}\n")
            f.write(f"Processing date: {os.popen('date').read().strip()}\n\n")
            
            f.write("ORIGINAL DATASET:\n")
            f.write(f"  Samples: {original_metadata['n_samples']}\n")
            f.write(f"  Full shape: {original_metadata['original_full_shape']}\n")
            f.write(f"  Core shape: {original_metadata['core_shape']}\n")
            f.write(f"  Core columns: {original_metadata['core_columns']}\n\n")
            
            f.write("MODIFICATIONS:\n")
            f.write(f"  Removed: {generation_metadata['removed_attribute']} (from index {generation_metadata['removed_from_index']})\n")
            f.write(f"  Added: {generation_metadata['added_attribute']} (at index {generation_metadata['added_at_index']})\n")
            f.write(f"  Uniform range: {generation_metadata['uniform_range']}\n")
            f.write(f"  Uniform stats: {generation_metadata['uniform_stats']}\n\n")
            
            f.write("NEW LABEL FORMAT (8 columns):\n")
            for idx, col_name in generation_metadata['new_column_mapping'].items():
                f.write(f"  {idx}: {col_name}\n")
            f.write("\n")
            
            f.write("KEY CHANGES:\n")
            for change in full_metadata['key_changes']:
                f.write(f"  • {change}\n")
        
        logger.info(f"Saved metadata to {output_path}")
    
    def process_split(self, input_dir: str, output_dir: str, split_name: str) -> None:
        """
        Process a complete dataset split (train/val/test).
        
        Args:
            input_dir: Directory containing original dataset
            output_dir: Directory for modified dataset output
            split_name: Name of the split to process
        """
        logger.info(f"Processing {split_name} split...")
        
        # Define paths
        input_labels_path = os.path.join(input_dir, f"{split_name}_labels.npy")
        input_images_dir = os.path.join(input_dir, split_name)
        
        output_labels_path = os.path.join(output_dir, f"{split_name}_labels_nosize.npy")
        output_images_dir = os.path.join(output_dir, split_name)
        output_metadata_path = os.path.join(output_dir, f"{split_name}_metadata.txt")
        
        # Step 1: Load original labels
        try:
            core_labels, original_metadata = self.load_original_labels(input_labels_path)
        except Exception as e:
            logger.error(f"Failed to load {split_name} labels: {e}")
            return
        
        # Step 2: Create modified labels (remove size, add uniform random)
        try:
            new_labels, generation_metadata = self.create_nosize_labels(core_labels)
        except Exception as e:
            logger.error(f"Failed to create no-size labels for {split_name}: {e}")
            return
        
        # Step 3: Save modified labels
        try:
            self.save_nosize_labels(new_labels, output_labels_path)
        except Exception as e:
            logger.error(f"Failed to save {split_name} labels: {e}")
            return
        
        # Step 4: Symlink original images
        try:
            self.symlink_images(input_images_dir, output_images_dir)
        except Exception as e:
            logger.error(f"Failed to symlink {split_name} images: {e}")
            # Continue even if image symlinking fails
        
        # Step 5: Save metadata
        try:
            self.save_metadata(output_metadata_path, original_metadata, 
                             generation_metadata, split_name)
        except Exception as e:
            logger.error(f"Failed to save {split_name} metadata: {e}")
            # Continue even if metadata saving fails
        
        logger.info(f"Completed processing {split_name} split")
    
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
        labels_path = os.path.join(output_dir, f"{split_name}_labels_nosize.npy")
        images_dir = os.path.join(output_dir, split_name)
        metadata_path = os.path.join(output_dir, f"{split_name}_metadata.txt")
        
        if not os.path.exists(labels_path):
            logger.error(f"Missing labels file: {labels_path}")
            return False
        
        if not os.path.exists(images_dir):
            logger.error(f"Missing images directory: {images_dir}")
            return False
        
        if not os.path.exists(metadata_path):
            logger.warning(f"Missing metadata file: {metadata_path}")
        
        # Validate label array properties
        try:
            labels = np.load(labels_path)
            
            # Check shape
            if labels.shape[1] != 8:
                logger.error(f"Expected 8 columns, got {labels.shape[1]}")
                return False
            
            # Check uniform random column (should be last column)
            uniform_col = labels[:, -1]
            if uniform_col.min() < 0 or uniform_col.max() > 1:
                logger.error(f"Uniform random column out of range [0,1]: [{uniform_col.min()}, {uniform_col.max()}]")
                return False
            
            # Basic statistical check for uniformity (rough heuristic)
            expected_std = np.sqrt(1/12)  # Standard deviation of U(0,1)
            actual_std = uniform_col.std()
            if abs(actual_std - expected_std) > 0.1:
                logger.warning(f"Uniform random std deviation check: expected≈{expected_std:.3f}, got {actual_std:.3f}")
            
            logger.info(f"Validation passed for {split_name}: shape={labels.shape}, uniform_range=[{uniform_col.min():.3f}, {uniform_col.max():.3f}]")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed for {split_name}: {e}")
            return False


def main():
    """Main function to run the no-size dataset modification process."""
    
    parser = argparse.ArgumentParser(
        description="Generate SimpleShapes dataset with size attribute removed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script modifies SimpleShapes datasets by:
1. Removing the 'size' attribute from attribute vectors
2. Adding a uniform random variable U(0,1) as replacement
3. Keeping visual images exactly the same (size still visible)
4. Symlinking images to preserve disk space

Example usage:
  python generate_nosize_labels.py \\
      --input-dir data/simple_shapes_original \\
      --output-dir data/simple_shapes_nosize \\
      --splits train val test \\
      --seed 42
        """
    )
    
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing original SimpleShapes data")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save modified dataset")
    parser.add_argument("--splits", type=str, nargs="+", 
                       default=["train", "val", "test"],
                       help="Dataset splits to process (default: train val test)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible uniform random generation")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize modifier
    modifier = NoSizeDatasetModifier(seed=args.seed)
    
    # Process each split
    successful_splits = []
    failed_splits = []
    
    for split in args.splits:
        try:
            modifier.process_split(args.input_dir, args.output_dir, split)
            
            # Validate the output
            if modifier.validate_output(args.output_dir, split):
                successful_splits.append(split)
            else:
                failed_splits.append(split)
                
        except Exception as e:
            logger.error(f"Error processing {split} split: {e}")
            failed_splits.append(split)
    
    # Print final summary
    print("\n" + "="*70)
    print("NO-SIZE SIMPLESHAPES DATASET GENERATION COMPLETE")
    print("="*70)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed:      {args.seed}")
    print(f"Successful splits: {successful_splits}")
    if failed_splits:
        print(f"Failed splits:     {failed_splits}")
    
    print("\nKey modifications:")
    print("• Size attribute removed from attribute vectors")
    print("• Uniform random variable U(0,1) added as replacement")
    print("• Visual images unchanged (size information preserved visually)")
    print("• Images symlinked to save disk space")
    
    print("\nFiles generated per split:")
    print("• <split>_labels_nosize.npy - Modified attribute vectors (8 cols)")
    print("• <split>/                  - Symlinked images (unchanged)")
    print("• <split>_metadata.txt      - Processing metadata")
    
    if failed_splits:
        print(f"\nWARNING: {len(failed_splits)} splits failed. Check logs above.")
        sys.exit(1)
    else:
        print(f"\nSUCCESS: All {len(successful_splits)} splits processed successfully!")


if __name__ == "__main__":
    main() 