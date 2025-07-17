#!/usr/bin/env python3
"""
Create Independent Attribute and XOR Targets

This script:
1. Loads modified SimpleShapes dataset
2. Replaces the background scalar attribute (column 8) with independent random values
3. Extracts actual background colors from images
4. Creates XOR-like targets that require both visual and attribute information
5. Saves results as DataFrames for each split

The goal is to create targets that cannot be determined from either:
- Visual information (background color) alone
- Attribute information (new random attribute) alone
But require both sources jointly.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IndependentAttributeGenerator:
    """Generates independent attribute values and XOR targets."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        logger.info(f"Set random seed to {seed}")
    
    def load_labels(self, labels_path: str) -> np.ndarray:
        """Load label array from .npy file."""
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        labels = np.load(labels_path)
        logger.info(f"Loaded labels from {labels_path}: shape {labels.shape}")
        return labels
    
    def replace_background_attribute(self, labels: np.ndarray) -> np.ndarray:
        """
        Replace column 8 (background scalar) with independent random values.
        
        Args:
            labels: Original label array with shape (n_samples, 9)
            
        Returns:
            Modified label array with independent attribute in column 8
        """
        if labels.shape[1] != 9:
            raise ValueError(f"Expected 9 columns, got {labels.shape[1]}")
        
        n_samples = labels.shape[0]
        
        # Create independent random attribute values in [0, 1]
        independent_attribute = np.random.uniform(0, 1, size=n_samples)
        
        # Replace column 8
        modified_labels = labels.copy()
        modified_labels[:, 8] = independent_attribute
        
        logger.info(f"Replaced background attribute with independent values")
        logger.info(f"New attribute range: [{independent_attribute.min():.3f}, {independent_attribute.max():.3f}]")
        
        return modified_labels
    
    def extract_background_colors_from_images(self, images_dir: str, n_samples: int) -> np.ndarray:
        """
        Extract background colors from images by sampling corner pixels.
        
        Args:
            images_dir: Directory containing images (0.png, 1.png, ...)
            n_samples: Number of samples to process
            
        Returns:
            Array of background grayscale values normalized to [0, 1]
        """
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        background_colors = []
        
        logger.info(f"Extracting background colors from {n_samples} images...")
        
        for i in tqdm(range(n_samples), desc="Processing images"):
            image_path = os.path.join(images_dir, f"{i}.png")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Extract background color from corners (should be grayscale)
            # Sample multiple corner pixels to be robust
            corners = [
                img_array[0, 0],      # top-left
                img_array[0, -1],     # top-right  
                img_array[-1, 0],     # bottom-left
                img_array[-1, -1],    # bottom-right
            ]
            
            # Take mean of corner pixels (should all be same grayscale value)
            corner_mean = np.mean(corners, axis=0)
            
            # Convert to grayscale (should already be R=G=B, but take mean to be safe)
            grayscale_value = np.mean(corner_mean)
            
            # Normalize to [0, 1]
            normalized_bg = grayscale_value / 255.0
            background_colors.append(normalized_bg)
        
        background_colors = np.array(background_colors)
        logger.info(f"Extracted background colors: range [{background_colors.min():.3f}, {background_colors.max():.3f}]")
        
        return background_colors
    
    def create_xor_targets(self, visual_values: np.ndarray, attribute_values: np.ndarray, 
                          n_bins: int = 8) -> np.ndarray:
        """
        Create XOR-like targets from visual and attribute values.
        
        Both inputs are discretized into bins, then XORed to create targets
        that require both sources of information.
        
        Args:
            visual_values: Normalized visual background values [0, 1]
            attribute_values: Normalized attribute values [0, 1] 
            n_bins: Number of bins for discretization
            
        Returns:
            XOR target values
        """
        if len(visual_values) != len(attribute_values):
            raise ValueError("Visual and attribute arrays must have same length")
        
        # Discretize both inputs into bins
        visual_bins = np.digitize(visual_values, bins=np.linspace(0, 1, n_bins + 1)) - 1
        attribute_bins = np.digitize(attribute_values, bins=np.linspace(0, 1, n_bins + 1)) - 1
        
        # Clip to valid range [0, n_bins-1]
        visual_bins = np.clip(visual_bins, 0, n_bins - 1)
        attribute_bins = np.clip(attribute_bins, 0, n_bins - 1)
        
        # Create XOR targets (addition modulo n_bins)
        xor_targets = (visual_bins + attribute_bins) % n_bins
        
        # Normalize back to [0, 1] range
        normalized_targets = xor_targets.astype(np.float32) / (n_bins - 1)
        
        logger.info(f"Created XOR targets with {n_bins} bins")
        logger.info(f"Visual bins range: [{visual_bins.min()}, {visual_bins.max()}]")
        logger.info(f"Attribute bins range: [{attribute_bins.min()}, {attribute_bins.max()}]") 
        logger.info(f"XOR targets range: [{normalized_targets.min():.3f}, {normalized_targets.max():.3f}]")
        logger.info(f"XOR target distribution: {np.bincount(xor_targets)}")
        
        return normalized_targets
    
    def process_split(self, data_dir: str, split_name: str, n_bins: int = 8) -> pd.DataFrame:
        """
        Process a single dataset split.
        
        Args:
            data_dir: Base directory containing the dataset
            split_name: Name of split (train/val/test)
            n_bins: Number of bins for XOR discretization
            
        Returns:
            DataFrame with columns: [index, visual_bg, attribute_value, xor_target]
        """
        logger.info(f"Processing {split_name} split...")
        
        # Load and modify labels
        labels_path = os.path.join(data_dir, f"{split_name}_labels.npy")
        original_labels = self.load_labels(labels_path)
        modified_labels = self.replace_background_attribute(original_labels)
        
        # Save modified labels
        modified_labels_path = os.path.join(data_dir, f"{split_name}_labels_independent.npy")
        np.save(modified_labels_path, modified_labels)
        logger.info(f"Saved modified labels to {modified_labels_path}")
        
        # Extract visual background colors
        images_dir = os.path.join(data_dir, split_name)
        n_samples = len(modified_labels)
        visual_backgrounds = self.extract_background_colors_from_images(images_dir, n_samples)
        
        # Get new independent attribute values
        attribute_values = modified_labels[:, 8]  # Column 8 now contains independent values
        
        # Create XOR targets
        xor_targets = self.create_xor_targets(visual_backgrounds, attribute_values, n_bins)
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'index': np.arange(n_samples),
            'visual_background': visual_backgrounds,
            'attribute_value': attribute_values,
            'xor_target': xor_targets
        })
        
        logger.info(f"Created results DataFrame for {split_name}: shape {results_df.shape}")
        
        return results_df
    
    def process_all_splits(self, data_dir: str, splits: list = None, n_bins: int = 8) -> dict:
        """
        Process all dataset splits.
        
        Args:
            data_dir: Base directory containing dataset
            splits: List of split names to process
            n_bins: Number of bins for XOR discretization
            
        Returns:
            Dictionary mapping split names to DataFrames
        """
        if splits is None:
            splits = ['train', 'val', 'test']
        
        results = {}
        
        for split in splits:
            # Check if split exists
            labels_path = os.path.join(data_dir, f"{split}_labels.npy")
            if not os.path.exists(labels_path):
                logger.warning(f"Skipping {split} split - labels file not found")
                continue
            
            try:
                results[split] = self.process_split(data_dir, split, n_bins)
                
                # Save DataFrame
                output_path = os.path.join(data_dir, f"{split}_xor_targets.csv")
                results[split].to_csv(output_path, index=False)
                logger.info(f"Saved {split} XOR targets to {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {split} split: {e}")
                continue
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create independent attributes and XOR targets")
    
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing modified SimpleShapes dataset")
    parser.add_argument("--splits", type=str, nargs="+", 
                       default=["train", "val", "test"],
                       help="Dataset splits to process")
    parser.add_argument("--n-bins", type=int, default=8,
                       help="Number of bins for XOR discretization")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = IndependentAttributeGenerator(seed=args.seed)
    
    # Process all splits
    results = generator.process_all_splits(args.data_dir, args.splits, args.n_bins)
    
    # Print summary
    print("\n" + "="*60)
    print("INDEPENDENT ATTRIBUTE AND XOR TARGET GENERATION COMPLETE")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Processed splits: {list(results.keys())}")
    print(f"XOR bins: {args.n_bins}")
    print(f"Random seed: {args.seed}")
    
    for split, df in results.items():
        print(f"\n{split.upper()} split:")
        print(f"  Samples: {len(df)}")
        print(f"  Visual background range: [{df['visual_background'].min():.3f}, {df['visual_background'].max():.3f}]")
        print(f"  Attribute range: [{df['attribute_value'].min():.3f}, {df['attribute_value'].max():.3f}]")
        print(f"  XOR target range: [{df['xor_target'].min():.3f}, {df['xor_target'].max():.3f}]")
        print(f"  Unique XOR values: {df['xor_target'].nunique()}")
    
    print("\nFiles generated:")
    print("• <split>_labels_independent.npy - Modified labels with independent attribute")
    print("• <split>_xor_targets.csv - DataFrames with visual/attribute/XOR data")
    print("\nKey properties:")
    print("• Attribute column 8 now independent of visual background")
    print("• XOR targets require BOTH visual + attribute information") 
    print("• Neither visual nor attribute alone can determine XOR target")


if __name__ == "__main__":
    main() 