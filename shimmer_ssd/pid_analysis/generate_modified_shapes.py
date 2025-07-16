#!/usr/bin/env python3
"""
Generate Modified SimpleShapes Dataset

This script creates a new version of the SimpleShapes dataset with:
1. RGB space partitioned into grayscale and non-grayscale regions
2. Shapes colored from non-grayscale RGB region
3. Backgrounds colored from grayscale region
4. Extended attribute vectors (9 components: original 8 + background grayscale scalar)

The script loads existing SimpleShapes data and generates new versions without
modifying the original data in place.

Author: AI Assistant
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, List, Optional
import argparse
import logging
from typing import cast

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModifiedDataset:
    """Extended dataset structure with background colors"""
    classes: np.ndarray          # Shape categories
    locations: np.ndarray        # Shape positions (x, y)
    sizes: np.ndarray           # Shape sizes
    rotations: np.ndarray       # Shape rotations
    colors: np.ndarray          # Shape colors (RGB)
    background_colors: np.ndarray  # Background colors (grayscale RGB)
    background_scalars: np.ndarray # Background grayscale scalars [0,1]
    unpaired: np.ndarray        # Unpaired attribute

class RGBSpacePartitioner:
    """
    Partitions RGB space into grayscale and non-grayscale regions.
    
    Grayscale region: R = G = B (forms a line from black to white)
    Non-grayscale region: All other RGB combinations
    """
    
    def __init__(self, rgb_resolution: int = 256):
        """
        Initialize the RGB space partitioner.
        
        Args:
            rgb_resolution: Resolution of RGB space (typically 256 for 8-bit)
        """
        self.rgb_resolution = rgb_resolution
        self.max_val = rgb_resolution - 1
        
    def sample_grayscale_colors(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample colors uniformly from the grayscale region.
        
        Args:
            n_samples: Number of colors to sample
            
        Returns:
            Tuple of (rgb_colors, grayscale_scalars)
            - rgb_colors: Array of shape (n_samples, 3) with R=G=B
            - grayscale_scalars: Array of shape (n_samples,) with values [0,1]
        """
        # Sample grayscale values uniformly from [0, max_val]
        gray_values = np.random.randint(0, self.rgb_resolution, size=n_samples)
        
        # Create RGB colors where R=G=B
        rgb_colors = np.stack([gray_values, gray_values, gray_values], axis=1)
        
        # Normalize to [0,1] for scalars
        grayscale_scalars = gray_values.astype(np.float32) / self.max_val
        
        return rgb_colors, grayscale_scalars
    
    def sample_non_grayscale_colors(self, n_samples: int) -> np.ndarray:
        """
        Sample colors uniformly from the non-grayscale region.
        
        This samples from the entire RGB cube and rejects grayscale colors
        (where R=G=B) to ensure uniform sampling from non-grayscale region.
        
        Args:
            n_samples: Number of colors to sample
            
        Returns:
            Array of shape (n_samples, 3) with RGB colors
        """
        colors = []
        
        # Use rejection sampling to avoid grayscale colors
        while len(colors) < n_samples:
            # Sample more than needed to account for rejections
            batch_size = min(n_samples * 2, 10000)  # Avoid memory issues
            
            # Sample RGB values uniformly
            candidates = np.random.randint(0, self.rgb_resolution, size=(batch_size, 3))
            
            # Reject grayscale colors (where R=G=B)
            non_grayscale_mask = ~((candidates[:, 0] == candidates[:, 1]) & 
                                 (candidates[:, 1] == candidates[:, 2]))
            
            valid_candidates = candidates[non_grayscale_mask]
            
            # Add to our collection
            remaining_needed = n_samples - len(colors)
            colors.extend(valid_candidates[:remaining_needed])
        
        return np.array(colors[:n_samples])
    
    def is_grayscale_color(self, rgb_color: np.ndarray) -> bool:
        """Check if an RGB color is grayscale (R=G=B)"""
        return rgb_color[0] == rgb_color[1] == rgb_color[2]

class ModifiedImageGenerator:
    """
    Generates images with colored backgrounds and shapes from different RGB regions.
    """
    
    def __init__(self, imsize: int = 32):
        """
        Initialize the image generator.
        
        Args:
            imsize: Size of generated images in pixels
        """
        self.imsize = imsize
    
    def get_transformed_coordinates(self, coordinates: np.ndarray, origin: np.ndarray, 
                                  scale: float, rotation: float) -> np.ndarray:
        """Transform shape coordinates with scale, rotation, and translation."""
        center = np.array([[0.5, 0.5]])
        rotation_m = np.array([
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ])
        rotated_coordinates = (coordinates - center) @ rotation_m.T
        return origin + scale * rotated_coordinates
    
    def get_diamond_patch(self, location: np.ndarray, scale: int, rotation: float, 
                         color: np.ndarray) -> patches.Polygon:
        """Create a diamond shape patch."""
        coordinates = np.array([[0.5, 0.0], [1, 0.3], [0.5, 1], [0, 0.3]])
        transformed_coords = self.get_transformed_coordinates(
            coordinates, location, scale, rotation
        )
        return patches.Polygon(transformed_coords, closed=True, facecolor=color, 
                             edgecolor='none')
    
    def get_egg_patch(self, location: np.ndarray, scale: int, rotation: float, 
                     color: np.ndarray) -> patches.Polygon:
        """Create an egg shape patch."""
        # Create egg-like shape (ellipse-like but asymmetric)
        theta = np.linspace(0, 2*np.pi, 20)
        # Asymmetric radial function for egg shape
        r = 0.3 + 0.2 * np.cos(theta) - 0.1 * np.cos(2*theta)
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta) * 0.8  # Slightly compressed vertically
        coordinates = np.column_stack([x, y])
        
        transformed_coords = self.get_transformed_coordinates(
            coordinates, location, scale, rotation
        )
        return patches.Polygon(transformed_coords, closed=True, facecolor=color, 
                             edgecolor='none')
    
    def get_triangle_patch(self, location: np.ndarray, scale: int, rotation: float, 
                          color: np.ndarray) -> patches.Polygon:
        """Create a triangle shape patch."""
        coordinates = np.array([[0.5, 0.1], [0.9, 0.9], [0.1, 0.9]])
        transformed_coords = self.get_transformed_coordinates(
            coordinates, location, scale, rotation
        )
        return patches.Polygon(transformed_coords, closed=True, facecolor=color, 
                             edgecolor='none')
    
    def generate_image(self, ax: Axes, cls: int, location: np.ndarray, scale: int,
                      rotation: float, shape_color: np.ndarray, background_color: np.ndarray) -> None:
        """
        Generate an image with colored background and shape.
        
        Args:
            ax: Matplotlib axes to draw on
            cls: Shape class (0=diamond, 1=egg, 2=triangle)
            location: Shape position
            scale: Shape size
            rotation: Shape rotation
            shape_color: RGB color for shape [0,255]
            background_color: RGB color for background [0,255]
        """
        # Normalize colors to [0,1] for matplotlib
        shape_color_norm = shape_color.astype(np.float32) / 255
        background_color_norm = background_color.astype(np.float32) / 255
        
        # Set background color
        ax.set_facecolor(background_color_norm)
        
        # Create shape patch based on class
        if cls == 0:
            patch = self.get_diamond_patch(location, scale, rotation, shape_color_norm)
        elif cls == 1:
            patch = self.get_egg_patch(location, scale, rotation, shape_color_norm)
        elif cls == 2:
            patch = self.get_triangle_patch(location, scale, rotation, shape_color_norm)
        else:
            raise ValueError(f"Unknown shape class: {cls}")
        
        # Add shape to axes
        ax.add_patch(patch)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlim(0, self.imsize)
        ax.set_ylim(0, self.imsize)

class DatasetModifier:
    """
    Main class for modifying SimpleShapes datasets with new color scheme.
    """
    
    def __init__(self, rgb_resolution: int = 256, imsize: int = 32):
        """
        Initialize the dataset modifier.
        
        Args:
            rgb_resolution: RGB space resolution
            imsize: Image size in pixels
        """
        self.partitioner = RGBSpacePartitioner(rgb_resolution)
        self.image_generator = ModifiedImageGenerator(imsize)
        self.imsize = imsize
    
    def load_original_dataset(self, labels_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load original SimpleShapes dataset labels.
        
        Args:
            labels_path: Path to .npy file with labels
            
        Returns:
            Tuple of (labels_array, metadata_dict)
        """
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        labels = np.load(labels_path)
        logger.info(f"Loaded dataset with {len(labels)} samples from {labels_path}")
        logger.info(f"Label array shape: {labels.shape}")
        
        # Parse label structure (based on save_labels function)
        # [category, x, y, size, rotation, r, g, b, h, l, s, unpaired]
        metadata = {
            'n_samples': len(labels),
            'original_shape': labels.shape,
            'column_mapping': {
                0: 'category',
                1: 'x_location', 
                2: 'y_location',
                3: 'size',
                4: 'rotation',
                5: 'color_r',
                6: 'color_g', 
                7: 'color_b',
                8: 'hls_h',
                9: 'hls_l',
                10: 'hls_s',
                11: 'unpaired'
            }
        }
        
        return labels, metadata
    
    def create_modified_dataset(self, original_labels: np.ndarray) -> ModifiedDataset:
        """
        Create modified dataset with new color scheme.
        
        Args:
            original_labels: Original label array
            
        Returns:
            ModifiedDataset with new colors and extended attributes
        """
        n_samples = len(original_labels)
        logger.info(f"Creating modified dataset for {n_samples} samples")
        
        # Extract original attributes
        classes = original_labels[:, 0].astype(int)
        locations = original_labels[:, 1:3]  # x, y
        sizes = original_labels[:, 3].astype(int)
        rotations = original_labels[:, 4]
        unpaired = original_labels[:, 11]
        
        # Generate new shape colors from non-grayscale region
        logger.info("Generating new shape colors from non-grayscale RGB region...")
        new_shape_colors = self.partitioner.sample_non_grayscale_colors(n_samples)
        
        # Generate background colors from grayscale region
        logger.info("Generating background colors from grayscale RGB region...")
        background_colors, background_scalars = self.partitioner.sample_grayscale_colors(n_samples)
        
        # Log color statistics
        logger.info(f"Shape colors - R: [{new_shape_colors[:, 0].min()}, {new_shape_colors[:, 0].max()}]")
        logger.info(f"Shape colors - G: [{new_shape_colors[:, 1].min()}, {new_shape_colors[:, 1].max()}]")
        logger.info(f"Shape colors - B: [{new_shape_colors[:, 2].min()}, {new_shape_colors[:, 2].max()}]")
        logger.info(f"Background scalars: [{background_scalars.min():.3f}, {background_scalars.max():.3f}]")
        
        return ModifiedDataset(
            classes=classes,
            locations=locations,
            sizes=sizes,
            rotations=rotations,
            colors=new_shape_colors,
            background_colors=background_colors,
            background_scalars=background_scalars,
            unpaired=unpaired
        )
    
    def save_modified_images(self, dataset: ModifiedDataset, output_dir: str) -> None:
        """
        Save modified images to disk.
        
        Args:
            dataset: Modified dataset
            output_dir: Directory to save images
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving {len(dataset.classes)} modified images to {output_dir}")
        
        dpi = 1
        
        for i in tqdm(range(len(dataset.classes)), desc="Generating images"):
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(self.imsize / dpi, self.imsize / dpi), dpi=dpi)
            ax = cast(Axes, ax)
            
            # Generate image with new colors
            self.image_generator.generate_image(
                ax=ax,
                cls=dataset.classes[i],
                location=dataset.locations[i],
                scale=dataset.sizes[i],
                rotation=dataset.rotations[i],
                shape_color=dataset.colors[i],
                background_color=dataset.background_colors[i]
            )
            
            # Save image
            output_path = Path(output_dir) / f"{i}.png"
            plt.tight_layout(pad=0)
            plt.savefig(output_path, dpi=dpi, format="png")
            plt.close(fig)
    
    def save_modified_labels(self, dataset: ModifiedDataset, output_path: str) -> None:
        """
        Save modified labels with extended attributes.
        
        The new label format includes 9 attributes:
        [category, x, y, size, rotation, shape_r, shape_g, shape_b, background_scalar]
        
        Args:
            dataset: Modified dataset
            output_path: Path to save labels
        """
        # Create extended label array (9 components instead of 8)
        extended_labels = np.column_stack([
            dataset.classes.reshape(-1, 1),                    # 0: category
            dataset.locations,                                  # 1-2: x, y
            dataset.sizes.reshape(-1, 1),                      # 3: size
            dataset.rotations.reshape(-1, 1),                  # 4: rotation
            dataset.colors,                                     # 5-7: shape RGB
            dataset.background_scalars.reshape(-1, 1)          # 8: background scalar
        ]).astype(np.float32)
        
        # Save to file
        np.save(output_path, extended_labels)
        logger.info(f"Saved extended labels to {output_path}")
        logger.info(f"Extended label array shape: {extended_labels.shape}")
        logger.info("Label format: [category, x, y, size, rotation, shape_r, shape_g, shape_b, background_scalar]")
    
    def process_dataset(self, input_labels_path: str, output_dir: str, 
                       split_name: str = "modified") -> None:
        """
        Process a complete dataset split.
        
        Args:
            input_labels_path: Path to original labels file
            output_dir: Directory for output
            split_name: Name of the split (train/val/test)
        """
        logger.info(f"Processing {split_name} split...")
        
        # Load original data
        original_labels, metadata = self.load_original_dataset(input_labels_path)
        
        # Create modified dataset
        modified_dataset = self.create_modified_dataset(original_labels)
        
        # Create output directories
        images_dir = os.path.join(output_dir, split_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save images
        self.save_modified_images(modified_dataset, images_dir)
        
        # Save labels
        labels_output_path = os.path.join(output_dir, f"{split_name}_labels.npy")
        self.save_modified_labels(modified_dataset, labels_output_path)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"{split_name}_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Modified SimpleShapes Dataset - {split_name} split\n")
            f.write(f"Original samples: {metadata['n_samples']}\n")
            f.write(f"Original shape: {metadata['original_shape']}\n")
            f.write(f"Modified samples: {len(modified_dataset.classes)}\n")
            f.write(f"Modified attributes: 9 (original 8 + background_scalar)\n")
            f.write(f"Image size: {self.imsize}x{self.imsize}\n")
            f.write("\nModified label format:\n")
            f.write("0: category\n1: x_location\n2: y_location\n3: size\n4: rotation\n")
            f.write("5: shape_color_r\n6: shape_color_g\n7: shape_color_b\n8: background_scalar\n")
            f.write(f"\nColor scheme:\n")
            f.write(f"- Shapes: Non-grayscale RGB region (R≠G or G≠B or R≠B)\n")
            f.write(f"- Backgrounds: Grayscale region (R=G=B), scalar ∈ [0,1]\n")
        
        logger.info(f"Completed processing {split_name} split")

def main():
    """Main function to run the dataset modification process."""
    
    parser = argparse.ArgumentParser(description="Generate modified SimpleShapes dataset")
    
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing original SimpleShapes data")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save modified dataset")
    parser.add_argument("--splits", type=str, nargs="+", 
                       default=["train", "val", "test"],
                       help="Dataset splits to process")
    parser.add_argument("--imsize", type=int, default=32,
                       help="Image size in pixels")
    parser.add_argument("--rgb-resolution", type=int, default=256,
                       help="RGB space resolution")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Initialize modifier
    modifier = DatasetModifier(rgb_resolution=args.rgb_resolution, imsize=args.imsize)
    
    # Process each split
    for split in args.splits:
        input_labels_path = os.path.join(args.input_dir, f"{split}_labels.npy")
        
        if not os.path.exists(input_labels_path):
            logger.warning(f"Skipping {split} split - labels file not found: {input_labels_path}")
            continue
        
        try:
            modifier.process_dataset(input_labels_path, args.output_dir, split)
        except Exception as e:
            logger.error(f"Error processing {split} split: {e}")
            continue
    
    logger.info("Dataset modification complete!")
    
    # Print summary
    print("\n" + "="*60)
    print("MODIFIED SIMPLESHAPES DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processed splits: {args.splits}")
    print(f"Image size: {args.imsize}x{args.imsize}")
    print(f"RGB resolution: {args.rgb_resolution}")
    print("\nKey changes:")
    print("• Shape colors: Sampled from non-grayscale RGB region")
    print("• Background colors: Sampled from grayscale RGB region") 
    print("• Extended attributes: 9 components (8 original + background_scalar)")
    print("• Color space partitioning: Disjoint grayscale/non-grayscale regions")
    print("\nFiles generated per split:")
    print("• <split>_labels.npy - Extended attribute vectors")
    print("• <split>/ - Directory with modified images")
    print("• <split>_metadata.txt - Processing metadata")

if __name__ == "__main__":
    main() 