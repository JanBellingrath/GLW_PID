#!/usr/bin/env python3
"""
SimpleShapes Image Viewer

This script helps you view and compare original SimpleShapes images 
with your modified versions.
"""

import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import argparse
from pathlib import Path

def view_images(image_dir, start_idx=0, count=10, title_prefix="Images"):
    """View a range of images from a directory"""
    
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return False
    
    # Calculate grid size
    cols = min(5, count)
    rows = (count + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if count == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flat if hasattr(axes, 'flat') else [axes]
    
    fig.suptitle(f"{title_prefix} from {os.path.basename(image_dir)}", fontsize=14)
    
    for i in range(count):
        img_path = os.path.join(image_dir, f"{start_idx + i}.png")
        
        try:
            img = Image.open(img_path)
            if i < len(axes):
                axes[i].imshow(img)
                axes[i].set_title(f"Image {start_idx + i}\n{img.size} {img.mode}")
                axes[i].axis('off')
        except Exception as e:
            if i < len(axes):
                axes[i].text(0.5, 0.5, f"Error loading\n{start_idx + i}.png\n{str(e)[:30]}", 
                           ha='center', va='center', fontsize=8)
                axes[i].axis('off')
    
    # Hide empty subplots
    for j in range(count, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    return fig

def compare_images(original_dir, modified_dir, start_idx=0, count=5):
    """Compare original and modified images side by side"""
    
    if not os.path.exists(original_dir):
        print(f"Original directory not found: {original_dir}")
        return False
    
    if not os.path.exists(modified_dir):
        print(f"Modified directory not found: {modified_dir}")
        return False
    
    fig, axes = plt.subplots(2, count, figsize=(count*3, 6))
    if count == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle("Original (top) vs Modified (bottom) SimpleShapes", fontsize=16)
    
    for i in range(count):
        # Original image (top row)
        orig_path = os.path.join(original_dir, f"{start_idx + i}.png")
        try:
            orig_img = Image.open(orig_path)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f"Original {start_idx + i}\n{orig_img.size} {orig_img.mode}")
            axes[0, i].axis('off')
        except Exception as e:
            axes[0, i].text(0.5, 0.5, f"Original {start_idx + i}\nNot found", 
                           ha='center', va='center', fontsize=8)
            axes[0, i].axis('off')
        
        # Modified image (bottom row)
        mod_path = os.path.join(modified_dir, f"{start_idx + i}.png")
        try:
            mod_img = Image.open(mod_path)
            axes[1, i].imshow(mod_img)
            axes[1, i].set_title(f"Modified {start_idx + i}\n{mod_img.size} {mod_img.mode}")
            axes[1, i].axis('off')
        except Exception as e:
            axes[1, i].text(0.5, 0.5, f"Modified {start_idx + i}\nNot found", 
                           ha='center', va='center', fontsize=8)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig

def find_dataset_dirs():
    """Find available dataset directories"""
    potential_paths = [
        "/home/janerik/shimmer-ssd/full_shapes_dataset/simple_shapes_dataset",
        "/home/janerik/GLW_PID/simple_shapes_dataset/simple_shapes_dataset",
        "/home/janerik/datasets/simple_shapes/simple_shapes_dataset",
        "/home/janerik/shimmer-ssd/simple_shapes_dataset",
    ]
    
    found_paths = []
    for path in potential_paths:
        val_dir = os.path.join(path, "val")
        if os.path.exists(val_dir):
            found_paths.append(path)
    
    return found_paths

def main():
    parser = argparse.ArgumentParser(description="View SimpleShapes images")
    parser.add_argument("--original-dir", type=str, 
                       help="Directory containing original SimpleShapes images")
    parser.add_argument("--modified-dir", type=str, 
                       default="shimmer_ssd/pid_analysis/test_high_dpi/val",
                       help="Directory containing modified images")
    parser.add_argument("--start", type=int, default=0,
                       help="Starting image index")
    parser.add_argument("--count", type=int, default=10,
                       help="Number of images to view")
    parser.add_argument("--compare", action="store_true",
                       help="Compare original vs modified images")
    parser.add_argument("--save", action="store_true",
                       help="Save the plot as PNG file")
    
    args = parser.parse_args()
    
    # Find original directory if not specified
    if not args.original_dir:
        found_datasets = find_dataset_dirs()
        if found_datasets:
            args.original_dir = os.path.join(found_datasets[0], "val")
            print(f"🔍 Auto-detected original images at: {args.original_dir}")
        else:
            print("❌ No SimpleShapes datasets found. Please specify --original-dir")
            return
    
    print("🖼️  SimpleShapes Image Viewer")
    print("=" * 50)
    
    if args.compare:
        print(f"📊 Comparing original vs modified images...")
        print(f"   Original: {args.original_dir}")
        print(f"   Modified: {args.modified_dir}")
        
        fig = compare_images(args.original_dir, args.modified_dir, 
                           args.start, min(args.count, 5))
        
        if fig and args.save:
            output_file = "image_comparison.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"💾 Saved comparison to {output_file}")
    
    else:
        print(f"👀 Viewing original images from: {args.original_dir}")
        
        fig = view_images(args.original_dir, args.start, args.count, "Original Images")
        
        if fig and args.save:
            output_file = "original_images.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"💾 Saved images to {output_file}")
    
    if fig:
        print("✅ Images loaded successfully!")
        plt.show()
    else:
        print("❌ Failed to load images")

if __name__ == "__main__":
    main() 