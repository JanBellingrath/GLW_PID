#!/usr/bin/env python3
"""
Quick Image Viewer - No arguments needed!
Just run this to see the original SimpleShapes images immediately.
"""

import matplotlib.pyplot as plt
from PIL import Image
import os

def main():
    print("🚀 Quick SimpleShapes Image Viewer")
    print("=" * 40)
    
    # Auto-find the original images
    potential_paths = [
        "/home/janerik/shimmer-ssd/full_shapes_dataset/simple_shapes_dataset/val",
        "/home/janerik/GLW_PID/simple_shapes_dataset/simple_shapes_dataset/val",
        "/home/janerik/datasets/simple_shapes/simple_shapes_dataset/val",
        "/home/janerik/shimmer-ssd/simple_shapes_dataset/val",
    ]
    
    original_dir = None
    for path in potential_paths:
        if os.path.exists(path):
            original_dir = path
            break
    
    if not original_dir:
        print("❌ Could not find original SimpleShapes images")
        print("💡 Make sure you have the dataset downloaded")
        return
    
    print(f"✅ Found images at: {original_dir}")
    
    # Display first 6 images
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Original SimpleShapes Images", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        img_path = os.path.join(original_dir, f"{i}.png")
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"Image {i} ({img.size}, {img.mode})")
            ax.axis('off')
            print(f"   📸 Loaded image {i}: {img.size} {img.mode}")
        except Exception as e:
            ax.text(0.5, 0.5, f"Image {i}\nNot found", ha='center', va='center')
            ax.axis('off')
            print(f"   ❌ Could not load image {i}")
    
    plt.tight_layout()
    
    # Save the result
    output_file = "original_images_preview.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"💾 Saved preview to: {output_file}")
    
    print("🎉 Displaying images now...")
    plt.show()

if __name__ == "__main__":
    main() 