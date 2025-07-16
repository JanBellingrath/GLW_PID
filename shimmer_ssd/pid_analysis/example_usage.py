#!/usr/bin/env python3
"""
Example usage of the Modified SimpleShapes Dataset Generator

This script demonstrates how to use the generate_modified_shapes.py script
to create a modified SimpleShapes dataset with RGB space partitioning.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_modification_example():
    """Run an example of the dataset modification process."""
    
    # Example paths (adjust these to your actual paths)
    input_dir = "/home/janerik/shimmer-ssd/simple-shapes-dataset/sample_dataset"
    output_dir = "./modified_simpleshapes_dataset"
    
    print("="*60)
    print("MODIFIED SIMPLESHAPES DATASET GENERATION EXAMPLE")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        print("Please update the input_dir variable to point to your SimpleShapes dataset")
        return False
    
    # Check for required label files
    required_files = ["train_labels.npy", "val_labels.npy", "test_labels.npy"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(input_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing label files: {missing_files}")
        print("Available splits will be processed, missing ones will be skipped.")
    
    # Construct command
    script_path = "generate_modified_shapes.py"
    cmd = [
        "python3", script_path,
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--imsize", "32",
        "--rgb-resolution", "256", 
        "--seed", "42"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Run the script
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dataset modification completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dataset modification: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_path}")
        print("Make sure generate_modified_shapes.py is in the current directory")
        return False

def inspect_results(output_dir: str):
    """Inspect the results of the dataset modification."""
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return
    
    print(f"\n📁 Inspecting results in {output_dir}:")
    print("="*50)
    
    # List contents
    for item in sorted(os.listdir(output_dir)):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            num_images = len([f for f in os.listdir(item_path) if f.endswith('.png')])
            print(f"📂 {item}/ ({num_images} images)")
        else:
            size_mb = os.path.getsize(item_path) / 1024 / 1024
            print(f"📄 {item} ({size_mb:.2f} MB)")
    
    # Check a sample split
    for split in ["train", "val", "test"]:
        labels_path = os.path.join(output_dir, f"{split}_labels.npy")
        if os.path.exists(labels_path):
            import numpy as np
            labels = np.load(labels_path)
            print(f"\n🔍 {split}_labels.npy analysis:")
            print(f"   Shape: {labels.shape}")
            print(f"   Samples: {len(labels)}")
            print(f"   Attributes: {labels.shape[1]}")
            print(f"   Sample attribute vector: {labels[0]}")
            print(f"   Background scalar range: [{labels[:, 8].min():.3f}, {labels[:, 8].max():.3f}]")
            break

if __name__ == "__main__":
    # Run the example
    success = run_modification_example()
    
    if success:
        # Inspect results
        output_dir = "./modified_simpleshapes_dataset"
        inspect_results(output_dir)
        
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("1. Check the generated images in the output directories")
        print("2. Examine the extended attribute vectors with 9 components")
        print("3. Use the new dataset in your ML experiments")
        print("4. Note the RGB space partitioning: shapes (non-grayscale) vs backgrounds (grayscale)")
    else:
        print("\n❌ Example failed. Please check the error messages above.") 