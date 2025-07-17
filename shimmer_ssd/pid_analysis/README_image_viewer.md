# SimpleShapes Image Viewer Scripts

## Quick Start (Easiest Way)

```bash
cd shimmer_ssd/pid_analysis
python3 quick_view.py
```

This will automatically find your original SimpleShapes images and show the first 6 images.

## Advanced Usage

```bash
cd shimmer_ssd/pid_analysis
python3 view_images.py --help
```

### View Original Images
```bash
# View first 10 original images
python3 view_images.py

# View specific range
python3 view_images.py --start 100 --count 8

# Save to file
python3 view_images.py --save
```

### Compare Original vs Modified
```bash
# Compare original and modified images side by side
python3 view_images.py --compare

# Compare different range
python3 view_images.py --compare --start 50 --count 3

# Compare and save
python3 view_images.py --compare --save
```

### Specify Custom Directories
```bash
# Use specific directories
python3 view_images.py \
  --original-dir "/path/to/original/val" \
  --modified-dir "/path/to/modified/val" \
  --compare

# View images from different modified version
python3 view_images.py \
  --modified-dir "shimmer_ssd/pid_analysis/small_test_output/small_test" \
  --compare
```

## Output Files

- `original_images_preview.png` - Preview from quick_view.py
- `original_images.png` - Saved from view_images.py
- `image_comparison.png` - Saved comparison from --compare mode

## Features

- ✅ Auto-detects original SimpleShapes dataset location
- ✅ Shows image properties (size, mode)
- ✅ Side-by-side comparison of original vs modified
- ✅ Handles missing images gracefully
- ✅ Saves high-quality PNG outputs
- ✅ Customizable start index and count 