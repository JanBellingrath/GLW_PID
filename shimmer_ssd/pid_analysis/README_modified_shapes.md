# Modified SimpleShapes Dataset Generator

This module provides tools to generate modified versions of the SimpleShapes dataset with RGB space partitioning and extended attribute vectors.

## Overview

The modified dataset introduces:

1. **RGB Space Partitioning**: Divides RGB color space into two disjoint regions
   - **Grayscale region**: R = G = B (black to white line)
   - **Non-grayscale region**: All other RGB combinations

2. **New Color Scheme**:
   - **Shape colors**: Sampled uniformly from non-grayscale RGB region
   - **Background colors**: Sampled uniformly from grayscale region

3. **Extended Attributes**: 9 components instead of 8
   - Original 8: `[category, x, y, size, rotation, shape_r, shape_g, shape_b]`
   - New 9th: `background_scalar` (grayscale value ∈ [0,1])

## Files

- `generate_modified_shapes.py` - Main script for dataset generation
- `example_usage.py` - Example script showing usage
- `README_modified_shapes.md` - This documentation

## Usage

### Image Quality Note

The default DPI setting is 100, which produces high-quality images. You can adjust this with the `--dpi` parameter:
- `--dpi 50`: Lower quality, smaller files
- `--dpi 100`: Default, good quality 
- `--dpi 150` or higher: Maximum quality, larger files

### Basic Usage

```bash
python3 generate_modified_shapes.py \
    --input-dir /path/to/original/simpleshapes \
    --output-dir /path/to/modified/dataset \
    --splits train val test \
    --imsize 32 \
    --dpi 100 \
    --seed 42
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | Required | Directory containing original SimpleShapes data |
| `--output-dir` | Required | Directory to save modified dataset |
| `--splits` | `["train", "val", "test"]` | Dataset splits to process |
| `--imsize` | `32` | Image size in pixels |
| `--dpi` | `100` | DPI for image generation (higher = better quality) |
| `--rgb-resolution` | `256` | RGB space resolution (8-bit = 256) |
| `--seed` | `42` | Random seed for reproducibility |

### Input Requirements

The input directory should contain:
- `train_labels.npy` - Training labels
- `val_labels.npy` - Validation labels  
- `test_labels.npy` - Test labels

### Output Structure

```
output_dir/
├── train/                    # Modified training images
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── val/                      # Modified validation images  
│   └── ...
├── test/                     # Modified test images
│   └── ...
├── train_labels.npy          # Extended training labels (9 attributes)
├── val_labels.npy            # Extended validation labels (9 attributes)
├── test_labels.npy           # Extended test labels (9 attributes)
├── train_metadata.txt        # Processing metadata
├── val_metadata.txt          # Processing metadata
└── test_metadata.txt         # Processing metadata
```

## RGB Space Partitioning

### Mathematical Definition

**Grayscale Region**: 
```
G = {(r,g,b) ∈ [0,255]³ : r = g = b}
```
This forms a line from (0,0,0) to (255,255,255).

**Non-grayscale Region**:
```
NG = {(r,g,b) ∈ [0,255]³ : r ≠ g ∨ g ≠ b ∨ r ≠ b}
```
This is the complement of the grayscale region.

### Sampling Strategy

1. **Grayscale Colors**: 
   - Sample gray value `g ~ Uniform(0, 255)`
   - Set RGB = (g, g, g)
   - Scalar = g / 255 ∈ [0,1]

2. **Non-grayscale Colors**:
   - Rejection sampling from RGB cube
   - Sample (r,g,b) ~ Uniform([0,255]³)
   - Reject if r = g = b, otherwise accept

## Extended Attribute Format

The new 9-component attribute vectors:

| Index | Component | Range | Description |
|-------|-----------|-------|-------------|
| 0 | `category` | {0,1,2} | Shape type (diamond/egg/triangle) |
| 1 | `x` | [0, imsize] | X position |
| 2 | `y` | [0, imsize] | Y position |
| 3 | `size` | [min_scale, max_scale] | Shape size |
| 4 | `rotation` | [0, 2π] | Rotation angle |
| 5 | `shape_r` | [0, 255] | Shape red component |
| 6 | `shape_g` | [0, 255] | Shape green component |
| 7 | `shape_b` | [0, 255] | Shape blue component |
| 8 | `background_scalar` | [0, 1] | Background grayscale scalar |

## Implementation Details

### Class Structure

```python
# Core classes
RGBSpacePartitioner     # Handles color space partitioning
ModifiedImageGenerator  # Generates images with colored backgrounds  
DatasetModifier        # Main orchestrator class
ModifiedDataset        # Data structure for extended dataset

# Key methods
sample_grayscale_colors()      # Sample from grayscale region
sample_non_grayscale_colors()  # Sample from non-grayscale region
generate_image()               # Create image with background color
save_modified_labels()         # Save extended attribute vectors
```

### Color Space Properties

- **Disjoint regions**: G ∩ NG = ∅
- **Complete coverage**: G ∪ NG = [0,255]³
- **Uniform sampling**: Both regions sampled uniformly
- **No color conflicts**: Shape and background colors never overlap

## Example Code

### Loading Modified Dataset

```python
import numpy as np

# Load extended labels
labels = np.load('output_dir/train_labels.npy')

# Extract components
categories = labels[:, 0].astype(int)
positions = labels[:, 1:3]  # x, y
sizes = labels[:, 3]
rotations = labels[:, 4]
shape_colors = labels[:, 5:8]  # RGB
background_scalars = labels[:, 8]

print(f"Dataset shape: {labels.shape}")
print(f"Background scalar range: [{background_scalars.min():.3f}, {background_scalars.max():.3f}]")
```

### Analyzing Color Distribution

```python
# Check color space partitioning
shape_rgb = labels[:, 5:8]
bg_scalars = labels[:, 8]

# Verify no grayscale shapes
is_grayscale = (shape_rgb[:, 0] == shape_rgb[:, 1]) & (shape_rgb[:, 1] == shape_rgb[:, 2])
print(f"Grayscale shapes: {is_grayscale.sum()} (should be 0)")

# Background scalar distribution
print(f"Background scalar stats: mean={bg_scalars.mean():.3f}, std={bg_scalars.std():.3f}")
```

## Running the Example

```bash
cd shimmer_ssd/pid_analysis/
python3 example_usage.py
```

This will:
1. Check for required input files
2. Run the dataset modification
3. Inspect and analyze the results
4. Display summary statistics

## Integration with Existing Code

To use the modified dataset in your existing SimpleShapes pipeline:

1. **Update attribute processing**: Extend from 8 to 9 components
2. **Handle new color ranges**: Account for different shape color distribution
3. **Use background information**: Leverage the new background_scalar attribute

### Example Integration

```python
# In your existing attribute processing code:
def process_extended_attributes(data):
    if hasattr(data, 'background_scalar'):
        # Handle 9-component vectors
        attrs = torch.tensor([
            data.category.float(),
            data.x.float(), 
            data.y.float(),
            data.size.float(),
            data.rotation.float(),
            data.color_r.float(),
            data.color_g.float(), 
            data.color_b.float(),
            data.background_scalar.float()  # New component
        ])
    else:
        # Fallback to 8-component vectors
        attrs = torch.tensor([...])  # Original processing
    
    return attrs
```

## Benefits

1. **Richer Feature Space**: 9D instead of 8D attribute vectors
2. **Controlled Color Distribution**: Explicit separation of shape/background colors
3. **Uniform Sampling**: Both color regions sampled uniformly
4. **Backward Compatibility**: Can still use first 8 components
5. **Enhanced Visualization**: Colored backgrounds improve visual distinction
6. **Better Clustering**: Background scalar provides additional clustering dimension

## Citation

If you use this modified dataset in your research, please cite the original SimpleShapes dataset and mention the modifications:

```
Modified SimpleShapes Dataset with RGB Space Partitioning
Extended from original SimpleShapes with:
- Disjoint grayscale/non-grayscale color regions
- 9-component attribute vectors  
- Uniform color sampling strategy
``` 