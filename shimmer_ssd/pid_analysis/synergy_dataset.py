#!/usr/bin/env python3
"""
Synergy Dataset for Global Workspace Training

This dataset loads real synergistic data where:
- Inputs: VAE visual latents + non-synergistic attributes
- Targets: VAE visual latents + attributes INCLUDING synergistic features

The key insight: synergistic features should NOT be in inputs, only in targets.
The model must learn to predict synergistic combinations from other modalities.

Visual data is loaded from pre-saved VAE latents for performance.
Attribute data is loaded from modified attribute files with XOR synergistic features.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging

try:
    import torchvision.transforms as T
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

logger = logging.getLogger(__name__)


class SynergyDataset(Dataset):
    """
    Dataset for synergistic Global Workspace training.
    
    Loads XOR target data and creates proper input/target separation:
    - Input attributes: original attributes WITHOUT synergistic features
    - Target attributes: original attributes WITH synergistic features appended
    - Visual data: unchanged in both input and target
    
    This enables training models to predict synergistic combinations from
    other modalities (vision + non-synergistic attributes).
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str,
        synergy_config: Dict[str, Any],
        image_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        transform: Optional[callable] = None,
        load_images: bool = True,
        cache_data: bool = True,
    ):
        """
        Initialize synergy dataset.
        
        Args:
            data_dir: Directory containing XOR target files
            split: Dataset split ('train', 'val', 'test')
            synergy_config: Configuration for synergistic features per domain
            image_dir: Directory containing images (if loading raw images)
            device: Device to load tensors on
            transform: Optional transform for images
            load_images: Whether to load image data
            cache_data: Whether to cache loaded data in memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.synergy_config = synergy_config
        self.image_dir = Path(image_dir) if image_dir else None
        self.device = device or torch.device('cpu')
        self.transform = transform
        self.load_images = load_images
        self.cache_data = cache_data
        
        # Validate synergy config
        self._validate_synergy_config()
        
        # Load data
        self._load_data()
        
        # Cache for performance if requested
        self._cached_items = {} if cache_data else None
        
        logger.info(f"Initialized SynergyDataset for {split} split with {len(self)} samples")
        logger.info(f"Synergy domains: {list(self.synergy_config.keys())}")
    
    def _validate_synergy_config(self):
        """Validate synergy configuration structure."""
        required_keys = ['domains', 'feature_indices']
        for key in required_keys:
            if key not in self.synergy_config:
                raise ValueError(f"synergy_config missing required key: {key}")
        
        # Validate domain specifications
        domains = self.synergy_config['domains']
        feature_indices = self.synergy_config['feature_indices']
        
        for domain in domains:
            if domain in feature_indices:
                indices = feature_indices[domain]
                if not isinstance(indices, (list, tuple)):
                    raise ValueError(f"Feature indices for domain '{domain}' must be list/tuple")
            else:
                # Visual domains may not need synergistic features
                if domain not in ['v', 'v_latents']:
                    raise ValueError(f"Domain '{domain}' has no feature indices specified")
        
        logger.info(f"Validated synergy config for domains: {domains}")
    
    def _load_data(self):
        """Load XOR target data and prepare input/target tensors."""
        # Load XOR targets
        xor_file = self.data_dir / f"{self.split}_xor_targets.npy"
        if not xor_file.exists():
            raise FileNotFoundError(f"XOR targets file not found: {xor_file}")
        
        # Load as structured array
        xor_data = np.load(xor_file)
        
        # Convert to DataFrame for easier manipulation
        if xor_data.dtype.names:
            self.xor_df = pd.DataFrame(xor_data)
        else:
            # Fallback: load CSV version
            csv_file = self.data_dir / f"{self.split}_xor_targets.csv"
            if csv_file.exists():
                self.xor_df = pd.read_csv(csv_file)
            else:
                raise ValueError("Could not load XOR data as structured array or CSV")
        
        # Load original attribute data (for input features)
        self._load_original_attributes()
        
        # Always try to load visual data (VAE latents or images)
        self._load_image_data()
        
        # Prepare input and target tensors
        self._prepare_tensors()
        
        logger.info(f"Loaded {len(self.xor_df)} samples for {self.split} split")
    
    def _load_original_attributes(self):
        """Load original attribute data (without synergistic features) and size values."""
        # Load no-size attributes
        attr_file = self.data_dir / f"{self.split}_labels_nosize.npy"
        if not attr_file.exists():
            # Try parent directory
            attr_file = self.data_dir.parent / f"{self.split}_labels_nosize.npy"
        
        if not attr_file.exists():
            raise FileNotFoundError(f"Original attribute file not found: {attr_file}")
        
        self.original_attrs = np.load(attr_file)
        logger.info(f"Loaded original attributes: shape {self.original_attrs.shape}")
        
        # Load original labels with size information
        # Look for original labels in the dataset directory structure
        original_labels_paths = [
            self.data_dir.parent / f"{self.split}_labels.npy",  # Try parent directory first
            self.data_dir / f"{self.split}_labels.npy",  # Try data directory
            self.data_dir.parent.parent / "simple_shapes_dataset" / f"{self.split}_labels.npy",  # Try dataset root
            Path("/home/janerik/GLW_PID/simple_shapes_dataset/simple_shapes_dataset") / f"{self.split}_labels.npy",  # Direct path
        ]
        
        self.original_labels_with_size = None
        for labels_path in original_labels_paths:
            if labels_path.exists():
                try:
                    self.original_labels_with_size = np.load(labels_path)
                    logger.info(f"Loaded original labels with size: {labels_path}, shape {self.original_labels_with_size.shape}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {labels_path}: {e}")
                    continue
        
        if self.original_labels_with_size is None:
            raise FileNotFoundError(
                f"Could not find original labels with size. Searched: {[str(p) for p in original_labels_paths]}"
            )
        
        # Extract size values (column 3 in original labels)
        self.size_values = self.original_labels_with_size[:, 3].astype(np.float32)
        logger.info(f"Extracted size values: range [{self.size_values.min():.3f}, {self.size_values.max():.3f}]")
        
        # Validate sample count matches
        if len(self.original_attrs) != len(self.xor_df):
            raise ValueError(
                f"Sample count mismatch: attrs={len(self.original_attrs)}, "
                f"xor={len(self.xor_df)}"
            )
        if len(self.size_values) != len(self.xor_df):
            raise ValueError(
                f"Sample count mismatch: size_values={len(self.size_values)}, "
                f"xor={len(self.xor_df)}"
            )
    
    def _load_image_data(self):
        """Load VAE latents or image data (prefer VAE latents for performance)."""
        # First try to load VAE latents from saved_latents directory
        vae_latents_loaded = self._try_load_vae_latents()
        
        if not vae_latents_loaded:
            # Try to load precomputed visual features
            visual_file = self.data_dir / f"{self.split}_visual_features.npy"
            if visual_file.exists():
                self.visual_features = np.load(visual_file)
                logger.info(f"Loaded precomputed visual features: {self.visual_features.shape}")
            elif self.load_images and self.image_dir:
                # Only load raw images if specifically requested
                self.image_paths = []
                for idx in range(len(self.xor_df)):
                    img_path = self.image_dir / f"{idx:06d}.png"
                    if not img_path.exists():
                        raise FileNotFoundError(f"Image not found: {img_path}")
                    self.image_paths.append(img_path)
            else:
                raise FileNotFoundError(
                    f"No visual data found! Expected either:\n"
                    f"  - VAE latents in saved_latents directory\n"
                    f"  - Precomputed features at {visual_file}\n"
                    f"  - Image directory specified with load_images=True for raw images"
                )
    
    def _prepare_tensors(self):
        """Prepare input and target tensors with proper synergy separation."""
        self.inputs = {}
        self.targets = {}
        
        # Process each domain
        domains = self.synergy_config['domains']
        feature_indices = self.synergy_config['feature_indices']
        
        for domain in domains:
            if domain == 'attr':
                # Attribute domain: separate input from target
                if domain in feature_indices:
                    input_attrs, target_attrs = self._prepare_attribute_tensors(feature_indices[domain])
                else:
                    # No synergistic features for this domain - still preprocess to 11D
                    processed_attrs = self._preprocess_attributes(self.original_attrs)
                    input_attrs = torch.tensor(processed_attrs, dtype=torch.float32, device=self.device)
                    target_attrs = input_attrs  # Same for both input and target
                self.inputs[domain] = input_attrs
                self.targets[domain] = target_attrs
                
            elif domain in ['v', 'v_latents']:
                # Visual domain: same for input and target (no synergistic features)
                visual_data = self._prepare_visual_tensors()
                self.inputs[domain] = visual_data
                self.targets[domain] = visual_data
                
            else:
                raise ValueError(f"Unsupported domain: {domain}")
        
        logger.info("Prepared input/target tensors for all domains")
        for domain in domains:
            logger.info(f"  {domain}: input {self.inputs[domain].shape}, "
                       f"target {self.targets[domain].shape}")
    
    def _preprocess_attributes(self, raw_attrs: np.ndarray) -> np.ndarray:
        """
        Preprocess 8D raw attributes to 11D processed attributes.
        
        Transformations:
        - Category (1D): 0,1,2 → one-hot encoding (3D)
        - Coordinates (2D): [7,24] → [0,1] normalization (2D)
        - Rotation (1D): radians → [cos(θ), sin(θ)] (2D) 
        - Colors (3D): [0,255] → [0,1] normalization (3D)
        - Keep uniform_random as-is: [0,1] (1D)
        
        Args:
            raw_attrs: Array of shape [N, 8] with columns:
                [category, x, y, rotation, r, g, b, uniform_random]
                
        Returns:
            Processed array of shape [N, 11] with columns:
                [cat_0, cat_1, cat_2, x, y, cos_rot, sin_rot, r_norm, g_norm, b_norm, uniform_random]
        """
        N = raw_attrs.shape[0]
        processed = np.zeros((N, 11), dtype=np.float32)
        
        # One-hot encode category (columns 0-2)
        categories = raw_attrs[:, 0].astype(int)
        for i in range(3):
            processed[:, i] = (categories == i).astype(np.float32)
        
        # Normalize x, y coordinates from [7,24] to [0,1] (columns 3-4)
        x_coords = raw_attrs[:, 1]  # x coordinates
        y_coords = raw_attrs[:, 2]  # y coordinates
        processed[:, 3] = (x_coords - 7.0) / (24.0 - 7.0)  # x normalized to [0,1]
        processed[:, 4] = (y_coords - 7.0) / (24.0 - 7.0)  # y normalized to [0,1]
        
        # Convert rotation to cos/sin (columns 5-6)
        rotation = raw_attrs[:, 3]
        processed[:, 5] = np.cos(rotation)  # cos(rotation)
        processed[:, 6] = np.sin(rotation)  # sin(rotation)
        
        # Normalize color values from [0,255] to [0,1] and keep random as-is (columns 7-10)
        processed[:, 7] = raw_attrs[:, 4] / 255.0   # r (normalized)
        processed[:, 8] = raw_attrs[:, 5] / 255.0   # g (normalized)
        processed[:, 9] = raw_attrs[:, 6] / 255.0   # b (normalized)
        processed[:, 10] = raw_attrs[:, 7]  # uniform_random (already [0,1])
        
        return processed

    def _prepare_attribute_tensors(self, synergy_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare attribute tensors with synergy separation.
        
        Args:
            synergy_indices: Not used - kept for compatibility. XOR feature is hardcoded.
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        # Preprocess raw 8D attributes to 11D processed attributes
        processed_attrs = self._preprocess_attributes(self.original_attrs)
        
        # Input: processed attributes (no synergistic features)
        input_attrs = torch.tensor(processed_attrs, dtype=torch.float32, device=self.device)
        
        # Target: processed attributes + synergistic features
        # Extract XOR synergy feature by name (always "xor_target_normalized")
        synergy_feature = self.xor_df["xor_target_normalized"].values
        
        synergy_tensor = torch.tensor(
            synergy_feature.reshape(-1, 1),  # Ensure 2D shape
            dtype=torch.float32, 
            device=self.device
        )
        
        # CLEAN SOLUTION: Return input (without synergy) and target (with synergy)
        # The synergy feature will be at index 11 in the 12D target tensor
        target_attrs = torch.cat([input_attrs, synergy_tensor], dim=1)
        
        return input_attrs, target_attrs
    
    def _try_load_vae_latents(self) -> bool:
        """Try to load VAE latents from saved_latents directory."""
        # Default VAE latents filename from GLW training
        latent_filename = "calmip-822888_epoch=282-step=1105680_future.npy"
        
        # Look for VAE latents in the dataset structure
        potential_paths = []
        
        # Add direct path to simple shapes dataset saved latents (VAE)
        simple_shapes_vae_latent = Path("/home/janerik/GLW_PID/simple_shapes_dataset/simple_shapes_dataset/saved_latents") / self.split / latent_filename
        potential_paths.append(simple_shapes_vae_latent)
        
        if self.image_dir:
            # Check if image_dir contains saved_latents
            base_dataset_path = self.image_dir
            if isinstance(base_dataset_path, str):
                base_dataset_path = Path(base_dataset_path)
            
            # Try dataset_path/saved_latents/split/filename
            latents_path = base_dataset_path / "saved_latents" / self.split / latent_filename
            potential_paths.append(latents_path)
            
            # Try parent directory structure
            parent_latents_path = base_dataset_path.parent / "saved_latents" / self.split / latent_filename
            potential_paths.append(parent_latents_path)
        
        # Also check relative to data_dir
        if hasattr(self, 'data_dir'):
            relative_latents_path = self.data_dir.parent / "saved_latents" / self.split / latent_filename
            potential_paths.append(relative_latents_path)
        
        for latents_path in potential_paths:
            if latents_path.exists():
                try:
                    logger.info(f"Loading VAE latents from: {latents_path}")
                    latents_data = np.load(latents_path)
                    
                    # Validate latent data
                    if len(latents_data) < len(self.xor_df):
                        logger.warning(
                            f"VAE latents ({len(latents_data)}) have fewer samples than "
                            f"XOR data ({len(self.xor_df)}). Using first {len(latents_data)} samples."
                        )
                        # Trim XOR data to match latents
                        self.xor_df = self.xor_df.iloc[:len(latents_data)]
                        self.original_attrs = self.original_attrs[:len(latents_data)]
                    elif len(latents_data) > len(self.xor_df):
                        logger.info(
                            f"VAE latents ({len(latents_data)}) have more samples than "
                            f"XOR data ({len(self.xor_df)}). Using first {len(self.xor_df)} latents."
                        )
                        latents_data = latents_data[:len(self.xor_df)]
                    
                    self.visual_features = latents_data
                    logger.info(
                        f"✅ Loaded {len(latents_data)} VAE latents: shape {latents_data.shape}, "
                        f"range [{latents_data.min():.3f}, {latents_data.max():.3f}]"
                    )
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to load VAE latents from {latents_path}: {e}")
                    continue
        
        logger.warning(
            f"Could not find VAE latents. Searched paths:\n" + 
            "\n".join(f"  - {p}" for p in potential_paths)
        )
        return False
    
    def _prepare_visual_tensors(self) -> torch.Tensor:
        """Prepare visual tensors from VAE latents + size values."""
        if hasattr(self, 'visual_features'):
            visual_tensor = torch.tensor(self.visual_features, dtype=torch.float32, device=self.device)
            
            # Handle VAE latents: extract mean vector, drop logvar
            if visual_tensor.dim() == 3 and visual_tensor.shape[1] == 2:
                # VAE latents shape [N, 2, latent_dim] -> [N, latent_dim] (take mean, drop logvar)
                visual_tensor = visual_tensor[:, 0, :]
                logger.info(f"Extracted VAE mean vectors: {visual_tensor.shape}")
            else:
                logger.info(f"Using visual tensors: {visual_tensor.shape}")
            
            # Add actual size values as an additional dimension
            if hasattr(self, 'size_values'):
                # Normalize size values to match the range expectation
                size_min = self.size_values.min()
                size_max = self.size_values.max()
                size_range = size_max - size_min
                
                if size_range > 0:
                    normalized_size = (self.size_values - size_min) / size_range
                else:
                    normalized_size = np.full_like(self.size_values, 0.5)
                
                # Convert to tensor and add as extra dimension
                size_tensor = torch.tensor(normalized_size, dtype=torch.float32, device=self.device).unsqueeze(1)
                
                # Concatenate VAE latents with size values: [N, latent_dim] + [N, 1] = [N, latent_dim+1]
                visual_tensor = torch.cat([visual_tensor, size_tensor], dim=1)
                
                logger.info(f"Added size values to visual tensor: {visual_tensor.shape} (VAE latents + size)")
                logger.info(f"Size range: [{size_min:.3f}, {size_max:.3f}] → [0, 1]")
            else:
                logger.warning("No size values available - using VAE latents only")
            
            return visual_tensor
        else:
            raise RuntimeError(
                "No visual data available. This should not happen if _load_image_data() succeeded."
            )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.xor_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with separated inputs and targets.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'inputs' and 'targets' keys, each containing domain data
        """
        # Check cache
        if self._cached_items is not None and idx in self._cached_items:
            return self._cached_items[idx]
        
        # Prepare sample
        sample = {
            'inputs': {},
            'targets': {},
            'metadata': {
                'idx': idx,
                'split': self.split
            }
        }
        
        # Get data for each domain
        for domain in self.synergy_config['domains']:
            sample['inputs'][domain] = self.inputs[domain][idx]
            sample['targets'][domain] = self.targets[domain][idx]
        
        # Load image if needed
        if self.load_images and hasattr(self, 'image_paths'):
            img_path = self.image_paths[idx]
            # Load and transform image
            import PIL.Image
            img = PIL.Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                # Default transform: resize and normalize
                if not TORCHVISION_AVAILABLE:
                    raise ImportError("torchvision is required for image transforms but not available")
                transform = T.Compose([
                    T.Resize((64, 64)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                img = transform(img)
            
            sample['inputs']['v'] = img.to(self.device)
            sample['targets']['v'] = img.to(self.device)
        
        # Cache if enabled
        if self._cached_items is not None:
            self._cached_items[idx] = sample
        
        return sample
    
    def get_synergy_info(self) -> Dict[str, Any]:
        """Get information about synergistic features."""
        info = {
            'domains': self.synergy_config['domains'],
            'feature_indices': self.synergy_config['feature_indices'],
            'input_shapes': {domain: tensor.shape[1:] for domain, tensor in self.inputs.items()},
            'target_shapes': {domain: tensor.shape[1:] for domain, tensor in self.targets.items()},
        }
        
        # Add feature names if available
        if hasattr(self.xor_df, 'columns'):
            info['feature_names'] = list(self.xor_df.columns)
        
        return info
    
    @staticmethod
    def create_synergy_config(
        domains: List[str],
        synergy_features: Dict[str, Union[List[int], List[str]]]
    ) -> Dict[str, Any]:
        """
        Create a synergy configuration.
        
        Args:
            domains: List of domain names
            synergy_features: Mapping from domain to feature indices/names
            
        Returns:
            Synergy configuration dictionary
        """
        return {
            'domains': domains,
            'feature_indices': synergy_features
        }


def synergy_collate_fn(batch):
    """Custom collate function for synergy dataset batches."""
    from torch.utils.data._utils.collate import default_collate
    import torch
    
    try:
        # Handle empty batch
        if not batch:
            return {}
            
        # Separate inputs, targets, and metadata
        inputs_batch = {}
        targets_batch = {}
        metadata_batch = []
        
        # Collect data from all samples in batch
        for sample in batch:
            if not isinstance(sample, dict):
                raise ValueError(f"Expected dict sample, got {type(sample)}")
                
            # Collect inputs
            if 'inputs' in sample:
                for domain, data in sample['inputs'].items():
                    if domain not in inputs_batch:
                        inputs_batch[domain] = []
                    inputs_batch[domain].append(data)
            
            # Collect targets
            if 'targets' in sample:
                for domain, data in sample['targets'].items():
                    if domain not in targets_batch:
                        targets_batch[domain] = []
                    targets_batch[domain].append(data)
                    
            # Collect metadata
            if 'metadata' in sample:
                metadata_batch.append(sample['metadata'])
        
        # Batch the tensors using default collate with error handling
        batched_inputs = {}
        for domain, tensors in inputs_batch.items():
            try:
                # Use default_collate which handles tensors properly
                batched_inputs[domain] = default_collate(tensors)
            except Exception as e:
                print(f"Warning: Collation error for input domain {domain}: {e}")
                # Fallback: return as list if collation fails
                batched_inputs[domain] = tensors
        
        batched_targets = {}
        for domain, tensors in targets_batch.items():
            try:
                batched_targets[domain] = default_collate(tensors)
            except Exception as e:
                print(f"Warning: Collation error for target domain {domain}: {e}")
                batched_targets[domain] = tensors
        
        return {
            'inputs': batched_inputs,
            'targets': batched_targets,
            'metadata': metadata_batch
        }
        
    except Exception as e:
        print(f"Error in synergy_collate_fn: {e}")
        # Final fallback: return batch as-is
        return batch


def create_synergy_dataloaders(
    data_dir: str,
    synergy_config: Dict[str, Any],
    batch_size: int = 32,
    image_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_workers: int = 4,
    **dataset_kwargs
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create data loaders for all splits.
    
    Args:
        data_dir: Directory containing XOR target files
        synergy_config: Configuration for synergistic features
        batch_size: Batch size for training
        image_dir: Directory containing images
        device: Device to load data on
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for SynergyDataset
        
    Returns:
        Dictionary mapping split names to DataLoaders
    """
    splits = ['train', 'val', 'test']
    dataloaders = {}
    
    for split in splits:
        # Check if split exists
        split_file = Path(data_dir) / f"{split}_xor_targets.npy"
        if not split_file.exists():
            logger.warning(f"Skipping {split} split - file not found: {split_file}")
            continue
        
        try:
            # Create dataset
            dataset = SynergyDataset(
                data_dir=data_dir,
                split=split,
                synergy_config=synergy_config,
                image_dir=image_dir,
                device=device,
                **dataset_kwargs
            )
            
            # Create dataloader
            shuffle = (split == 'train')
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=(device is not None and device.type == 'cuda'),
                drop_last=False,
                collate_fn=synergy_collate_fn
            )
            
            dataloaders[split] = dataloader
            logger.info(f"Created {split} dataloader with {len(dataset)} samples")
            
        except Exception as e:
            logger.error(f"Failed to create {split} dataloader: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue to next split instead of failing completely
            continue
    
    return dataloaders


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    synergy_config = SynergyDataset.create_synergy_config(
        domains=['attr', 'v'],
        synergy_features={
            'attr': ['xor_target_normalized'],  # Column name from XOR data
            # 'v': []  # No synergy features in vision for now
        }
    )
    
    # Create dataset
    dataset = SynergyDataset(
        data_dir="data/simple_shapes_xor_targets",
        split="train",
        synergy_config=synergy_config,
        load_images=False,
        cache_data=True
    )
    
    # Test sample
    sample = dataset[0]
    print("Sample structure:")
    print(f"  Inputs: {list(sample['inputs'].keys())}")
    print(f"  Targets: {list(sample['targets'].keys())}")
    
    for domain in sample['inputs']:
        input_shape = sample['inputs'][domain].shape
        target_shape = sample['targets'][domain].shape
        print(f"  {domain}: input {input_shape}, target {target_shape}")
    
    # Print synergy info
    print("\nSynergy info:")
    import json
    print(json.dumps(dataset.get_synergy_info(), indent=2)) 