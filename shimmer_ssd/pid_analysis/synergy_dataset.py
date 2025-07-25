#!/usr/bin/env python3
"""
Synergy Dataset for Global Workspace Training

This dataset loads synthetic synergistic data where:
- Inputs: visual data + non-synergistic attributes
- Targets: visual data + attributes INCLUDING synergistic features

The key insight: synergistic features should NOT be in inputs, only in targets.
The model must learn to predict synergistic combinations from other modalities.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging

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
            if domain not in feature_indices:
                raise ValueError(f"Domain '{domain}' has no feature indices specified")
            
            indices = feature_indices[domain]
            if not isinstance(indices, (list, tuple)):
                raise ValueError(f"Feature indices for domain '{domain}' must be list/tuple")
        
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
        
        # Load image paths/data
        if self.load_images:
            self._load_image_data()
        
        # Prepare input and target tensors
        self._prepare_tensors()
        
        logger.info(f"Loaded {len(self.xor_df)} samples for {self.split} split")
    
    def _load_original_attributes(self):
        """Load original attribute data (without synergistic features)."""
        # Try to find original no-size labels
        attr_file = self.data_dir / f"{self.split}_labels_nosize.npy"
        if not attr_file.exists():
            # Try parent directory
            attr_file = self.data_dir.parent / f"{self.split}_labels_nosize.npy"
        
        if not attr_file.exists():
            raise FileNotFoundError(f"Original attribute file not found: {attr_file}")
        
        self.original_attrs = np.load(attr_file)
        logger.info(f"Loaded original attributes: shape {self.original_attrs.shape}")
        
        # Validate sample count matches
        if len(self.original_attrs) != len(self.xor_df):
            raise ValueError(
                f"Sample count mismatch: attrs={len(self.original_attrs)}, "
                f"xor={len(self.xor_df)}"
            )
    
    def _load_image_data(self):
        """Load image data (either paths or precomputed features)."""
        if self.image_dir:
            # Load from image files
            self.image_paths = []
            for idx in range(len(self.xor_df)):
                img_path = self.image_dir / f"{idx:06d}.png"
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                self.image_paths.append(img_path)
        else:
            # Try to load precomputed visual features
            visual_file = self.data_dir / f"{self.split}_visual_features.npy"
            if visual_file.exists():
                self.visual_features = np.load(visual_file)
                logger.info(f"Loaded precomputed visual features: {self.visual_features.shape}")
            else:
                logger.warning("No visual data found, will use dummy features")
                self.visual_features = np.zeros((len(self.xor_df), 64))  # Dummy features
    
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
                input_attrs, target_attrs = self._prepare_attribute_tensors(
                    feature_indices[domain]
                )
                self.inputs[domain] = input_attrs
                self.targets[domain] = target_attrs
                
            elif domain == 'v':
                # Visual domain: same for input and target (for now)
                visual_data = self._prepare_visual_tensors()
                self.inputs[domain] = visual_data
                self.targets[domain] = visual_data
                
            else:
                raise ValueError(f"Unsupported domain: {domain}")
        
        logger.info("Prepared input/target tensors for all domains")
        for domain in domains:
            logger.info(f"  {domain}: input {self.inputs[domain].shape}, "
                       f"target {self.targets[domain].shape}")
    
    def _prepare_attribute_tensors(self, synergy_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare attribute tensors with synergy separation.
        
        Args:
            synergy_indices: Indices of synergistic features in XOR data
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        # Input: original attributes (no synergistic features)
        input_attrs = torch.tensor(self.original_attrs, dtype=torch.float32, device=self.device)
        
        # Target: original attributes + synergistic features
        # Extract synergistic features from XOR data
        synergy_features = []
        for idx in synergy_indices:
            if isinstance(idx, str):
                # Column name
                synergy_features.append(self.xor_df[idx].values)
            else:
                # Numeric index
                synergy_features.append(self.xor_df.iloc[:, idx].values)
        
        synergy_tensor = torch.tensor(
            np.column_stack(synergy_features), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Concatenate original attributes with synergistic features
        target_attrs = torch.cat([input_attrs, synergy_tensor], dim=1)
        
        return input_attrs, target_attrs
    
    def _prepare_visual_tensors(self) -> torch.Tensor:
        """Prepare visual tensors."""
        if hasattr(self, 'visual_features'):
            return torch.tensor(self.visual_features, dtype=torch.float32, device=self.device)
        else:
            # Will load images on-demand
            return torch.zeros((len(self.xor_df), 3, 64, 64), dtype=torch.float32, device=self.device)
    
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
                import torchvision.transforms as T
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
            drop_last=False
        )
        
        dataloaders[split] = dataloader
        logger.info(f"Created {split} dataloader with {len(dataset)} samples")
    
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