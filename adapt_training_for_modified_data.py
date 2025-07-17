#!/usr/bin/env python3
"""
Adaptation Guide: Using the Simple Shapes Training Script for Modified Data
==========================================================================

This script shows how to modify the original simple_shapes_dataset_training.py
to work with your custom "changed data" instead of the original dataset.

Key Changes Needed:
1. Custom Dataset class
2. Modified data loading
3. Updated configuration
4. Proper output naming
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple

class ModifiedDataset(Dataset):
    """
    Custom dataset class for your modified/changed data.
    Adapt this to match your specific data format.
    """
    
    def __init__(self, data_path: str, split: str = "train", transform=None):
        """
        Args:
            data_path: Path to your modified dataset
            split: 'train', 'val', or 'test'
            transform: Optional data transformations
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        
        # Load your modified data
        self.image_files = self._load_image_paths()
        self.attributes = self._load_attributes()
        
    def _load_image_paths(self) -> List[Path]:
        """Load paths to your modified images"""
        image_dir = self.data_path / self.split / "images"
        return list(image_dir.glob("*.png"))  # Adjust extension as needed
        
    def _load_attributes(self) -> Dict[str, Any]:
        """Load attribute data for your modified dataset"""
        attr_file = self.data_path / self.split / "attributes.npy"
        if attr_file.exists():
            return np.load(attr_file, allow_pickle=True).item()
        else:
            # Generate dummy attributes if not available
            return {}
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns data in the format expected by the training script:
        {
            "v": image_tensor,  # [3, 32, 32]
            "attr": [categories_one_hot, attributes_vector]
        }
        """
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to match original script
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        # Load attributes (modify this based on your data format)
        sample_id = image_path.stem
        if sample_id in self.attributes:
            attr_data = self.attributes[sample_id]
            categories = torch.tensor(attr_data["categories"])  # One-hot encoded
            attributes = torch.tensor(attr_data["attributes"])  # Continuous values
        else:
            # Default/dummy attributes if not available
            categories = torch.zeros(3)  # 3 shape categories
            attributes = torch.zeros(8)  # 8 attribute values
            
        return {
            "v": image,
            "attr": [categories, attributes]
        }


class ModifiedDataModule:
    """
    Data module that replaces SimpleShapesDataModule for your custom data.
    """
    
    def __init__(self, data_path: str, batch_size: int = 32, num_workers: int = 4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create datasets
        self.train_dataset = ModifiedDataset(data_path, "train")
        self.val_dataset = ModifiedDataset(data_path, "val")
        self.test_dataset = ModifiedDataset(data_path, "test")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def get_samples(self, split: str, num_samples: int) -> Dict[frozenset, Dict[str, Any]]:
        """
        Get samples in the format expected by the training script.
        This matches the interface of SimpleShapesDataModule.
        """
        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset
            
        samples = []
        for i in range(min(num_samples, len(dataset))):
            samples.append(dataset[i])
        
        # Group samples by domain combination
        batch = {
            frozenset(["v"]): {
                "v": torch.stack([s["v"] for s in samples])
            },
            frozenset(["attr"]): {
                "attr": [
                    torch.stack([s["attr"][0] for s in samples]),  # categories
                    torch.stack([s["attr"][1] for s in samples])   # attributes
                ]
            }
        }
        
        return batch


def modify_training_script_for_custom_data():
    """
    Step-by-step guide to modify the original training script.
    """
    
    modifications = [
        {
            "section": "Data Loading",
            "original": "data_module = SimpleShapesDataModule(...)",
            "modified": """
# Replace SimpleShapesDataModule with your custom data module
data_module = ModifiedDataModule(
    data_path="path/to/your/modified/data",
    batch_size=config.training.batch_size,
    num_workers=config.training.num_workers,
)
            """,
            "line_range": "~150-160"
        },
        {
            "section": "Configuration",
            "original": "config.dataset.path",
            "modified": """
# Update config to point to your modified data
config.dataset.path = "path/to/your/modified/data"
            """,
            "line_range": "~100-110"
        },
        {
            "section": "Latent Extraction",
            "original": '--latent_name "domain_v_tuto.npy"',
            "modified": '--latent_name "modified_data_v_latents.npy"',
            "line_range": "~400"
        },
        {
            "section": "Checkpoint Paths",
            "original": "checkpoints/vision/version_X/last.ckpt",
            "modified": "checkpoints/modified_vision/version_X/last.ckpt",
            "line_range": "Various locations"
        }
    ]
    
    return modifications


# Example usage in the training script:
if __name__ == "__main__":
    print("Adaptation Guide for Modified Data Training")
    print("=" * 50)
    
    modifications = modify_training_script_for_custom_data()
    
    for i, mod in enumerate(modifications, 1):
        print(f"{i}. {mod['section']} (Lines {mod['line_range']})")
        print(f"   Original: {mod['original']}")
        print(f"   Modified: {mod['modified']}")
        print() 