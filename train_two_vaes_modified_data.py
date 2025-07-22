#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Simple Shapes VAE Training Script
==========================================

This script trains ONLY the two VAE domain modules (visual + attribute) 
on modified/new data and saves both models and extracted latents.

Workflow:
1. Train Visual VAE on modified image data
2. Train Attribute VAE on modified attribute data  
3. Extract and save latents for both domains (train + val sets)
4. Save trained models

Modified from original simple-shapes-dataset-training.ipynb
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast
import os
import numpy as np
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision

from shimmer import DomainModule, LossOutput
from shimmer.modules.vae import (
    VAE,
    VAEDecoder,
    VAEEncoder,
    gaussian_nll,
    kl_divergence_loss,
)
from shimmer_ssd import DEBUG_MODE, LOGGER, PROJECT_DIR
from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig, load_config
from shimmer_ssd.dataset.pre_process import TokenizeCaptions
from shimmer_ssd.logging import (
    LogAttributesCallback,
    LogVisualCallback,
    batch_to_device,
)
from shimmer_ssd.modules.vae import RAEDecoder, RAEEncoder
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR

# Set non-interactive backend for matplotlib
matplotlib.use('Agg')

# Enable debug mode if needed
torch.autograd.set_detect_anomaly(True)#TODO this works like that?

#############################################################################
# CONFIGURATION & PATHS
#############################################################################

MODIFIED_DATA_PATH = "shimmer_ssd/pid_analysis/modified_simpleshapes_full/"

# Output directories
OUTPUT_DIR = PROJECT_DIR / "vae_training_outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LATENTS_DIR = OUTPUT_DIR / "saved_latents"

# Create output directories
for dir_path in [OUTPUT_DIR, CHECKPOINT_DIR, LATENTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Training configuration
CONFIG = {
    'seed': 42,
    'batch_size': 32,
    'num_workers': 4,
    'max_steps': 300000,  # ~13 epochs - train long, early stop on val loss
    'learning_rate': 1e-3,
    'weight_decay': 0.0,
    'visual': {
        'latent_dim': 12,
        'ae_dim': 64,
        'beta': 1.0,
        'background_weight': None  # Will be auto-calculated if None
    },
    'attribute': {
        'latent_dim': 12,
        'hidden_dim': 64,
        'beta': 1.0
    }
}

#############################################################################
# BACKGROUND WEIGHTING FUNCTIONS
#############################################################################

def compute_foreground_mask(image: torch.Tensor, color_threshold: float = 0.05, deviation_threshold: float = 0.05) -> torch.Tensor:
    """
    Compute a binary foreground mask for an image.
    
    Args:
        image: RGB image tensor of shape [3, H, W] with values in [0, 1]
        color_threshold: Threshold for color deviation to detect colored pixels
        deviation_threshold: Threshold for brightness deviation from modal grey
        
    Returns:
        Binary mask of shape [1, H, W] where 1 = foreground, 0 = background
    """
    # Method 1: Detect colored pixels (RGB std > threshold)
    color_mask = image.std(dim=0, keepdim=True) > color_threshold  # [1, H, W]
    
    # Method 2: Detect deviation from average grey level
    bg_level = image.mean()  # Average grey level across entire image
    brightness_deviation = torch.abs(image.mean(dim=0, keepdim=True) - bg_level) > deviation_threshold
    
    # Combine both methods
    foreground_mask = (color_mask | brightness_deviation).float()
    
    return foreground_mask

def calculate_global_background_weight(data_module, num_samples: int = 1000) -> float:
    """
    Pre-calculate the global background weight by sampling from training data.
    
    Args:
        data_module: ModifiedShapesDataModule with training data
        num_samples: Number of samples to use for calculation
        
    Returns:
        Background weight (w_bg) to balance foreground vs background loss
    """
    print(f"🔍 Calculating global background weight from {num_samples} training samples...")
    
    # Setup data module
    data_module.setup()
    train_loader = data_module.train_dataloader()
    
    total_pixels = 0
    total_foreground_pixels = 0
    samples_processed = 0
    
    with torch.no_grad():
        for batch in train_loader:
            if samples_processed >= num_samples:
                break
                
            images = batch[frozenset(["v"])]["v"]  # [batch_size, 3, H, W]
            batch_size = images.size(0)
            
            for i in range(batch_size):
                if samples_processed >= num_samples:
                    break
                    
                image = images[i]  # [3, H, W]
                fg_mask = compute_foreground_mask(image)  # [1, H, W]
                
                total_pixels += fg_mask.numel()
                total_foreground_pixels += fg_mask.sum().item()
                samples_processed += 1
    
    foreground_ratio = total_foreground_pixels / total_pixels
    background_weight = foreground_ratio / (1 - foreground_ratio)  # N_fg / N_bg
    
    print(f"✅ Global statistics:")
    print(f"  Foreground ratio: {foreground_ratio:.4f}")
    print(f"  Background weight: {background_weight:.4f}")
    
    return background_weight

#############################################################################
# DATA PREPROCESSING FUNCTIONS
#############################################################################

def preprocess_attributes(raw_attributes: np.ndarray) -> torch.Tensor:
    """
    Convert 9-feature modified format to VAE-compatible normalized format.
    
    Input format (9 features): [category, x, y, size, rotation, r, g, b, background_scalar]
    Output format (11 features): [cat_0, cat_1, cat_2, x_norm, y_norm, size_norm, rotation_norm, r_norm, g_norm, b_norm, background_scalar]
    
    NOTE: All continuous attributes are normalized to the [0, 1] range to match the decoder's Sigmoid output range [0, 1].
    
    Args:
        raw_attributes: Array of shape [N, 9] with modified attribute format
        
    Returns:
        Tensor of shape [N, 11] with one-hot categories + 8 normalized attributes
    """
    N = raw_attributes.shape[0]
    
    # Extract components
    categories = raw_attributes[:, 0].astype(int)  # Single values {0, 1, 2}
    other_attributes = raw_attributes[:, 1:].copy()  # [x, y, size, rotation, r, g, b, background_scalar]
    
    # CRITICAL NORMALIZATION: Scale all attributes to [0, 1] range
    # This prevents the color values [0, 255] from dominating the loss
    
    # Normalize spatial attributes (CORRECTED ranges based on actual data)
    other_attributes[:, 0] = (other_attributes[:, 0] - 7.0) / 17.0   # x: [7, 24] → [0, 1]
    other_attributes[:, 1] = (other_attributes[:, 1] - 7.0) / 17.0   # y: [7, 24] → [0, 1]  
    other_attributes[:, 2] = (other_attributes[:, 2] - 7.0) / 7.0    # size: [7, 14] → [0, 1]
    other_attributes[:, 3] /= (2 * np.pi)  # rotation: [0, 2π] → [0, 1]
    
    # Normalize color attributes (CRITICAL!)
    other_attributes[:, 4] /= 255.0     # r: [0, 255] → [0, 1]
    other_attributes[:, 5] /= 255.0     # g: [0, 255] → [0, 1]
    other_attributes[:, 6] /= 255.0     # b: [0, 255] → [0, 1]
    
    # background_scalar is already in [0, 1] range - no change needed
    # other_attributes[:, 7] = other_attributes[:, 7]  # background_scalar: [0, 1] → [0, 1]
    
    # One-hot encode categories
    category_onehot = np.zeros((N, 3))
    category_onehot[np.arange(N), categories] = 1.0
    
    # Concatenate: [cat_0, cat_1, cat_2] + [x_norm, y_norm, size_norm, rotation_norm, r_norm, g_norm, b_norm, background_scalar]
    processed = np.concatenate([category_onehot, other_attributes], axis=1)
    
    return torch.tensor(processed, dtype=torch.float32)

def convert_rgba_to_rgb(image_path: str) -> torch.Tensor:
    """
    Load image and convert from RGBA to RGB format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tensor of shape [3, H, W] with RGB channels only
    """
    # Load image
    image = Image.open(image_path)
    
    # Convert RGBA to RGB (removes alpha channel)
    if image.mode == 'RGBA':
        # Create white background and paste RGBA image on it
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[-1])  # Use alpha as mask
        image = rgb_image
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] and changes to [C, H, W]
    ])
    
    return transform(image)

#############################################################################
# CUSTOM DATASET CLASSES
#############################################################################

class ModifiedShapesDataset(Dataset):
    """
    Dataset for loading modified SimpleShapes data with preprocessing.
    """
    
    def __init__(self, data_dir: str, split: str = "train", domain: str = "both"):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to modified dataset directory
            split: Dataset split ('train', 'val', 'test')
            domain: Which domain to return ('visual', 'attributes', 'both')
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.domain = domain
        
        # Load attributes
        labels_path = self.data_dir / f"{split}_labels_independent.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        self.raw_attributes = np.load(labels_path)
        print(f"Loaded {len(self.raw_attributes)} samples from {labels_path}")
        
        # Preprocess attributes (convert to one-hot + 9 attributes = 11 total)
        self.processed_attributes = preprocess_attributes(self.raw_attributes)
        
        # Image directory
        self.image_dir = self.data_dir / split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        print(f"Dataset initialized: {len(self)} samples")
        print(f"Attribute shape: {self.processed_attributes.shape}")
    
    def __len__(self):
        return len(self.raw_attributes)
    
    def __getitem__(self, idx):
        # Load and preprocess image (RGBA -> RGB)
        image_path = self.image_dir / f"{idx}.png"
        image = convert_rgba_to_rgb(str(image_path))
        
        # Get preprocessed attributes - split into categories and continuous
        attributes = self.processed_attributes[idx]
        categories = attributes[:3]  # One-hot encoded categories
        continuous_attrs = attributes[3:]  # 8 continuous attributes
        
        # Return format expected by shimmer domain modules
        if self.domain == "visual":
            return {frozenset(["v"]): {"v": image}}
        elif self.domain == "attributes":
            return {frozenset(["attr"]): {"attr": [categories, continuous_attrs]}}
        else:  # both
            return {
                'image': image,
                'attributes': [categories, continuous_attrs],
                'idx': idx
            }

class ModifiedShapesDataModule(LightningDataModule):
    """
    Data module for modified SimpleShapes dataset.
    """
    
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, domain: str = "both"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.domain = domain
        
    def setup(self, stage: str = None):
        # Create datasets
        self.train_dataset = ModifiedShapesDataset(self.data_dir, "train", self.domain)
        self.val_dataset = ModifiedShapesDataset(self.data_dir, "val", self.domain)
        
        print(f"Setup complete:")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )

#############################################################################
# COMPREHENSIVE VAE LOGGING CALLBACKS
#############################################################################

class VAEReconstructionCallback(Callback):
    """Log reconstruction samples and metrics for VAE training."""
    
    def __init__(self, num_samples=8, log_every_n_epochs=5):
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.val_samples = None
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self._log_reconstructions(trainer, pl_module)
            self._log_latent_statistics(trainer, pl_module)
    
    def _log_reconstructions(self, trainer, pl_module):
        """Log original vs reconstructed samples."""
        if self.val_samples is None:
            # Get validation samples once
            val_loader = trainer.datamodule.val_dataloader()
            batch = next(iter(val_loader))
            if isinstance(pl_module, VisualDomainModule):
                self.val_samples = batch[frozenset(["v"])]["v"][:self.num_samples]
            else:  # AttributeDomainModule
                self.val_samples = batch[frozenset(["attr"])]["attr"]
                
        pl_module.eval()
        with torch.no_grad():
            # Handle both single tensor (visual) and list of tensors (attributes)
            if isinstance(self.val_samples, (list, tuple)):
                # Attribute domain: list of [categories, continuous] tensors
                samples = [t[:self.num_samples].to(pl_module.device) for t in self.val_samples]
            else:
                # Visual domain: single tensor
                samples = self.val_samples[:self.num_samples].to(pl_module.device)
            
            if isinstance(pl_module, VisualDomainModule):
                # Visual reconstructions
                (mean, logvar), reconstruction = pl_module.vae(samples)
                
                # Create grid of original vs reconstructed images
                comparison = torch.cat([samples, reconstruction], dim=0)
                grid = torchvision.utils.make_grid(comparison, nrow=self.num_samples, normalize=True)
                
                trainer.logger.log_image(
                    key="visual_reconstructions", 
                    images=[grid], 
                    caption=["Top: Original, Bottom: Reconstructed"]
                )
                
                # Log sample generations from prior
                z_sample = torch.randn(self.num_samples, pl_module.latent_dim).to(pl_module.device)
                generated = pl_module.decode(z_sample)
                gen_grid = torchvision.utils.make_grid(generated, nrow=self.num_samples, normalize=True)
                
                trainer.logger.log_image(
                    key="visual_generations", 
                    images=[gen_grid], 
                    caption=["Generated from random latents"]
                )
                
            else:
                # Attribute reconstructions
                (mean, logvar), reconstruction = pl_module.vae(samples)
                
                # Log attribute reconstruction accuracy
                categories_target = samples[0].argmax(dim=1)
                categories_pred = reconstruction[0].argmax(dim=1)
                attr_accuracy = (categories_pred == categories_target).float().mean()
                
                continuous_mse = F.mse_loss(reconstruction[1], samples[1])
                
                # Log through Lightning's logger
                trainer.logger.log_metrics({
                    "attr_category_accuracy": attr_accuracy,
                    "attr_continuous_mse": continuous_mse,
                }, step=trainer.global_step)
    
    def _log_latent_statistics(self, trainer, pl_module):
        """Log latent space statistics."""
        if hasattr(pl_module, '_last_mean') and hasattr(pl_module, '_last_logvar'):
            mean_norm = pl_module._last_mean.norm(dim=1).mean()
            logvar_mean = pl_module._last_logvar.mean()
            
            trainer.logger.log_metrics({
                f"{pl_module.__class__.__name__.lower()}_latent_mean_norm": mean_norm,
                f"{pl_module.__class__.__name__.lower()}_latent_logvar_mean": logvar_mean,
            }, step=trainer.global_step)

class VAEMetricsCallback(Callback):
    """Track comprehensive VAE training metrics."""
    
    def __init__(self):
        super().__init__()
        
    def on_train_epoch_start(self, trainer, pl_module):
        # Log learning rate
        if trainer.optimizers:
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            trainer.logger.log_metrics({
                "learning_rate": current_lr,
            }, step=trainer.global_step)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Log additional metrics
        trainer.logger.log_metrics({
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "beta": pl_module.vae.beta if hasattr(pl_module.vae, 'beta') else 1.0,
        }, step=trainer.global_step)

#############################################################################
# VISUAL DOMAIN MODULE (updated for RGB only)
#############################################################################

class VisualDomainModule(DomainModule):
    def __init__(
        self,
        num_channels: int,
        latent_dim: int,
        ae_dim: int,
        beta: float = 1,
        background_weight: float = 1.0,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: Mapping[str, Any] | None = None,
    ):
        """Visual domain module with VAE for image encoding."""
        super().__init__(latent_dim)
        self.save_hyperparameters()
        
        # Store latent_dim for callback access
        self.latent_dim = latent_dim
        self.background_weight = background_weight

        vae_encoder = RAEEncoder(num_channels, ae_dim, latent_dim, use_batchnorm=True)
        vae_decoder = RAEDecoder(num_channels, latent_dim, ae_dim)
        self.vae = VAE(vae_encoder, vae_decoder, beta)
        
        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args: dict[str, Any] = {
            "max_lr": optim_lr,
            "total_steps": 1,
        }
        self.scheduler_args.update(scheduler_args or {})

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode from the image to the latent representation."""
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the unimodal latent into the original domain."""
        return self.vae.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any) -> LossOutput:
        """Compute MSE loss in the latent domain."""
        loss = mse_loss(pred, target, reduction="mean")
        return LossOutput(loss)

    def training_step(self, batch: Mapping[frozenset[str], Mapping[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        x = batch[frozenset(["v"])]["v"]
        return self.generic_step(x, "train")

    def validation_step(self, batch: Mapping[frozenset[str], Mapping[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        x = batch[frozenset(["v"])]["v"]
        return self.generic_step(x, "val")

    def generic_step(self, x: torch.Tensor, mode: str = "train") -> torch.Tensor:
        """Computes the loss given image data"""
        (mean, logvar), reconstruction = self.vae(x)
        
        # Store latent statistics for logging
        self._last_mean = mean.detach()
        self._last_logvar = logvar.detach()
        
        # Compute weighted reconstruction loss to balance foreground vs background
        if self.background_weight != 1.0:
            # Compute foreground/background weights for each image in batch
            batch_size = x.size(0)
            weight_maps = []
            
            for i in range(batch_size):
                fg_mask = compute_foreground_mask(x[i])  # [1, H, W]
                bg_mask = 1.0 - fg_mask  # [1, H, W]
                weight_map = fg_mask + self.background_weight * bg_mask  # [1, H, W]
                weight_maps.append(weight_map)
            
            weight_maps = torch.stack(weight_maps, dim=0)  # [batch_size, 1, H, W]
            weight_maps = weight_maps.expand_as(x)  # [batch_size, 3, H, W]
            
            # Compute weighted reconstruction loss
            err_squared = (reconstruction - x).pow(2)  # [batch_size, 3, H, W]
            reconstruction_loss = (weight_maps * err_squared).sum() * 0.5
        else:
            # Use original unweighted loss
            reconstruction_loss = gaussian_nll(reconstruction, torch.zeros_like(reconstruction), x).sum()
        
        kl_loss = kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.vae.beta * kl_loss

        # Enhanced logging
        batch_size = x.size(0)
        self.log(f"{mode}/reconstruction_loss", reconstruction_loss, batch_size=batch_size)
        self.log(f"{mode}/kl_loss", kl_loss, batch_size=batch_size)
        self.log(f"{mode}/total_loss", total_loss, batch_size=batch_size)
        self.log(f"{mode}/beta", self.vae.beta, batch_size=batch_size)
        self.log(f"{mode}/background_weight", self.background_weight, batch_size=batch_size)
        
        # Additional VAE metrics
        if mode == "val":
            kl_per_dim = kl_loss / self.latent_dim
            reconstruction_per_pixel = reconstruction_loss / (x.numel())
            
            self.log(f"{mode}/kl_per_dimension", kl_per_dim, batch_size=batch_size)
            self.log(f"{mode}/reconstruction_per_pixel", reconstruction_per_pixel, batch_size=batch_size)
            self.log(f"{mode}/elbo", -total_loss, batch_size=batch_size)  # Evidence Lower BOund
            
            # Latent statistics
            latent_mean_norm = mean.norm(dim=1).mean()
            latent_std_mean = torch.exp(0.5 * logvar).mean()
            
            self.log(f"{mode}/latent_mean_norm", latent_mean_norm, batch_size=batch_size)
            self.log(f"{mode}/latent_std_mean", latent_std_mean, batch_size=batch_size)

        return total_loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        lr_scheduler = OneCycleLR(optimizer, **self.scheduler_args)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

#############################################################################
# ATTRIBUTE DOMAIN MODULE (keep function definitions)
#############################################################################

class Encoder(VAEEncoder):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.encoder = nn.Sequential(
            nn.Linear(11, hidden_dim),  # 3 one-hot categories + 8 attributes (including background_scalar)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

        self.q_mean = nn.Linear(self.out_dim, self.out_dim)
        self.q_logvar = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.cat(list(x), dim=-1)
        out = self.encoder(out)
        return self.q_mean(out), self.q_logvar(out)

class Decoder(VAEDecoder):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.decoder_categories = nn.Sequential(nn.Linear(self.hidden_dim, 3))
        
        # Output 8 attributes: [x, y, size, rotation, r, g, b, background_scalar]
        self.decoder_attributes = nn.Sequential(
            nn.Linear(self.hidden_dim, 8),
            nn.Sigmoid(),  # FIXED: Output [0, 1] range to match normalized input attributes
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = self.decoder(x)
        return [self.decoder_categories(out), self.decoder_attributes(out)]

class AttributeDomainModule(DomainModule):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        beta: float = 1,
        coef_categories: float = 1,
        coef_attributes: float = 1,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: dict[str, Any] | None = None,
    ):
        """Attribute domain module with VAE."""
        super().__init__(latent_dim)
        self.save_hyperparameters()
        
        # Store latent_dim for callback access
        self.latent_dim = latent_dim

        self.hidden_dim = hidden_dim
        self.coef_categories = coef_categories
        self.coef_attributes = coef_attributes

        vae_encoder = Encoder(self.hidden_dim, self.latent_dim)
        vae_decoder = Decoder(self.latent_dim, self.hidden_dim)
        self.vae = VAE(vae_encoder, vae_decoder, beta)

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args = {
            "max_lr": optim_lr,
            "total_steps": 1,
        }
        self.scheduler_args.update(scheduler_args or {})

    def encode(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        """Encodes the attributes into the latent representation."""
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Decodes the latent representation to the shape category and attributes."""
        return self.vae.decode(z)

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return self.decode(self.encode(x))

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any) -> LossOutput:
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

    def training_step(self, batch: Mapping[frozenset[str], Mapping[str, Sequence[torch.Tensor]]], batch_idx: int) -> torch.Tensor:
        x = batch[frozenset(["attr"])]["attr"]
        return self.generic_step(x, "train")

    def validation_step(self, batch: Mapping[frozenset[str], Mapping[str, Sequence[torch.Tensor]]], batch_idx: int) -> torch.Tensor:
        x = batch[frozenset(["attr"])]["attr"]
        return self.generic_step(x, "val")

    def generic_step(self, x: Sequence[torch.Tensor], mode: str = "train") -> torch.Tensor:
        x_categories, x_attributes = x[0], x[1]

        (mean, logvar), reconstruction = self.vae(x)
        reconstruction_categories = reconstruction[0]
        reconstruction_attributes = reconstruction[1]
        
        # Store latent statistics for logging
        self._last_mean = mean.detach()
        self._last_logvar = logvar.detach()

        # Compute losses
        reconstruction_loss_categories = F.cross_entropy(
            reconstruction_categories, x_categories.argmax(dim=1), reduction="sum"
        )
        reconstruction_loss_attributes = gaussian_nll(
            reconstruction_attributes,
            torch.zeros_like(reconstruction_attributes),
            x_attributes,
        ).sum()

        reconstruction_loss = (
            self.coef_categories * reconstruction_loss_categories
            + self.coef_attributes * reconstruction_loss_attributes
        )
        kl_loss = kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.vae.beta * kl_loss

        # Enhanced logging
        batch_size = x_categories.size(0)
        self.log(f"{mode}/reconstruction_loss_categories", reconstruction_loss_categories, batch_size=batch_size)
        self.log(f"{mode}/reconstruction_loss_attributes", reconstruction_loss_attributes, batch_size=batch_size)
        self.log(f"{mode}/reconstruction_loss", reconstruction_loss, batch_size=batch_size)
        self.log(f"{mode}/kl_loss", kl_loss, batch_size=batch_size)
        self.log(f"{mode}/total_loss", total_loss, batch_size=batch_size)
        self.log(f"{mode}/beta", self.vae.beta, batch_size=batch_size)
        
        # Additional VAE metrics
        if mode == "val":
            # Category accuracy
            cat_pred = reconstruction_categories.argmax(dim=1)
            cat_target = x_categories.argmax(dim=1)
            category_accuracy = (cat_pred == cat_target).float().mean()
            
            # Continuous attribute MSE
            attr_mse = F.mse_loss(reconstruction_attributes, x_attributes)
            
            # VAE specific metrics
            kl_per_dim = kl_loss / self.latent_dim
            
            self.log(f"{mode}/category_accuracy", category_accuracy, batch_size=batch_size)
            self.log(f"{mode}/attribute_mse", attr_mse, batch_size=batch_size)
            self.log(f"{mode}/kl_per_dimension", kl_per_dim, batch_size=batch_size)
            self.log(f"{mode}/elbo", -total_loss, batch_size=batch_size)
            
            # Latent statistics
            latent_mean_norm = mean.norm(dim=1).mean()
            latent_std_mean = torch.exp(0.5 * logvar).mean()
            
            self.log(f"{mode}/latent_mean_norm", latent_mean_norm, batch_size=batch_size)
            self.log(f"{mode}/latent_std_mean", latent_std_mean, batch_size=batch_size)
            
        return total_loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        lr_scheduler = OneCycleLR(optimizer, **self.scheduler_args)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

#############################################################################
# LATENT EXTRACTION FUNCTIONS
#############################################################################

def extract_latents_from_dataloader(model, dataloader, domain_key, device):
    """Extract latents from a dataloader using trained model."""
    model.eval()
    all_latents = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict) and frozenset([domain_key]) in batch:
                # Training format: {frozenset(['v']): {'v': data}}
                data = batch[frozenset([domain_key])][domain_key]
            elif isinstance(batch, dict) and domain_key in batch:
                # Validation format: {'v': data}
                data = batch[domain_key]
            else:
                continue
                
            # Move data to device
            if isinstance(data, (list, tuple)):
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
            latents = model.encode(data)
            all_latents.append(latents.cpu().numpy())
    
    return np.concatenate(all_latents, axis=0)

def save_latents_for_domain(model, data_module, domain_key, model_name, device):
    """Extract and save latents for train and validation sets."""
    print(f"\n🔍 Extracting latents for {model_name} domain...")
    
    # Extract train latents
    train_dataloader = data_module.train_dataloader()
    train_latents = extract_latents_from_dataloader(model, train_dataloader, domain_key, device)
    train_file = LATENTS_DIR / f"{model_name}_train_latents.npy"
    np.save(train_file, train_latents)
    print(f"✅ Saved train latents: {train_file} (shape: {train_latents.shape})")
    
    # Extract validation latents
    val_dataloader = data_module.val_dataloader()
    val_latents = extract_latents_from_dataloader(model, val_dataloader, domain_key, device)
    val_file = LATENTS_DIR / f"{model_name}_val_latents.npy"
    np.save(val_file, val_latents)
    print(f"✅ Saved val latents: {val_file} (shape: {val_latents.shape})")
    
    return train_latents, val_latents

#############################################################################
# MAIN TRAINING WORKFLOW
#############################################################################

def train_visual_vae():
    """Train the visual VAE on modified data."""
    print("🖼️  Training Visual VAE...")
    
    # Set random seed
    seed_everything(CONFIG['seed'], workers=True)
    
    # Create data module for visual domain only
    data_module = ModifiedShapesDataModule(
        data_dir=MODIFIED_DATA_PATH,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        domain="visual",
    )
    
    # Calculate background weight if not explicitly set
    if CONFIG['visual']['background_weight'] is None:
        background_weight = calculate_global_background_weight(data_module, num_samples=1000)
        CONFIG['visual']['background_weight'] = background_weight
    else:
        background_weight = CONFIG['visual']['background_weight']
    
    # Create visual domain module
    v_domain_module = VisualDomainModule(
        num_channels=3,
        ae_dim=CONFIG['visual']['ae_dim'],
        latent_dim=CONFIG['visual']['latent_dim'],
        beta=CONFIG['visual']['beta'],
        background_weight=background_weight,
        optim_lr=CONFIG['learning_rate'],
        optim_weight_decay=CONFIG['weight_decay'],
        scheduler_args={
            "max_lr": CONFIG['learning_rate'],
            "total_steps": CONFIG['max_steps'],
        },
    )
    
    # Setup logging and callbacks
    logger = WandbLogger(
        project="modified_shapes_vae", 
        name="visual_vae",
        config={
            **CONFIG,
            "model_type": "VisualVAE",
            "domain": "visual", 
            "architecture": "RAE",
            "input_channels": 3,
            "image_size": "32x32",
            "encoder": "RAEEncoder",
            "decoder": "RAEDecoder",
            "likelihood": "Gaussian",
            "dataset": "modified_simpleshapes",
            "data_path": str(MODIFIED_DATA_PATH),
            "background_weight": background_weight,
            "model_parameters": sum(p.numel() for p in v_domain_module.parameters()),
            "trainable_parameters": sum(p.numel() for p in v_domain_module.parameters() if p.requires_grad),
        },
        tags=["vae", "visual", "modified_shapes", "training", "background_weighted"],
        notes=f"Training visual VAE with background weighting (w_bg={background_weight:.4f}) on modified SimpleShapes dataset"
    )
    
    # Get samples for logging
    # val_samples = data_module.get_samples("val", 32)[frozenset(["v"])]["v"]
    # train_samples = data_module.get_samples("train", 32)[frozenset(["v"])]["v"]
    
    visual_checkpoint_dir = CHECKPOINT_DIR / "visual"
    visual_checkpoint_dir.mkdir(exist_ok=True)
    
    callbacks = [
        # LogVisualCallback(
        #     val_samples,
        #     log_key="val_images",
        #     mode="val",
        #     every_n_epochs=1,
        #     ncols=8,
        # ),
        ModelCheckpoint(
            dirpath=visual_checkpoint_dir,
            filename="visual_vae_best",
            monitor="val/total_loss",
            mode="min",
            save_last=True,
            save_top_k=1,
        ),
        VAEReconstructionCallback(num_samples=8, log_every_n_epochs=5),
        VAEMetricsCallback(),
    ]
    
    # Create trainer and train
    trainer = Trainer(
        logger=logger,
        max_steps=CONFIG['max_steps'],
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        val_check_interval=0.25,  # Validate 4 times per epoch
    )
    
    trainer.fit(v_domain_module, data_module)
    trainer.validate(v_domain_module, data_module, "best")
    
    return v_domain_module, data_module, visual_checkpoint_dir / "visual_vae_best.ckpt"

def train_attribute_vae():
    """Train the attribute VAE on modified data."""
    print("📊 Training Attribute VAE...")
    
    # Set random seed
    seed_everything(CONFIG['seed'], workers=True)
    
    # Create data module for attribute domain only
    data_module = ModifiedShapesDataModule(
        data_dir=MODIFIED_DATA_PATH,
        batch_size=CONFIG['batch_size'], 
        num_workers=CONFIG['num_workers'],
        domain="attributes",
    )
    
    # Create attribute domain module
    attr_domain_module = AttributeDomainModule(
        latent_dim=CONFIG['attribute']['latent_dim'],
        hidden_dim=CONFIG['attribute']['hidden_dim'],
        beta=CONFIG['attribute']['beta'],
        optim_lr=CONFIG['learning_rate'],
        optim_weight_decay=CONFIG['weight_decay'],
        scheduler_args={
            "max_lr": CONFIG['learning_rate'],
            "total_steps": CONFIG['max_steps'],
        },
    )
    
    # Setup logging and callbacks
    logger = WandbLogger(
        project="modified_shapes_vae", 
        name="attribute_vae",
        config={
            **CONFIG,
            "model_type": "AttributeVAE",
            "domain": "attributes",
            "architecture": "MLP",
            "input_features": 11,  # 3 categories + 8 continuous
            "output_categories": 3,
            "output_continuous": 8,
            "encoder": "Custom MLP",
            "decoder": "Custom MLP + Sigmoid",
            "categorical_likelihood": "Categorical (CrossEntropy)",
            "continuous_likelihood": "Gaussian",
            "dataset": "modified_simpleshapes",
            "data_path": str(MODIFIED_DATA_PATH),
            "model_parameters": sum(p.numel() for p in attr_domain_module.parameters()),
            "trainable_parameters": sum(p.numel() for p in attr_domain_module.parameters() if p.requires_grad),
        },
        tags=["vae", "attributes", "modified_shapes", "training"],
        notes="Training attribute VAE on modified SimpleShapes dataset with normalized attributes"
    )
    
    attribute_checkpoint_dir = CHECKPOINT_DIR / "attribute"
    attribute_checkpoint_dir.mkdir(exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=attribute_checkpoint_dir,
            filename="attribute_vae_best",
            monitor="val/total_loss",
            mode="min",
            save_last=True,
            save_top_k=1,
        ),
        VAEReconstructionCallback(num_samples=8, log_every_n_epochs=5),
        VAEMetricsCallback(),
    ]
    
    # Create trainer and train
    trainer = Trainer(
        logger=logger,
        max_steps=CONFIG['max_steps'],
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        val_check_interval=0.25,  # Validate 4 times per epoch
    )
    
    trainer.fit(attr_domain_module, data_module)
    trainer.validate(attr_domain_module, data_module, "best")
    
    return attr_domain_module, data_module, attribute_checkpoint_dir / "attribute_vae_best.ckpt"

def main():
    """Main training workflow."""
    print("🚀 Starting Modified Data VAE Training Workflow")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📁 Modified data path: {MODIFIED_DATA_PATH}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Initialize timing for CUDA devices
    start_time = None
    end_time = None
    if torch.cuda.is_available() and device.type == "cuda":
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
    
    try:
        # Step 1: Train Visual VAE
        print("\n" + "="*60)
        visual_model, visual_data, visual_checkpoint = train_visual_vae()
        
        # Step 2: Train Attribute VAE  
        print("\n" + "="*60)
        attr_model, attr_data, attr_checkpoint = train_attribute_vae()
        
        # Step 3: Extract and save latents
        print("\n" + "="*60)
        print("💾 Extracting and saving latents...")
        
        # Load best models for latent extraction
        visual_model = VisualDomainModule.load_from_checkpoint(visual_checkpoint)
        attr_model = AttributeDomainModule.load_from_checkpoint(attr_checkpoint)
        
        visual_model.to(device)
        attr_model.to(device)
        
        # Extract latents
        visual_train_latents, visual_val_latents = save_latents_for_domain(visual_model, visual_data, "v", "visual", device)
        attr_train_latents, attr_val_latents = save_latents_for_domain(attr_model, attr_data, "attr", "attribute", device)
        
        # Calculate timing if CUDA was used
        if start_time and end_time:
            end_time.record()
            torch.cuda.synchronize()
            total_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            print(f"⏱️ Total training time: {total_time:.2f} seconds")
        
        print("\n🎉 Training workflow completed!")
        print(f"📁 All outputs saved to: {OUTPUT_DIR}")
        print(f"🏗️  Model checkpoints: {CHECKPOINT_DIR}")
        print(f"🔢 Latent vectors: {LATENTS_DIR}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 