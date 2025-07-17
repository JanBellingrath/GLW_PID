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

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from shimmer import DomainModule, LossOutput
from shimmer.modules.domain import DomainModule
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
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

# Custom data loading for modified data
from adapt_training_for_modified_data import ModifiedDataModule

#############################################################################
# CONFIGURATION AND SETUP
#############################################################################

# PLACEHOLDER: Update these paths to your modified data
MODIFIED_DATA_PATH = "/path/to/your/modified/dataset"  # UPDATE THIS
OUTPUT_DIR = Path("./modified_data_outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LATENTS_DIR = OUTPUT_DIR / "latents"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
LATENTS_DIR.mkdir(exist_ok=True)

# Training configuration
CONFIG = {
    'seed': 42,
    'batch_size': 32,
    'num_workers': 4,
    'max_steps': 10000,  # Adjust as needed
    'learning_rate': 1e-3,
    'weight_decay': 0.0,
    'visual': {
        'latent_dim': 12,
        'ae_dim': 64,
        'beta': 1.0
    },
    'attribute': {
        'latent_dim': 12,
        'hidden_dim': 64,
        'beta': 1.0
    }
}

#############################################################################
# VISUAL DOMAIN MODULE (keep function definitions)
#############################################################################

class VisualDomainModule(DomainModule):
    def __init__(
        self,
        num_channels: int,
        latent_dim: int,
        ae_dim: int,
        beta: float = 1,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: Mapping[str, Any] | None = None,
    ):
        """Visual domain module with VAE for image encoding."""
        super().__init__(latent_dim)
        self.save_hyperparameters()

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

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["v"]
        return self.generic_step(x, "val")

    def generic_step(self, x: torch.Tensor, mode: str = "train") -> torch.Tensor:
        """Computes the loss given image data"""
        (mean, logvar), reconstruction = self.vae(x)
        
        reconstruction_loss = gaussian_nll(reconstruction, torch.tensor(0), x).sum()
        kl_loss = kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.vae.beta * kl_loss

        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)
        self.log(f"{mode}/kl_loss", kl_loss)
        self.log(f"{mode}/loss", total_loss)

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
            nn.Linear(11, hidden_dim),  # 3 categories + 8 attributes
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
        self.decoder_attributes = nn.Sequential(
            nn.Linear(self.hidden_dim, 8),
            nn.Tanh(),
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

    def validation_step(self, batch: Mapping[str, Sequence[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        x = batch["attr"]
        return self.generic_step(x, "val")

    def generic_step(self, x: Sequence[torch.Tensor], mode: str = "train") -> torch.Tensor:
        x_categories, x_attributes = x[0], x[1]

        (mean, logvar), reconstruction = self.vae(x)
        reconstruction_categories = reconstruction[0]
        reconstruction_attributes = reconstruction[1]

        reconstruction_loss_categories = F.cross_entropy(
            reconstruction_categories, x_categories.argmax(dim=1), reduction="sum"
        )
        reconstruction_loss_attributes = gaussian_nll(reconstruction_attributes, torch.tensor(0), x_attributes).sum()

        reconstruction_loss = (
            self.coef_categories * reconstruction_loss_categories
            + self.coef_attributes * reconstruction_loss_attributes
        )
        kl_loss = kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.vae.beta * kl_loss

        self.log(f"{mode}/reconstruction_loss_categories", reconstruction_loss_categories)
        self.log(f"{mode}/reconstruction_loss_attributes", reconstruction_loss_attributes)
        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)
        self.log(f"{mode}/kl_loss", kl_loss)
        self.log(f"{mode}/loss", total_loss)
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
    data_module = ModifiedDataModule(
        data_path=MODIFIED_DATA_PATH,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
    )
    
    # Create visual domain module
    v_domain_module = VisualDomainModule(
        num_channels=3,
        ae_dim=CONFIG['visual']['ae_dim'],
        latent_dim=CONFIG['visual']['latent_dim'],
        beta=CONFIG['visual']['beta'],
        optim_lr=CONFIG['learning_rate'],
        optim_weight_decay=CONFIG['weight_decay'],
        scheduler_args={
            "max_lr": CONFIG['learning_rate'],
            "total_steps": CONFIG['max_steps'],
        },
    )
    
    # Setup logging and callbacks
    logger = TensorBoardLogger(OUTPUT_DIR / "logs", name="visual_vae")
    
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
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=1,
        ),
    ]
    
    # Create trainer and train
    trainer = Trainer(
        logger=logger,
        max_steps=CONFIG['max_steps'],
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
    )
    
    # trainer.fit(v_domain_module, data_module)
    # trainer.validate(v_domain_module, data_module, "best")
    
    return v_domain_module, data_module, visual_checkpoint_dir / "visual_vae_best.ckpt"

def train_attribute_vae():
    """Train the attribute VAE on modified data."""
    print("📊 Training Attribute VAE...")
    
    # Set random seed
    seed_everything(CONFIG['seed'], workers=True)
    
    # Create data module for attribute domain only
    data_module = ModifiedDataModule(
        data_path=MODIFIED_DATA_PATH,
        batch_size=CONFIG['batch_size'], 
        num_workers=CONFIG['num_workers'],
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
    logger = TensorBoardLogger(OUTPUT_DIR / "logs", name="attribute_vae")
    
    attribute_checkpoint_dir = CHECKPOINT_DIR / "attribute"
    attribute_checkpoint_dir.mkdir(exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=attribute_checkpoint_dir,
            filename="attribute_vae_best",
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=1,
        ),
    ]
    
    # Create trainer and train
    trainer = Trainer(
        logger=logger,
        max_steps=CONFIG['max_steps'],
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
    )
    
    # trainer.fit(attr_domain_module, data_module)
    # trainer.validate(attr_domain_module, data_module, "best")
    
    return attr_domain_module, data_module, attribute_checkpoint_dir / "attribute_vae_best.ckpt"

def main():
    """Main training workflow."""
    print("🚀 Starting Modified Data VAE Training Workflow")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📁 Modified data path: {MODIFIED_DATA_PATH}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Step 1: Train Visual VAE
    print("\n" + "="*60)
    # visual_model, visual_data, visual_checkpoint = train_visual_vae()
    
    # Step 2: Train Attribute VAE  
    print("\n" + "="*60)
    # attr_model, attr_data, attr_checkpoint = train_attribute_vae()
    
    # Step 3: Extract and save latents
    print("\n" + "="*60)
    print("💾 Extracting and saving latents...")
    
    # # Load best models for latent extraction
    # visual_model = VisualDomainModule.load_from_checkpoint(visual_checkpoint)
    # attr_model = AttributeDomainModule.load_from_checkpoint(attr_checkpoint)
    
    # visual_model.to(device)
    # attr_model.to(device)
    
    # # Extract latents
    # save_latents_for_domain(visual_model, visual_data, "v", "visual", device)
    # save_latents_for_domain(attr_model, attr_data, "attr", "attribute", device)
    
    print("\n🎉 Training workflow completed!")
    print(f"📁 All outputs saved to: {OUTPUT_DIR}")
    print(f"🏗️  Model checkpoints: {CHECKPOINT_DIR}")
    print(f"🔢 Latent vectors: {LATENTS_DIR}")

if __name__ == "__main__":
    # Uncomment the line below to run the full training workflow
    # main()
    
    # For now, just show the configuration
    print("Configuration loaded. Update MODIFIED_DATA_PATH and uncomment main() to start training.")
    print(f"Current config: {CONFIG}") 