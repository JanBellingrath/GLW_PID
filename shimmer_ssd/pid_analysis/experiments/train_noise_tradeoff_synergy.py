#!/usr/bin/env python3
"""
Noise–Synergy Tradeoff Experiment for Global Workspace

This script reuses the existing synergy-aware training pipeline and adds:
- Configurable Gaussian noise injection into the post-fusion (post-tanh) latent
- Optional third decoder head 'syn' (3-layer MLP) for synergy classification
- Evaluation comparing direct synergy decoding vs broadcast-cycle re-fuse decoding under noise

Usage examples:
  # Train with syn head, disable attr synergy logits, inject noise during training and eval
  python train_noise_tradeoff_synergy.py \
    --config ../synergy_config.json \
    --enable-syn-head --disable-attr-synergy \
    --train-noise-std 0.15 --eval-noise-std 0.15 --synergy-bins 8

  # Evaluate a sweep of noise stds after training
  python train_noise_tradeoff_synergy.py \
    --config ../synergy_config.json \
    --evaluate-only \
    --eval-noise-stds 0.0,0.05,0.1,0.2
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch

# Ensure parent directory (pid_analysis) is importable
_SCRIPT_DIR = Path(__file__).parent
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

# Local imports from existing pipeline
from train_synergy_glw import SynergyExperimentConfig, SynergyTrainer
from synergy_losses import convert_synergy_targets_to_classes, extract_synergy_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Noise–Synergy Tradeoff Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to synergy configuration JSON")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--eval-noise-stds", type=str, default=None, help="Comma-separated list of eval noise stds to sweep")
    # Pass-through flags to enable the syn setup and noise (mirrors train_synergy_glw.py)
    parser.add_argument("--enable-syn-head", action="store_true")
    parser.add_argument("--disable-attr-synergy", action="store_true")
    parser.add_argument("--train-noise-std", type=float, default=None)
    parser.add_argument("--eval-noise-std", type=float, default=None)
    parser.add_argument("--noise-site", type=str, default=None)
    parser.add_argument("--synergy-bins", type=int, default=None)
    parser.add_argument("--syn-head-n-layers", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, default="fusion_only", help="Which loss config to run (must exist in config)")
    return parser.parse_args()


def apply_overrides(config: SynergyExperimentConfig, args: argparse.Namespace) -> None:
    # Enable syn head and disable attr synergy logits per request
    if args.enable_syn_head:
        config.synergy_config['enable_syn_head'] = True
    if args.disable_attr_synergy:
        config.synergy_config['attr_includes_synergy'] = False
    # Noise site/std overrides
    noise = config.synergy_config.setdefault('noise', {})
    if args.noise_site is not None:
        noise['site'] = args.noise_site
    if args.train_noise_std is not None:
        noise['train_std'] = float(args.train_noise_std)
    if args.eval_noise_std is not None:
        noise['eval_std'] = float(args.eval_noise_std)
    # Bins and syn head depth
    if args.synergy_bins is not None:
        config.synergy_config['n_bins'] = int(args.synergy_bins)
    if args.syn_head_n_layers is not None:
        config.synergy_config['syn_head_n_layers'] = int(args.syn_head_n_layers)
    # Restrict to a single experiment if provided
    if args.experiment_name:
        config.loss_configs = [cfg for cfg in config.loss_configs if cfg['name'] == args.experiment_name]


@torch.no_grad()
def evaluate_tradeoff(trainer: SynergyTrainer, eval_noise_stds: List[float]) -> Dict[str, Any]:
    model = trainer.model
    device = trainer.device
    noise_cfg = trainer.config.synergy_config.get('noise', {})
    site = noise_cfg.get('site', 'post_fusion_post_tanh')
    n_bins = int(trainer.config.synergy_config.get('n_bins', 8))

    results = {}

    if 'val' not in trainer.dataloaders:
        return {"error": "No validation dataloader available"}

    val_loader = trainer.create_synergy_dataloader_wrapper('val')
    model.eval()

    def inject_noise(gw_state: torch.Tensor, std: float) -> torch.Tensor:
        if 'post_tanh' in str(site):
            gw_state = torch.tanh(gw_state)
        if std and std > 0.0:
            gw_state = gw_state + std * torch.randn_like(gw_state)
        return gw_state

    for std in eval_noise_stds:
        total = 0
        correct_direct = 0
        correct_cycle = 0

        for batch in val_loader:
            # Separate inputs and targets from flattened batch
            inputs = {}
            targets = {}
            for k, v in batch.items():
                if k.startswith('_target_'):
                    targets[k.replace('_target_', '')] = v.to(device)
                else:
                    inputs[k] = v.to(device)

            # Encode inputs (bypassing domain modules for 'attr' and 'v')
            latents = {}
            for domain, data in inputs.items():
                if domain in model.gw_encoders:
                    latents[domain] = model.gw_encoders[domain](data)

            if not latents:
                continue

            # Fuse and inject eval noise
            gw_state = model.fuse(latents, selection_scores={})
            gw_state_noisy = inject_noise(gw_state, std)

            # Direct syn prediction
            if 'syn' not in model.gw_decoders:
                raise RuntimeError("'syn' decoder head not available. Enable with --enable-syn-head and retrain.")
            syn_logits_direct = model.gw_decoders['syn'](gw_state_noisy)

            # Broadcast-cycle: decode to each modality base, re-encode, and re-fuse
            decoded_attr = model.gw_decoders['attr'](gw_state_noisy) if 'attr' in model.gw_decoders else None
            decoded_v = model.gw_decoders['v'](gw_state_noisy) if 'v' in model.gw_decoders else None

            re_latents = {}
            if decoded_attr is not None and 'attr' in model.gw_encoders:
                re_latents['attr'] = model.gw_encoders['attr'](decoded_attr)
            if decoded_v is not None and 'v' in model.gw_encoders:
                re_latents['v'] = model.gw_encoders['v'](decoded_v)

            if not re_latents:
                continue

            gw_state_refused = model.fuse(re_latents, selection_scores={})
            syn_logits_cycle = model.gw_decoders['syn'](gw_state_refused)

            # Build target class indices from attr target
            if 'attr' not in targets:
                continue
            syn_target_scalar, _ = extract_synergy_features(targets['attr'], 'attr', trainer.config.synergy_config, is_model_output=False, n_synergy_classes=n_bins)
            # If synergy target is one-hot (B, n_bins), reduce to class indices
            if syn_target_scalar.shape[-1] == n_bins:
                target_classes = syn_target_scalar.argmax(dim=-1).view(-1)
            else:
                target_classes = convert_synergy_targets_to_classes(syn_target_scalar.view(-1), n_bins=n_bins)

            # Handle logits shape [B, n_bins] or [B, n_features*n_bins]
            def logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
                if logits.shape[-1] == n_bins:
                    return logits.argmax(dim=-1).view(-1)
                else:
                    # Assume single feature for simplicity; take first group of n_bins
                    first_group = logits[..., :n_bins]
                    return first_group.argmax(dim=-1).view(-1)

            pred_direct = logits_to_pred(syn_logits_direct)
            pred_cycle = logits_to_pred(syn_logits_cycle)

            batch_size = target_classes.shape[0]
            total += batch_size
            correct_direct += (pred_direct == target_classes).sum().item()
            correct_cycle += (pred_cycle == target_classes).sum().item()

        results[str(std)] = {
            'direct_accuracy': (correct_direct / total) if total else 0.0,
            'cycle_accuracy': (correct_cycle / total) if total else 0.0,
            'num_samples': total,
        }

    return results


def main():
    args = parse_args()

    # Load and override config
    config = SynergyExperimentConfig.from_file(args.config)
    apply_overrides(config, args)

    # Device override
    device = torch.device(args.device) if args.device else None

    # Initialize trainer and data/model
    trainer = SynergyTrainer(config, device=device)
    trainer.setup_model()
    trainer.setup_data()

    # Train unless evaluate-only
    if not args.evaluate_only:
        trainer.run_all_experiments()

    # Build noise std sweep
    if args.eval_noise_std is not None and args.eval_noise_stds is None:
        eval_stds = [float(args.eval_noise_std)]
    elif args.eval_noise_stds is not None:
        eval_stds = [float(x) for x in args.eval_noise_stds.split(',') if x.strip()]
    else:
        eval_stds = [float(config.synergy_config.get('noise', {}).get('eval_std', 0.0))]

    # Evaluate direct vs broadcast-cycle under noise
    tradeoff_results = evaluate_tradeoff(trainer, eval_stds)

    # Persist results next to standard outputs
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'tradeoff_results.json'
    with open(out_file, 'w') as f:
        json.dump(tradeoff_results, f, indent=2)
    print(f"Saved tradeoff results to {out_file}")
    print(json.dumps(tradeoff_results, indent=2))


if __name__ == "__main__":
    main()


