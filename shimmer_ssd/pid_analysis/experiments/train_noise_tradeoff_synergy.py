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
import torch.nn.functional as F

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
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained model checkpoint to load for evaluation-only")
    parser.add_argument("--eval-noise-stds", type=str, default=None, help="Comma-separated list of eval noise stds to sweep")
    # Pass-through flags to enable the syn setup and noise (mirrors train_synergy_glw.py)
    parser.add_argument("--enable-syn-head", action="store_true")
    parser.add_argument("--disable-attr-synergy", action="store_true")
    parser.add_argument("--train-noise-std", type=float, default=None)
    parser.add_argument("--eval-noise-std", type=float, default=None)
    parser.add_argument("--noise-site", type=str, default=None)
    parser.add_argument("--synergy-bins", type=int, default=None)
    parser.add_argument("--syn-head-n-layers", type=int, default=None)
    parser.add_argument("--syn-loss-type", type=str, default=None, choices=["ce", "mse"], help="Loss type for syn head: ce or mse")
    # Capacity and demi-cycle style overrides
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override encoder hidden width")
    parser.add_argument("--decoder-hidden-dim", type=int, default=None, help="Override decoder hidden width")
    parser.add_argument("--demi-cycle-style", type=str, default=None, choices=["encode_decode", "decode_encode", "auto"], help="Demi-cycle style or auto schedule-switch")
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
    if args.syn_loss_type is not None:
        config.synergy_config['syn_loss_type'] = str(args.syn_loss_type)
    if args.hidden_dim is not None:
        config.hidden_dim = int(args.hidden_dim)
    if args.decoder_hidden_dim is not None:
        config.decoder_hidden_dim = int(args.decoder_hidden_dim)
    if args.demi_cycle_style is not None:
        config.synergy_config['demi_cycle_style'] = str(args.demi_cycle_style)
    # Restrict to a single experiment if provided
    if args.experiment_name:
        config.loss_configs = [cfg for cfg in config.loss_configs if cfg['name'] == args.experiment_name]


@torch.no_grad()
def evaluate_tradeoff(trainer: SynergyTrainer, eval_noise_stds: List[float], path_mode: str = 'both') -> Dict[str, Any]:
    model = trainer.model
    device = trainer.device
    noise_cfg = trainer.config.synergy_config.get('noise', {})
    site = noise_cfg.get('site', 'post_fusion_post_tanh')
    n_bins = int(trainer.config.synergy_config.get('n_bins', 8))
    syn_loss_type = str(trainer.config.synergy_config.get('syn_loss_type', 'ce')).lower()

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
        # For regression mode
        sse_direct = 0.0
        sse_cycle = 0.0
        count_direct = 0
        count_cycle = 0

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

            # Prepare logits per requested path(s)
            if 'syn' not in model.gw_decoders:
                raise RuntimeError("'syn' decoder head not available. Enable with --enable-syn-head and retrain.")

            syn_logits_direct = None
            syn_logits_cycle = None
            if path_mode in ('both', 'direct'):
                syn_logits_direct = model.gw_decoders['syn'](gw_state_noisy)

            if path_mode in ('both', 'cycle'):
                decoded_attr = model.gw_decoders['attr'](gw_state_noisy) if 'attr' in model.gw_decoders else None
                decoded_v = model.gw_decoders['v'](gw_state_noisy) if 'v' in model.gw_decoders else None

                re_latents = {}
                if decoded_attr is not None and 'attr' in model.gw_encoders:
                    re_latents['attr'] = model.gw_encoders['attr'](decoded_attr)
                if decoded_v is not None and 'v' in model.gw_encoders:
                    re_latents['v'] = model.gw_encoders['v'](decoded_v)

                if not re_latents:
                    # If no cycle path possible, skip cycle for this batch
                    pass
                else:
                    gw_state_refused = model.fuse(re_latents, selection_scores={})
                    syn_logits_cycle = model.gw_decoders['syn'](gw_state_refused)

            # Build target values from attr target
            if 'attr' not in targets:
                continue
            syn_target_scalar, _ = extract_synergy_features(targets['attr'], 'attr', trainer.config.synergy_config, is_model_output=False, n_synergy_classes=n_bins)

            if syn_loss_type == 'mse':
                # Compute regression SSE and counts for mean at the end
                def sse_from_logits(logits: torch.Tensor):
                    if logits is None:
                        return 0.0, 0
                    pred = torch.sigmoid(logits[..., :1] if logits.shape[-1] > 1 else logits)
                    tgt = syn_target_scalar.view_as(pred)
                    sse = F.mse_loss(pred, tgt, reduction='sum').item()
                    return sse, pred.shape[0]

                sse_d, cnt_d = sse_from_logits(syn_logits_direct)
                sse_c, cnt_c = sse_from_logits(syn_logits_cycle)
                sse_direct += sse_d
                sse_cycle += sse_c
                count_direct += cnt_d
                count_cycle += cnt_c
                total += syn_target_scalar.shape[0]
            else:
                # Classification accuracy
                if syn_target_scalar.shape[-1] == n_bins:
                    target_classes = syn_target_scalar.argmax(dim=-1).view(-1)
                else:
                    target_classes = convert_synergy_targets_to_classes(syn_target_scalar.view(-1), n_bins=n_bins)

                # Handle logits shape [B, n_bins] or [B, n_features*n_bins]
                def logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
                    if logits.shape[-1] == n_bins:
                        return logits.argmax(dim=-1).view(-1)
                    else:
                        first_group = logits[..., :n_bins]
                        return first_group.argmax(dim=-1).view(-1)

                pred_direct = logits_to_pred(syn_logits_direct) if syn_logits_direct is not None else None
                pred_cycle = logits_to_pred(syn_logits_cycle) if syn_logits_cycle is not None else None

                batch_size = target_classes.shape[0]
                total += batch_size
                if pred_direct is not None:
                    correct_direct += (pred_direct == target_classes).sum().item()
                if pred_cycle is not None:
                    correct_cycle += (pred_cycle == target_classes).sum().item()

        # Use None when a path was not computed
        if syn_loss_type == 'mse':
            direct_mse = (sse_direct / count_direct) if count_direct > 0 else None
            cycle_mse = (sse_cycle / count_cycle) if count_cycle > 0 else None
            results[str(std)] = {
                'mode': 'mse',
                'direct_mse': direct_mse,
                'cycle_mse': cycle_mse,
                'direct_accuracy': None,
                'cycle_accuracy': None,
                'num_samples_direct': int(count_direct),
                'num_samples_cycle': int(count_cycle),
            }
        else:
            direct_acc = (correct_direct / total) if (total and path_mode in ('both', 'direct')) else None
            cycle_acc = (correct_cycle / total) if (total and path_mode in ('both', 'cycle')) else None
            results[str(std)] = {
                'mode': 'ce',
                'direct_accuracy': direct_acc,
                'cycle_accuracy': cycle_acc,
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
        # Ensure model has epoch counters for epoch-based scheduling
        try:
            setattr(trainer.model, 'current_epoch', 0)
            setattr(trainer.model, 'total_epochs', int(trainer.config.training.get('epochs', 0)))
        except Exception:
            pass
        trainer.run_all_experiments()
    else:
        # Load checkpoint if provided or discover from results
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            # Try to discover from experiment_results.json for given experiment name
            results_file = Path(config.output_dir) / 'experiment_results.json'
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    # pick last matching experiment or any if only one
                    matches = [r for r in results if r.get('experiment_name') == args.experiment_name]
                    if not matches and results:
                        matches = [results[-1]]
                    if matches:
                        ckpt_path = matches[-1].get('checkpoint_path')
                except Exception:
                    ckpt_path = None
        if ckpt_path:
            try:
                state = torch.load(ckpt_path, map_location=trainer.device)
                if isinstance(state, dict):
                    # common keys
                    if 'state_dict' in state:
                        sd = state['state_dict']
                    elif 'model_state_dict' in state:
                        sd = state['model_state_dict']
                    else:
                        sd = state
                else:
                    sd = state
                try:
                    trainer.model.load_state_dict(sd, strict=False)
                except Exception:
                    # Try stripping 'module.' prefixes
                    from collections import OrderedDict
                    new_sd = OrderedDict()
                    for k, v in sd.items():
                        new_k = k.replace('module.', '') if k.startswith('module.') else k
                        new_sd[new_k] = v
                    trainer.model.load_state_dict(new_sd, strict=False)
                print(f"Loaded checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"Warning: failed to load checkpoint {ckpt_path}: {e}")

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


