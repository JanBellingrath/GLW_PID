#!/usr/bin/env python3
"""
Sweep training over train-time noise std and evaluate tradeoff curves.

Wrapper around the existing training/eval pipeline that:
- Iterates over a list of train-time noise std values
- Trains a model per value using the standard trainer
- Evaluates direct vs broadcast-cycle accuracy across a given eval-noise range
- Logs per-model plots and a joint plot to W&B and saves JSON/PNGs to disk

Minimal new code; reuses SynergyTrainer and evaluate_tradeoff.
"""

import sys
import json
import argparse
from pathlib import Path
from glob import glob
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import torch

# Ensure pid_analysis is importable
_SCRIPT_DIR = Path(__file__).parent
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from train_synergy_glw import SynergyExperimentConfig, SynergyTrainer
from train_noise_tradeoff_synergy import evaluate_tradeoff

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False


def parse_floats_list(spec: str) -> List[float]:
    spec = spec.strip()
    if ':' in spec:
        # format: start:stop:step
        parts = spec.split(':')
        if len(parts) != 3:
            raise ValueError("Range must be start:stop:step")
        start, stop, step = map(float, parts)
        vals = []
        v = start
        # include stop with tolerance
        while (step > 0 and v <= stop + 1e-12) or (step < 0 and v >= stop - 1e-12):
            vals.append(round(v, 10))
            v += step
        return vals
    # CSV
    return [float(x) for x in spec.split(',') if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep over train noise std and evaluate tradeoff curves")
    parser.add_argument("--config", type=str, required=True, help="Path to synergy configuration JSON")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--train-noise-stds", type=str, required=True, help="CSV or start:stop:step for train noise std sweep")
    parser.add_argument("--eval-noise-stds", type=str, required=True, help="CSV or start:stop:step for eval noise std sweep")
    parser.add_argument("--experiment-name", type=str, default="all_losses", help="Loss config name to use")
    parser.add_argument("--enable-syn-head", action="store_true")
    parser.add_argument("--disable-attr-synergy", action="store_true")
    parser.add_argument("--syn-cycle-ratio", type=float, default=None, help="0..1 weight for syn CE on broadcast-cycle path")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training; load checkpoints per train std and only run eval")
    parser.add_argument("--last-n", type=int, default=None, help="If set, only evaluate the last N train noise stds from the provided list")
    parser.add_argument("--syn-training-style", type=str, default="together", choices=["together", "separate"],
                        help="'together' trains one model per train std with syn_cycle_ratio; 'separate' trains two models (direct-only and cycle-only)")
    parser.add_argument("--variant-filter", type=str, default="both", choices=["both", "direct-only", "cycle-only"],
                        help="When syn-training-style is 'separate', optionally run only one variant")
    parser.add_argument("--wandb-group", type=str, default="train-noise-sweep", help="W&B group for this sweep")
    parser.add_argument("--syn-loss-type", type=str, default=None, choices=["ce", "mse"], help="Loss type for syn head: ce or mse")
    # Broadcast-only modules
    parser.add_argument("--use-broadcast-modules", action="store_true", help="Use separate broadcast-only encoders/decoders for demi-cycle denoising")
    parser.add_argument("--broadcast-hidden-dim", type=int, default=None, help="Hidden dim for broadcast-only modules")
    parser.add_argument("--broadcast-n-layers", type=int, default=None, help="Number of layers for broadcast-only modules")
    parser.add_argument("--eval-use-broadcast-cycle", action="store_true", help="At eval, use broadcast-only modules for the cycle path")
    # Two-stage syn training
    parser.add_argument("--syn-two-stage", action="store_true", help="Enable two-stage training: pretrain system (no syn), then syn-only")
    parser.add_argument("--syn-pretrain-epochs", type=int, default=None, help="Epochs for pretraining system without syn")
    parser.add_argument("--syn-finetune-epochs", type=int, default=None, help="Epochs for syn-only finetune with system frozen")
    parser.add_argument("--cycle-noise-only", action="store_true", help="If set, run cycle-only noise only for all variants")
    return parser.parse_args()


def plot_tradeoff(std_list: List[float], direct: List[float], cycle: List[float], title: str, out_file: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(std_list, direct, label='direct', marker='o')
    plt.plot(std_list, cycle, label='cycle', marker='o')
    plt.xlabel('eval noise std')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_compare_variants(xs: List[float],
                          direct_only_direct: List[float],
                          cycle_only_cycle: List[float],
                          title: str,
                          out_file: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(xs, direct_only_direct, label='direct-only (direct)', marker='o', linestyle='-')
    plt.plot(xs, cycle_only_cycle, label='cycle-only (cycle)', marker='o', linestyle='--')
    plt.xlabel('eval noise std')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def main():
    args = parse_args()
    train_noise_vals = parse_floats_list(args.train_noise_stds)
    eval_noise_vals = parse_floats_list(args.eval_noise_stds)

    config = SynergyExperimentConfig.from_file(args.config)
    base_output = Path(config.output_dir)

    # We will create per-variant eval runs (and a final summary run) instead of one monolithic sweep run

    # If requested, only process the last N train stds
    if args.last_n is not None and args.last_n > 0 and args.last_n < len(train_noise_vals):
        train_noise_vals = train_noise_vals[-args.last_n:]

    # Accumulate for joint plot
    joint_curves = []  # list of dicts: {label, eval_stds, direct, cycle}

    for train_std in train_noise_vals:
        # Fresh config per run
        run_config = SynergyExperimentConfig.from_file(args.config)
        # Overrides
        if args.enable_syn_head:
            run_config.synergy_config['enable_syn_head'] = True
        if args.disable_attr_synergy:
            run_config.synergy_config['attr_includes_synergy'] = False
        if args.syn_cycle_ratio is not None:
            ratio = max(0.0, min(1.0, float(args.syn_cycle_ratio)))
            run_config.synergy_config['syn_cycle_ratio'] = ratio
        if args.syn_loss_type is not None:
            run_config.synergy_config['syn_loss_type'] = str(args.syn_loss_type)
        if args.syn_two_stage:
            run_config.synergy_config.setdefault('syn_two_stage', {})
            run_config.synergy_config['syn_two_stage']['enabled'] = True
        if args.syn_pretrain_epochs is not None:
            run_config.synergy_config.setdefault('syn_two_stage', {})
            run_config.synergy_config['syn_two_stage']['pretrain_epochs'] = int(args.syn_pretrain_epochs)
        if args.syn_finetune_epochs is not None:
            run_config.synergy_config.setdefault('syn_two_stage', {})
            run_config.synergy_config['syn_two_stage']['finetune_epochs'] = int(args.syn_finetune_epochs)
        if args.use_broadcast_modules:
            run_config.synergy_config.setdefault('broadcast', {})
            run_config.synergy_config['broadcast']['use_separate_modules'] = True
        if args.broadcast_hidden_dim is not None:
            run_config.synergy_config.setdefault('broadcast', {})
            run_config.synergy_config['broadcast']['hidden_dim'] = int(args.broadcast_hidden_dim)
        if args.broadcast_n_layers is not None:
            run_config.synergy_config.setdefault('broadcast', {})
            run_config.synergy_config['broadcast']['n_layers'] = int(args.broadcast_n_layers)
        if args.eval_use_broadcast_cycle:
            run_config.synergy_config.setdefault('broadcast', {})
            run_config.synergy_config['broadcast']['eval_use_broadcast'] = True
        # Noise: configure train_std for training branch; eval-only path will ignore
        run_config.synergy_config.setdefault('noise', {})
        run_config.synergy_config['noise']['train_std'] = float(train_std)
        run_config.synergy_config['noise']['eval_std'] = float(train_std)
        # Cycle-only noise flag
        if hasattr(args, 'cycle_noise_only') and args.cycle_noise_only:
            run_config.synergy_config['noise']['cycle_noise_only'] = True
        # Choose training variants
        variants: List[Dict[str, Any]]
        if args.syn_training_style == 'separate':
            base_variants = [
                {'name': 'direct-only', 'ratio': 0.0},
                {'name': 'cycle-only', 'ratio': 1.0},
            ]
            if args.variant_filter == 'direct-only':
                variants = [base_variants[0]]
            elif args.variant_filter == 'cycle-only':
                variants = [base_variants[1]]
            else:
                variants = base_variants
        else:
            variants = [{'name': 'together', 'ratio': run_config.synergy_config.get('syn_cycle_ratio', 0.5)}]

        # Collect variant results for optional cross-model comparison plot
        variant_cache: Dict[str, Dict[str, Any]] = {}

        for var in variants:
            # Variant-specific config
            var_config = SynergyExperimentConfig.from_file(args.config)
            # Apply same general overrides
            if args.enable_syn_head:
                var_config.synergy_config['enable_syn_head'] = True
            if args.disable_attr_synergy:
                var_config.synergy_config['attr_includes_synergy'] = False
            if args.syn_loss_type is not None:
                var_config.synergy_config['syn_loss_type'] = str(args.syn_loss_type)
            if args.syn_two_stage:
                var_config.synergy_config.setdefault('syn_two_stage', {})
                var_config.synergy_config['syn_two_stage']['enabled'] = True
            if args.syn_pretrain_epochs is not None:
                var_config.synergy_config.setdefault('syn_two_stage', {})
                var_config.synergy_config['syn_two_stage']['pretrain_epochs'] = int(args.syn_pretrain_epochs)
            if args.syn_finetune_epochs is not None:
                var_config.synergy_config.setdefault('syn_two_stage', {})
                var_config.synergy_config['syn_two_stage']['finetune_epochs'] = int(args.syn_finetune_epochs)
            if args.use_broadcast_modules:
                var_config.synergy_config.setdefault('broadcast', {})
                var_config.synergy_config['broadcast']['use_separate_modules'] = True
            if args.broadcast_hidden_dim is not None:
                var_config.synergy_config.setdefault('broadcast', {})
                var_config.synergy_config['broadcast']['hidden_dim'] = int(args.broadcast_hidden_dim)
            if args.broadcast_n_layers is not None:
                var_config.synergy_config.setdefault('broadcast', {})
                var_config.synergy_config['broadcast']['n_layers'] = int(args.broadcast_n_layers)
            if args.eval_use_broadcast_cycle:
                var_config.synergy_config.setdefault('broadcast', {})
                var_config.synergy_config['broadcast']['eval_use_broadcast'] = True
            if args.syn_cycle_ratio is not None and args.syn_training_style != 'separate':
                var_config.synergy_config['syn_cycle_ratio'] = max(0.0, min(1.0, float(args.syn_cycle_ratio)))
            # For separate, force ratio
            if args.syn_training_style == 'separate':
                var_config.synergy_config['syn_cycle_ratio'] = float(var['ratio'])
                # Variant-specific demi-cycle style and schedule usage
                if var['name'] == 'direct-only':
                    # constant encode_decode; remove schedule to freeze weights
                    var_config.synergy_config['demi_cycle_style'] = 'encode_decode'
                    if 'schedule' in var_config.synergy_config:
                        del var_config.synergy_config['schedule']
                else:
                    # cycle-only uses auto style with schedule (decode_encode after t0)
                    var_config.synergy_config['demi_cycle_style'] = 'auto'
            # Noise: training std for this loop value
            var_config.synergy_config.setdefault('noise', {})
            var_config.synergy_config['noise']['train_std'] = float(train_std)
            var_config.synergy_config['noise']['eval_std'] = float(train_std)
            if hasattr(args, 'cycle_noise_only') and args.cycle_noise_only:
                var_config.synergy_config['noise']['cycle_noise_only'] = True
            # Output subdir includes variant
            run_output = base_output / f"sweep_trainstd_{train_std}" / f"style_{var['name']}"
            var_config.output_dir = str(run_output)
            # Restrict experiments and rename
            var_config.loss_configs = [cfg for cfg in var_config.loss_configs if cfg['name'] == args.experiment_name]
            if not var_config.loss_configs:
                print(f"Warning: loss config {args.experiment_name} not found; skipping {var['name']}")
                continue
            var_config.loss_configs[0] = dict(var_config.loss_configs[0])
            var_config.loss_configs[0]['name'] = f"{args.experiment_name}_trainstd_{train_std}_{var['name']}"

            # Trainer
            device = torch.device(args.device) if args.device else None
            trainer = SynergyTrainer(var_config, device=device)
            trainer.setup_model()
            trainer.setup_data()

            # Train unless evaluate-only
            if not args.evaluate_only:
                trainer.run_all_experiments()
            else:
                # Try to load checkpoint from run_output folders
                ckpt = None
                results_file = run_output / 'experiment_results.json'
                if results_file.exists():
                    try:
                        with open(results_file, 'r') as f:
                            arr = json.load(f)
                        if isinstance(arr, list) and arr:
                            ckpt = arr[-1].get('checkpoint_path')
                    except Exception:
                        ckpt = None
                if ckpt is None:
                    candidates = sorted(glob(str(run_output / '**' / '*.pt')), reverse=True)
                    if candidates:
                        ckpt = candidates[0]
                if ckpt:
                    try:
                        state = torch.load(ckpt, map_location=trainer.device)
                        sd = state.get('state_dict', state.get('model_state_dict', state)) if isinstance(state, dict) else state
                        try:
                            trainer.model.load_state_dict(sd, strict=False)
                        except Exception:
                            from collections import OrderedDict
                            new_sd = OrderedDict()
                            for k, v in sd.items():
                                new_sd[k.replace('module.', '') if k.startswith('module.') else k] = v
                            trainer.model.load_state_dict(new_sd, strict=False)
                        print(f"Loaded checkpoint for train_std={train_std} variant={var['name']}: {ckpt}")
                    except Exception as e:
                        print(f"Warning: failed to load checkpoint for train_std={train_std} variant={var['name']}: {e}")

            # Evaluate tradeoff
            # Evaluate matching path(s) per training style:
            # - separate/direct-only: compute only 'direct'
            # - separate/cycle-only: compute only 'cycle'
            # - together: compute both
            if args.syn_training_style == 'separate' and var['name'] == 'direct-only':
                results = evaluate_tradeoff(trainer, eval_noise_vals, path_mode='direct')
            elif args.syn_training_style == 'separate' and var['name'] == 'cycle-only':
                results = evaluate_tradeoff(trainer, eval_noise_vals, path_mode='cycle')
            else:
                results = evaluate_tradeoff(trainer, eval_noise_vals, path_mode='both')
            # Save JSON
            out_json = run_output / 'tradeoff_results.json'
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved tradeoff results to {out_json}")

            # Prepare per-model plot
            xs = [float(k) for k in results.keys()]
            xs.sort()
            # Choose metric depending on mode
            first_key = f"{xs[0]}" if xs else None
            mode = results[first_key].get('mode', 'ce') if first_key else 'ce'
            if mode == 'mse':
                direct = [results[f"{x}"]['direct_mse'] if results[f"{x}"]['direct_mse'] is not None else None for x in xs]
                cycle = [results[f"{x}"]['cycle_mse'] if results[f"{x}"]['cycle_mse'] is not None else None for x in xs]
                # Replace None with NaN for plotting gaps
                direct = [d if d is not None else float('nan') for d in direct]
                cycle = [c if c is not None else float('nan') for c in cycle]
                title = f"eval curves (MSE) train_std={train_std}, {var['name']}"
                out_png = run_output / 'tradeoff_plot.png'
                # Reuse plotting but y-label will still say accuracy; create a local variant
                plt.figure(figsize=(6, 4))
                plt.plot(xs, direct, label='direct MSE', marker='o')
                plt.plot(xs, cycle, label='cycle MSE', marker='o')
                plt.xlabel('eval noise std')
                plt.ylabel('MSE')
                plt.title(title)
                plt.grid(True, alpha=0.3)
                plt.legend()
                out_png.parent.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(out_png)
                plt.close()
            else:
                # Use 0.0 for missing path for plotting consistency
                direct = [results[f"{x}"]['direct_accuracy'] if results[f"{x}"]['direct_accuracy'] is not None else 0.0 for x in xs]
                cycle = [results[f"{x}"]['cycle_accuracy'] if results[f"{x}"]['cycle_accuracy'] is not None else 0.0 for x in xs]
                title = f"eval curves (train_std={train_std}, {var['name']})"
                out_png = run_output / 'tradeoff_plot.png'
                plot_tradeoff(xs, direct, cycle, title, out_png)
            print(f"Saved tradeoff plot to {out_png}")

            # Save to cache for cross-model comparison plot
            variant_cache[var['name']] = {
                'xs': xs,
                'direct': direct,
                'cycle': cycle,
            }

            # Log to a dedicated W&B run for this variant (if enabled)
            if HAS_WANDB and config.experiment.get('log_to_wandb', True):
                try:
                    run_name = f"{args.experiment_name}_trainstd_{train_std}_{var['name']}_eval"
                    wb_cfg = {
                        'train_std': float(train_std),
                        'eval_noise_stds': xs,
                        'syn_training_style': args.syn_training_style,
                        'variant': var['name'],
                    }
                    r = wandb.init(project=config.wandb_project, group=args.wandb_group, name=run_name, config=wb_cfg, reinit=True)
                    wandb.log({
                        'eval_plot': wandb.Image(str(out_png)),
                        'eval_data': {
                            'eval_noise_std': xs,
                            'direct': direct,
                            'cycle': cycle,
                        }
                    })
                    wandb.finish()
                except Exception:
                    pass

            joint_curves.append({
                'label': f"train_std={train_std} {var['name']}",
                'xs': xs,
                'direct': direct,
                'cycle': cycle,
                'mode': mode,
            })

        # If separate models were evaluated, produce a single comparison plot per train_std
        if args.syn_training_style == 'separate' and 'direct-only' in variant_cache and 'cycle-only' in variant_cache:
            xs0 = variant_cache['direct-only']['xs']
            xs1 = variant_cache['cycle-only']['xs']
            # Assume same xs; if different, fall back to xs0
            xs_use = xs0
            direct_line = variant_cache['direct-only']['direct']
            cycle_line = variant_cache['cycle-only']['cycle']
            compare_png = (base_output / f"sweep_trainstd_{train_std}") / 'compare_models_plot.png'
            plot_compare_variants(xs_use, direct_line, cycle_line,
                                  title=f"eval curves (train_std={train_std}, direct-only vs cycle-only)",
                                  out_file=compare_png)
            print(f"Saved compare plot to {compare_png}")
            if HAS_WANDB and config.experiment.get('log_to_wandb', True):
                try:
                    run_name = f"compare_trainstd_{train_std}"
                    r = wandb.init(project=config.wandb_project, group=args.wandb_group, name=run_name, reinit=True)
                    wandb.log({'compare_plot': wandb.Image(str(compare_png))})
                    wandb.finish()
                except Exception:
                    pass

    # Joint plot
    if joint_curves:
        plt.figure(figsize=(7, 5))
        cmap = plt.get_cmap('tab10')
        # Separate classification and regression curves for proper axis labels
        any_mse = any(curve.get('mode', 'ce') == 'mse' for curve in joint_curves)
        if any_mse:
            # Plot MSE curves
            for idx, curve in enumerate([c for c in joint_curves if c.get('mode', 'ce') == 'mse']):
                color = cmap(idx % 10)
                plt.plot(curve['xs'], curve['direct'], label=f"direct MSE {curve['label']}", color=color, linestyle='-')
                plt.plot(curve['xs'], curve['cycle'], label=f"cycle MSE {curve['label']}", color=color, linestyle='--')
            plt.xlabel('eval noise std')
            plt.ylabel('MSE')
            plt.title('joint eval curves (MSE)')
            plt.grid(True, alpha=0.3)
            plt.legend(ncol=2, fontsize=8)
        else:
            for idx, curve in enumerate(joint_curves):
                color = cmap(idx % 10)
                plt.plot(curve['xs'], curve['direct'], label=f"direct {curve['label']}", color=color, linestyle='-')
                plt.plot(curve['xs'], curve['cycle'], label=f"cycle {curve['label']}", color=color, linestyle='--')
            plt.xlabel('eval noise std')
            plt.ylabel('accuracy')
            plt.ylim(0.0, 1.0)
            plt.title('joint eval curves')
            plt.grid(True, alpha=0.3)
            plt.legend(ncol=2, fontsize=8)
        base_output.mkdir(parents=True, exist_ok=True)
        joint_png = base_output 
        if any_mse:
            joint_png = joint_png / 'sweep_joint_tradeoff_plot_mse.png'
        else:
            joint_png = joint_png / 'sweep_joint_tradeoff_plot.png'
        plt.tight_layout()
        plt.savefig(joint_png)
        plt.close()
        print(f"Saved joint tradeoff plot to {joint_png}")
        if HAS_WANDB and config.experiment.get('log_to_wandb', True):
            try:
                run_name = f"sweep_summary_{args.experiment_name}_{args.syn_training_style}"
                r = wandb.init(project=config.wandb_project, group=args.wandb_group, name=run_name, reinit=True)
                wandb.log({"eval_plot/joint": wandb.Image(str(joint_png))})
                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()


