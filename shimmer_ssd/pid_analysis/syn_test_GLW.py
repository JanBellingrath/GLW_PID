#!/usr/bin/env python3
# two_branch_add_mod_r.py
# Tiny experiment: can a hard-wired tanh bottleneck learn (a+b) mod r ?

import argparse, torch, torch.nn as nn, torch.nn.functional as F, random, numpy as np
import os, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional

# ------------- command-line -----------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--mod',     type=int,   default=8,   help='modulus r')
p.add_argument('--epochs',  type=int,   default=10000, help='# training epochs per latent size')
p.add_argument('--batch',   type=int,   default=256, help='mini-batch size')
p.add_argument('--opwidth', type=int,   default=64,  help='hidden width in each operand branch')
p.add_argument('--decwidth',type=int,   default=64,  help='hidden width in decoder')
p.add_argument('--latent',  type=int,   default=12,  help='latent bottleneck dimensionality (per branch output and decoder input)')
p.add_argument('--latent_list', type=str, default='', help='comma-separated list of latent sizes to sweep; overrides --latent if provided')
p.add_argument('--lr',      type=float, default=3e-3)
p.add_argument('--loss_scale', type=float, default=1.0, help='scaling factor for classification loss (higher means stronger modulus addition)')
p.add_argument('--loss_scale_list', type=str, default='', help='comma-separated loss_scale values to sweep; overrides logspace sweep')
p.add_argument('--loss_scale_min', type=float, default=0.1, help='min loss scale (logspace) when sweeping')
p.add_argument('--loss_scale_max', type=float, default=10.0, help='max loss scale (logspace) when sweeping')
p.add_argument('--loss_scale_points', type=int, default=21, help='# points in loss scale sweep (logspace)')
p.add_argument('--acc_threshold', type=float, default=0.95, help='classification accuracy threshold for success')
p.add_argument('--rec_threshold', type=float, default=0.02, help='reconstruction MSE threshold for success (for both A and B)')
p.add_argument('--results_dir', type=str, default='results', help='directory to save plots and CSV logs')
p.add_argument('--eval_every', type=int, default=0, help='evaluate during training every N epochs (0 disables mid-epoch eval)')
p.add_argument('--wandb', action='store_true', help='enable Weights & Biases logging')
p.add_argument('--wandb_project', type=str, default='glw_synergy', help='wandb project name')
p.add_argument('--wandb_entity', type=str, default='cerco_neuro_ai', help='wandb entity (team/user)')
p.add_argument('--wandb_mode', type=str, default='online', choices=['online','offline','disabled'], help='wandb mode')
p.add_argument('--device',  type=str,   default='cuda' if torch.cuda.is_available() else 'cpu')
p.add_argument('--seed',    type=int,   default=2025)
p.add_argument('--gpu_mem_frac', type=float, default=0.10, help='max fraction of selected CUDA device memory to allow (via PyTorch caching allocator)')
args = p.parse_args()

# reproducibility
torch.manual_seed(args.seed);  np.random.seed(args.seed);  random.seed(args.seed)
device = args.device
# Optionally cap CUDA memory usage to a fraction of the selected device
if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
    if ':' in device:
        device_index = int(device.split(':')[1])
    else:
        device_index = 0
    torch.cuda.set_device(device_index)
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        try:
            torch.cuda.set_per_process_memory_fraction(max(0.0, min(1.0, args.gpu_mem_frac)), device=device_index)
        except Exception as _e:
            # Silently continue if not supported on this build
            pass
r      = args.mod

# ------------------ model pieces ------------------------------------------
class OperandBranch(nn.Module):
    def __init__(self, width: int, latent_dim: int, input_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, latent_dim)      # output dim = latent bottleneck
        )
    def forward(self, x):            # x shape (B,input_dim)
        return self.net(x)

class Model(nn.Module):
    def __init__(self, op_width: int, dec_width: int, latent_dim: int, modulus_r: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.modulus_r = modulus_r
        self.branchA = OperandBranch(op_width, latent_dim)
        self.branchB = OperandBranch(op_width, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dec_width),
            nn.ReLU(),
            nn.Linear(dec_width, modulus_r + 22)  # r for classification + 11 for noise_a + 11 for noise_b
        )
        # fixed merge weights: no parameters needed

    def forward(self, x):            # x shape (B,24)  -> split into two 12-D views
        xa, xb = x[:, :12], x[:, 12:]
        ha = self.branchA(xa)        # (B,latent_dim)
        hb = self.branchB(xb)        # (B,latent_dim)
        h  = 0.5*(ha + hb)
        h  = torch.tanh(h)           # (B,latent_dim)
        return self.decoder(h)       # (B, r+22)

def build_model(latent_dim: int) -> Model:
    model = Model(args.opwidth, args.decwidth, latent_dim, r).to(device)
    return model

# ------------------ synthetic data sampler --------------------------------
def sample_batch(n):
    # operands
    a = torch.randint(0, r, (n,1), device=device)
    b = torch.randint(0, r, (n,1), device=device)
    # random noise
    noise_a = torch.rand(n,11, device=device)          # 11 independent noise dimensions ~ uniform[0,1]
    # Make branch noises identical so symmetric merge preserves reconstructable info
    noise_b = noise_a.clone()
    # build branch inputs
    xa = torch.cat(((a.float()/(r-1)), noise_a), dim=1)  # (n,12)
    xb = torch.cat(((b.float()/(r-1)), noise_b), dim=1)  # (n,12)
    x  = torch.cat((xa, xb), dim=1)                      # (n,24)
    y_class = ((a + b) % r).squeeze(1)                   # integer class labels (n,)
    # return both noise targets for reconstruction
    return x, y_class, noise_a, noise_b

def accuracy(logits, y_indices):
    return (logits.argmax(1)==y_indices).float().mean().item()

def train_and_evaluate(latent_dim: int, loss_scale_value: float):
    # fresh model and optimizer per latent size
    model = build_model(latent_dim)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    # optional wandb setup (one run per combo)
    global wandb
    use_wandb = False
    if args.wandb and args.wandb_mode != 'disabled':
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                config={
                    'mod': r,
                    'epochs': args.epochs,
                    'batch': args.batch,
                    'opwidth': args.opwidth,
                    'decwidth': args.decwidth,
                    'latent': latent_dim,
                    'loss_scale': loss_scale_value,
                    'lr': args.lr,
                    'acc_threshold': args.acc_threshold,
                    'rec_threshold': args.rec_threshold,
                },
                tags=[f'latent:{latent_dim}', f'ls:{loss_scale_value}'],
                name=f'latent{latent_dim}_ls{loss_scale_value}',
                # finish previous run if any, per wandb deprecation note
                settings=wandb.Settings(start_method='thread', _service_wait=300),
            )
            use_wandb = True
            # Print run URL so it is always visible in terminal
            try:
                print(f"wandb run URL: {wandb.run.url}")
            except Exception:
                pass
        except Exception as e:
            use_wandb = False
            print(f"[wandb] disabled: {e}")

    for epoch in range(1, args.epochs+1):
        x, y_class, y_recon_a, y_recon_b = sample_batch(args.batch)
        output  = model(x)
        # Split output into classification and reconstruction parts
        logits_class = output[:, :r]               # (B, r)
        recon_a_pred = output[:, r:r+11]          # (B, 11)
        recon_b_pred = output[:, r+11:r+22]       # (B, 11)

        # Combined loss: scaled classification + reconstruction for both A and B
        loss_class = F.cross_entropy(logits_class, y_class)
        loss_recon_a = F.mse_loss(recon_a_pred, y_recon_a)
        loss_recon_b = F.mse_loss(recon_b_pred, y_recon_b)
        loss_recon = loss_recon_a + loss_recon_b
        loss = loss_scale_value * loss_class + loss_recon

        opt.zero_grad(); loss.backward(); opt.step()

        if epoch==1 or epoch%10==0:
            acc = accuracy(logits_class, y_class)
            print(
                f'[latent={latent_dim:3d} ls={loss_scale_value:.4g}] Epoch {epoch:4d} | loss {loss.item():.4f} '
                f'(cls: {loss_class.item():.4f}, recA: {loss_recon_a.item():.4f}, recB: {loss_recon_b.item():.4f}) '
                f'| acc {acc*100:5.1f}%'
            )

        # optional mid-epoch eval and logging
        if args.eval_every > 0 and (epoch % args.eval_every == 0):
            with torch.no_grad():
                x_val, y_val_idx, y_val_a, y_val_b = sample_batch(2048)
                val_out = model(x_val)
                val_logits = val_out[:, :r]
                val_rec_a = val_out[:, r:r+11]
                val_rec_b = val_out[:, r+11:r+22]
                val_acc = accuracy(val_logits, y_val_idx)
                val_cls_loss = F.cross_entropy(val_logits, y_val_idx).item()
                val_rec_a_mse = F.mse_loss(val_rec_a, y_val_a).item()
                val_rec_b_mse = F.mse_loss(val_rec_b, y_val_b).item()
                val_total = loss_scale_value * val_cls_loss + (val_rec_a_mse + val_rec_b_mse)
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'latent': latent_dim,
                    'loss_scale': loss_scale_value,
                    'train/total': float(loss.item()),
                    'train/cls': float(loss_class.item()),
                    'train/recA': float(loss_recon_a.item()),
                    'train/recB': float(loss_recon_b.item()),
                    'train/acc': float(accuracy(logits_class, y_class)),
                    'eval/total': float(val_total),
                    'eval/cls': float(val_cls_loss),
                    'eval/recA': float(val_rec_a_mse),
                    'eval/recB': float(val_rec_b_mse),
                    'eval/acc': float(val_acc),
                })

    # ------------------ final evaluation ----------------------------------
    with torch.no_grad():
        x_test, y_class_test, y_recon_a_test, y_recon_b_test = sample_batch(2048)
        test_output = model(x_test)
        test_logits_class = test_output[:, :r]
        test_recon_a_pred = test_output[:, r:r+11]
        test_recon_b_pred = test_output[:, r+11:r+22]

        test_loss_class = F.cross_entropy(test_logits_class, y_class_test).item()
        test_loss_recon_a = F.mse_loss(test_recon_a_pred, y_recon_a_test).item()
        test_loss_recon_b = F.mse_loss(test_recon_b_pred, y_recon_b_test).item()
        test_acc = accuracy(test_logits_class, y_class_test)
        test_total_loss = loss_scale_value * test_loss_class + (test_loss_recon_a + test_loss_recon_b)

    print(
        f'\n[latent={latent_dim} ls={loss_scale_value:.4g}] Test on 2k samples | total {test_total_loss:.4f} '
        f'| cls_acc {test_acc*100:.2f}% (cls_loss {test_loss_class:.4f}) '
        f'| recA_mse {test_loss_recon_a:.6f} | recB_mse {test_loss_recon_b:.6f}\n'
    )
    if use_wandb:
        wandb.log({
            'latent': latent_dim,
            'loss_scale': loss_scale_value,
            'test/total': float(test_total_loss),
            'test/cls': float(test_loss_class),
            'test/recA': float(test_loss_recon_a),
            'test/recB': float(test_loss_recon_b),
            'test/acc': float(test_acc),
        })
        try:
            wandb.finish()
        except Exception:
            pass

    return {
        'latent': latent_dim,
        'loss_scale': loss_scale_value,
        'test_acc': test_acc,
        'test_cls_loss': test_loss_class,
        'test_recA_mse': test_loss_recon_a,
        'test_recB_mse': test_loss_recon_b,
        'test_total_loss': test_total_loss,
    }


def parse_latent_list() -> list:
    if args.latent_list.strip() == '':
        return [args.latent]
    sizes = []
    for token in args.latent_list.split(','):
        token = token.strip()
        if token:
            sizes.append(int(token))
    return sizes


def parse_loss_scale_list() -> list:
    if args.loss_scale_list.strip() != '':
        vals = []
        for token in args.loss_scale_list.split(','):
            token = token.strip()
            if token:
                vals.append(float(token))
        return vals
    # default: logspace sweep
    return list(np.logspace(np.log10(args.loss_scale_min), np.log10(args.loss_scale_max), args.loss_scale_points))


os.makedirs(args.results_dir, exist_ok=True)
all_results = []
latent_sizes = parse_latent_list()
loss_scales = parse_loss_scale_list()

for ld in latent_sizes:
    ld_results = []
    for ls in loss_scales:
        # re-seed per combination for deterministic comparisons
        combo_seed = args.seed + int(ld * 1000 + ls * 100)
        torch.manual_seed(combo_seed);  np.random.seed(combo_seed);  random.seed(combo_seed)
        res = train_and_evaluate(ld, ls)
        ld_results.append(res)
        all_results.append(res)

    # Save CSV per latent
    csv_path = os.path.join(args.results_dir, f'tradeoff_latent{ld}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['latent','loss_scale','test_acc','test_cls_loss','test_recA_mse','test_recB_mse','test_total_loss'])
        writer.writeheader()
        for row in ld_results:
            writer.writerow(row)

    # Plot per latent
    ls_vals = [row['loss_scale'] for row in ld_results]
    acc_vals = [row['test_acc'] for row in ld_results]
    recA_vals = [row['test_recA_mse'] for row in ld_results]
    recB_vals = [row['test_recB_mse'] for row in ld_results]

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
    axes[0].plot(ls_vals, acc_vals, marker='o', label=f'latent={ld}')
    axes[0].axhline(args.acc_threshold, color='gray', linestyle='--', linewidth=1)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ls_vals, recA_vals, marker='o', color='tab:orange', label='recA MSE')
    axes[1].axhline(args.rec_threshold, color='gray', linestyle='--', linewidth=1)
    axes[1].set_ylabel('recA MSE')
    axes[1].set_xscale('log')
    axes[1].invert_yaxis() if max(recA_vals) < 0.5 else None
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ls_vals, recB_vals, marker='o', color='tab:green', label='recB MSE')
    axes[2].axhline(args.rec_threshold, color='gray', linestyle='--', linewidth=1)
    axes[2].set_ylabel('recB MSE')
    axes[2].set_xlabel('loss_scale (log)')
    axes[2].set_xscale('log')
    axes[2].invert_yaxis() if max(recB_vals) < 0.5 else None
    axes[2].grid(True, alpha=0.3)

    # Panel 4: boolean success mask (both tasks succeed)
    success_both = [
        1.0 if (acc_vals[i] >= args.acc_threshold and recA_vals[i] <= args.rec_threshold and recB_vals[i] <= args.rec_threshold)
        else 0.0 for i in range(len(ls_vals))
    ]
    axes[3].plot(ls_vals, success_both, marker='s', linestyle='-', color='tab:purple')
    axes[3].set_ylabel('both_succeed')
    axes[3].set_yticks([0,1])
    axes[3].set_ylim(-0.05,1.05)
    axes[3].set_xscale('log')
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f'Loss-scale tradeoff (latent={ld})')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_path = os.path.join(args.results_dir, f'tradeoff_latent{ld}.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

# Text summary of regions where tasks are solved
print('\nSweep summary: solving regions by latent size')
for ld in latent_sizes:
    subset = [r for r in all_results if r['latent']==ld]
    subset.sort(key=lambda d: d['loss_scale'])
    both = [r for r in subset if (r['test_acc']>=args.acc_threshold and r['test_recA_mse']<=args.rec_threshold and r['test_recB_mse']<=args.rec_threshold)]
    cls_only = [r for r in subset if (r['test_acc']>=args.acc_threshold and not (r['test_recA_mse']<=args.rec_threshold and r['test_recB_mse']<=args.rec_threshold))]
    rec_only = [r for r in subset if ((r['test_recA_mse']<=args.rec_threshold and r['test_recB_mse']<=args.rec_threshold) and r['test_acc']<args.acc_threshold)]

    def span(vals):
        if not vals:
            return '(none)'
        lo, hi = min(v['loss_scale'] for v in vals), max(v['loss_scale'] for v in vals)
        return f'[{lo:.4g}, {hi:.4g}]'

    print(
        f'latent={ld} | both: {span(both)} | cls-only: {span(cls_only)} | rec-only: {span(rec_only)}'
    )

# Extra: overall heatmap-like CSV showing whether both succeed for (latent, loss_scale)
grid_csv = os.path.join(args.results_dir, 'both_succeed_grid.csv')
with open(grid_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['latent','loss_scale','both_succeed','acc','recA_mse','recB_mse'])
    for r in all_results:
        both_ok = int(r['test_acc']>=args.acc_threshold and r['test_recA_mse']<=args.rec_threshold and r['test_recB_mse']<=args.rec_threshold)
        writer.writerow([r['latent'], r['loss_scale'], both_ok, r['test_acc'], r['test_recA_mse'], r['test_recB_mse']])
