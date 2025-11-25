import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, Dict, List

# --- 1. Data Generation ---

def generate_data(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates batch of data:
    Ux, Uy, A, B ~ Bernoulli(0.5)
    X = [Ux, A]
    Y = [Uy, B]
    S = A XOR B
    """
    # Independent bits
    Ux = torch.randint(0, 2, (batch_size, 1), device=device).float()
    Uy = torch.randint(0, 2, (batch_size, 1), device=device).float()
    A = torch.randint(0, 2, (batch_size, 1), device=device).float()
    B = torch.randint(0, 2, (batch_size, 1), device=device).float()

    # Inputs
    X = torch.cat([Ux, A], dim=1) # (B, 2)
    Y = torch.cat([Uy, B], dim=1) # (B, 2)

    # Synergy Target (XOR)
    # A XOR B is equivalent to (A + B) % 2 or abs(A - B) for 0/1
    S = torch.abs(A - B) # (B, 1)

    return X, Y, S

# --- 2. Model Architecture ---

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SynergyBottleneckModel(nn.Module):
    def __init__(self, db: int, de: int = 32):
        super().__init__()
        self.db = db

        # 2.1 Encoders (2-layer MLP)
        self.enc_x = MLP(2, de, de)
        self.enc_y = MLP(2, de, de)

        # 2.2 Bottlenecks (Linear)
        self.bottleneck_x = nn.Linear(de, db, bias=False)
        self.bottleneck_y = nn.Linear(de, db, bias=False)

        # 2.3 Workspace (Shared) - bottlenecked directly
        # z = tanh(Wx h~x + Wy h~y + bz)
        # For minimal model: set dz = db, Wx = Wy = I
        # But to make it more interesting, let's keep separate projections but smaller
        self.proj_x = nn.Linear(db, db)
        self.proj_y = nn.Linear(db, db)

        # 2.4 Decoders (2-layer MLP) - from bottlenecked workspace
        self.dec_x = MLP(db, de, 2)
        self.dec_y = MLP(db, de, 2)

        # 2.5 Synergy Head (2-layer MLP as requested)
        self.synergy_head = MLP(db, de, 1)

    def forward(self, x, y):
        # Encode
        hx = self.enc_x(x)
        hy = self.enc_y(y)

        # Bottleneck
        h_tilde_x = self.bottleneck_x(hx)
        h_tilde_y = self.bottleneck_y(hy)

        # Workspace Integration (bottlenecked)
        z = torch.tanh(self.proj_x(h_tilde_x) + self.proj_y(h_tilde_y))

        # Decode from bottlenecked representation
        x_logits = self.dec_x(z)
        y_logits = self.dec_y(z)

        # Synergy Prediction from bottlenecked representation
        s_logits = self.synergy_head(z)

        return x_logits, y_logits, s_logits

# --- 3. Training Loop & Sweep ---

def evaluate(model: nn.Module, batch_size: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    X, Y, S = generate_data(batch_size, device)
    
    with torch.no_grad():
        x_logits, y_logits, s_logits = model(X, Y)
        
        loss_rec_x = criterion(x_logits, X)
        loss_rec_y = criterion(y_logits, Y)
        loss_syn = criterion(s_logits, S)
        loss = loss_rec_x + loss_rec_y + loss_syn
        
        s_pred = (torch.sigmoid(s_logits) > 0.5).float()
        acc_syn = (s_pred == S).float().mean().item()
        
        x_pred = (torch.sigmoid(x_logits) > 0.5).float()
        y_pred = (torch.sigmoid(y_logits) > 0.5).float()
        acc_rec_x = (x_pred == X).float().mean().item()
        acc_rec_y = (y_pred == Y).float().mean().item()
        acc_rec = (acc_rec_x + acc_rec_y) / 2
        
    return {
        'loss': loss.item(),
        'loss_syn': loss_syn.item(),
        'acc_syn': acc_syn,
        'acc_rec': acc_rec
    }

def train_one_config(db: int, epochs: int = 2000, batch_size: int = 256, lr: float = 0.001, device='cpu'):
    model = SynergyBottleneckModel(db=db).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss functions
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train() # Ensure training mode
        X, Y, S = generate_data(batch_size, device)
        
        optimizer.zero_grad()
        x_logits, y_logits, s_logits = model(X, Y)

        # Reconstruction Loss
        loss_rec_x = criterion(x_logits, X)
        loss_rec_y = criterion(y_logits, Y)
        
        # Synergy Loss
        loss_syn = criterion(s_logits, S)

        # Total Loss
        loss = loss_rec_x + loss_rec_y + loss_syn

        loss.backward()
        optimizer.step()
            
    # Final evaluation on new data (Test Set)
    test_metrics = evaluate(model, 1000, device)
    return test_metrics

def run_sweep(sweep_range: List[int], output_dir: str):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sweep on device: {device}")

    for db in sweep_range:
        print(f"Training with db={db}...")
        # Run multiple seeds per config? Let's do 5 runs.
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            metrics = train_one_config(db=db, device=device)
            metrics['db'] = db
            metrics['seed'] = seed
            results.append(metrics)
            
            print(f"  Seed {seed}: Acc Syn={metrics['acc_syn']:.4f}, Acc Rec={metrics['acc_rec']:.4f}")

    # Save Results
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'synergy_bottleneck_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    
    # Plot 1: Synergy Accuracy vs Bottleneck Dim
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x='db', y='acc_syn', marker='o', label='Synergy Acc')
    plt.axhline(0.5, color='r', linestyle='--', label='Chance')
    plt.axhline(1.0, color='g', linestyle='--', label='Perfect')
    plt.title('Synergy Accuracy vs Bottleneck Dim')
    plt.xlabel('Bottleneck Dimension (db)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)

    # Plot 2: Reconstruction Accuracy vs Bottleneck Dim
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x='db', y='acc_rec', marker='o', color='orange', label='Rec Acc')
    plt.axhline(0.5, color='r', linestyle='--', label='Chance')
    plt.axhline(1.0, color='g', linestyle='--', label='Perfect')
    plt.title('Reconstruction Accuracy vs Bottleneck Dim')
    plt.xlabel('Bottleneck Dimension (db)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'synergy_bottleneck_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    output_dir = "results/synergy_bottleneck"
    os.makedirs(output_dir, exist_ok=True)
    
    sweep_dbs = [1, 2, 3, 4, 5, 6]
    run_sweep(sweep_dbs, output_dir)

