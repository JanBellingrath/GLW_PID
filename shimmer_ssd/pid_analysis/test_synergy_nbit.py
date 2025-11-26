import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from typing import Tuple, Dict, List

# --- 1. Data Generation (N-bit Parity) ---

def generate_data(batch_size: int, n_bits: int, fixed_xor: bool, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates batch of data:
    Ux, Uy ~ Bernoulli(0.5) of shape (B, n_bits)
    A, B ~ Bernoulli(0.5) of shape (B, n_bits)
    
    X = [Ux, A] (Dim: 2*n_bits)
    Y = [Uy, B] (Dim: 2*n_bits)
    
    S = Parity(A) XOR Parity(B)
    
    If fixed_xor=True:
        Only the FIRST bit of A and B matters for synergy.
        S = A[:,0] XOR B[:,0]
        The other n_bits-1 bits of A and B are just random noise (distractors).
    Else:
        S = (sum(A) + sum(B)) % 2
    """
    # Unique bits
    Ux = torch.randint(0, 2, (batch_size, n_bits), device=device).float()
    Uy = torch.randint(0, 2, (batch_size, n_bits), device=device).float()
    
    # Synergistic bits candidates
    A = torch.randint(0, 2, (batch_size, n_bits), device=device).float()
    B = torch.randint(0, 2, (batch_size, n_bits), device=device).float()

    # Inputs
    X = torch.cat([Ux, A], dim=1) # (B, 2*n_bits)
    Y = torch.cat([Uy, B], dim=1) # (B, 2*n_bits)

    if fixed_xor:
        # Only first bit matters
        S = (A[:, 0:1] + B[:, 0:1]) % 2
    else:
        # All bits matter (N-bit parity)
        A_sum = A.sum(dim=1, keepdim=True)
        B_sum = B.sum(dim=1, keepdim=True)
        S = (A_sum + B_sum) % 2 
    
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
    def __init__(self, db: int, n_bits: int, de: int = 64):
        super().__init__()
        self.db = db
        self.n_bits = n_bits
        input_dim = 2 * n_bits # Ux + A
        
        # 2.1 Encoders (2-layer MLP)
        self.enc_x = MLP(input_dim, de, de)
        self.enc_y = MLP(input_dim, de, de)

        # 2.2 Bottlenecks (Linear)
        self.bottleneck_x = nn.Linear(de, db, bias=False)
        self.bottleneck_y = nn.Linear(de, db, bias=False)

        # 2.3 Workspace (Shared)
        self.proj_x = nn.Linear(db, db)
        self.proj_y = nn.Linear(db, db)
        
        # 2.4 Decoders (2-layer MLP)
        self.dec_x = MLP(db, de, input_dim)
        self.dec_y = MLP(db, de, input_dim)

        # 2.5 Synergy Head (2-layer MLP)
        self.synergy_head = MLP(db, de, 1)

    def forward(self, x, y):
        # Encode
        hx = self.enc_x(x)
        hy = self.enc_y(y)

        # Bottleneck
        h_tilde_x = self.bottleneck_x(hx)
        h_tilde_y = self.bottleneck_y(hy)

        # Workspace Integration
        z = torch.tanh(self.proj_x(h_tilde_x) + self.proj_y(h_tilde_y))

        # Decode
        x_logits = self.dec_x(z)
        y_logits = self.dec_y(z)

        # Synergy
        s_logits = self.synergy_head(z)

        return x_logits, y_logits, s_logits

# --- 3. Training & Evaluation ---

def evaluate(model: nn.Module, batch_size: int, n_bits: int, fixed_xor: bool, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    X, Y, S = generate_data(batch_size, n_bits, fixed_xor, device)
    
    with torch.no_grad():
        x_logits, y_logits, s_logits = model(X, Y)
        
        loss_rec_x = criterion(x_logits, X)
        loss_rec_y = criterion(y_logits, Y)
        loss_syn = criterion(s_logits, S)
        loss = loss_rec_x + loss_rec_y + loss_syn
        
        # Synergy Acc
        s_pred = (torch.sigmoid(s_logits) > 0.5).float()
        acc_syn = (s_pred == S).float().mean().item()
        
        # Split reconstruction targets
        # X = [Ux, A], Y = [Uy, B]
        # Each has n_bits
        x_pred = (torch.sigmoid(x_logits) > 0.5).float()
        y_pred = (torch.sigmoid(y_logits) > 0.5).float()
        
        # Reconstruction of Unique Features (Ux, Uy) - First n_bits
        rec_Ux = (x_pred[:, :n_bits] == X[:, :n_bits]).float().mean().item()
        rec_Uy = (y_pred[:, :n_bits] == Y[:, :n_bits]).float().mean().item()
        acc_rec_unique = (rec_Ux + rec_Uy) / 2
        
        # Reconstruction of Synergy Features (A, B) - Last n_bits
        rec_A = (x_pred[:, n_bits:] == X[:, n_bits:]).float().mean().item()
        rec_B = (y_pred[:, n_bits:] == Y[:, n_bits:]).float().mean().item()
        acc_rec_syn_features = (rec_A + rec_B) / 2
        
    return {
        'loss': loss.item(),
        'acc_syn': acc_syn,
        'acc_rec_unique': acc_rec_unique,
        'acc_rec_syn_features': acc_rec_syn_features
    }

def train_one_config(db: int, n_bits: int, fixed_xor: bool, epochs: int = 3000, batch_size: int = 256, lr: float = 0.001, device='cpu'):
    model = SynergyBottleneckModel(db=db, n_bits=n_bits).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        X, Y, S = generate_data(batch_size, n_bits, fixed_xor, device)
        
        optimizer.zero_grad()
        x_logits, y_logits, s_logits = model(X, Y)

        loss = criterion(x_logits, X) + criterion(y_logits, Y) + criterion(s_logits, S)

        loss.backward()
        optimizer.step()
            
    return evaluate(model, 1000, n_bits, fixed_xor, device)

def run_sweep(db_range: List[int], n_bits_range: List[int], fixed_xor: bool, output_dir: str):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sweep on device: {device}")
    print(f"Fixed XOR Task: {fixed_xor}")

    for n_bits in n_bits_range:
        for db in db_range:
            print(f"Training: n_bits={n_bits}, db={db}")
            for seed in range(3): 
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                metrics = train_one_config(db=db, n_bits=n_bits, fixed_xor=fixed_xor, device=device)
                metrics['db'] = db
                metrics['n_bits'] = n_bits
                metrics['seed'] = seed
                results.append(metrics)
                
                print(f"  Seed {seed}: Acc Syn={metrics['acc_syn']:.3f}, Acc Unique={metrics['acc_rec_unique']:.3f}, Acc SynFeat={metrics['acc_rec_syn_features']:.3f}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'synergy_nbit_results.csv')
    df.to_csv(csv_path, index=False)
    
    return df

def plot_results(df: pd.DataFrame, output_dir: str):
    # Plot 1: Acc vs N-bits
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Synergy Accuracy
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x='n_bits', y='acc_syn', hue='db', palette='viridis', marker='o')
    plt.title('Synergy Accuracy vs N-bits (by db)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    # Subplot 2: Unique Feature Reconstruction Accuracy
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x='n_bits', y='acc_rec_unique', hue='db', palette='viridis', marker='o')
    plt.title('Unique Feature Rec. Accuracy vs N-bits (by db)')
    plt.ylabel('Accuracy (Unique Features)')
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_acc_vs_nbits.png'))
    
    # Plot 2: 3D Plot
    fig = plt.figure(figsize=(12, 6))
    
    # 3D Surface for Synergy
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Pivot for surface plot (average over seeds)
    df_avg = df.groupby(['n_bits', 'db']).mean().reset_index()
    
    # We need a grid
    N_unique = sorted(df_avg['n_bits'].unique())
    DB_unique = sorted(df_avg['db'].unique())
    X, Y = np.meshgrid(N_unique, DB_unique)
    Z = np.zeros_like(X, dtype=float)
    
    for i, db_val in enumerate(DB_unique):
        for j, n_val in enumerate(N_unique):
            val = df_avg[(df_avg['n_bits'] == n_val) & (df_avg['db'] == db_val)]['acc_syn'].values
            if len(val) > 0:
                Z[i, j] = val[0]
                
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax1.set_title('Synergy Accuracy')
    ax1.set_xlabel('N bits')
    ax1.set_ylabel('Bottleneck (db)')
    ax1.set_zlabel('Accuracy')
    
    # 3D Surface for Reconstruction (UNIQUE FEATURES ONLY)
    ax2 = fig.add_subplot(122, projection='3d')
    Z_rec = np.zeros_like(X, dtype=float)
    
    for i, db_val in enumerate(DB_unique):
        for j, n_val in enumerate(N_unique):
            val = df_avg[(df_avg['n_bits'] == n_val) & (df_avg['db'] == db_val)]['acc_rec_unique'].values
            if len(val) > 0:
                Z_rec[i, j] = val[0]
                
    surf2 = ax2.plot_surface(X, Y, Z_rec, cmap='magma', edgecolor='none')
    ax2.set_title('Unique Rec. Accuracy')
    ax2.set_xlabel('N bits')
    ax2.set_ylabel('Bottleneck (db)')
    ax2.set_zlabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_3d_surfaces.png'))
    print("Plots saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed_xor', action='store_true', help='If set, synergy task is always 1-bit XOR regardless of N')
    args = parser.parse_args()
    
    output_dir = "results/synergy_bottleneck_nbit"
    if args.fixed_xor:
        output_dir += "_fixed_xor"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sweep ranges
    n_bits_range = [1, 2, 3, 4]
    db_range = [1, 2, 3, 4, 6, 8]
    
    df = run_sweep(db_range, n_bits_range, args.fixed_xor, output_dir)
    plot_results(df, output_dir)
