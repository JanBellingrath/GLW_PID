#!/usr/bin/env python3
# add_mod_r.py
# Train a tiny network to learn (a + b) mod r.

import argparse, torch, torch.nn as nn, torch.nn.functional as F

# -------------------- hyper-parameters --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--mod',     type=int, default=10,   help='modulus r')
parser.add_argument('--epochs',  type=int, default=10000, help='#epochs')
parser.add_argument('--batch',   type=int, default=128, help='batch size')
parser.add_argument('--width',   type=int, default=64,  help='hidden width')
parser.add_argument('--lr',      type=float, default=3e-3)
parser.add_argument('--device',  type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

r        = args.mod
device   = args.device

# -------------------- model -------------------------------
model = nn.Sequential(
    nn.Linear(2, args.width),
    nn.ReLU(),
    nn.Linear(args.width, r)          # logits for r classes
).to(device)

opt  = torch.optim.Adam(model.parameters(), lr=args.lr)

def sample_batch(n):
    a = torch.randint(0, r, (n,1), device=device)
    b = torch.randint(0, r, (n,1), device=device)
    x = torch.cat((a, b), dim=1).float() / (r-1)   # normalise inputs to [0,1]
    y = (a + b) % r                                # targets: 0 … r-1
    return x, y.squeeze(1)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

# -------------------- training loop -----------------------
for epoch in range(1, args.epochs+1):
    x, y = sample_batch(args.batch)
    logits = model(x)
    loss   = F.cross_entropy(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()

    if epoch % 10 == 0 or epoch == 1:
        acc = accuracy(logits, y)
        print(f'Epoch {epoch:4d}: loss {loss.item():.4f} | acc {acc*100:5.1f}%')

# -------------------- final test --------------------------
with torch.no_grad():
    x_test, y_test = sample_batch(1024)
    test_acc = accuracy(model(x_test), y_test)
print(f'\nTest accuracy on 1024 fresh samples: {test_acc*100:.2f}%')
