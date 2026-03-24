"""
DeepScan — CNN Training Script
Trains the Convolutional Neural Network on a labelled real/fake dataset.

Dataset structure required:
  dataset/
  ├── train/
  │   ├── real/    ← real photographs
  │   └── fake/    ← AI-generated / manipulated images
  └── val/
      ├── real/
      └── fake/

Usage:
  python train_model.py
  python train_model.py --dataset ./dataset --epochs 25 --batch 32 --lr 0.001

After training, set MODEL_PATH=best_model.pt before running app.py
"""

import argparse
import os
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH = True
except ImportError:
    TORCH = False

try:
    import matplotlib
    matplotlib.use('Agg')       # headless safe
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
if TORCH:
    class FakeImageCNN(nn.Module):
        """
        4-block CNN for binary classification (REAL / FAKE).
        Input : 128×128 RGB
        Output: 2-class logits
        """
        def __init__(self, dropout=0.5):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout2d(0.1),
                # Block 2
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout2d(0.1),
                # Block 3
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(512, 128),          nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 2),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device        : {device}')
    print(f'[INFO] Dataset path  : {args.dataset}')
    print(f'[INFO] Epochs        : {args.epochs}')
    print(f'[INFO] Batch size    : {args.batch}')
    print(f'[INFO] Learning rate : {args.lr}')
    print()

    # ── Transforms
    train_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # ── Datasets
    train_dir = os.path.join(args.dataset, 'train')
    val_dir   = os.path.join(args.dataset, 'val')

    if not os.path.isdir(train_dir):
        print(f'[ERROR] Training folder not found: {train_dir}')
        print('Please create:')
        print('  dataset/train/real/')
        print('  dataset/train/fake/')
        return

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfm)
    val_ds   = datasets.ImageFolder(val_dir, transform=val_tfm) \
               if os.path.isdir(val_dir) else None

    print(f'[INFO] Classes       : {train_ds.classes}')
    print(f'[INFO] Train samples : {len(train_ds)}')
    if val_ds:
        print(f'[INFO] Val samples   : {len(val_ds)}')
    print()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch,
        shuffle=False, num_workers=2,
    ) if val_ds else None

    # ── Model, optimiser, scheduler
    model     = FakeImageCNN(dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train phase
        model.train()
        t_loss = t_correct = t_total = 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)

        scheduler.step()
        tr_loss = t_loss / t_total
        tr_acc  = t_correct / t_total

        # ── Validation phase
        v_loss_avg = v_acc = 0.0
        if val_loader:
            model.eval()
            v_loss = v_correct = v_total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out  = model(imgs)
                    loss = criterion(out, labels)
                    v_loss    += loss.item() * imgs.size(0)
                    v_correct += (out.argmax(1) == labels).sum().item()
                    v_total   += imgs.size(0)

            v_loss_avg = v_loss / v_total
            v_acc      = v_correct / v_total

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save(model.state_dict(), 'best_model.pt')
                print(f'  ★ New best saved  (val_acc = {v_acc:.4f})')

        elapsed = time.time() - t0
        print(
            f'Epoch [{epoch:3d}/{args.epochs}] '
            f'train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  '
            f'val_loss={v_loss_avg:.4f}  val_acc={v_acc:.4f}  '
            f'({elapsed:.1f}s)'
        )

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(v_loss_avg)
        history['val_acc'].append(v_acc)

    # Save final weights
    torch.save(model.state_dict(), 'final_model.pt')
    print(f'\n[INFO] Training complete.')
    print(f'[INFO] Best val_acc : {best_val_acc:.4f}')
    print(f'[INFO] Weights saved: best_model.pt  /  final_model.pt')

    # ── Plot training curves
    if MPL:
        epochs_r = range(1, args.epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#0a0a0f')

        for ax in (ax1, ax2):
            ax.set_facecolor('#16161f')
            ax.tick_params(colors='#6b6b8a')
            ax.xaxis.label.set_color('#6b6b8a')
            ax.yaxis.label.set_color('#6b6b8a')
            ax.title.set_color('#e8e8f0')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e1e2e')

        ax1.plot(epochs_r, history['train_loss'], color='#4d9fff',
                 label='Train Loss', linewidth=2)
        ax1.plot(epochs_r, history['val_loss'],   color='#ff3c6e',
                 label='Val Loss',   linewidth=2)
        ax1.set_title('Loss Curve')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.legend(labelcolor='#e8e8f0', facecolor='#1e1e2e')

        ax2.plot(epochs_r, history['train_acc'], color='#4d9fff',
                 label='Train Acc', linewidth=2)
        ax2.plot(epochs_r, history['val_acc'],   color='#00e5a0',
                 label='Val Acc',   linewidth=2)
        ax2.set_title('Accuracy Curve')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
        ax2.legend(labelcolor='#e8e8f0', facecolor='#1e1e2e')

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight',
                    facecolor='#0a0a0f')
        print('[INFO] Training curves → training_curves.png')

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    if not TORCH:
        print('[ERROR] PyTorch is required for training.')
        print('Install with:')
        print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu')
    else:
        parser = argparse.ArgumentParser(description='DeepScan CNN Trainer')
        parser.add_argument('--dataset', type=str,   default='./dataset',
                            help='Root folder (contains train/ and val/)')
        parser.add_argument('--epochs',  type=int,   default=20)
        parser.add_argument('--batch',   type=int,   default=32)
        parser.add_argument('--lr',      type=float, default=1e-3)
        parser.add_argument('--dropout', type=float, default=0.5)
        args = parser.parse_args()
        train(args)
