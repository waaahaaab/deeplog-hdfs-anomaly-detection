"""
DeepLog - Étape 3 : Entraînement
=================================
v3 : vocab_size chargé dynamiquement depuis vocab_size.npy
     Compatible pipeline v2 (101 templates) et v3 CSV (30 templates)
     Pointer DATA_PATH vers le bon dossier selon le pipeline utilisé.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
import importlib.util

import sys

# Chemins relatifs au dossier racine du projet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


spec = importlib.util.spec_from_file_location(
    "model",
    os.path.join(os.path.dirname(__file__), "02_model.py")
)

model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

DeepLogLSTM                = model_module.DeepLogLSTM
LabelSmoothingCrossEntropy = model_module.LabelSmoothingCrossEntropy
model_summary              = model_module.model_summary
load_vocab_size            = model_module.load_vocab_size


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CFG = {
    "data_path"     : "data/processed/",
    "checkpoint_dir": "checkpoints/",
    # Architecture — vocab_size sera chargé dynamiquement
    "vocab_size"    : None,      # ← rempli automatiquement au runtime
    "embed_dim"     : 64,
    "hidden_size"   : 128,
    "num_layers"    : 2,
    "dropout"       : 0.2,

    # Entraînement — identique à v2
    "epochs"        : 30,
    "batch_size"    : 2048,
    "lr_max"        : 3e-3,
    "weight_decay"  : 1e-4,
    "grad_clip"     : 1.0,
    "label_smooth"  : 0.05,
    "val_ratio"     : 0.1,
    "patience"      : 5,
    "seed"          : 42,
}
# ──────────────────────────────────────────────────────────────────────────────


class LogDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("  ⚠️  GPU non disponible, utilisation du CPU.")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, scheduler,
                    criterion, device, grad_clip) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss = correct_top1 = correct_top5 = total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        logits  = model(X_batch)
        total_loss   += criterion(logits, y_batch).item() * len(X_batch)
        correct_top1 += (logits.argmax(dim=-1) == y_batch).sum().item()
        _, top5_idx   = torch.topk(logits, 5, dim=-1)
        correct_top5 += (top5_idx == y_batch.unsqueeze(1)).any(dim=-1).sum().item()
        total        += len(X_batch)

    return total_loss / total, correct_top1 / total, correct_top5 / total


def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 60)
    print("DeepLog — Entraînement v3")
    print("=" * 60)

    device = get_device()
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    # ── vocab_size dynamique ──────────────────────────────────────
    CFG["vocab_size"] = load_vocab_size(CFG["data_path"], default=30)
    print(f"  vocab_size utilisé : {CFG['vocab_size']}")

    # ── Chargement ────────────────────────────────────────────────
    print("\n📂 Chargement des données...")
    X = np.load(os.path.join(CFG["data_path"], "X_train.npy"))
    y = np.load(os.path.join(CFG["data_path"], "y_train.npy"))
    print(f"   X_train : {X.shape}, y_train : {y.shape}")

    # ── Split ─────────────────────────────────────────────────────
    full_dataset = LogDataset(X, y)
    n_val   = int(len(full_dataset) * CFG["val_ratio"])
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(CFG["seed"])
    )
    print(f"   Train : {n_train:,} | Val : {n_val:,}")

    # ── DataLoaders ───────────────────────────────────────────────
    n_workers    = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True, num_workers=n_workers,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2,
                              shuffle=False, num_workers=n_workers,
                              pin_memory=(device.type == "cuda"))

    # ── Modèle ────────────────────────────────────────────────────
    model = DeepLogLSTM(
        vocab_size   = CFG["vocab_size"],
        embed_dim    = CFG["embed_dim"],
        hidden_size  = CFG["hidden_size"],
        num_layers   = CFG["num_layers"],
        dropout      = CFG["dropout"],
    ).to(device)
    model_summary(model)

    # ── Loss, Optimizer, Scheduler ────────────────────────────────
    criterion = LabelSmoothingCrossEntropy(smoothing=CFG["label_smooth"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=CFG["lr_max"] / 10,
                                   weight_decay=CFG["weight_decay"])
    scheduler = OneCycleLR(optimizer, max_lr=CFG["lr_max"],
                            steps_per_epoch=len(train_loader),
                            epochs=CFG["epochs"], pct_start=0.3)

    # ── Boucle ────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    history          = []

    print(f"\n🚀 Début de l'entraînement ({CFG['epochs']} epochs max)\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} "
          f"{'Acc@1':>8} {'Acc@5':>8} {'LR':>10} {'Time':>7}")
    print("─" * 65)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                      scheduler, criterion, device, CFG["grad_clip"])
        val_loss, acc1, acc5 = evaluate(model, val_loader, criterion, device)
        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "acc1": acc1,
                         "acc5": acc5, "lr": current_lr})

        print(f"{epoch:>6} {train_loss:>11.4f} {val_loss:>10.4f} "
              f"{acc1:>8.3f} {acc5:>8.3f} {current_lr:>10.2e} {elapsed:>6.1f}s")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(CFG["checkpoint_dir"], "deeplog_best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "val_loss": val_loss, "cfg": CFG}, ckpt_path)
            print(f"         ✅ Checkpoint sauvegardé (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CFG["patience"]:
                print(f"\n⏹  Early stopping epoch {epoch}")
                break

    hist_path = os.path.join(CFG["checkpoint_dir"], "training_history.npy")
    np.save(hist_path, history)
    print(f"\n💾 Historique : {hist_path}")
    print(f"   Meilleure val_loss : {best_val_loss:.4f}")


if __name__ == "__main__":
    main()