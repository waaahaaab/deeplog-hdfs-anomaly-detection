"""
DeepLog — Visualisation complémentaire
========================================
Figures manquantes dans 05_visualize.py :
  fig10 : courbe apprentissage (si pas encore générée)
  fig_acc5 : Acc@1 et Acc@5 par epoch
  fig_lr : courbe du learning rate OneCycleLR
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import sys

# Chemins relatifs au dossier racine du projet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.titlesize" : 12,
    "axes.labelsize" : 11,
    "legend.fontsize": 9,
    "figure.dpi"     : 150,
    "axes.grid"      : True,
    "grid.alpha"     : 0.3,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

CKPT_DIR    = "checkpoints/"
FIGURES_DIR ="figures/"


def plot_full_training(save_path):
    """
    Figure complète entraînement :
    - Train loss / Val loss
    - Acc@1 / Acc@5
    - Learning rate
    3 sous-graphes sur une figure.
    """
    hist_path = os.path.join(CKPT_DIR, "training_history.npy")
    history   = np.load(hist_path, allow_pickle=True).tolist()

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    acc1       = [h["acc1"] * 100 for h in history]
    acc5       = [h["acc5"] * 100 for h in history]
    lr         = [h["lr"]         for h in history]

    best_epoch = epochs[int(np.argmin(val_loss))]
    best_val   = min(val_loss)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── Loss ──
    ax = axes[0]
    ax.plot(epochs, train_loss, "o-", color="#2563EB",
            lw=2, markersize=3, label="Train")
    ax.plot(epochs, val_loss,   "s-", color="#DC2626",
            lw=2, markersize=3, label="Validation")
    ax.axvline(best_epoch, color="#16A34A", lw=1.5, ls="--",
               label=f"Best epoch {best_epoch}\n"
                     f"val_loss={best_val:.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train / Val Loss")
    ax.legend(fontsize=8)

    # ── Accuracy ──
    ax = axes[1]
    ax.plot(epochs, acc1, "o-", color="#9333EA",
            lw=2, markersize=3, label="Acc@1")
    ax.plot(epochs, acc5, "s-", color="#16A34A",
            lw=2, markersize=3, label="Acc@5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy@1 et Accuracy@5")
    ax.set_ylim(85, 101)
    ax.legend()

    # ── LR ──
    ax = axes[2]
    ax.plot(epochs, [l * 1000 for l in lr],
            "o-", color="#F59E0B", lw=2, markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (×10⁻³)")
    ax.set_title("OneCycleLR — Évolution du LR")

    fig.suptitle(
        "DeepLog LSTM — Historique d'entraînement complet\n"
        f"(vocab=30, hidden=128, 2 couches, 237,214 params)",
        fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def main():
    print("="*55)
    print("DeepLog — Visualisation complémentaire")
    print("="*55)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plot_full_training(
        os.path.join(FIGURES_DIR, "fig10_training_complete.pdf"))

    print("\n✅ Terminé.")


if __name__ == "__main__":
    main()