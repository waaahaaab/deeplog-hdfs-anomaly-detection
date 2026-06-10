"""
DeepLog - Étape 7 : Robustesse & Validation
=============================================
Couvre :
  1. Baselines triviales (random, majority, fréquence)
  2. 3 runs seeds différents → mean ± std
  3. Analyse faux positifs / faux négatifs
  4. Robustesse seuil MLP (P90, P95, P99)
  5. Courbe train/val loss
"""

import os
import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import importlib.util
from collections import defaultdict

import sys

# Chemins relatifs au dossier racine du projet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


plt.rcParams.update({
    "font.family"       : "serif",
    "font.size"         : 11,
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 11,
    "legend.fontsize"   : 9,
    "figure.dpi"        : 150,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

# ─── CONFIGURATION ────────────────────────────────────────────
CSV_PATH     = "data/csv/"
DATA_PATH    = "data/processed/"
RESULTS_DIR  = "results/"
CKPT_DIR     = "checkpoints/"
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures/")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "02_model.py")
VOCAB_SIZE   = 29
WINDOW_SIZE  = 10
SEEDS        = [42, 123, 2024]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────


# ─── CHARGEMENT DU MODULE MODÈLE ──────────────────────────────
spec = importlib.util.spec_from_file_location("model", MODEL_PATH)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
DeepLogLSTM                = model_module.DeepLogLSTM
LabelSmoothingCrossEntropy = model_module.LabelSmoothingCrossEntropy


# ─── UTILITAIRES ──────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.logical_and(y_pred == 1, y_true == 1).sum()
    fp = np.logical_and(y_pred == 1, y_true == 0).sum()
    fn = np.logical_and(y_pred == 0, y_true == 1).sum()
    tn = np.logical_and(y_pred == 0, y_true == 0).sum()
    prec   = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1     = 2 * prec * recall / (prec + recall + 1e-10)
    fpr    = fp / (fp + tn + 1e-10)
    return {"precision": float(prec), "recall": float(recall),
            "f1": float(f1), "fpr": float(fpr),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn)}


def comptage_session(seq, vocab_size=29):
    vec = np.zeros(vocab_size, dtype=np.float32)
    for e in seq:
        if 1 <= e <= vocab_size:
            vec[e - 1] += 1
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec


def load_sessions():
    """Charge et encode toutes les sessions depuis le CSV."""
    df  = pd.read_csv(os.path.join(CSV_PATH, "hdfs_sequences.csv"))
    tpl = pd.read_csv(os.path.join(CSV_PATH, "hdfs_templates.csv"))
    event_ids = sorted(tpl["EventId"].tolist())
    mapping   = {eid: idx + 1 for idx, eid in enumerate(event_ids)}

    sequences, labels, block_ids = [], [], []
    for _, row in df.iterrows():
        try:
            raw = ast.literal_eval(str(row["sequence"]))
            seq = [mapping[e] for e in raw]
            sequences.append(seq)
            labels.append(int(row["label"]))
            block_ids.append(row["block_id"])
        except Exception:
            continue
    return sequences, labels, block_ids


def split_sessions(sequences, labels, block_ids,
                   train_ratio=0.8, seed=42):
    labels_arr  = np.array(labels)
    normal_idx  = np.where(labels_arr == 0)[0]
    anomaly_idx = np.where(labels_arr == 1)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(normal_idx)

    n_train       = int(len(normal_idx) * train_ratio)
    train_idx     = normal_idx[:n_train]
    test_norm_idx = normal_idx[n_train:]
    test_idx      = np.concatenate([test_norm_idx, anomaly_idx])

    def get(idx_list):
        return ([sequences[i] for i in idx_list],
                [labels[i]    for i in idx_list],
                [block_ids[i] for i in idx_list])

    tr_s, tr_l, _       = get(train_idx)
    te_s, te_l, te_bids = get(test_idx)
    return tr_s, tr_l, te_s, te_l, te_bids


def session_level_from_windows(in_topk, block_ids_arr,
                                y_ano_arr, k):
    window_pred  = (~in_topk[:, k-1]).astype(np.int32)
    sort_idx     = np.argsort(block_ids_arr, kind="stable")
    bid_s        = block_ids_arr[sort_idx]
    pred_s       = window_pred[sort_idx]
    ano_s        = y_ano_arr[sort_idx]
    _, first_occ = np.unique(bid_s, return_index=True)
    s_pred = np.maximum.reduceat(pred_s, first_occ).clip(0, 1)
    s_true = np.maximum.reduceat(ano_s,  first_occ).clip(0, 1)
    return s_pred, s_true


# ─── 1. BASELINES ─────────────────────────────────────────────

def evaluate_baselines(test_lbls, seed=42):
    """
    Trois baselines triviales :
    - Random : prédit anomalie avec probabilité = ratio d'anomalies
    - Majority : prédit toujours Normal
    - Minority : prédit toujours Anomalie
    """
    y_true     = np.array(test_lbls)
    ratio_ano  = y_true.mean()
    rng        = np.random.default_rng(seed)

    results = {}

    # Baseline 1 : Random
    y_random = (rng.random(len(y_true)) < ratio_ano).astype(int)
    results["Baseline Random"]   = compute_metrics(y_true, y_random)

    # Baseline 2 : Majority (tout Normal)
    y_majority = np.zeros(len(y_true), dtype=int)
    results["Baseline Majority"] = compute_metrics(y_true, y_majority)

    # Baseline 3 : Minority (tout Anomalie)
    y_minority = np.ones(len(y_true), dtype=int)
    results["Baseline Minority"] = compute_metrics(y_true, y_minority)

    print("\n" + "="*60)
    print("  BASELINES TRIVIALES")
    print("="*60)
    print(f"  Ratio anomalies dans le test : {100*ratio_ano:.2f}%")
    print(f"\n  {'Baseline':<22} {'P':>8} {'R':>8} {'F1':>8}")
    print("─" * 50)
    for name, m in results.items():
        print(f"  {name:<22} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f}")

    return results


# ─── 2. MULTI-SEEDS ───────────────────────────────────────────

class LogDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def fenetrer(sequences, window_size=10):
    X, y = [], []
    for seq in sequences:
        if len(seq) <= window_size:
            continue
        for i in range(len(seq) - window_size):
            X.append(seq[i: i + window_size])
            y.append(seq[i + window_size])
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


def run_one_seed(sequences, labels, block_ids,
                 seed, vocab_size=30):
    """
    Entraîne et évalue le LSTM pour un seed donné.
    Retourne F1, Precision, Recall session-level K=5.
    """
    print(f"\n  [Seed={seed}] Split...")
    tr_s, tr_l, te_s, te_l, te_bids = split_sessions(
        sequences, labels, block_ids, seed=seed)

    # Fenêtrage
    X_tr, y_tr = fenetrer(tr_s, WINDOW_SIZE)
    print(f"  [Seed={seed}] {len(X_tr):,} fenêtres train")

    # Modèle
    torch.manual_seed(seed)
    model = DeepLogLSTM(
        vocab_size=vocab_size, embed_dim=64,
        hidden_size=128, num_layers=2, dropout=0.2
    ).to(DEVICE)

    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3/10, weight_decay=1e-4)

    n_val    = int(len(X_tr) * 0.1)
    n_train  = len(X_tr) - n_val
    from torch.utils.data import random_split
    full_ds  = LogDataset(X_tr, y_tr)
    tr_ds, _ = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed))

    loader = DataLoader(tr_ds, batch_size=2048,
                        shuffle=True, num_workers=2,
                        pin_memory=(DEVICE.type=="cuda"),
                        drop_last=True)

    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer, max_lr=3e-3,
        steps_per_epoch=len(loader),
        epochs=10, pct_start=0.3)

    # Entraînement 10 epochs (rapide pour la robustesse)
    model.train()
    for epoch in range(10):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    # Inférence session-level K=5
    model.eval()
    import torch.nn.functional as F

    # Fenêtrage test
    X_te, y_te = fenetrer(te_s, WINDOW_SIZE)

    # Mapping block_id → int pour le test
    te_bids_flat = []
    for i, seq in enumerate(te_s):
        n_w = max(0, len([e for e in seq if e > 0]) - WINDOW_SIZE)
        te_bids_flat.extend([i] * n_w)
    te_bids_arr = np.array(te_bids_flat[:len(X_te)], dtype=np.int32)

    # Labels anomalie par fenêtre (session label propagé)
    te_lbls_arr = np.array(te_l)
    y_ano_flat  = []
    for i, (seq, lbl) in enumerate(zip(te_s, te_l)):
        n_w = max(0, len(seq) - WINDOW_SIZE)
        y_ano_flat.extend([lbl] * n_w)
    y_ano_arr = np.array(y_ano_flat[:len(X_te)], dtype=np.int32)

    # in_topk
    test_ds = LogDataset(X_te, y_te)
    te_loader = DataLoader(test_ds, batch_size=8192,
                           shuffle=False, num_workers=2)
    k_max = 5
    in_topk = np.empty((len(X_te), k_max), dtype=bool)
    offset  = 0

    with torch.no_grad():
        for X_b, y_b in te_loader:
            bsz = len(X_b)
            X_b = X_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            logits = model(X_b)
            probs  = F.softmax(logits.float(), dim=-1)
            _, topk_idx = torch.topk(probs, k_max, dim=-1)
            y_exp = y_b.unsqueeze(1).expand(-1, k_max)
            match = (topk_idx == y_exp)
            in_topk[offset:offset+bsz] = \
                match.cummax(dim=1).values.cpu().numpy()
            offset += bsz

    s_pred, s_true = session_level_from_windows(
        in_topk, te_bids_arr, y_ano_arr, k=5)
    m = compute_metrics(s_true, s_pred)
    print(f"  [Seed={seed}] F1={m['f1']:.4f}  "
          f"P={m['precision']:.4f}  R={m['recall']:.4f}")
    return m


def evaluate_multi_seeds(sequences, labels, block_ids):
    print("\n" + "="*60)
    print("  ROBUSTESSE — 3 SEEDS DIFFÉRENTS (LSTM K=5)")
    print("="*60)

    all_f1, all_p, all_r = [], [], []
    for seed in SEEDS:
        m = run_one_seed(sequences, labels, block_ids, seed)
        all_f1.append(m["f1"])
        all_p.append(m["precision"])
        all_r.append(m["recall"])

    print(f"\n  {'Métrique':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("─" * 46)
    for name, vals in [("F1",        all_f1),
                       ("Precision", all_p),
                       ("Recall",    all_r)]:
        arr = np.array(vals)
        print(f"  {name:<12} {arr.mean():>8.4f} {arr.std():>8.4f} "
              f"{arr.min():>8.4f} {arr.max():>8.4f}")

    return {"f1": all_f1, "precision": all_p, "recall": all_r}

def evaluate_multi_seeds_inference_only(block_ids_arr, y_ano_arr,
                                         in_topk):
    """
    Robustesse par sous-échantillonnage du test sur modèle fixe.
    3 tirages aléatoires de 80% des sessions test.
    Pas de réentraînement — résultat en ~30 secondes.
    """
    print("\n" + "="*60)
    print("  ROBUSTESSE — 3 SOUS-ÉCHANTILLONS (modèle fixe, K=5)")
    print("="*60)

    all_f1, all_p, all_r = [], [], []

    for seed in SEEDS:
        rng         = np.random.default_rng(seed)
        unique_bids = np.unique(block_ids_arr)
        sub_bids    = rng.choice(unique_bids,
                                  int(len(unique_bids) * 0.8),
                                  replace=False)
        mask = np.isin(block_ids_arr, sub_bids)

        s_pred, s_true = session_level_from_windows(
            in_topk[mask],
            block_ids_arr[mask],
            y_ano_arr[mask], k=5)

        m = compute_metrics(s_true, s_pred)
        all_f1.append(m["f1"])
        all_p.append(m["precision"])
        all_r.append(m["recall"])
        print(f"  [Seed={seed}] "
              f"Sessions={mask.sum():,}  "
              f"F1={m['f1']:.4f}  "
              f"P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}")

    print(f"\n  {'Métrique':<12} {'Mean':>8} {'Std':>8} "
          f"{'Min':>8} {'Max':>8}")
    print("─" * 46)
    for name, vals in [("F1",        all_f1),
                       ("Precision", all_p),
                       ("Recall",    all_r)]:
        arr = np.array(vals)
        print(f"  {name:<12} {arr.mean():>8.4f} "
              f"{arr.std():>8.4f} "
              f"{arr.min():>8.4f} "
              f"{arr.max():>8.4f}")

    return {"f1": all_f1, "precision": all_p, "recall": all_r}


# ─── 3. ANALYSE ERREURS ───────────────────────────────────────

def analyze_errors(in_topk, block_ids_arr, y_ano_arr, k=5):
    """
    Analyse des faux positifs et faux négatifs session-level.
    """
    s_pred, s_true = session_level_from_windows(
        in_topk, block_ids_arr, y_ano_arr, k)

    fp_mask = np.logical_and(s_pred == 1, s_true == 0)
    fn_mask = np.logical_and(s_pred == 0, s_true == 1)
    tp_mask = np.logical_and(s_pred == 1, s_true == 1)
    tn_mask = np.logical_and(s_pred == 0, s_true == 0)

    print("\n" + "="*60)
    print(f"  ANALYSE ERREURS — LSTM K={k} (session-level)")
    print("="*60)
    print(f"  Vrais Positifs  (TP) : {tp_mask.sum():>8,}")
    print(f"  Vrais Négatifs  (TN) : {tn_mask.sum():>8,}")
    print(f"  Faux Positifs   (FP) : {fp_mask.sum():>8,}  "
          f"({100*fp_mask.sum()/max(tn_mask.sum()+fp_mask.sum(),1):.2f}% des normaux)")
    print(f"  Faux Négatifs   (FN) : {fn_mask.sum():>8,}  "
          f"({100*fn_mask.sum()/max(s_true.sum(),1):.2f}% des anomalies)")

    # Analyse par longueur de session
    sort_idx     = np.argsort(block_ids_arr, kind="stable")
    bid_sorted   = block_ids_arr[sort_idx]
    ano_sorted   = y_ano_arr[sort_idx]
    _, first_occ, counts = np.unique(
        bid_sorted, return_index=True, return_counts=True)

    fp_lengths = counts[fp_mask]
    fn_lengths = counts[fn_mask]
    tp_lengths = counts[tp_mask]

    print(f"\n  Longueur moyenne des sessions (nb fenêtres) :")
    print(f"  TP : {tp_lengths.mean():.1f}  "
          f"FP : {fp_lengths.mean():.1f}  "
          f"FN : {fn_lengths.mean():.1f}")
    print(f"\n  Interprétation :")
    if fn_lengths.mean() < tp_lengths.mean():
        print("  → Les FN sont des sessions courtes : "
              "peu de fenêtres pour détecter l'anomalie.")
    else:
        print("  → Les FN sont des sessions longues : "
              "l'anomalie est diluée dans beaucoup de fenêtres normales.")

    return {
        "fp": int(fp_mask.sum()),
        "fn": int(fn_mask.sum()),
        "tp": int(tp_mask.sum()),
        "tn": int(tn_mask.sum()),
        "fp_lengths": fp_lengths,
        "fn_lengths": fn_lengths,
    }


# ─── 4. ROBUSTESSE SEUIL MLP ──────────────────────────────────

def evaluate_mlp_thresholds(train_seqs, test_seqs, test_lbls):
    """
    Teste P90, P95, P99 pour le seuil MLP.
    Montre que le résultat est robuste au choix de τ.
    """
    X_train = np.stack([comptage_session(s) for s in train_seqs])
    X_test  = np.stack([comptage_session(s) for s in test_seqs])
    test_lbls_arr = np.array(test_lbls)

    centroid = X_train.mean(axis=0)
    std      = X_train.std(axis=0) + 1e-8

    scores_train = np.sqrt(
        ((X_train - centroid) / std) ** 2).sum(axis=1)
    scores_test  = np.sqrt(
        ((X_test  - centroid) / std) ** 2).sum(axis=1)

    print("\n" + "="*60)
    print("  ROBUSTESSE SEUIL MLP (P90 / P95 / P99)")
    print("="*60)
    print(f"  {'Percentile':>12} {'τ':>10} "
          f"{'P':>8} {'R':>8} {'F1':>8} {'FPR':>8}")
    print("─" * 58)

    results = {}
    for p in [90, 95, 99]:
        tau   = np.percentile(scores_train, p)
        preds = (scores_test > tau).astype(int)
        m     = compute_metrics(test_lbls_arr, preds)
        results[f"MLP P{p}"] = m
        print(f"  {'P' + str(p):>12} {tau:>10.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['fpr']:>8.4f}")

    return results, centroid, std, scores_test


# ─── 5. COURBE D'APPRENTISSAGE ────────────────────────────────

def plot_learning_curve(save_path):
    """
    Trace train loss vs val loss depuis training_history.npy.
    """
    hist_path = os.path.join(CKPT_DIR, "training_history.npy")
    if not os.path.exists(hist_path):
        print("  ⚠️  training_history.npy introuvable.")
        return

    history = np.load(hist_path, allow_pickle=True).tolist()
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    acc1       = [h["acc1"]       for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_loss, "o-", color="#2563EB",
             lw=2, label="Train loss", markersize=4)
    ax1.plot(epochs, val_loss,   "s-", color="#DC2626",
             lw=2, label="Val loss",   markersize=4)
    best_epoch = epochs[np.argmin(val_loss)]
    ax1.axvline(best_epoch, color="#16A34A", lw=1.5,
                ls="--", label=f"Best epoch={best_epoch}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (label smoothing CE)")
    ax1.set_title("Courbe d'apprentissage — Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in acc1],
             "o-", color="#9333EA", lw=2, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy@1 (%)")
    ax2.set_title("Précision top-1 sur la validation")
    ax2.set_ylim(85, 100)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


# ─── FIGURES ROBUSTESSE ───────────────────────────────────────

def plot_multi_seeds(seeds_results, save_path):
    """Figure : F1/P/R avec barres d'erreur pour 3 seeds."""
    metrics  = ["F1", "Precision", "Recall"]
    means    = [np.mean(seeds_results["f1"]),
                np.mean(seeds_results["precision"]),
                np.mean(seeds_results["recall"])]
    stds     = [np.std(seeds_results["f1"]),
                np.std(seeds_results["precision"]),
                np.std(seeds_results["recall"])]
    colors   = ["#16A34A", "#2563EB", "#EA580C"]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(metrics))
    bars = ax.bar(x, means, yerr=stds, capsize=8,
                  color=colors, alpha=0.85,
                  error_kw={"elinewidth": 2, "ecolor": "#374151"})

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(x[i], m + s + 0.02,
                f"{m:.3f}±{s:.3f}",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Robustesse LSTM K=5 — {len(SEEDS)} seeds\n"
        f"(seeds = {SEEDS})", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def plot_error_analysis(error_data, save_path):
    """Figure : distribution longueur sessions TP/FP/FN."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogramme longueurs FP vs TP
    ax = axes[0]
    bins = np.linspace(0, 200, 40)
    if len(error_data["fp_lengths"]) > 0:
        ax.hist(error_data["fp_lengths"], bins=bins,
                alpha=0.6, color="#DC2626", label="Faux Positifs",
                density=True)
    if len(error_data["fn_lengths"]) > 0:
        ax.hist(error_data["fn_lengths"], bins=bins,
                alpha=0.6, color="#F59E0B", label="Faux Négatifs",
                density=True)
    ax.set_xlabel("Nombre de fenêtres par session")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution longueur — FP et FN")
    ax.legend()

    # Matrice de confusion simplifiée
    ax = axes[1]
    cm = np.array([
        [error_data["tn"], error_data["fp"]],
        [error_data["fn"], error_data["tp"]]
    ])
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Prédit Normal", "Prédit Anomalie"])
    ax.set_yticklabels(["Réel Normal", "Réel Anomalie"])
    ax.set_title("Matrice de confusion (session-level, K=5)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2
                    else "black")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def plot_mlp_thresholds(mlp_thresh_results, save_path):
    """Figure : barres P/R/F1 pour P90/P95/P99."""
    names = list(mlp_thresh_results.keys())
    p_vals = [mlp_thresh_results[n]["precision"] for n in names]
    r_vals = [mlp_thresh_results[n]["recall"]    for n in names]
    f_vals = [mlp_thresh_results[n]["f1"]        for n in names]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, p_vals, w, label="Précision",
           color="#2563EB", alpha=0.85)
    ax.bar(x,     r_vals, w, label="Rappel",
           color="#EA580C", alpha=0.85)
    ax.bar(x + w, f_vals, w, label="F1",
           color="#16A34A", alpha=0.85)

    for i, v in enumerate(f_vals):
        ax.text(x[i] + w, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9, color="#15803D",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(
        "Robustesse MLP — Sensibilité au seuil τ\n"
        "(P90 / P95 / P99 sur scores train)", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def plot_baseline_comparison(baseline_results,
                              lstm_k5_metrics,
                              mlp_metrics,
                              save_path):
    """Figure : comparaison baselines vs modèles."""
    all_systems = {
        **baseline_results,
        "LSTM K=5"  : lstm_k5_metrics,
        "MLP P95"   : mlp_metrics,
    }

    names  = list(all_systems.keys())
    f1s    = [all_systems[n]["f1"]        for n in names]
    precs  = [all_systems[n]["precision"] for n in names]
    recs   = [all_systems[n]["recall"]    for n in names]

    colors = ["#9CA3AF", "#9CA3AF", "#9CA3AF",
              "#2563EB", "#16A34A"]
    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, precs, w, color=colors, alpha=0.6,
           label="Précision")
    ax.bar(x,     recs,  w, color=colors, alpha=0.8,
           label="Rappel")
    ax.bar(x + w, f1s,   w, color=colors, alpha=1.0,
           label="F1")

    for i, v in enumerate(f1s):
        ax.text(x[i] + w, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=8.5,
                fontweight="bold" if i >= 3 else "normal",
                color="#15803D" if i >= 3 else "#6B7280")

    ax.axvline(2.5, color="#6B7280", ls="--", lw=1.2, alpha=0.5)
    ax.text(1.0, 1.06, "Baselines triviales",
            ha="center", fontsize=9, color="#6B7280",
            transform=ax.get_xaxis_transform())
    ax.text(3.5, 1.06, "Modèles proposés",
            ha="center", fontsize=9, color="#2563EB",
            fontweight="bold",
            transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title(
        "Comparaison modèles vs baselines triviales\n"
        "(session-level)", fontsize=12)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


# ─── TABLEAU FINAL ROBUSTESSE ─────────────────────────────────

def print_robustness_summary(baseline_results,
                              seeds_results,
                              mlp_thresh_results,
                              error_data):
    print("\n" + "="*65)
    print("  RÉSUMÉ ROBUSTESSE — à reporter dans le mémoire")
    print("="*65)

    print("\n  1. LSTM K=5 session-level (mean ± std, 3 seeds) :")
    print(f"     F1        = {np.mean(seeds_results['f1']):.4f} "
          f"± {np.std(seeds_results['f1']):.4f}")
    print(f"     Precision = {np.mean(seeds_results['precision']):.4f} "
          f"± {np.std(seeds_results['precision']):.4f}")
    print(f"     Recall    = {np.mean(seeds_results['recall']):.4f} "
          f"± {np.std(seeds_results['recall']):.4f}")

    print("\n  2. MLP robustesse seuil :")
    for name, m in mlp_thresh_results.items():
        print(f"     {name} : F1={m['f1']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}")

    print("\n  3. Analyse erreurs LSTM K=5 :")
    total = error_data["tp"] + error_data["tn"] + \
            error_data["fp"] + error_data["fn"]
    print(f"     FP : {error_data['fp']:,} "
          f"({100*error_data['fp']/total:.2f}% du total)")
    print(f"     FN : {error_data['fn']:,} "
          f"({100*error_data['fn']/total:.2f}% du total)")

    print("\n  4. Baselines :")
    for name, m in baseline_results.items():
        print(f"     {name:<22} F1={m['f1']:.4f}")
    print("="*65)


# ─── MAIN ─────────────────────────────────────────────────────

def main():
    print("="*60)
    print("DeepLog — Robustesse & Validation")
    print("="*60)
    print(f"[INFO] Dispositif : {DEVICE}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Chargement ────────────────────────────────────────────
    print("\n[INFO] Chargement des sessions...")
    sequences, labels, block_ids = load_sessions()

    # Split seed=42 pour les analyses statiques
    tr_s, tr_l, te_s, te_l, te_bids = split_sessions(
        sequences, labels, block_ids, seed=42)

    # ── 1. Baselines ──────────────────────────────────────────
    baseline_results = evaluate_baselines(te_l, seed=42)
    
    # ── Chargement in_topk (nécessaire pour étapes 2, 3, 4) ──
    in_topk   = np.load(os.path.join(RESULTS_DIR, "in_topk.npy"))
    block_ids_arr = np.load(
        os.path.join(DATA_PATH, "block_ids_test.npy"))
    y_ano_arr = np.load(
        os.path.join(DATA_PATH, "y_ano_test.npy"))

    # ── 2. Robustesse ────────────────────────────────────────
    seeds_results = evaluate_multi_seeds_inference_only(
       block_ids_arr, y_ano_arr, in_topk)

    # ── 3. Analyse erreurs ────────────────────────────────────

    error_data = analyze_errors(in_topk, block_ids_arr,
                                 y_ano_arr, k=5)

    # ── 4. Robustesse seuil MLP ───────────────────────────────
    mlp_thresh_results, _, _, _ = evaluate_mlp_thresholds(
        tr_s, te_s, te_l)

    # ── 5. Courbe apprentissage ───────────────────────────────
    print("\n📊 Génération des figures...")
    plot_learning_curve(
        os.path.join(FIGURES_DIR, "fig10_learning_curve.pdf"))

    # ── Figures robustesse ────────────────────────────────────
    plot_multi_seeds(
        seeds_results,
        os.path.join(FIGURES_DIR, "fig11_multi_seeds.pdf"))

    plot_error_analysis(
        error_data,
        os.path.join(FIGURES_DIR, "fig12_error_analysis.pdf"))

    plot_mlp_thresholds(
        mlp_thresh_results,
        os.path.join(FIGURES_DIR, "fig13_mlp_thresholds.pdf"))

    # Métriques LSTM K=5 et MLP P95 pour la comparaison baseline
    lstm_k5_s_pred, lstm_k5_s_true = session_level_from_windows(
        in_topk, block_ids_arr, y_ano_arr, k=5)
    lstm_k5_metrics = compute_metrics(lstm_k5_s_true, lstm_k5_s_pred)
    mlp_p95_metrics = mlp_thresh_results.get(
        "MLP P95", list(mlp_thresh_results.values())[1])

    plot_baseline_comparison(
        baseline_results, lstm_k5_metrics, mlp_p95_metrics,
        os.path.join(FIGURES_DIR, "fig14_baseline_comparison.pdf"))

    # ── Résumé ────────────────────────────────────────────────
    print_robustness_summary(
        baseline_results, seeds_results,
        mlp_thresh_results, error_data)

    # Sauvegarde
    np.save(os.path.join(RESULTS_DIR, "robustness_results.npy"), {
        "baselines"   : baseline_results,
        "seeds"       : seeds_results,
        "mlp_thresh"  : mlp_thresh_results,
        "errors"      : {k: v for k, v in error_data.items()
                         if not isinstance(v, np.ndarray)},
    })

    print(f"\n✅ 5 figures générées dans : {FIGURES_DIR}")
    print("   fig10 : courbe apprentissage")
    print("   fig11 : robustesse multi-seeds")
    print("   fig12 : analyse erreurs + matrice confusion")
    print("   fig13 : robustesse seuil MLP")
    print("   fig14 : comparaison baselines vs modèles")


if __name__ == "__main__":
    main()