"""
DeepLog - Étape 6 : MLP Comptage fréquentiel
=============================================
Implémentation honnête sans data leakage.
Entraînement : sessions normales du train UNIQUEMENT.
Détection finale : LSTM OR MLP (session-level).
"""

import os
import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# ─── CONFIGURATION ────────────────────────────────────────────
CSV_PATH    = "data/csv/"
DATA_PATH   = "data/processed/"
RESULTS_DIR = "results/"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures/")
VOCAB_SIZE  = 29   # nombre d'events (sans padding)
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
# ──────────────────────────────────────────────────────────────




# ─── VECTEUR DE COMPTAGE ──────────────────────────────────────

def comptage_session(seq, vocab_size=29):
    """
    Vecteur fréquentiel normalisé d'une session.
    seq : liste d'entiers (1 à 29)
    """
    vec = np.zeros(vocab_size, dtype=np.float32)
    for e in seq:
        if 1 <= e <= vocab_size:
            vec[e - 1] += 1
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec


# ─── CHARGEMENT CSV ───────────────────────────────────────────

def load_sessions_from_csv():
    """
    Charge hdfs_sequences.csv et retourne les sessions
    encodées en entiers + leurs labels.
    """
    print("[INFO] Chargement hdfs_sequences.csv...")
    df = pd.read_csv(os.path.join(CSV_PATH, "hdfs_sequences.csv"))

    # Mapping EventId → entier
    tpl = pd.read_csv(os.path.join(CSV_PATH, "hdfs_templates.csv"))
    event_ids = sorted(tpl["EventId"].tolist())
    mapping   = {eid: idx + 1 for idx, eid in enumerate(event_ids)}

    sequences = []
    labels    = []
    block_ids = []
    n_skipped = 0

    for _, row in df.iterrows():
        try:
            raw  = ast.literal_eval(str(row["sequence"]))
            seq  = [mapping[e] for e in raw]
            lbl  = int(row["label"])
            sequences.append(seq)
            labels.append(lbl)
            block_ids.append(row["block_id"])
        except Exception:
            n_skipped += 1
            continue

    print(f"  Sessions chargées : {len(sequences):,}  "
          f"(ignorées : {n_skipped})")
    print(f"  Normales : {sum(l==0 for l in labels):,}  "
          f"Anormales : {sum(l==1 for l in labels):,}")

    return sequences, labels, block_ids


# ─── SPLIT COHÉRENT AVEC 01_data_preparation.py ──────────────

def split_sessions(sequences, labels, block_ids, train_ratio=0.8):
    """
    Reproduit exactement le split de 01_data_preparation.py :
    train = 80% des sessions normales (seed=42, même ordre).
    """
    labels_arr = np.array(labels)
    normal_idx  = np.where(labels_arr == 0)[0]
    anomaly_idx = np.where(labels_arr == 1)[0]

    rng = np.random.default_rng(42)
    rng.shuffle(normal_idx)

    n_train      = int(len(normal_idx) * train_ratio)
    train_idx    = normal_idx[:n_train]
    test_norm_idx= normal_idx[n_train:]

    print(f"\n  Train (normaux)    : {len(train_idx):,}")
    print(f"  Test normaux       : {len(test_norm_idx):,}")
    print(f"  Test anormaux      : {len(anomaly_idx):,}")

    def get(idx_list):
        seqs = [sequences[i] for i in idx_list]
        lbls = [labels[i]    for i in idx_list]
        bids = [block_ids[i] for i in idx_list]
        return seqs, lbls, bids

    train_seqs, train_lbls, _ = get(train_idx)

    test_idx  = np.concatenate([test_norm_idx, anomaly_idx])
    test_seqs, test_lbls, test_bids = get(test_idx)

    return train_seqs, train_lbls, test_seqs, test_lbls, test_bids


# ─── MÉTRIQUES ────────────────────────────────────────────────

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


# ─── ENTRAÎNEMENT MLP ─────────────────────────────────────────

def fit_oneclass_centroid(train_seqs, train_lbls):
    """
    One-class : calcule le centroïde des vecteurs normaux.
    Détection : distance euclidienne au centroïde.
    Pas de réseau nécessaire — plus honnête et plus efficace.
    """
    print("\n" + "="*55)
    print("  Entraînement MLP Comptage (centroïde one-class)")
    print("="*55)

    X_train = np.stack([comptage_session(s) for s in train_seqs])
    print(f"  X_train shape : {X_train.shape}")

    # Centroïde = profil fréquentiel moyen des sessions normales
    centroid = X_train.mean(axis=0)
    std      = X_train.std(axis=0) + 1e-8

    print(f"  Centroïde calculé sur {len(X_train):,} sessions normales")
    print("  ✅ Modèle one-class prêt.")

    return centroid, std


def predict_mlp_oneclass(centroid, std, train_seqs,
                          test_seqs, test_lbls):
    """
    Seuil calibré sur le train (percentile 95 des normaux).
    Aucun leakage : le test n'est pas utilisé pour choisir τ.
    """   
    # Score sur le train pour calibrer le seuil
    X_train = np.stack([comptage_session(s) for s in train_seqs])
    scores_train = np.sqrt(
        ((X_train - centroid) / std) ** 2
    ).sum(axis=1)

    # Seuil = percentile 95 des scores normaux
    # → on accepte 5% de faux positifs sur les normaux
    tau = np.percentile(scores_train, 95)
    print(f"  Seuil calibré sur train (P95) : τ={tau:.4f}")

    # Score sur le test
    X_test = np.stack([comptage_session(s) for s in test_seqs])
    scores_test = np.sqrt(
        ((X_test - centroid) / std) ** 2
    ).sum(axis=1)

    test_lbls_arr = np.array(test_lbls)
    preds = (scores_test > tau).astype(int)
    m     = compute_metrics(test_lbls_arr, preds)

    print(f"  MLP one-class — "
          f"P={m['precision']:.4f}  "
          f"R={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}")

    return preds, scores_test, m




# ─── COMBINAISON LSTM OR MLP ──────────────────────────────────

def combine_lstm_mlp(lstm_preds_session, mlp_preds_session,
                     test_lbls_session):
    """
    Détection finale = LSTM OR MLP au niveau session.
    """
    y_true    = np.array(test_lbls_session)
    combined  = np.logical_or(
        lstm_preds_session, mlp_preds_session
    ).astype(int)
    return compute_metrics(y_true, combined)


# ─── LSTM SESSION-LEVEL DEPUIS in_topk ───────────────────────

def get_lstm_session_preds(k=5):
    """
    Récupère les prédictions LSTM session-level pour un K donné,
    en utilisant les résultats sauvegardés par 04_evaluate.py.
    """
    in_topk   = np.load(os.path.join(RESULTS_DIR, "in_topk.npy"))
    block_ids = np.load(os.path.join(DATA_PATH,   "block_ids_test.npy"))
    y_ano     = np.load(os.path.join(DATA_PATH,   "y_ano_test.npy"))

    window_pred = (~in_topk[:, k - 1]).astype(np.int32)

    sort_idx    = np.argsort(block_ids, kind="stable")
    bid_sorted  = block_ids[sort_idx]
    pred_sorted = window_pred[sort_idx]
    ano_sorted  = y_ano[sort_idx]

    _, first_occ = np.unique(bid_sorted, return_index=True)
    session_pred = np.maximum.reduceat(pred_sorted, first_occ).clip(0, 1)
    session_true = np.maximum.reduceat(ano_sorted,  first_occ).clip(0, 1)

    return session_pred, session_true


# ─── VISUALISATION COMPARATIVE ────────────────────────────────

def plot_comparison(results_dict, save_path):
    """
    Figure 8 : Comparaison LSTM seul vs MLP seul vs LSTM+MLP.
    """
    systems   = list(results_dict.keys())
    precision = [results_dict[s]["precision"] for s in systems]
    recall    = [results_dict[s]["recall"]    for s in systems]
    f1        = [results_dict[s]["f1"]        for s in systems]

    x = np.arange(len(systems))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, precision, w, label="Précision",
           color="#2563EB", alpha=0.85)
    ax.bar(x,     recall,    w, label="Rappel",
           color="#EA580C", alpha=0.85)
    ax.bar(x + w, f1,        w, label="F1-score",
           color="#16A34A", alpha=0.85)

    for i, v in enumerate(f1):
        ax.text(x[i] + w, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=9,
                color="#15803D", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(
        "Comparaison des systèmes — Session-level\n"
        "(LSTM seul vs MLP seul vs LSTM+MLP combiné)",
        fontsize=12)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def plot_pr_comparison(results_dict, save_path):
    """
    Figure 9 : Scatter Precision/Recall pour tous les systèmes.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    colors_map = {
        "LSTM K=1"    : "#93C5FD",
        "LSTM K=3"    : "#60A5FA",
        "LSTM K=5"    : "#2563EB",
        "LSTM K=10"   : "#1D4ED8",
        "LSTM K=20"   : "#1E3A8A",
        "MLP seul"    : "#F87171",
        "LSTM+MLP OR" : "#16A34A",
    }

    for name, m in results_dict.items():
        color = colors_map.get(name, "#6B7280")
        ax.scatter(m["recall"], m["precision"],
                   s=150, zorder=5, color=color,
                   label=f"{name} (F1={m['f1']:.3f})")
        ax.annotate(name,
                    xy=(m["recall"], m["precision"]),
                    xytext=(m["recall"] + 0.01,
                            m["precision"] + 0.01),
                    fontsize=8, color=color)

    ax.set_xlabel("Rappel (session-level)")
    ax.set_ylabel("Précision (session-level)")
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Espace Précision/Rappel — Tous les systèmes\n"
        "(session-level)", fontsize=12)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


# ─── TABLEAU RÉCAPITULATIF ────────────────────────────────────

def print_final_table(results_dict):
    print("\n" + "="*70)
    print("  TABLEAU FINAL — Comparaison complète session-level")
    print("="*70)
    print(f"  {'Système':<20} {'Precision':>10} {'Recall':>8} "
          f"{'F1':>8} {'FPR':>8}")
    print("─" * 58)
    for name, m in results_dict.items():
        marker = "  ← meilleur F1" if m["f1"] == max(
            v["f1"] for v in results_dict.values()) else ""
        print(f"  {name:<20} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} "
              f"{m['fpr']:>8.4f}{marker}")
    print("="*70)


# ─── MAIN ─────────────────────────────────────────────────────

def main():
    print("="*55)
    print("DeepLog — MLP Comptage + Comparaison finale")
    print("="*55)
    print(f"[INFO] Dispositif : {DEVICE}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Charger les sessions ───────────────────────────────
    sequences, labels, block_ids = load_sessions_from_csv()

    # ── 2. Split cohérent ─────────────────────────────────────
    (train_seqs, train_lbls,
     test_seqs,  test_lbls,
     test_bids) = split_sessions(sequences, labels, block_ids)

    test_lbls_arr = np.array(test_lbls)

    # ── 3. Entraîner MLP (centroïde one-class) ────────────────
    centroid, std = fit_oneclass_centroid(train_seqs, train_lbls)

    # ── 4. Prédictions MLP ────────────────────────────────────
    print("\n[INFO] Inférence MLP one-class sur le test...")
    best_mlp_preds, mlp_scores, m_mlp = predict_mlp_oneclass(
        centroid, std, train_seqs, test_seqs, test_lbls)

    # ── 5. Résultats LSTM depuis in_topk ──────────────────────
    # IMPORTANT : on charge les block_ids du test LSTM
    # et on réaligne le MLP sur les mêmes sessions
    print("\n[INFO] Chargement résultats LSTM session-level...")

    in_topk        = np.load(os.path.join(RESULTS_DIR, "in_topk.npy"))
    block_ids_lstm = np.load(os.path.join(DATA_PATH, "block_ids_test.npy"))
    y_ano_lstm     = np.load(os.path.join(DATA_PATH, "y_ano_test.npy"))

    # Calcul session-level LSTM pour tous les K
    results_all = {}

    def lstm_session(k):
        window_pred = (~in_topk[:, k-1]).astype(np.int32)
        sort_idx    = np.argsort(block_ids_lstm, kind="stable")
        bid_sorted  = block_ids_lstm[sort_idx]
        pred_sorted = window_pred[sort_idx]
        ano_sorted  = y_ano_lstm[sort_idx]
        _, first_occ = np.unique(bid_sorted, return_index=True)
        s_pred = np.maximum.reduceat(pred_sorted, first_occ).clip(0,1)
        s_true = np.maximum.reduceat(ano_sorted,  first_occ).clip(0,1)
        return s_pred, s_true

    for k in [1, 3, 5, 10, 20]:
        s_pred, s_true = lstm_session(k)
        m = compute_metrics(s_true, s_pred)
        results_all[f"LSTM K={k}"] = m
        print(f"  LSTM K={k} — "
              f"P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}")

    # ── 6. Combinaison LSTM OR MLP alignée par block_id ──────
    print("\n[INFO] Combinaison LSTM(K=5) OR MLP (alignement block_id)...")

    # Charger les block_ids LSTM (entiers) et reconstruire
    # la correspondance avec les block_ids CSV (strings)
    in_topk        = np.load(os.path.join(RESULTS_DIR, "in_topk.npy"))
    block_ids_lstm = np.load(os.path.join(DATA_PATH, "block_ids_test.npy"))
    y_ano_lstm     = np.load(os.path.join(DATA_PATH, "y_ano_test.npy"))

    # Session-level LSTM K=5
    window_pred  = (~in_topk[:, 4]).astype(np.int32)
    sort_idx     = np.argsort(block_ids_lstm, kind="stable")
    bid_sorted   = block_ids_lstm[sort_idx]
    pred_sorted  = window_pred[sort_idx]
    ano_sorted   = y_ano_lstm[sort_idx]
    _, first_occ = np.unique(bid_sorted, return_index=True)
    unique_bids_lstm = bid_sorted[first_occ]
    lstm_s_pred  = np.maximum.reduceat(pred_sorted, first_occ).clip(0,1)
    lstm_s_true  = np.maximum.reduceat(ano_sorted,  first_occ).clip(0,1)

    # Mapping block_id_int → index dans unique_bids_lstm
    bid_int_to_idx = {bid: i for i, bid in enumerate(unique_bids_lstm)}

    # Charger le mapping block_id_string → int
    # depuis 01_data_preparation (block_to_int)
    df_csv = pd.read_csv(os.path.join(CSV_PATH, "hdfs_sequences.csv"))
    unique_blocks_csv = df_csv["block_id"].unique()
    block_str_to_int  = {b: i for i, b in enumerate(unique_blocks_csv)}

    # MLP session-level : une prédiction par session test (string)
    # Construire mapping block_id_string → mlp_pred
    mlp_session_map = {}
    for bid_str, pred in zip(test_bids, best_mlp_preds):
        mlp_session_map[bid_str] = int(pred)

    # Aligner : pour chaque session LSTM, trouver la prédiction MLP
    mlp_aligned  = np.zeros(len(unique_bids_lstm), dtype=np.int32)
    n_found = 0
    for bid_str, bid_int in block_str_to_int.items():
        if bid_int in bid_int_to_idx and bid_str in mlp_session_map:
            idx = bid_int_to_idx[bid_int]
            mlp_aligned[idx] = mlp_session_map[bid_str]
            n_found += 1

    print(f"  Sessions alignées : {n_found:,} / {len(unique_bids_lstm):,}")

    combined   = np.logical_or(lstm_s_pred, mlp_aligned).astype(int)
    m_combined = compute_metrics(lstm_s_true, combined)
    results_all["LSTM+MLP OR"] = m_combined
    results_all["MLP seul"]    = m_mlp

    print(f"  Combiné OR — "
          f"P={m_combined['precision']:.4f}  "
          f"R={m_combined['recall']:.4f}  "
          f"F1={m_combined['f1']:.4f}")

    # ── 7. Tableau final ──────────────────────────────────────
    print_final_table(results_all)

    # ── 8. Figures ────────────────────────────────────────────
    print("\n📊 Génération des figures comparatives...")

    results_main = {
        "LSTM K=5\n(seul)" : results_all["LSTM K=5"],
        "MLP\n(seul)"      : results_all["MLP seul"],
        "LSTM+MLP\n(OR)"   : results_all["LSTM+MLP OR"],
    }
    plot_comparison(
        results_main,
        os.path.join(FIGURES_DIR, "fig8_comparison.pdf"))

    plot_pr_comparison(
        results_all,
        os.path.join(FIGURES_DIR, "fig9_pr_comparison.pdf"))

   # ── 9. Sauvegarde ─────────────────────────────────────────
    np.save(os.path.join(RESULTS_DIR, "results_all.npy"), results_all)
    np.save(os.path.join(RESULTS_DIR, "mlp_centroid.npy"), centroid)
    np.save(os.path.join(RESULTS_DIR, "mlp_std.npy"),      std)
    print(f"\n✅ Terminé. Figures : {FIGURES_DIR}")

if __name__ == "__main__":
    main()
