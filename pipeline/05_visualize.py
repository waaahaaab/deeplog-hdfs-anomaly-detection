"""
DeepLog - Visualisation complète pour mémoire
===============================================
v4 : pipeline LogPai vocab=30, résultats mis à jour
     Génère fig1 à fig7 + tableau récapitulatif console
"""

import os
import numpy as np
import matplotlib.pyplot as plt

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

COLORS = {
    "topk"    : "#2563EB",
    "thresh"  : "#DC2626",
    "or"      : "#16A34A",
    "and"     : "#9333EA",
    "normal"  : "#60A5FA",
    "anomaly" : "#F87171",
}

RESULTS_DIR = "results/"
DATA_PATH   = "data/processed/"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures/")

# ─── DONNÉES FENÊTRE-LEVEL (issues de 04_evaluate.py) ────────────────────────
K_VALUES   = [1,     3,      5,      10,     20    ]
F1_WINDOW  = [0.270, 0.149,  0.099,  0.052,  0.038 ]
P_WINDOW   = [0.320, 0.578,  0.984,  0.993,  0.993 ]
R_WINDOW   = [0.234, 0.085,  0.052,  0.027,  0.019 ]
# ──────────────────────────────────────────────────────────────────────────────


# ─── CALCUL SESSION-LEVEL EN LIVE ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    tp = np.logical_and(y_pred == 1, y_true == 1).sum()
    fp = np.logical_and(y_pred == 1, y_true == 0).sum()
    fn = np.logical_and(y_pred == 0, y_true == 1).sum()
    prec   = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1     = 2 * prec * recall / (prec + recall + 1e-10)
    return float(prec), float(recall), float(f1)


def compute_all_session_metrics(in_topk, y_ano, block_ids, k_values):
    results = {}
    for k in k_values:
        window_pred = (~in_topk[:, k - 1]).astype(np.int32)
        sort_idx    = np.argsort(block_ids, kind="stable")
        bid_sorted  = block_ids[sort_idx]
        pred_sorted = window_pred[sort_idx]
        ano_sorted  = y_ano[sort_idx]
        _, first_occ = np.unique(bid_sorted, return_index=True)
        session_pred = np.maximum.reduceat(pred_sorted, first_occ).clip(0, 1)
        session_true = np.maximum.reduceat(ano_sorted,  first_occ).clip(0, 1)
        p, r, f1 = compute_metrics(session_true, session_pred)
        results[k] = {"precision": p, "recall": r, "f1": f1}
    return results


# ─── FIGURES ──────────────────────────────────────────────────────────────────

def plot_f1_vs_k_window(save_path: str):
    """Figure 1 : F1, Precision, Recall fenêtre-level vs K."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(K_VALUES, F1_WINDOW, "o-",  color=COLORS["topk"],  label="F1-score",  lw=2)
    ax.plot(K_VALUES, P_WINDOW,  "s--", color="#F59E0B",        label="Precision", lw=1.5, alpha=0.8)
    ax.plot(K_VALUES, R_WINDOW,  "^--", color="#6B7280",        label="Recall",    lw=1.5, alpha=0.8)

    best_idx = int(np.argmax(F1_WINDOW))
    best_k   = K_VALUES[best_idx]
    best_f1  = F1_WINDOW[best_idx]
    ax.axvline(best_k, color=COLORS["topk"], lw=1, ls=":", alpha=0.5)
    ax.annotate(f"K={best_k}\nF1={best_f1:.3f}",
                xy=(best_k, best_f1),
                xytext=(best_k + 1, best_f1 + 0.05),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

    ax.set_xlabel("Valeur de K")
    ax.set_ylabel("Score")
    ax.set_title("DeepLog — Métriques fenêtre-level vs K")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def plot_precision_recall_curve(thresh_results, save_path: str):
    """Figure 2 : Courbe Precision-Recall (seuil de probabilité)."""
    prec = thresh_results[:, 0]
    rec  = thresh_results[:, 1]
    f1s  = thresh_results[:, 2]

    sort_idx = np.argsort(rec)
    prec_s   = prec[sort_idx]
    rec_s    = rec[sort_idx]
    auc_pr = np.trapezoid(prec_s, rec_s)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(rec_s, prec_s, c=f1s[sort_idx],
                    cmap="RdYlGn", s=8, alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="F1-score")

    best_idx = f1s.argmax()
    ax.scatter(rec[best_idx], prec[best_idx], c="red", s=120,
               zorder=5, marker="*",
               label=f"Optimal (F1={f1s[best_idx]:.3f})")

    y_ano = np.load(os.path.join(DATA_PATH, "y_ano_test.npy"))
    baseline = y_ano.mean()
    ax.axhline(baseline, color="gray", lw=1, ls="--",
               label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Courbe Precision-Recall (AUC-PR ≈ {auc_pr:.3f})")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def plot_f1_vs_threshold(thresh_results, res_or, res_and,
                          thresholds, save_path: str):
    """Figure 3 : F1 vs τ pour les trois stratégies."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, thresh_results[:, 2],
            color=COLORS["thresh"], lw=2, label="Seuil probabilité seul")
    ax.plot(thresholds, res_or[:, 2],
            color=COLORS["or"],    lw=2, label="Top-5 OR seuil")
    ax.plot(thresholds, res_and[:, 2],
            color=COLORS["and"],   lw=2, label="Top-5 AND seuil")

    for results, color in [(thresh_results, COLORS["thresh"]),
                            (res_or,         COLORS["or"]),
                            (res_and,        COLORS["and"])]:
        best_idx = results[:, 2].argmax()
        ax.axvline(thresholds[best_idx], color=color, lw=1, ls=":", alpha=0.5)

    ax.set_xlabel("Seuil de probabilité τ")
    ax.set_ylabel("F1-score")
    ax.set_title("F1 en fonction du seuil τ — Comparaison des stratégies")
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def plot_probability_distribution(true_probs, save_path: str):
    """Figure 4 : Distribution P(vrai event) normal vs anomalie."""
    y_ano = np.load(os.path.join(DATA_PATH, "y_ano_test.npy"))
    probs = true_probs.astype(np.float32)

    norm_probs = probs[y_ano == 0]
    abn_probs  = probs[y_ano == 1]

    if len(norm_probs) > 500_000:
        rng = np.random.default_rng(42)
        norm_probs = rng.choice(norm_probs, 500_000, replace=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 0.5, 100)
    ax.hist(norm_probs, bins=bins, density=True, alpha=0.6,
            color=COLORS["normal"],  label="Normal")
    ax.hist(abn_probs,  bins=bins, density=True, alpha=0.6,
            color=COLORS["anomaly"], label="Anomalie")

    ax.set_xlabel("P(prochain event | contexte)")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution des probabilités — Normal vs Anomalie")
    ax.legend()

    ax.text(0.98, 0.95,
            f"Normal  μ={norm_probs.mean():.3f}\n"
            f"Anomalie μ={abn_probs.mean():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def plot_session_vs_window(session_results: dict, save_path: str):
    """
    Figure 5 : Comparaison fenêtre-level vs session-level.
    Résultat principal : K optimal selon F1 session.
    """
    # Trouver K optimal session
    best_k   = max(session_results, key=lambda k: session_results[k]["f1"])
    best_m   = session_results[best_k]
    best_f1w = F1_WINDOW[K_VALUES.index(best_k)]

    strategies = [
        "Top-1\n(fenêtre)",
        f"Top-{K_VALUES[np.argmax(F1_WINDOW)]}\n(fenêtre, best)",
        "Seuil τ\n(fenêtre)",
        "Combinée OR\n(fenêtre)",
        f"Session K={best_k}\n* optimal",
    ]

    # Charger best threshold pour OR
    thresh_results = np.load(os.path.join(RESULTS_DIR, "thresh_results.npy"))
    res_or         = np.load(os.path.join(RESULTS_DIR, "res_or.npy"))
    thresholds     = np.load(os.path.join(RESULTS_DIR, "thresholds.npy"))

    best_thresh_idx = thresh_results[:, 2].argmax()
    best_or_idx     = res_or[:, 2].argmax()

    precision = [P_WINDOW[0],
                 P_WINDOW[np.argmax(F1_WINDOW)],
                 float(thresh_results[best_thresh_idx, 0]),
                 float(res_or[best_or_idx, 0]),
                 best_m["precision"]]
    recall    = [R_WINDOW[0],
                 R_WINDOW[np.argmax(F1_WINDOW)],
                 float(thresh_results[best_thresh_idx, 1]),
                 float(res_or[best_or_idx, 1]),
                 best_m["recall"]]
    f1        = [F1_WINDOW[0],
                 max(F1_WINDOW),
                 float(thresh_results[best_thresh_idx, 2]),
                 float(res_or[best_or_idx, 2]),
                 best_m["f1"]]

    x = np.arange(len(strategies))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, precision, w, label="Précision", color="#2563EB", alpha=0.85)
    ax.bar(x,     recall,    w, label="Rappel",    color="#EA580C", alpha=0.85)
    ax.bar(x + w, f1,        w, label="F1-score",  color="#16A34A", alpha=0.85)

    for i, v in enumerate(f1):
        weight = "bold" if i == 4 else "normal"
        color  = "#15803D" if i == 4 else "#16A34A"
        ax.text(x[i] + w, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=8.5, color=color, fontweight=weight)

    ax.axvspan(3.5, 4.5, alpha=0.07, color="#16A34A", zorder=0)
    ax.axvline(x=3.5, color="#6B7280", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.text(1.5, 1.04, "Évaluation fenêtre-level",
            ha="center", fontsize=9, color="#6B7280",
            transform=ax.get_xaxis_transform())
    ax.text(4.0, 1.04, "Session-level",
            ha="center", fontsize=9, color="#16A34A", fontweight="bold",
            transform=ax.get_xaxis_transform())

    gain = f1[4] - f1[3]
    ax.annotate("",
        xy=(x[4] + w, f1[4] + 0.04),
        xytext=(x[3] + w, f1[3] + 0.04),
        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.8))
    ax.text(3.85, f1[4] + 0.06, f"+{gain:.3f}",
            ha="center", fontsize=9, color="#16A34A", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title(
        "DeepLog HDFS — Impact de l'agrégation session-level sur les performances",
        fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def plot_topk_window_vs_session(session_results: dict, save_path: str):
    """Figure 6 : F1 fenêtre vs F1 session pour chaque K."""
    f1_session = [session_results[k]["f1"] for k in K_VALUES]
    best_k     = max(session_results, key=lambda k: session_results[k]["f1"])

    x = np.arange(len(K_VALUES))
    w = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, F1_WINDOW,  w, label="F1 fenêtre-level",
           color="#6B7280", alpha=0.75)
    ax.bar(x + w/2, f1_session, w, label="F1 session-level",
           color="#16A34A", alpha=0.85)

    for i, (fw, fs) in enumerate(zip(F1_WINDOW, f1_session)):
        ax.text(x[i] - w/2, fw + 0.008, f"{fw:.3f}",
                ha="center", fontsize=8, color="#6B7280")
        weight = "bold" if K_VALUES[i] == best_k else "normal"
        ax.text(x[i] + w/2, fs + 0.008, f"{fs:.3f}",
                ha="center", fontsize=8, color="#15803D", fontweight=weight)

        gain = fs - fw
        pct  = gain / (fw + 1e-10) * 100
        ax.annotate("",
            xy=(x[i] + w/2, fs + 0.04),
            xytext=(x[i] - w/2, fw + 0.04),
            arrowprops=dict(arrowstyle="->", color="#EA580C",
                            lw=1.2, connectionstyle="arc3,rad=-0.25"))
        ax.text(x[i], max(fw, fs) + 0.07, f"+{pct:.0f}%",
                ha="center", fontsize=7.5, color="#EA580C", fontweight="bold")

    ax.axvspan(x[K_VALUES.index(best_k)] - 0.5,
               x[K_VALUES.index(best_k)] + 0.5,
               alpha=0.06, color="#16A34A", zorder=0)
    ax.text(x[K_VALUES.index(best_k)], 0.02, "optimal",
            ha="center", fontsize=8, color="#16A34A", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES], fontsize=10)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("F1-score")
    ax.set_title(
        "DeepLog HDFS — F1 fenêtre vs session selon K\n"
        "(agrégation OR session-level, He et al. 2016)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def plot_precision_recall_session(session_results: dict, save_path: str):
    """Figure 7 : Trade-off Precision/Recall session-level selon K."""
    p_session = [session_results[k]["precision"] for k in K_VALUES]
    r_session = [session_results[k]["recall"]    for k in K_VALUES]
    f1_session= [session_results[k]["f1"]        for k in K_VALUES]
    best_k    = max(session_results, key=lambda k: session_results[k]["f1"])
    best_f1   = session_results[best_k]["f1"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(r_session, p_session, c=f1_session,
                    cmap="RdYlGn", s=180, zorder=5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="F1 session-level")

    offsets = {1: (-0.06, 0.015), 3: (-0.06, 0.015),
               5: (0.01, 0.015), 10: (0.01, -0.025), 20: (0.01, 0.015)}
    for k, p, r in zip(K_VALUES, p_session, r_session):
        dx, dy = offsets.get(k, (0.01, 0.015))
        weight = "bold" if k == best_k else "normal"
        ax.text(r + dx, p + dy, f"K={k}", fontsize=9,
                fontweight=weight,
                color="#15803D" if k == best_k else "#374151")

    recall_range = np.linspace(0.01, 1.0, 300)
    prec_iso = best_f1 * recall_range / (
        2 * recall_range - best_f1 + 1e-10)
    mask = (prec_iso >= 0) & (prec_iso <= 1)
    ax.plot(recall_range[mask], prec_iso[mask],
            "--", color="#16A34A", lw=1.2, alpha=0.5,
            label=f"Iso-F1 = {best_f1:.3f}")

    ax.set_xlabel("Rappel (session-level)")
    ax.set_ylabel("Précision (session-level)")
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Trade-off Précision/Rappel session-level selon K\n"
        "(contexte opérationnel SOC)", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {save_path}")


def print_summary_table(session_results: dict,
                         thresh_results, thresholds,
                         res_or, res_and):
    """Tableau récapitulatif console."""
    best_k = max(session_results, key=lambda k: session_results[k]["f1"])

    print("\n" + "=" * 70)
    print("  TABLEAU RÉCAPITULATIF — Fenêtre-level")
    print("=" * 70)
    print(f"  {'K':>4} {'F1 fenêtre':>12} {'Precision':>10} {'Recall':>8}")
    print("─" * 40)
    for k, fw, p, r in zip(K_VALUES, F1_WINDOW, P_WINDOW, R_WINDOW):
        print(f"  {k:>4} {fw:>12.3f} {p:>10.3f} {r:>8.3f}")

    best_thresh_idx = thresh_results[:, 2].argmax()
    best_or_idx     = res_or[:, 2].argmax()
    best_and_idx    = res_and[:, 2].argmax()
    print(f"\n  Seuil τ={thresholds[best_thresh_idx]:.3f} : "
          f"P={thresh_results[best_thresh_idx,0]:.3f}  "
          f"R={thresh_results[best_thresh_idx,1]:.3f}  "
          f"F1={thresh_results[best_thresh_idx,2]:.3f}")
    print(f"  OR    τ={thresholds[best_or_idx]:.3f}    : "
          f"P={res_or[best_or_idx,0]:.3f}  "
          f"R={res_or[best_or_idx,1]:.3f}  "
          f"F1={res_or[best_or_idx,2]:.3f}")

    print("\n" + "=" * 70)
    print("  TABLEAU RÉCAPITULATIF — Session-level")
    print("=" * 70)
    print(f"  {'K':>4} {'F1 session':>12} {'Precision':>10} {'Recall':>8}")
    print("─" * 40)
    for k in K_VALUES:
        m = session_results[k]
        marker = "  ← optimal" if k == best_k else ""
        print(f"  {k:>4} {m['f1']:>12.3f} "
              f"{m['precision']:>10.3f} {m['recall']:>8.3f}{marker}")
    print("=" * 70)


def main():
    print("=" * 60)
    print("DeepLog — Visualisation complète v4 (LogPai vocab=30)")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Chargement des résultats ──────────────────────────────────
    print("\n📂 Chargement des résultats...")
    in_topk        = np.load(os.path.join(RESULTS_DIR, "in_topk.npy"))
    true_probs     = np.load(os.path.join(RESULTS_DIR, "true_probs.npy"))
    thresh_results = np.load(os.path.join(RESULTS_DIR, "thresh_results.npy"))
    res_or         = np.load(os.path.join(RESULTS_DIR, "res_or.npy"))
    res_and        = np.load(os.path.join(RESULTS_DIR, "res_and.npy"))
    thresholds     = np.load(os.path.join(RESULTS_DIR, "thresholds.npy"))
    block_ids      = np.load(os.path.join(DATA_PATH,   "block_ids_test.npy"))
    y_ano          = np.load(os.path.join(DATA_PATH,   "y_ano_test.npy"))

    # ── Calcul session-level en live ──────────────────────────────
    print("⚙️  Calcul session-level pour tous les K...")
    session_results = compute_all_session_metrics(
        in_topk, y_ano, block_ids, K_VALUES)

    best_k = max(session_results, key=lambda k: session_results[k]["f1"])
    print(f"  K optimal session : K={best_k}  "
          f"F1={session_results[best_k]['f1']:.4f}  "
          f"P={session_results[best_k]['precision']:.4f}  "
          f"R={session_results[best_k]['recall']:.4f}")

    # ── Génération des figures ────────────────────────────────────
    print("\n📊 Génération des figures...")

    plot_f1_vs_k_window(
        os.path.join(FIGURES_DIR, "fig1_f1_vs_k.pdf"))

    plot_precision_recall_curve(
        thresh_results,
        os.path.join(FIGURES_DIR, "fig2_precision_recall.pdf"))

    plot_f1_vs_threshold(
        thresh_results, res_or, res_and, thresholds,
        os.path.join(FIGURES_DIR, "fig3_f1_vs_threshold.pdf"))

    plot_probability_distribution(
        true_probs,
        os.path.join(FIGURES_DIR, "fig4_prob_distribution.pdf"))

    plot_session_vs_window(
        session_results,
        os.path.join(FIGURES_DIR, "fig5_session_vs_window.pdf"))

    plot_topk_window_vs_session(
        session_results,
        os.path.join(FIGURES_DIR, "fig6_topk_window_vs_session.pdf"))

    plot_precision_recall_session(
        session_results,
        os.path.join(FIGURES_DIR, "fig7_precision_recall_session.pdf"))

    # ── Tableau récapitulatif ─────────────────────────────────────
    print_summary_table(
        session_results, thresh_results, thresholds, res_or, res_and)

    print(f"\n✅ 7 figures sauvegardées dans : {FIGURES_DIR}")


if __name__ == "__main__":
    main()