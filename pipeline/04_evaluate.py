"""
DeepLog - Étape 4 : Inférence & Stratégies de détection
=========================================================
v3 : session-level évalué sur tous les K
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
# Chemins relatifs au dossier racine du projet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import importlib.util

spec = importlib.util.spec_from_file_location(
    "model",
    os.path.join(os.path.dirname(__file__), "02_model.py")
)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

DeepLogLSTM = model_module.DeepLogLSTM


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CFG = {
    "data_path"      : "data/processed/",
    "checkpoint_dir" : "checkpoints/",
    "results_dir"    : "results/",
    "batch_size"     : 8192,
    "num_workers"    : 2,
    "k_values" : [1, 3, 5, 7, 9, 10, 20],
    "thresholds"     : np.linspace(0.001, 0.5, 200),
}
# ──────────────────────────────────────────────────────────────────────────────


class TestDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_model(cfg: dict, device: torch.device) -> DeepLogLSTM:
    ckpt_path = os.path.join(cfg["checkpoint_dir"], "deeplog_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = ckpt.get("cfg", cfg)
    model = DeepLogLSTM(
        vocab_size  = saved_cfg["vocab_size"],
        embed_dim   = saved_cfg["embed_dim"],
        hidden_size = saved_cfg["hidden_size"],
        num_layers  = saved_cfg["num_layers"],
        dropout     = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  ✅ Modèle chargé (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model


@torch.no_grad()
def run_inference(
    model: DeepLogLSTM,
    loader: DataLoader,
    device: torch.device,
    k_max: int = 20,
) -> tuple:
    N = len(loader.dataset)
    true_probs = np.empty(N, dtype=np.float16)
    in_topk    = np.empty((N, k_max), dtype=bool)

    offset = 0
    for X_batch, y_batch in tqdm(loader, desc="  Inférence", ncols=80):
        bsz = len(X_batch)
        X_batch     = X_batch.to(device, non_blocking=True)
        y_batch_dev = y_batch.to(device, non_blocking=True)

        logits = model(X_batch)
        probs  = F.softmax(logits.float(), dim=-1)

        true_p = probs[torch.arange(bsz, device=device), y_batch_dev]
        true_probs[offset: offset + bsz] = true_p.cpu().numpy().astype(np.float16)

        _, topk_idx   = torch.topk(probs, k_max, dim=-1)
        y_exp         = y_batch_dev.unsqueeze(1).expand(-1, k_max)
        match         = (topk_idx == y_exp)
        in_topk_batch = match.cummax(dim=1).values
        in_topk[offset: offset + bsz] = in_topk_batch.cpu().numpy()

        offset += bsz

    return true_probs, in_topk


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = np.logical_and(y_pred == 1, y_true == 1).sum()
    fp = np.logical_and(y_pred == 1, y_true == 0).sum()
    fn = np.logical_and(y_pred == 0, y_true == 1).sum()
    prec   = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1     = 2 * prec * recall / (prec + recall + 1e-10)
    return {"precision": prec, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def evaluate_topk(in_topk, y_ano, k_values):
    results = []
    print(f"\n{'K':>4} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("─" * 35)
    for k in k_values:
        y_pred = (~in_topk[:, k - 1]).astype(np.int32)
        m = compute_metrics(y_ano, y_pred)
        results.append({"k": k, **m})
        print(f"{k:>4} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
    return results


def evaluate_threshold(true_probs, y_ano, thresholds):
    results = np.zeros((len(thresholds), 3), dtype=np.float32)
    for i, tau in enumerate(thresholds):
        y_pred = (true_probs < tau).astype(np.int32)
        m = compute_metrics(y_ano, y_pred)
        results[i] = [m["precision"], m["recall"], m["f1"]]
    return results


def evaluate_topk_plus_threshold(in_topk, true_probs, y_ano, k, thresholds):
    not_in_topk = ~in_topk[:, k - 1]
    results_or  = np.zeros((len(thresholds), 3), dtype=np.float32)
    results_and = np.zeros((len(thresholds), 3), dtype=np.float32)
    for i, tau in enumerate(thresholds):
        low_prob = true_probs < tau
        y_or  = (not_in_topk | low_prob).astype(np.int32)
        y_and = (not_in_topk & low_prob).astype(np.int32)
        m = compute_metrics(y_ano, y_or)
        results_or[i]  = [m["precision"], m["recall"], m["f1"]]
        m = compute_metrics(y_ano, y_and)
        results_and[i] = [m["precision"], m["recall"], m["f1"]]
    return results_or, results_and


def find_best_threshold(results, thresholds):
    best_idx = results[:, 2].argmax()
    return {
        "threshold": thresholds[best_idx],
        "precision": results[best_idx, 0],
        "recall"   : results[best_idx, 1],
        "f1"       : results[best_idx, 2],
    }


def evaluate_session_level(
    in_topk: np.ndarray,
    y_ano: np.ndarray,
    block_ids: np.ndarray,
    k: int,
) -> dict:
    """
    Agrégation session-level vectorisée — He et al. 2016.
    Une session est anormale si AU MOINS UNE fenêtre est anormale (OR).
    """
    window_pred = (~in_topk[:, k - 1]).astype(np.int32)

    sort_idx    = np.argsort(block_ids, kind="stable")
    bid_sorted  = block_ids[sort_idx]
    pred_sorted = window_pred[sort_idx]
    ano_sorted  = y_ano[sort_idx]

    _, first_occ = np.unique(bid_sorted, return_index=True)

    session_pred = np.maximum.reduceat(pred_sorted, first_occ).clip(0, 1)
    session_true = np.maximum.reduceat(ano_sorted,  first_occ).clip(0, 1)

    return compute_metrics(session_true, session_pred)


def main():
    print("=" * 60)
    print("DeepLog — Inférence & Évaluation v3")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG["results_dir"], exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────
    print("\n📂 Chargement des données de test...")
    X_test    = np.load(os.path.join(CFG["data_path"], "X_test.npy"))
    y_test    = np.load(os.path.join(CFG["data_path"], "y_test.npy"))
    y_ano     = np.load(os.path.join(CFG["data_path"], "y_ano_test.npy"))
    block_ids = np.load(os.path.join(CFG["data_path"], "block_ids_test.npy"))

    print(f"   X_test    : {X_test.shape}")
    print(f"   block_ids : {block_ids.shape}  "
          f"sessions={len(np.unique(block_ids)):,}")
    print(f"   Anomalies : {y_ano.sum():,} ({100*y_ano.mean():.1f}%)")

    assert len(block_ids) == len(X_test), \
        f"MISMATCH block_ids={len(block_ids)} vs X_test={len(X_test)}"

    test_ds = TestDataset(X_test, y_test)
    loader  = DataLoader(
        test_ds,
        batch_size  = CFG["batch_size"],
        shuffle     = False,
        num_workers = CFG["num_workers"],
        pin_memory  = (device.type == "cuda"),
    )

    # ── Modèle ────────────────────────────────────────────────────
    model = load_model(CFG, device)

    # ── Inférence ─────────────────────────────────────────────────
    k_max = max(CFG["k_values"])
    print(f"\n⚙️  Inférence sur {len(test_ds):,} séquences...")
    true_probs, in_topk = run_inference(model, loader, device, k_max)
    np.save(os.path.join(CFG["results_dir"], "true_probs.npy"), true_probs)
    np.save(os.path.join(CFG["results_dir"], "in_topk.npy"),   in_topk)
    print("  ✅ Scores bruts sauvegardés.")

    # ── Stratégie 1 : Top-K fenêtre ───────────────────────────────
    print("\n📊 [Stratégie 1] Top-K — Fenêtre-level")
    topk_results = evaluate_topk(in_topk, y_ano, CFG["k_values"])

    # ── Stratégie 2 : Seuil fenêtre ───────────────────────────────
    print("\n📊 [Stratégie 2] Seuil de probabilité — Fenêtre-level")
    thresh_results = evaluate_threshold(true_probs, y_ano, CFG["thresholds"])
    best_thresh    = find_best_threshold(thresh_results, CFG["thresholds"])
    print(f"   Meilleur seuil τ = {best_thresh['threshold']:.4f}")
    print(f"   P={best_thresh['precision']:.4f}  "
          f"R={best_thresh['recall']:.4f}  "
          f"F1={best_thresh['f1']:.4f}")

    # ── Stratégie 3 : Combinée fenêtre ────────────────────────────
    print("\n📊 [Stratégie 3] Combinée Top-5 + seuil — Fenêtre-level")
    res_or, res_and = evaluate_topk_plus_threshold(
        in_topk, true_probs, y_ano, k=5, thresholds=CFG["thresholds"]
    )
    best_or  = find_best_threshold(res_or,  CFG["thresholds"])
    best_and = find_best_threshold(res_and, CFG["thresholds"])
    print(f"   [OR]  τ={best_or['threshold']:.4f}  "
          f"P={best_or['precision']:.4f}  "
          f"R={best_or['recall']:.4f}  "
          f"F1={best_or['f1']:.4f}")
    print(f"   [AND] τ={best_and['threshold']:.4f}  "
          f"P={best_and['precision']:.4f}  "
          f"R={best_and['recall']:.4f}  "
          f"F1={best_and['f1']:.4f}")

    # ── Stratégie 4 : Session-level tous les K ────────────────────
    print("\n📊 [Stratégie 4] Agrégation Session-Level — tous les K")
    print(f"\n{'K':>4} {'Precision':>10} {'Recall':>8} {'F1':>8} "
          f"{'TP':>8} {'FP':>8} {'FN':>8}")
    print("─" * 60)
    session_results = []
    for k in CFG["k_values"]:
        m = evaluate_session_level(in_topk, y_ano, block_ids, k=k)
        session_results.append({"k": k, **m})
        print(f"{k:>4} {m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {int(m['tp']):>8} "
              f"{int(m['fp']):>8} {int(m['fn']):>8}")

    # ── Sauvegarde ────────────────────────────────────────────────
    np.save(os.path.join(CFG["results_dir"], "topk_results.npy"),     topk_results)
    np.save(os.path.join(CFG["results_dir"], "thresh_results.npy"),   thresh_results)
    np.save(os.path.join(CFG["results_dir"], "thresholds.npy"),       CFG["thresholds"])
    np.save(os.path.join(CFG["results_dir"], "res_or.npy"),           res_or)
    np.save(os.path.join(CFG["results_dir"], "res_and.npy"),          res_and)
    np.save(os.path.join(CFG["results_dir"], "session_results.npy"),  session_results)

    print("\n✅ Tous les résultats sauvegardés dans", CFG["results_dir"])


if __name__ == "__main__":
    main()