"""
DeepLog - Étape 1 : Préparation des données
============================================
v3 : adapté pour hdfs_sequences.csv + hdfs_templates.csv (LogPai officiel)
"""

import numpy as np
import pandas as pd
import os
import ast
from typing import Tuple, List

import sys

# Chemins relatifs au dossier racine du projet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_PATH   = "data/raw/"
CSV_PATH    = "data/csv/"
SAVE_PATH   = "data/processed/"
WINDOW_SIZE = 10

FILE_SEQUENCES = "hdfs_sequences.csv"
FILE_TEMPLATES = "hdfs_templates.csv"
# ──────────────────────────────────────────────────────────────────────────────


def load_templates(csv_path: str) -> dict:
    path = os.path.join(csv_path, FILE_TEMPLATES)
    print(f"[TEMPLATES] Lecture de {path}...")
    df = pd.read_csv(path)

    print(f"  Colonnes disponibles : {list(df.columns)}")
    print(f"  Nombre de templates  : {len(df)}")
    print(f"  Aperçu :\n{df.head(3)}")

    event_ids = sorted(df["EventId"].tolist())
    mapping = {eid: idx + 1 for idx, eid in enumerate(event_ids)}

    print(f"  Mapping créé : {len(mapping)} events → IDs 1 à {len(mapping)}")
    print(f"  Exemple : {list(mapping.items())[:5]}")

    return mapping


def load_sequences(csv_path: str, template_mapping: dict) -> Tuple[List, List, List, List]:
    path = os.path.join(csv_path, FILE_SEQUENCES)
    print(f"\n[SEQUENCES] Lecture de {path}...")
    df = pd.read_csv(path)

    print(f"  Colonnes disponibles : {list(df.columns)}")
    print(f"  Nombre de sessions   : {len(df)}")
    print(f"  Aperçu :\n{df.head(3)}")

    col_block    = _find_column(df, ["block_id", "BlockId", "Block"])
    col_sequence = _find_column(df, ["sequence", "EventSequence",
                                     "event_sequence", "Sequence", "EventId"])
    col_label    = _find_column(df, ["label", "Label", "Anomaly"])

    print(f"  Colonnes identifiées : block={col_block}, "
          f"seq={col_sequence}, label={col_label}")

    X_list, y_list, y_ano_list, block_id_list = [], [], [], []

    unique_blocks = df[col_block].unique()
    block_to_int  = {b: i for i, b in enumerate(unique_blocks)}

    n_skipped = 0

    for _, row in df.iterrows():

        # ── 1. Parser la séquence JSON ─────────────────────────────
        raw_seq = str(row[col_sequence]).strip()
        try:
            event_ids_str = ast.literal_eval(raw_seq)
        except (ValueError, SyntaxError):
            n_skipped += 1
            continue  # ← indenté DANS le except

        # ── 2. Convertir EventId → entier ──────────────────────────
        try:
            events = [template_mapping[e] for e in event_ids_str]
        except KeyError:
            n_skipped += 1
            continue

        # ── 3. Ignorer les séquences trop courtes ──────────────────
        if len(events) <= WINDOW_SIZE:
            n_skipped += 1
            continue

        # ── 4. Label de la session ─────────────────────────────────
        label_str  = str(row[col_label]).strip().lower()
        is_anomaly = label_str in ["anomaly", "1", "true", "abnormal"]

        block_int     = block_to_int[row[col_block]]
        n_windows     = len(events) - WINDOW_SIZE
        anomaly_start = 0 if is_anomaly else n_windows
        # ── 5. Générer les fenêtres ────────────────────────────────
        for i in range(n_windows):
            X_list.append(events[i: i + WINDOW_SIZE])
            y_list.append(events[i + WINDOW_SIZE])
            label = 1 if (is_anomaly and i >= anomaly_start) else 0
            y_ano_list.append(label)
            block_id_list.append(block_int)

    print(f"  Sessions ignorées : {n_skipped}")
    return X_list, y_list, y_ano_list, block_id_list


def _find_column(df: pd.DataFrame, candidates: list) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Aucune colonne trouvée parmi {candidates}. "
        f"Colonnes disponibles : {list(df.columns)}"
    )


def validate_vocab(X: np.ndarray, y: np.ndarray, vocab_size: int):
    max_x = X.max()
    max_y = y.max()
    print(f"  Token max dans X : {max_x}  (vocab_size={vocab_size})")
    print(f"  Token max dans y : {max_y}  (vocab_size={vocab_size})")
    assert max_x < vocab_size, f"Token {max_x} hors vocabulaire!"
    assert max_y < vocab_size, f"Token {max_y} hors vocabulaire!"
    print("  ✅ Vocabulaire validé.")


def split_train_test(
    X: List, y: List, y_ano: List, block_ids: List,
    train_ratio: float = 0.8
) -> Tuple:

    block_ids_arr = np.array(block_ids, dtype=np.int32)
    y_ano_arr     = np.array(y_ano,     dtype=np.int32)

    unique_blocks  = np.unique(block_ids_arr)
    
    # Identifier les blocs anormaux vectorisé
    anomaly_flags = np.zeros(len(unique_blocks), dtype=bool)
    for i, bid in enumerate(unique_blocks):
        if y_ano_arr[block_ids_arr == bid].any():
            anomaly_flags[i] = True

    normal_blocks  = unique_blocks[~anomaly_flags]
    anomaly_blocks = unique_blocks[anomaly_flags]

    print(f"\n  Sessions normales  : {len(normal_blocks):,}")
    print(f"  Sessions anormales : {len(anomaly_blocks):,}")

    # Split
    n_train = int(len(normal_blocks) * train_ratio)
    rng     = np.random.default_rng(42)
    rng.shuffle(normal_blocks)

    train_blocks        = normal_blocks[:n_train]
    test_normal_blocks  = normal_blocks[n_train:]
    test_anomaly_blocks = anomaly_blocks

    # Masques vectorisés — beaucoup plus rapide qu'une compréhension
    train_set       = set(train_blocks.tolist())
    test_normal_set = set(test_normal_blocks.tolist())
    test_ano_set    = set(test_anomaly_blocks.tolist())

    # Vectorisation via np.isin
    train_mask = np.isin(block_ids_arr, train_blocks)
    test_mask  = np.isin(block_ids_arr, 
                         np.concatenate([test_normal_blocks, 
                                        test_anomaly_blocks]))

    X_arr = np.array(X, dtype=np.int32)
    y_arr = np.array(y, dtype=np.int32)

    return (
        X_arr[train_mask],    y_arr[train_mask],
        X_arr[test_mask],     y_arr[test_mask],
        y_ano_arr[test_mask], block_ids_arr[test_mask]
    )

def main():
    print("=" * 60)
    print("DeepLog — Préparation données v3 (CSV LogPai officiel)")
    print("=" * 60)

    os.makedirs(SAVE_PATH, exist_ok=True)

    # ── Étape 1 : Templates ───────────────────────────────────────
    template_mapping = load_templates(CSV_PATH)
    VOCAB_SIZE = len(template_mapping) + 1
    print(f"\n  VOCAB_SIZE détecté : {VOCAB_SIZE}")

    # ── Étape 2 : Séquences ───────────────────────────────────────
    X, y, y_ano, block_ids = load_sequences(CSV_PATH, template_mapping)
    print(f"\n  Total fenêtres générées : {len(X):,}")

    if len(X) == 0:
        raise RuntimeError("❌ Aucune fenêtre générée — vérifier le CSV.")

    # ── Étape 3 : Split ───────────────────────────────────────────
    print("\n⏳ Split train/test...")
    X_train, y_train, X_test, y_test, y_ano_test, block_ids_test = \
        split_train_test(X, y, y_ano, block_ids, train_ratio=0.8)

    # ── Étape 4 : Validation ──────────────────────────────────────
    print("\n🔍 Validation vocabulaire (train) :")
    validate_vocab(X_train, y_train, VOCAB_SIZE)
    print("🔍 Validation vocabulaire (test) :")
    validate_vocab(X_test, y_test, VOCAB_SIZE)

    # ── Étape 5 : Sauvegarde ──────────────────────────────────────
    print("\n💾 Sauvegarde dans", SAVE_PATH)
    np.save(os.path.join(SAVE_PATH, "X_train.npy"),        X_train)
    np.save(os.path.join(SAVE_PATH, "y_train.npy"),        y_train)
    np.save(os.path.join(SAVE_PATH, "X_test.npy"),         X_test)
    np.save(os.path.join(SAVE_PATH, "y_test.npy"),         y_test)
    np.save(os.path.join(SAVE_PATH, "y_ano_test.npy"),     y_ano_test)
    np.save(os.path.join(SAVE_PATH, "block_ids_test.npy"), block_ids_test)
    np.save(os.path.join(SAVE_PATH, "vocab_size.npy"),     np.array([VOCAB_SIZE]))

    # ── Bilan ─────────────────────────────────────────────────────
    n_ano      = y_ano_test.sum()
    n_tot      = len(y_ano_test)
    n_sessions = len(np.unique(block_ids_test))
    n_sessions_ano = len([
        bid for bid in np.unique(block_ids_test)
        if y_ano_test[block_ids_test == bid].any()
    ])

    print("\n✅ TERMINÉ — Bilan :")
    print(f"   VOCAB_SIZE       : {VOCAB_SIZE}")
    print(f"   X_train          : {X_train.shape}")
    print(f"   X_test           : {X_test.shape}")
    print(f"   Sessions test    : {n_sessions:,}")
    print(f"   Sessions ano     : {n_sessions_ano:,}")
    print(f"   Fenêtres ano     : {n_ano:,} / {n_tot:,}  "
          f"({100 * n_ano / n_tot:.1f}%)")
    print(f"\n   ⚠ Mettre à jour VOCAB_SIZE={VOCAB_SIZE} dans "
          f"02_model.py et 03_train.py !")


if __name__ == "__main__":
    main()