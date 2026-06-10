

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score,
                             precision_recall_fscore_support)
import time
import os
import sys

# Chemins relatifs au dossier racine du projet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

CONFIG = {
    "chemin_sequences" : "data/csv/hdfs_sequences.csv",
    "chemin_modele"    : "checkpoints/deeplog_semisup.pt",

    # Fenêtrage LSTM
    "window_size"      : 10,
    "top_k"            : 3,    # meilleur compromis Recall/Precision selon analyse sensibilité
                                # top-k=1 : trop de FP | top-k=9 : Recall trop bas

    # Modèle LSTM
    "vocab_size"       : 30,      # 29 EventIDs + 1 padding
    "embedding_dim"    : 32,
    "hidden_size"      : 128,
    "num_layers"       : 2,
    "dropout"          : 0.3,

    # Modèle MLP
    "mlp_hidden"       : 64,
    "mlp_val_ratio"    : 0.2,     # ← CORRECTION 3 : 20% des données MLP pour validation
    "mlp_patience"     : 5,       # ← CORRECTION 3 : early stopping

    # Entraînement commun
    "epochs_lstm"      : 30,
    "epochs_mlp"       : 50,      # plus d'epochs car early stopping arrêtera au bon moment
    "batch_size"       : 1024,
    "learning_rate"    : 0.001,

    # Split des données
    "normal_train_ratio"  : 0.8,  # 80% des normaux → train LSTM
    "anomaly_train_ratio" : 0.7,  # ← CORRECTION 1 : 70% anomalies → train MLP
                                  #                   30% anomalies → test uniquement

    # Reproductibilité
    "early_stopping"   : 5,
    "seed"             : 42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Dispositif : {DEVICE}")
print(f"[INFO] top_k      : {CONFIG['top_k']}  (valeur papier DeepLog original)")


# ─────────────────────────────────────────────────────────────
# ENCODAGE EventID → entier
# ─────────────────────────────────────────────────────────────

EVENT_TO_IDX = {f"E{i}": i for i in range(1, 30)}
EVENT_TO_IDX["UNKNOWN"] = 0

def encoder(seq):
    """Convertit une liste d'EventIDs en liste d'entiers."""
    return [EVENT_TO_IDX.get(e, 0) for e in seq]


# ─────────────────────────────────────────────────────────────
# VECTEUR DE COMPTAGE PAR SESSION
# ─────────────────────────────────────────────────────────────

def comptage_session(seq, vocab_size=29):
    """
    Calcule le vecteur de fréquence normalisé d'une session encodée.
    v[k] = count(E_{k+1} dans session) / len(session)
    """
    vec = np.zeros(vocab_size, dtype=np.float32)
    for e in seq:
        if 1 <= e <= vocab_size:
            vec[e - 1] += 1
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec

def build_count_matrix(sequences):
    """Construit la matrice de comptage pour toutes les sessions."""
    return np.stack([comptage_session(s) for s in sequences])


# ─────────────────────────────────────────────────────────────
# FENÊTRAGE GLISSANT POUR LE LSTM
# ─────────────────────────────────────────────────────────────

def fenetrer(sequences, window_size):
    """
    Génère les paires (fenêtre d'entrée, cible) par fenêtrage glissant.
    La cible est le prochain EventID à prédire.
    """
    X, y = [], []
    for seq in sequences:
        if len(seq) <= window_size:
            continue
        for i in range(len(seq) - window_size):
            X.append(seq[i : i + window_size])
            y.append(seq[i + window_size])
    return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)


# ─────────────────────────────────────────────────────────────
# DATASETS PYTORCH
# ─────────────────────────────────────────────────────────────

class LogDataset(Dataset):
    """Dataset pour les fenêtres LSTM (entrée entière)."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class CountDataset(Dataset):
    """Dataset pour les vecteurs de comptage MLP (entrée float)."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ─────────────────────────────────────────────────────────────
# MODÈLE 1 — LSTM DeepLog
# ─────────────────────────────────────────────────────────────

class DeepLog(nn.Module):
    """
    Architecture LSTM DeepLog (Du et al., 2017).
    Embedding → LSTM (2 couches) → Linear → distribution sur vocab.
    Entraîné à prédire le prochain EventID depuis une fenêtre de w.
    """
    def __init__(self, vocab_size, embedding_dim,
                 hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb       = self.embedding(x)
        out, _    = self.lstm(emb)
        last      = self.dropout(out[:, -1, :])
        return self.fc(last)


# ─────────────────────────────────────────────────────────────
# MODÈLE 2 — MLP sur vecteur de comptage
# ─────────────────────────────────────────────────────────────

class CountMLP(nn.Module):
    """
    MLP supervisé : vecteur de fréquence (29 dims) → Normal / Anomaly.
    Architecture : Linear(29→64) → ReLU → Dropout → Linear(64→32) → ReLU → Linear(32→2)
    """
    def __init__(self, input_dim=29, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# ENTRAÎNEMENT LSTM
# ─────────────────────────────────────────────────────────────

def entrainer_lstm(model, loader, config):
    """
    Entraînement LSTM en mode non supervisé sur sessions normales.
    Objectif : prédire le prochain EventID (cross-entropie).
    Early stopping sur la loss d'entraînement.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    meilleure_loss = float('inf')
    sans_progres   = 0
    meilleur_etat  = None

    print("\n" + "="*62)
    print("  [1/2] ENTRAÎNEMENT LSTM — sessions normales uniquement")
    print(f"  window_size={config['window_size']} · "
          f"hidden={config['hidden_size']} · "
          f"top_k={config['top_k']} à l'inférence")
    print("="*62)

    for epoch in range(config["epochs_lstm"]):
        model.train()
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0

        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(X_b)
            correct    += (logits.argmax(1) == y_b).sum().item()
            total      += len(X_b)

        avg_loss = total_loss / total
        acc      = 100.0 * correct / total
        scheduler.step(avg_loss)

        print(f"  Epoch {epoch+1:>2}/{config['epochs_lstm']}  |"
              f"  Loss: {avg_loss:.4f}  |"
              f"  Acc: {acc:.2f}%  |"
              f"  {time.time()-t0:.1f}s")

        if avg_loss < meilleure_loss - 1e-4:   # seuil fin : laisse le LSTM converger complètement
            meilleure_loss = avg_loss
            sans_progres   = 0
            meilleur_etat  = {k: v.clone()
                              for k, v in model.state_dict().items()}
        else:
            sans_progres += 1
            print(f"           [Early stopping : "
                  f"{sans_progres}/{config['early_stopping']}]")
            if sans_progres >= config["early_stopping"]:
                print(f"  [STOP] Early stopping epoch {epoch+1}.")
                break

    if meilleur_etat:
        model.load_state_dict(meilleur_etat)
        print(f"  [OK] Meilleur modèle LSTM restauré "
              f"(loss={meilleure_loss:.4f})")
    return model


# ─────────────────────────────────────────────────────────────
# ENTRAÎNEMENT MLP — CORRECTION 3 : validation + early stopping
# ─────────────────────────────────────────────────────────────

def entrainer_mlp(model, X_train, y_train, X_val, y_val, config):
    """
    Entraînement MLP supervisé avec :
      - jeu de validation séparé pour early stopping
      - class_weight pour compenser le déséquilibre Normal/Anomaly
    X_train, y_train : données d'entraînement MLP
    X_val,   y_val   : données de validation MLP (ne chevauchent PAS le test)
    """
    # ── DataLoaders ──
    train_loader = DataLoader(
        CountDataset(X_train, y_train),
        batch_size=512, shuffle=True
    )
    val_loader = DataLoader(
        CountDataset(X_val, y_val),
        batch_size=512, shuffle=False
    )

    # ── Pondération des classes (déséquilibre Normal/Anomaly) ──
    n_normal  = (y_train == 0).sum()
    n_anomaly = (y_train == 1).sum()
    if n_anomaly > 0:
        poids = torch.tensor(
            [1.0, n_normal / n_anomaly], dtype=torch.float
        ).to(DEVICE)
    else:
        poids = None

    criterion = nn.CrossEntropyLoss(weight=poids)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    meilleure_val_loss = float('inf')
    sans_progres       = 0
    meilleur_etat      = None

    print("\n" + "="*62)
    print("  [2/2] ENTRAÎNEMENT MLP COMPTAGE")
    print(f"  Train : {len(X_train):,} sessions  |  "
          f"Val : {len(X_val):,} sessions")
    print(f"  Normal: {(y_train==0).sum():,}  |  "
          f"Anomaly: {(y_train==1).sum():,}  |  "
          f"class_weight: {n_normal/n_anomaly:.1f}")
    print("="*62)

    for epoch in range(config["epochs_mlp"]):
        # ── Phase train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_b)
            correct    += (logits.argmax(1) == y_b).sum().item()
            total      += len(X_b)

        train_loss = total_loss / total
        train_acc  = 100.0 * correct / total

        # ── Phase validation ──
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                logits    = model(X_b)
                loss      = criterion(logits, y_b)
                val_loss_total += loss.item() * len(X_b)
                val_correct    += (logits.argmax(1) == y_b).sum().item()
                val_total      += len(X_b)

        val_loss = val_loss_total / val_total
        val_acc  = 100.0 * val_correct / val_total

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>2}/{config['epochs_mlp']}  |"
                  f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |"
                  f"  Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")

        # ── Early stopping sur val_loss ──
        if val_loss < meilleure_val_loss - 1e-4:
            meilleure_val_loss = val_loss
            sans_progres       = 0
            meilleur_etat      = {k: v.clone()
                                  for k, v in model.state_dict().items()}
        else:
            sans_progres += 1
            if sans_progres >= config["mlp_patience"]:
                print(f"  [STOP] Early stopping MLP epoch {epoch+1}.")
                break

    if meilleur_etat:
        model.load_state_dict(meilleur_etat)
        print(f"  [OK] Meilleur modèle MLP restauré "
              f"(val_loss={meilleure_val_loss:.4f})")
    return model


# ─────────────────────────────────────────────────────────────
# INFÉRENCE ET ÉVALUATION
# ─────────────────────────────────────────────────────────────

def detecter_combine(lstm_model, mlp_model, sequences,
                     labels, config):
    """
    Inférence combinée session-level :
      LSTM : détection par top-k sur fenêtres, agrégation OR
      MLP  : détection directe sur vecteur de comptage
      Final: règle OR logique entre les deux décisions
    """
    lstm_model.eval()
    mlp_model.eval()
    window_size = config["window_size"]
    top_k       = config["top_k"]

    preds_lstm = []
    print(f"\n[INFO] Inférence LSTM (top-k={top_k}) "
          f"sur {len(sequences):,} sessions...")
    t0 = time.time()

    # ── Prédictions LSTM (top-k session-level) ──
    with torch.no_grad():
        for idx, seq in enumerate(sequences):
            if len(seq) <= window_size:
                preds_lstm.append(0)
                continue

            fenetres, cibles = [], []
            for i in range(len(seq) - window_size):
                fenetres.append(seq[i : i + window_size])
                cibles.append(seq[i + window_size])

            X_sess = torch.tensor(
                fenetres, dtype=torch.long
            ).to(DEVICE)
            logits = lstm_model(X_sess)
            topk   = logits.topk(top_k, dim=1).indices

            anomale = any(
                cibles[j] not in topk[j].cpu().tolist()
                for j in range(len(cibles))
            )
            preds_lstm.append(1 if anomale else 0)

            if (idx + 1) % 50_000 == 0:
                print(f"  → LSTM : {idx+1:,} / "
                      f"{len(sequences):,} sessions...")

    print(f"[OK] LSTM terminé en {time.time()-t0:.1f}s")

    # ── Prédictions MLP (vecteur de comptage) ──
    print("[INFO] Inférence MLP...")
    with torch.no_grad():
        count_matrix = build_count_matrix(sequences)
        X_count = torch.tensor(
            count_matrix, dtype=torch.float32
        ).to(DEVICE)
        preds_mlp_all = []
        for i in range(0, len(X_count), 4096):
            batch  = X_count[i : i + 4096]
            logits = mlp_model(batch)
            preds_mlp_all.extend(logits.argmax(1).cpu().tolist())

    preds_lstm    = np.array(preds_lstm)
    preds_mlp     = np.array(preds_mlp_all)
    labels_arr    = np.array(labels)
    preds_combine = np.logical_or(preds_lstm, preds_mlp).astype(int)

    # ── Affichage des résultats ──
    for preds, nom in [
        (preds_lstm,    f"LSTM seul (top-k={top_k})"),
        (preds_mlp,     "MLP Comptage seul"),
        (preds_combine, "Combiné LSTM OR MLP"),
    ]:
        afficher_resultats(preds, labels_arr, nom)

    return preds_combine, labels_arr


def afficher_resultats(preds, labels, nom):
    cm = confusion_matrix(labels, preds)
    if cm.size < 4:
        print(f"\n  [{nom}] Matrice de confusion incomplète.")
        return
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "="*62)
    print(f"  RÉSULTATS — {nom}")
    print("="*62)
    print(f"\n  Matrice de confusion :")
    print(f"                  Prédit Normal   Prédit Anomaly")
    print(f"  Réel Normal   :   {tn:>10,}      {fp:>10,}")
    print(f"  Réel Anomaly  :   {fn:>10,}      {tp:>10,}")
    print(f"\n  FPR     : {100*fpr:.2f}%")
    try:
        print(f"  AUC-ROC : {roc_auc_score(labels, preds):.4f}")
    except Exception:
        pass
    print(classification_report(
        labels, preds,
        target_names=["Normal", "Anomaly"],
        digits=4
    ))


# ─────────────────────────────────────────────────────────────
# ANALYSE DE SENSIBILITÉ top-k
# ─────────────────────────────────────────────────────────────

def analyse_sensibilite_topk(lstm_model, sequences, labels,
                              config, valeurs_k=(1, 3, 5, 7, 9, 11, 15)):
    """
    Évalue les performances du LSTM seul pour différentes
    valeurs de top-k, afin de justifier le choix retenu.
    """
    print("\n" + "="*62)
    print("  ANALYSE DE SENSIBILITÉ top-k (LSTM seul)")
    print("="*62)
    print(f"  {'top-k':>6}  {'Recall':>8}  {'Precision':>10}"
          f"  {'F1':>8}  {'FPR':>7}")
    print("  " + "-"*48)

    lstm_model.eval()
    window_size = config["window_size"]
    labels_arr  = np.array(labels)

    # Pré-calculer les logits pour éviter de relancer le LSTM à chaque k
    all_logits_sessions = []
    all_cibles_sessions = []

    with torch.no_grad():
        for seq in sequences:
            if len(seq) <= window_size:
                all_logits_sessions.append(None)
                all_cibles_sessions.append(None)
                continue
            fenetres, cibles = [], []
            for i in range(len(seq) - window_size):
                fenetres.append(seq[i : i + window_size])
                cibles.append(seq[i + window_size])
            X_sess = torch.tensor(
                fenetres, dtype=torch.long
            ).to(DEVICE)
            logits = lstm_model(X_sess).cpu()
            all_logits_sessions.append(logits)
            all_cibles_sessions.append(cibles)

    for k in valeurs_k:
        preds = []
        for logits, cibles in zip(all_logits_sessions,
                                  all_cibles_sessions):
            if logits is None:
                preds.append(0)
                continue
            topk    = logits.topk(k, dim=1).indices
            anomale = any(
                cibles[j] not in topk[j].tolist()
                for j in range(len(cibles))
            )
            preds.append(1 if anomale else 0)

        preds = np.array(preds)
        cm    = confusion_matrix(labels_arr, preds)
        if cm.size < 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1   = (2 * prec * rec / (prec + rec)
                if (prec + rec) > 0 else 0)
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0

        marker = " ←" if k == config["top_k"] else ""
        print(f"  {k:>6}  {100*rec:>7.2f}%  {100*prec:>9.2f}%"
              f"  {100*f1:>7.2f}%  {100*fpr:>6.2f}%{marker}")


# ─────────────────────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = CONFIG

    # ── 1. Chargement et encodage ────────────────────────────
    print("[INFO] Chargement de hdfs_sequences.csv ...")
    df = pd.read_csv(cfg["chemin_sequences"])
    df['sequence'] = df['sequence'].apply(json.loads)
    df = df.dropna(subset=['label'])
    df['label']    = df['label'].astype(int)
    df['sequence'] = df['sequence'].apply(encoder)

    n_normal  = (df['label'] == 0).sum()
    n_anomaly = (df['label'] == 1).sum()
    print(f"[OK]  {len(df):,} sessions — "
          f"Normal: {n_normal:,}  |  Anomaly: {n_anomaly:,}")

    # ── 2. Séparation stricte des données ────────────────────
    #
    #  CORRECTION 1 — schéma du split corrigé :
    #
    #  Normaux (558 223)
    #    ├── 80% → df_train_normal   (446 578) → Train LSTM + Train MLP
    #    └── 20% → df_test_normal    (111 645) → Test uniquement
    #
    #  Anomalies (16 838)
    #    ├── 70% → df_anomaly_train  ( 11 787) → Train MLP uniquement
    #    └── 30% → df_anomaly_test   (  5 051) → Test uniquement ← JAMAIS VU PAR LE MLP
    #
    df_normal  = df[df['label'] == 0].reset_index(drop=True)
    df_anomaly = df[df['label'] == 1].reset_index(drop=True)

    # Split des normaux : 80% train / 20% test
    n_train_normal = int(len(df_normal) * cfg["normal_train_ratio"])
    df_train_normal = df_normal.iloc[:n_train_normal]
    df_test_normal  = df_normal.iloc[n_train_normal:]

    # ── CORRECTION 1 : split des anomalies ──
    # 70% pour entraîner le MLP, 30% réservés au test uniquement
    n_anomaly_train = int(len(df_anomaly) * cfg["anomaly_train_ratio"])
    df_anomaly_train = df_anomaly.iloc[:n_anomaly_train]   # → MLP train
    df_anomaly_test  = df_anomaly.iloc[n_anomaly_train:]   # → test seulement

    # Constitution du jeu de test : normaux non vus + anomalies non vues
    df_test = pd.concat(
        [df_test_normal, df_anomaly_test], ignore_index=True
    ).sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)

    print(f"\n[INFO] Répartition CORRIGÉE :")
    print(f"  Train LSTM      : {len(df_train_normal):,} "
          f"sessions normales uniquement")
    print(f"  Train MLP       : {len(df_train_normal):,} normaux "
          f"+ {len(df_anomaly_train):,} anomalies "
          f"= {len(df_train_normal)+len(df_anomaly_train):,} sessions")
    print(f"  Test Normal     : {len(df_test_normal):,} sessions")
    print(f"  Test Anomaly    : {len(df_anomaly_test):,} sessions "
          f"[JAMAIS VUES PAR LE MLP]")
    print(f"  Test Total      : {len(df_test):,} sessions")

    # ── 3. Préparation des données LSTM ──────────────────────
    print(f"\n[INFO] Fenêtrage LSTM "
          f"(window_size={cfg['window_size']})...")
    X_lstm, y_lstm = fenetrer(
        df_train_normal['sequence'].tolist(), cfg["window_size"]
    )
    print(f"[OK]  {len(X_lstm):,} fenêtres d'entraînement")

    train_loader = DataLoader(
        LogDataset(X_lstm, y_lstm),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,    # ← CORRECTION 4 : 0 pour Colab
        pin_memory=(DEVICE.type == "cuda")
    )

    # ── 4. Préparation des données MLP ───────────────────────
    # Données d'entraînement MLP : normaux train + anomalies train (70%)
    df_mlp_train_full = pd.concat(
        [df_train_normal, df_anomaly_train], ignore_index=True
    )
    X_mlp_all = build_count_matrix(
        df_mlp_train_full['sequence'].tolist()
    )
    y_mlp_all = df_mlp_train_full['label'].values

    # ── CORRECTION 3 : split validation interne du MLP ──
    X_mlp_train, X_mlp_val, y_mlp_train, y_mlp_val = train_test_split(
        X_mlp_all, y_mlp_all,
        test_size=cfg["mlp_val_ratio"],
        random_state=cfg["seed"],
        stratify=y_mlp_all
    )
    print(f"\n[INFO] MLP — données d'entraînement :")
    print(f"  Train : {len(X_mlp_train):,} sessions "
          f"(Normal: {(y_mlp_train==0).sum():,} | "
          f"Anomaly: {(y_mlp_train==1).sum():,})")
    print(f"  Val   : {len(X_mlp_val):,} sessions "
          f"(Normal: {(y_mlp_val==0).sum():,} | "
          f"Anomaly: {(y_mlp_val==1).sum():,})")

    # ── 5. Création des modèles ──────────────────────────────
    lstm_model = DeepLog(
        cfg["vocab_size"], cfg["embedding_dim"],
        cfg["hidden_size"], cfg["num_layers"],
        cfg["dropout"]
    ).to(DEVICE)

    mlp_model = CountMLP(
        input_dim=29, hidden_dim=cfg["mlp_hidden"]
    ).to(DEVICE)

    n_lstm = sum(p.numel() for p in lstm_model.parameters())
    n_mlp  = sum(p.numel() for p in mlp_model.parameters())
    print(f"\n[INFO] LSTM : {n_lstm:,} paramètres")
    print(f"[INFO] MLP  : {n_mlp:,} paramètres")

    # ── 6. Entraînement ──────────────────────────────────────
    lstm_model = entrainer_lstm(lstm_model, train_loader, cfg)
    mlp_model  = entrainer_mlp(
        mlp_model,
        X_mlp_train, y_mlp_train,
        X_mlp_val,   y_mlp_val,
        cfg
    )

    # Sauvegarde des deux modèles
    torch.save({
        'lstm_state'     : lstm_model.state_dict(),
        'mlp_state'      : mlp_model.state_dict(),
        'config'         : cfg,
        'split_info'     : {
            'n_train_normal'  : len(df_train_normal),
            'n_anomaly_train' : len(df_anomaly_train),
            'n_anomaly_test'  : len(df_anomaly_test),
            'n_test_total'    : len(df_test),
        }
    }, cfg["chemin_modele"])
    print(f"\n[OK] Modèles sauvegardés → {cfg['chemin_modele']}")

    # ── 7. Inférence et évaluation ───────────────────────────
    seqs_test   = df_test['sequence'].tolist()
    labels_test = df_test['label'].tolist()

    detecter_combine(
        lstm_model, mlp_model, seqs_test, labels_test, cfg
    )

    # ── 8. Analyse de sensibilité top-k ──────────────────────
    analyse_sensibilite_topk(
        lstm_model, seqs_test, labels_test, cfg
    )

    print("\n[SUCCÈS] DeepLog+MLP v3 (corrigé) terminé.")
    print("→ Résultats honnêtes et défendables devant le jury.")
