"""
=============================================================
  MÉMOIRE MASTER 2 — Détection d'anomalies sur HDFS.log
  Script autonome : HDFS.log → Templates → Séquences
  (sans passer par l'étape 1)
=============================================================
"""

import pandas as pd
import re
import json
import os
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# PARTIE 1 — 29 TEMPLATES OFFICIELS HDFS (benchmark LogPai)
# ─────────────────────────────────────────────────────────────
# ATTENTION à l'ordre : du plus SPÉCIFIQUE au plus GÉNÉRAL
# Ex: E8 (Interrupted) et E10 (Exception) doivent être testés
# AVANT E11 (terminating) car les messages peuvent contenir
# plusieurs mots-clés.

HDFS_TEMPLATES = [
    ("E1",  r"Adding an already existing block"),
    ("E2",  r"Verification succeeded for"),
    ("E4",  r"Got exception while serving .+ to"),
    ("E7",  r"writeBlock .+ received exception"),
    ("E8",  r"PacketResponder .+ for block .+ Interrupted"),
    ("E10", r"PacketResponder .+ Exception"),
    ("E11", r"PacketResponder .+ for block .+ terminating"),
    ("E3",  r"Served block .+ to"),
    ("E6",  r"Received block .+ src: .+ dest: .+ of size"),
    ("E5",  r"Receiving block .+ src: .+ dest:"),
    ("E9",  r"Received block .+ of size .+ from"),
    ("E12", r"Exception writing block .+ to mirror"),
    ("E13", r"Receiving empty packet for block"),
    ("E14", r"Exception in receiveBlock for block"),
    ("E15", r"Changing block file offset of block .+ from .+ to .+ meta file offset to"),
    ("E16", r"Transmitted block .+ to"),
    ("E17", r"Failed to transfer .+ to .+ got"),
    ("E18", r"Starting thread to transfer block .+ to"),
    ("E19", r"Reopen Block"),
    ("E20", r"Unexpected error trying to delete block .+ BlockInfo not found in volumeMap"),
    ("E21", r"Deleting block .+ file"),
    ("E22", r"BLOCK\* NameSystem.*allocateBlock:"),
    ("E27", r"BLOCK\* NameSystem.*addStoredBlock: Redundant addStoredBlock request received for"),
    ("E28", r"BLOCK\* NameSystem.*addStoredBlock: addStoredBlock request received for.*But it does not belong to any file"),
    ("E26", r"BLOCK\* NameSystem.*addStoredBlock: blockMap updated:.*is added to.*size"),
    ("E23", r"BLOCK\* NameSystem.*delete:.*is added to invalidSet of"),
    ("E24", r"BLOCK\* Removing block .+ from neededReplications as it does not belong to any file"),
    ("E25", r"BLOCK\* ask .+ to replicate"),
    ("E29", r"PendingReplicationMonitor timed out block"),
]

# Template lisible associé à chaque EventID
TEMPLATE_LABEL = {
    "E1" : "[*]Adding an already existing block[*]",
    "E2" : "[*]Verification succeeded for[*]",
    "E3" : "[*]Served block[*]to[*]",
    "E4" : "[*]Got exception while serving[*]to[*]",
    "E5" : "[*]Receiving block[*]src:[*]dest:[*]",
    "E6" : "[*]Received block[*]src:[*]dest:[*]of size[*]",
    "E7" : "[*]writeBlock[*]received exception[*]",
    "E8" : "[*]PacketResponder[*]for block[*]Interrupted[*]",
    "E9" : "[*]Received block[*]of size[*]from[*]",
    "E10": "[*]PacketResponder[*]Exception[*]",
    "E11": "[*]PacketResponder[*]for block[*]terminating[*]",
    "E12": "[*]:Exception writing block[*]to mirror[*]",
    "E13": "[*]Receiving empty packet for block[*]",
    "E14": "[*]Exception in receiveBlock for block[*]",
    "E15": "[*]Changing block file offset of block[*]from[*]to[*]meta file offset to[*]",
    "E16": "[*]:Transmitted block[*]to[*]",
    "E17": "[*]:Failed to transfer[*]to[*]got[*]",
    "E18": "[*]Starting thread to transfer block[*]to[*]",
    "E19": "[*]Reopen Block[*]",
    "E20": "[*]Unexpected error trying to delete block[*]BlockInfo not found in volumeMap[*]",
    "E21": "[*]Deleting block[*]file[*]",
    "E22": "[*]BLOCK* NameSystem[*]allocateBlock:[*]",
    "E23": "[*]BLOCK* NameSystem[*]delete:[*]is added to invalidSet of[*]",
    "E24": "[*]BLOCK* Removing block[*]from neededReplications as it does not belong to any file[*]",
    "E25": "[*]BLOCK* ask[*]to replicate[*]to[*]",
    "E26": "[*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]",
    "E27": "[*]BLOCK* NameSystem[*]addStoredBlock: Redundant addStoredBlock request received for[*]on[*]size[*]",
    "E28": "[*]BLOCK* NameSystem[*]addStoredBlock: addStoredBlock request received for[*]on[*]size[*]But it does not belong to any file[*]",
    "E29": "[*]PendingReplicationMonitor timed out block[*]",
}

# Précompiler toutes les regex
COMPILED = [
    (eid, re.compile(pattern, re.IGNORECASE))
    for eid, pattern in HDFS_TEMPLATES
]

# Regex pour parser une ligne brute HDFS.log
# Ex: "081109 203518 143 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_123 terminating"
LINE_PATTERN = re.compile(
    r'(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([\w.$]+):\s+(.*)'
)

# Regex pour extraire le BlockID
BLOCK_PATTERN = re.compile(r'(blk_-?\d+)')


# ─────────────────────────────────────────────────────────────
# PARTIE 2 — LECTURE DIRECTE DE HDFS.log
# ─────────────────────────────────────────────────────────────

def lire_et_parser(chemin_log: str):
    """
    Lit HDFS.log ligne par ligne.
    Pour chaque ligne :
      1. Extrait le message avec la regex de parsing
      2. Extrait le BlockID depuis le message
      3. Mappe le message vers un EventID (29 templates)

    Retourne :
      - sessions : dict  { block_id → [EventID, EventID, ...] }
      - compteurs: dict  { EventID  → count }
    """

    # sessions[block_id] = liste ordonnée d'EventIDs
    sessions  = defaultdict(list)
    # compteur d'occurrences par EventID
    compteurs = defaultdict(int)

    n_total   = 0
    n_matchés = 0
    n_ignorés = 0

    print(f"[INFO] Lecture de {chemin_log} ...")

    with open(chemin_log, 'r', encoding='utf-8', errors='ignore') as f:
        for i, ligne in enumerate(f):
            ligne = ligne.strip()
            if not ligne:
                continue

            n_total += 1

            # ── Étape A : parser la structure de la ligne ──
            m = LINE_PATTERN.match(ligne)
            if not m:
                n_ignorés += 1
                continue

            message = m.group(6).strip()

            # ── Étape B : extraire le BlockID ──
            block_match = BLOCK_PATTERN.search(message)
            if not block_match:
                continue                    # ligne sans BlockID → ignorée
            block_id = block_match.group(1)

            # ── Étape C : trouver l'EventID ──
            event_id = None
            for eid, pattern in COMPILED:
                if pattern.search(message):
                    event_id = eid
                    break

            if event_id is None:
                continue                    # message hors des 29 templates

            # ── Étape D : enregistrer ──
            sessions[block_id].append(event_id)
            compteurs[event_id] += 1
            n_matchés += 1

            # Progression
            if (i + 1) % 1_000_000 == 0:
                print(f"  → {i+1:,} lignes lues | {len(sessions):,} sessions en cours...")

    print(f"\n[OK] Lecture terminée.")
    print(f"     Lignes totales     : {n_total:,}")
    print(f"     Événements mappés  : {n_matchés:,}")
    print(f"     Lignes ignorées    : {n_ignorés:,}")
    print(f"     Sessions (BlockIDs): {len(sessions):,}")

    return sessions, compteurs


# ─────────────────────────────────────────────────────────────
# PARTIE 3 — AFFICHER LES TEMPLATES (format benchmark)
# ─────────────────────────────────────────────────────────────

def afficher_templates(compteurs: dict):
    """
    Affiche le tableau EventId / EventTemplate / Count
    dans l'ordre E1 → E29, comme le benchmark LogPai.
    """
    print("\n" + "="*70)
    print("  RÉSULTAT — TEMPLATES HDFS OFFICIELS")
    print("="*70)
    print(f"  {'EventId':<8}  {'EventTemplate':<55}  {'Count':>10}")
    print("  " + "-"*67)

    for i in range(1, 30):
        eid   = f"E{i}"
        tmpl  = TEMPLATE_LABEL.get(eid, "")
        count = compteurs.get(eid, 0)
        if count > 0:
            print(f"  {eid:<8}  {tmpl:<55}  {count:>10,}")

    # EventIDs absents
    absents = [f"E{i}" for i in range(1, 30) if compteurs.get(f"E{i}", 0) == 0]
    if absents:
        print(f"\n  [!] EventIDs absents du dataset : {absents}")
    else:
        print(f"\n  [OK] Les 29 EventIDs sont tous présents.")


# ─────────────────────────────────────────────────────────────
# PARTIE 4 — CONSTRUIRE LE DataFrame DES SÉQUENCES
# ─────────────────────────────────────────────────────────────

def construire_sequences(sessions: dict) -> pd.DataFrame:
    """
    Convertit le dict sessions en DataFrame :
      block_id  |  sequence              |  longueur
      blk_123   |  [E5, E9, E11, E26]   |  4
    """
    print("\n[INFO] Construction du DataFrame des séquences...")

    rows = [
        {'block_id': bid, 'sequence': seq, 'longueur': len(seq)}
        for bid, seq in sessions.items()
    ]
    df = pd.DataFrame(rows)

    print(f"[OK] {len(df):,} sessions.")
    print(f"     Longueur moyenne : {df['longueur'].mean():.1f} événements")
    print(f"     Min : {df['longueur'].min()}  |  Max : {df['longueur'].max()}")

    # Aperçu des 5 premières séquences
    print("\n  Aperçu des 5 premières séquences :")
    print(f"  {'BlockID':<30}  {'Séquence'}")
    print("  " + "-"*65)
    for _, row in df.head(5).iterrows():
        seq_str = str(row['sequence'])[:45] + ("..." if len(str(row['sequence'])) > 45 else "")
        print(f"  {row['block_id']:<30}  {seq_str}")

    return df


# ─────────────────────────────────────────────────────────────
# PARTIE 5 — AJOUTER LES LABELS (anomaly_label.csv)
# ─────────────────────────────────────────────────────────────

def ajouter_labels(df: pd.DataFrame, chemin_labels: str) -> pd.DataFrame:
    """
    Fusionne le DataFrame des séquences avec anomaly_label.csv.
    Ajoute une colonne 'label' : 0 = Normal, 1 = Anomaly.
    """
    if not os.path.exists(chemin_labels):
        print(f"\n[!] {chemin_labels} introuvable — séquences sans labels.")
        return df

    labels = pd.read_csv(chemin_labels)
    labels.columns = ['block_id', 'label']
    labels['label'] = labels['label'].map({'Normal': 0, 'Anomaly': 1})

    df = df.merge(labels, on='block_id', how='left')

    n_anomaly = int(df['label'].sum())
    n_normal  = int((df['label'] == 0).sum())
    n_nan     = int(df['label'].isna().sum())
    total     = n_anomaly + n_normal

    print(f"\n[INFO] Distribution des labels :")
    print(f"       Normal  : {n_normal:>8,}  ({100*n_normal/total:.1f}%)")
    print(f"       Anomaly : {n_anomaly:>8,}  ({100*n_anomaly/total:.1f}%)")
    if n_nan:
        print(f"       Sans label : {n_nan:,} sessions (BlockIDs absents du CSV labels)")

    return df


# ─────────────────────────────────────────────────────────────
# PARTIE 6 — SAUVEGARDER LES RÉSULTATS
# ─────────────────────────────────────────────────────────────

def sauvegarder(df: pd.DataFrame, compteurs: dict):
    """
    Sauvegarde :
      - hdfs_sequences.csv  : séquences + labels (pour le modèle)
      - hdfs_templates.csv  : tableau EventId / EventTemplate
    """
    # Séquences : convertir la liste Python en string JSON
    df_save = df.copy()
    df_save['sequence'] = df_save['sequence'].apply(json.dumps)
    os.makedirs("data/csv", exist_ok=True)
    df_save.to_csv("data/csv/hdfs_sequences.csv", index=False)
    taille = os.path.getsize("data/csv/hdfs_sequences.csv") / (1024 * 1024)
    print(f"\n[OK] hdfs_sequences.csv   → {len(df_save):,} lignes  ({taille:.1f} Mo)")

    # Templates
    rows = []
    for i in range(1, 30):
        eid = f"E{i}"
        if compteurs.get(eid, 0) > 0:
            rows.append({'EventId': eid, 'EventTemplate': TEMPLATE_LABEL[eid]})
    pd.DataFrame(rows).to_csv("data/csv/hdfs_templates.csv", index=False)
    print(f"[OK] hdfs_templates.csv   → {len(rows)} templates")

    print("\n→ Prochaine étape : 03_encodage_lstm.py")


# ─────────────────────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ↓ Modifie ces chemins si nécessaire
    CHEMIN_LOG    = "data/raw/HDFS.log"
    CHEMIN_LABELS = "data/raw/anomaly_label.csv"

    # Vérification des fichiers
    for f in [CHEMIN_LOG, CHEMIN_LABELS]:
        if not os.path.exists(f):
            print(f"[ERREUR] Fichier introuvable : {f}")
            print("         Place HDFS.log et anomaly_label.csv dans le même dossier que ce script.")
            exit(1)

    # Pipeline complet
    sessions,  compteurs = lire_et_parser(CHEMIN_LOG)
    afficher_templates(compteurs)
    df = construire_sequences(sessions)
    df = ajouter_labels(df, CHEMIN_LABELS)
    sauvegarder(df, compteurs)

    print("\n[SUCCÈS] Pipeline terminé.")
    print("         Fichiers produits : hdfs_sequences.csv  |  hdfs_templates.csv")