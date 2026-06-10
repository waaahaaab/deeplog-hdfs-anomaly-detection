# DeepLog — Détection d'anomalies dans les logs HDFS
## Mémoire de fin d'études — Master 2 Informatique

**Établissement :** Centre de Recherche en Information
Scientifique et Technique (CERIST), Alger
**Auteurs :** [BOUZELBOUDJEN Mohamed Abdelwahab] · [BOUBCHIR ABDERRAZEK]
**Encadrants :** [RAHMANI Amine] · [DERKI Mohamed]
**Année :** 2026

---

## Résumé

Ce dépôt contient l'implémentation complète du pipeline de
détection d'anomalies dans les logs HDFS développé dans le
cadre de notre mémoire de Master 2. Le système explore deux
protocoles complémentaires :

**Protocole one-class (non supervisé)**
- DeepLog LSTM entraîné uniquement sur sessions normales
- MLP one-class par centroïde fréquentiel
- Combinaison LSTM OR MLP session-level

**Protocole semi-supervisé**
- DeepLog LSTM + MLP supervisé avec 70% des anomalies
  disponibles à l'entraînement
- Comparaison directe avec le protocole one-class

**Résultats principaux — Protocole one-class (session-level)**

| Système | Précision | Rappel | F1 | FPR |
|---------|-----------|--------|----|-----|
| LSTM seul (K=5) | 0.973 | 0.439 | 0.605 | 0.001 |
| MLP one-class (P99) | 0.948 | 1.000 | 0.973 | 0.008 |
| LSTM OR MLP | 0.707 | 1.000 | 0.829 | 0.039 |

**Résultats principaux — Protocole semi-supervisé (session-level)**

| Système | Précision | Rappel | F1 |
|---------|-----------|--------|----|
| LSTM seul (K=3) | 0.769 | 0.655 | 0.707 |
| MLP supervisé | 0.999 | 0.998 | 0.999 |
| LSTM OR MLP supervisé | 0.996 | 0.998 | 0.997 |

> **Note :** Les deux protocoles ont des ensembles de test
> différents (10 647 vs 5 052 anomalies). Leur comparaison
> directe n'est pas statistiquement valide. Voir
> section 5.4 du mémoire.

**Contribution analytique principale :** quantification du
mismatch fenêtre-level vs session-level — pour K=5, le F1
passe de 0.099 à 0.605, soit un facteur multiplicatif de 6.1.

---

## Dataset

Le dataset **HDFS v1** est disponible sur le dépôt officiel :

```
https://github.com/logpai/loghub/tree/master/HDFS
```

Fichiers nécessaires :
- `HDFS.log` — logs bruts (~1.5 Go)
- `anomaly_label.csv` — labels par BlockID

> Ces fichiers ne sont **pas inclus** dans ce dépôt.
> Téléchargez-les depuis le lien ci-dessus et placez-les
> dans `data/raw/` avant d'exécuter le pipeline.

---

## Structure du dépôt

```
deeplog-hdfs-anomaly-detection/
│
├── 00_parse_hdfs_log.py     # Parsing HDFS.log → séquences CSV
│                            # (commun aux deux protocoles)
│
├── pipeline/                # Protocole one-class
│   ├── 01_data_preparation.py  # Fenêtrage, split train/test
│   ├── 02_model.py             # Architecture DeepLog LSTM
│   ├── 03_train.py             # Entraînement LSTM
│   ├── 04_evaluate.py          # Inférence Top-K + métriques
│   ├── 05_visualize.py         # Figures principales (fig1–fig9)
│   ├── 06_mlp_counting.py      # MLP one-class + combinaison
│   ├── 07_robustness.py        # Robustesse + baselines
│   └── 08_viz_complementaires.py # ROC, profils fréquentiels
│
├── semisupervised/          # Protocole semi-supervisé
│   ├── deeplog_combined.py     # LSTM + MLP supervisé
│   └── fig_complementaire.py   # Courbes d'apprentissage
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/[username]/deeplog-hdfs-anomaly-detection.git
cd deeplog-hdfs-anomaly-detection
pip install -r requirements.txt
```

---

## Ordre d'exécution

### Étape préalable — Télécharger le dataset

Télécharge `HDFS.log` et `anomaly_label.csv` depuis :
https://github.com/logpai/loghub/tree/master/HDFS

Crée le dossier et place-y les fichiers :
```
data/
└── raw/
    ├── HDFS.log
    └── anomaly_label.csv
```

---

### Étape 0 — Parsing (commun aux deux protocoles)

```bash
python 00_parse_hdfs_log.py
```

Produit dans `data/csv/` :
- `hdfs_sequences.csv` — séquences par BlockID + labels
- `hdfs_templates.csv` — 29 templates EventID

---

### Protocole one-class (pipeline/)

```bash
# Fenêtrage + split train/test
python pipeline/01_data_preparation.py

# Entraînement LSTM (nécessite GPU recommandé)
python pipeline/03_train.py

# Évaluation Top-K session-level
python pipeline/04_evaluate.py

# Figures principales
python pipeline/05_visualize.py

# MLP one-class centroïde + combinaison LSTM OR MLP
python pipeline/06_mlp_counting.py

# Robustesse + baselines triviales
python pipeline/07_robustness.py

# Figures complémentaires (ROC, profils fréquentiels)
python pipeline/08_viz_complementaires.py
```

---

### Protocole semi-supervisé (semisupervised/)

```bash
# LSTM + MLP supervisé (70% anomalies à l'entraînement)
python semisupervised/deeplog_combined.py

# Courbes d'apprentissage
python semisupervised/fig_complementaire.py
```

---

### Structure complète après exécution

```
data/
├── raw/
│   ├── HDFS.log              ← téléchargé manuellement
│   └── anomaly_label.csv     ← téléchargé manuellement
├── csv/
│   ├── hdfs_sequences.csv    ← produit par 00
│   └── hdfs_templates.csv    ← produit par 00
└── processed/
    ├── X_train.npy           ← produit par 01
    ├── y_train.npy
    ├── X_test.npy
    ├── y_test.npy
    ├── y_ano_test.npy
    ├── block_ids_test.npy
    └── vocab_size.npy

checkpoints/
├── deeplog_best.pt           ← produit par 03
├── training_history.npy      ← produit par 03
└── deeplog_semisup.pt        ← produit par semisupervised/

results/
├── in_topk.npy               ← produit par 04
├── true_probs.npy            ← produit par 04
└── figures/                  ← produit par 05, 06, 07, 08
```

---

## Architecture du modèle LSTM

```
Entrée : séquence de 10 EventIDs        [batch × 10]
    ↓
Embedding (30 → 64)                     [batch × 10 × 64]
    ↓
LSTM couche 1 (64 → 128)               [batch × 10 × 128]
    ↓
Dropout (p=0.2)
    ↓
LSTM couche 2 (128 → 128)              [batch × 10 × 128]
    ↓
Extraction dernier état  [:, −1, :]    [batch × 128]
    ↓
Linéaire + Softmax (128 → 30)          [batch × 30]

Paramètres totaux : 237 214
Entraînement     : ~27 min sur Tesla T4 (26 epochs)
```

---

## Hyperparamètres

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `window_size` | 10 | Taille fenêtre glissante |
| `vocab_size` | 30 | 29 EventIDs + 1 padding |
| `embed_dim` | 64 | Dimension embedding |
| `hidden_size` | 128 | Neurones par couche LSTM |
| `num_layers` | 2 | Couches LSTM empilées |
| `dropout` | 0.2 | Taux dropout inter-couches |
| `batch_size` | 2048 | Taille batch entraînement |
| `lr_max` | 3e-3 | LR max OneCycleLR |
| `epochs` | 30 | Epochs max (early stop p=5) |
| `K_values` | [1,3,5,7,9,10,20] | Valeurs Top-K évaluées |
| `percentile_seuil` | 95 / 99 | Calibration seuil MLP |

---

## Résultats détaillés — Mismatch fenêtre vs session

| K | F1 fenêtre | F1 session | Gain |
|---|-----------|------------|------|
| 1 | 0.270 | 0.379 | +40% |
| 3 | 0.149 | 0.491 | +230% |
| 5 | 0.099 | 0.605 | +511% |
| 9 | 0.054 | 0.367 | +579% |
| 20 | 0.038 | 0.282 | +642% |

---

## Robustesse LSTM K=5 (3 sous-échantillons)

| Métrique | Moyenne | Écart-type | Min | Max |
|----------|---------|------------|-----|-----|
| F1 | 0.606 | 0.000 | 0.606 | 0.606 |
| Précision | 0.973 | 0.001 | 0.972 | 0.974 |
| Rappel | 0.440 | 0.000 | 0.440 | 0.441 |
| FPR | 0.001 | 0.000 | 0.001 | 0.001 |

---

## Références

```bibtex
@inproceedings{du2017deeplog,
  author    = {Du, Min and Li, Feifei and Zheng, Guineng
               and Srikumar, Vivek},
  title     = {DeepLog: Anomaly Detection and Diagnosis
               from System Logs through Deep Learning},
  booktitle = {ACM CCS},
  year      = {2017},
  pages     = {1285--1298}
}

@inproceedings{he2016experience,
  author    = {He, Pinjia and Zhu, Jieming and He, Shilin
               and Li, Jian and Lyu, Michael R.},
  title     = {An Evaluation Study on Log Parsing and Its
               Use in Log Mining},
  booktitle = {DSN},
  year      = {2016}
}

@inproceedings{meng2019loganomaly,
  author    = {Meng, Weibin and others},
  title     = {LogAnomaly: Unsupervised Detection of
               Sequential and Quantitative Anomalies
               in Unstructured Logs},
  booktitle = {IJCAI},
  year      = {2019}
}

@inproceedings{guo2021logbert,
  author    = {Guo, Haixuan and Yuan, Shuhan
               and Wu, Xintao},
  title     = {LogBERT: Log Anomaly Detection via BERT},
  booktitle = {IJCNN},
  year      = {2021}
}
```

---

## Licence

Code publié à des fins académiques — mémoire Master 2.
Toute réutilisation doit citer ce travail et les
références associées.

