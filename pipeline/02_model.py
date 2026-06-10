"""
DeepLog - Étape 2 : Architecture du modèle
===========================================
v3 : vocab_size dynamique (chargé depuis vocab_size.npy)
     Compatible CSV LogPai (vocab_size=30) et ancien pipeline (vocab_size=101)
     Aucun changement d'architecture — seul vocab_size change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os



class DeepLogLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int  = 101,
        embed_dim: int   = 64,
        hidden_size: int = 128,
        num_layers: int  = 2,
        dropout: float   = 0.2,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds      = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

    def get_topk_probs(self, logits: torch.Tensor, k: int):
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_idx = torch.topk(probs, k, dim=-1)
        return probs, topk_idx, topk_p


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        log_probs  = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(
                log_probs, self.smoothing / (vocab_size - 1)
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_targets * log_probs).sum(dim=-1).mean()


def model_summary(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*50}")
    print(f"  DeepLogLSTM — Résumé des paramètres")
    print(f"{'='*50}")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<20} {params:>10,} params")
    print(f"{'─'*50}")
    print(f"  {'TOTAL':<20} {total:>10,} params")
    print(f"  {'Entraînables':<20} {trainable:>10,} params")
    print(f"{'='*50}\n")


def load_vocab_size(data_path: str, default: int = 101) -> int:
    """
    Charge vocab_size depuis vocab_size.npy si disponible,
    sinon retourne la valeur par défaut.
    Permet de switcher automatiquement entre pipeline v2 (101)
    et pipeline v3 CSV (30).
    """
    vocab_path = os.path.join(data_path, "vocab_size.npy")
    if os.path.exists(vocab_path):
        vocab_size = int(np.load(vocab_path)[0])
        print(f"  vocab_size chargé depuis fichier : {vocab_size}")
        return vocab_size
    print(f"  vocab_size non trouvé → défaut : {default}")
    return default


if __name__ == "__main__":
    # Test avec les deux configurations
    for vs in [101, 30]:
        print(f"\n--- Test vocab_size={vs} ---")
        model = DeepLogLSTM(vocab_size=vs, embed_dim=64,
                            hidden_size=128, num_layers=2, dropout=0.2)
        model_summary(model)
        batch  = torch.randint(1, vs, (32, 10))
        logits = model(batch)
        assert logits.shape == (32, vs), f"Shape incorrecte : {logits.shape}"
        print(f"  ✅ Forward pass OK — output shape : {logits.shape}")