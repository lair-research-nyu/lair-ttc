from __future__ import annotations

import torch
import torch.nn as nn


class CausalTTCTransformer(nn.Module):
    """
    Causal transformer predicting log1p(TTC) from a window of K DINO frames.

    Architecture:
        Linear(768 → d_model)          input projection
        Embedding(K, d_model)           learned positional embedding
        N × TransformerEncoderLayer     pre-norm, causal mask
        LayerNorm + Linear(d_model → 1) output head on last token
    """

    def __init__(
        self,
        d_input:     int   = 768,
        d_model:     int   = 256,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        d_ff:        int   = 512,
        dropout:     float = 0.1,
        K:           int   = 32,
        predict_all: bool  = False,
    ):
        super().__init__()
        self.K           = K
        self.predict_all = predict_all

        self.input_proj = nn.Linear(d_input, d_model)
        self.pos_emb    = nn.Embedding(K, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable on small datasets
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm     = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, 1)

        # Causal mask: upper triangle = True → these positions are masked out
        # Shape (K, K) — registered as buffer so it moves with the model
        causal = torch.triu(torch.ones(K, K), diagonal=1).bool()
        self.register_buffer("causal_mask", causal)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, K, 768)
        returns : (B,)  predicted log1p(ttc_s) for the last frame in window
        """
        B, K, _ = x.shape
        pos = torch.arange(K, device=x.device)

        h = self.input_proj(x) + self.pos_emb(pos)          # (B, K, d_model)
        h = self.encoder(h, mask=self.causal_mask)           # causal attention
        h = self.norm(h)
        if self.predict_all:
            return self.head(h).squeeze(-1)                  # (B, K) — all positions
        return self.head(h[:, -1, :]).squeeze(-1)            # (B,)  — last position only
