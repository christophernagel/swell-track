"""
wave_transformer.py
--------------------
Implements the EnhancedWaveTransformer model, which processes sequences of
buoy data (with advanced temporal embeddings) and produces multi-task outputs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Any, Dict
from datetime import datetime


class TemporalEmbedding(nn.Module):
    """
    Combines:
      - Sinusoidal positional encoding
      - Learnable hour-of-day and day-of-week embeddings
    Expects a list of datetime objects for each sequence.
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Sinusoidal encoding
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Learnable embeddings for time-of-day and day-of-week
        self.hour_embedding = nn.Parameter(torch.randn(24, d_model))
        self.day_embedding = nn.Parameter(torch.randn(7, d_model))

    def forward(self, timestamps: List[datetime]) -> torch.Tensor:
        device = self.hour_embedding.device
        seq_len = len(timestamps)

        # Convert to hour/day indices
        hours = torch.tensor([ts.hour for ts in timestamps], dtype=torch.long, device=device)
        days = torch.tensor([ts.weekday() for ts in timestamps], dtype=torch.long, device=device)

        # Combine sinusoidal + learned embeddings
        out = self.pe[:seq_len].to(device)
        out = out + self.hour_embedding[hours] + self.day_embedding[days]
        return out  # (seq_len, d_model)


class EnhancedTransformerLayer(nn.TransformerEncoderLayer):
    """
    A standard TransformerEncoderLayer with an additional gating mechanism.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = kwargs.get('d_model', 256)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        att_output = super().forward(src, *args, **kwargs)
        gate_values = torch.sigmoid(self.gate(src))
        return att_output * gate_values


class EnhancedBuoyHead(nn.Module):
    """
    Predicts buoy states from final token representations.
    """
    def __init__(self, d_model: int, feature_dim: int, dropout: float):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, feature_dim)
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor, missing: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.projector(x)
        if missing is not None:
            out = out * (1 - missing.unsqueeze(-1).float())
        return self.norm(out)


class EnhancedNetworkHead(nn.Module):
    """
    Aggregates the network state using a global token that attends over all buoys.
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, missing: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        global_tokens = self.global_token.repeat(batch_size, 1, 1)
        out, _ = self.attention(global_tokens, x, x)
        out = self.dropout(out)
        return self.norm(out)  # (batch_size, 1, d_model)


class EnhancedSurfSpotHead(nn.Module):
    """
    Uses learnable surf spot tokens to attend over all buoys and produce surf conditions.
    """
    def __init__(self, d_model: int, num_spots: int, num_conditions: int, dropout: float):
        super().__init__()
        self.spot_tokens = nn.Parameter(torch.randn(num_spots, 1, d_model))
        self.attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_conditions)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, missing: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        spot_tokens = self.spot_tokens.repeat(1, batch_size, 1).transpose(0, 1)
        attended, _ = self.attention(spot_tokens, x, x)
        attended = self.norm(attended)
        conditions = self.predictor(attended)  # (batch_size, num_spots, num_conditions)
        return conditions


class EnhancedWaveTransformer(nn.Module):
    """
    A transformer that processes sequences of shape:
      (batch_size, seq_len, num_buoys, feature_dim)
    and integrates advanced temporal embeddings, buoy embeddings, and a final attention step.
    """
    def __init__(
        self,
        num_buoys: int,
        feature_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_surf_spots: Optional[int] = None,
        dropout: float = 0.1,
        max_sequence_length: int = 168
    ):
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.buoy_embedding = nn.Parameter(torch.randn(num_buoys, d_model))
        self.temporal_embedding = TemporalEmbedding(max_sequence_length, d_model)

        encoder_layer = EnhancedTransformerLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        self.final_attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

        # Heads
        self.buoy_head = EnhancedBuoyHead(d_model, feature_dim, dropout)
        self.network_head = EnhancedNetworkHead(d_model, dropout)
        if num_surf_spots is not None:
            self.surf_head = EnhancedSurfSpotHead(d_model, num_surf_spots, 4, dropout)

    def _create_attention_mask(
        self,
        sequence_mask: Optional[torch.Tensor],
        missing: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if sequence_mask is None and missing is None:
            return None
        masks = []
        if sequence_mask is not None:
            masks.append(sequence_mask.bool())
        if missing is not None:
            masks.append(missing.bool())
        final_mask = torch.stack(masks).any(dim=0)
        return final_mask

    def forward(
        self,
        x: torch.Tensor,
        timestamps: List[List[datetime]],
        mask: Optional[torch.Tensor] = None,
        missing_buoys: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        x: (batch_size, seq_len, num_buoys, feature_dim)
        timestamps: list of lists of datetime objects, shape = (batch_size, seq_len)
        mask: optional boolean mask of shape (batch_size, seq_len, num_buoys)
        missing_buoys: optional float mask of shape (batch_size, seq_len, num_buoys), 1 = missing
        """
        bsz, seq_len, num_buoys, feat_dim = x.shape

        # Project features to d_model
        x = self.feature_projection(x)  # (bsz, seq_len, num_buoys, d_model)

        # Add buoy embeddings
        buoy_emb = self.buoy_embedding.unsqueeze(0).unsqueeze(0)  # (1,1,num_buoys,d_model)
        if missing_buoys is not None:
            buoy_emb = buoy_emb * (1 - missing_buoys.unsqueeze(-1).float())
        x = x + buoy_emb

        # Temporal embeddings per sequence
        device = x.device
        batch_temporal = []
        for i in range(bsz):
            temp_enc = self.temporal_embedding(timestamps[i])  # (seq_len, d_model)
            batch_temporal.append(temp_enc)
        temporal_stack = torch.stack(batch_temporal, dim=0).unsqueeze(2)  # (bsz, seq_len, 1, d_model)
        x = x + temporal_stack

        # Flatten time+buoys
        x = x.view(bsz, seq_len*num_buoys, -1)

        # Flatten masks
        if mask is not None:
            mask = mask.view(bsz, seq_len*num_buoys)
        if missing_buoys is not None:
            missing_buoys = missing_buoys.view(bsz, seq_len*num_buoys)
        attn_mask = self._create_attention_mask(mask, missing_buoys)

        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=attn_mask)  # (bsz, seq_len*num_buoys, d_model)

        # Reshape back to (bsz, seq_len, num_buoys, d_model)
        encoded = encoded.view(bsz, seq_len, num_buoys, -1)

        # Final attention step: for each buoy, the last timestep's token attends to the entire sequence
        encoded_perm = encoded.permute(0, 2, 1, 3)  # (bsz, num_buoys, seq_len, d_model)
        encoded_flat = encoded_perm.reshape(bsz*num_buoys, seq_len, -1)
        query = encoded_flat[:, -1:, :]  # (bsz*num_buoys, 1, d_model)
        attn_out, _ = self.final_attention(query, encoded_flat, encoded_flat)
        final_attended = attn_out.view(bsz, num_buoys, -1)

        # Prediction heads
        if missing_buoys is not None:
            final_missing = missing_buoys[:, -num_buoys:]
        else:
            final_missing = None

        buoy_states = self.buoy_head(final_attended, final_missing)
        network_state = self.network_head(final_attended, final_missing)
        outputs = {
            'buoy_states': buoy_states,
            'network_state': network_state
        }
        if hasattr(self, 'surf_head'):
            outputs['surf_conditions'] = self.surf_head(final_attended, final_missing)
        return outputs
