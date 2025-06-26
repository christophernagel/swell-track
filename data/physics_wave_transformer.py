# physics_wave_transformer.py
# Fixed version with numerical stability improvements

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict
import math
from itertools import chain

class PhysicsInformedEmbedding(nn.Module):
    """
    Feature embedding layer that respects wave physics relationships.
    Treats different feature groups (spectral, directional, meteorological) appropriately.
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Define projection sizes that sum up correctly
        d_spec = d_model // 2
        d_dir = d_model // 4
        d_met = d_model // 8
        d_val = d_model - d_spec - d_dir - d_met # The rest

        # Separate embeddings for different physics domains
        self.spectral_proj = nn.Linear(10, d_spec)      # Features 0-9
        self.directional_proj = nn.Linear(5, d_dir)    # Features 10-14  
        self.meteorological_proj = nn.Linear(3, d_met) # Features 15-17
        self.validation_proj = nn.Linear(2, d_val)     # Features 18-19
        
        # Final projection to combine all physics domains
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_stations, seq_len, 20)
        returns: (batch_size, n_stations, seq_len, d_model)
        """
        # Split features by physics domain
        spectral = x[..., :10]
        directional = x[..., 10:15]
        meteorological = x[..., 15:18]
        validation = x[..., 18:20]
        
        # Project each domain
        spectral_emb = self.spectral_proj(spectral)
        directional_emb = self.directional_proj(directional)
        meteorological_emb = self.meteorological_proj(meteorological)
        validation_emb = self.validation_proj(validation)
        
        # Combine embeddings
        combined = torch.cat([
            spectral_emb, 
            directional_emb, 
            meteorological_emb, 
            validation_emb
        ], dim=-1)
        
        return self.dropout(self.norm(combined))

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for modeling wave propagation between stations.
    Uses geographic coordinates to bias attention scores.
    """
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model) 
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.distance_encoding = nn.Sequential(
            nn.Linear(1, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, self.num_heads)
        )
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, coords: torch.Tensor, 
                station_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch_size, n_stations, seq_len, d_model)
        coords: (batch_size, n_stations, 2) - lat/lon coordinates
        station_mask: (batch_size, n_stations) - valid station mask
        """
        B, N, T, D = x.shape
        
        # We perform attention over stations. Use mean over time as representative state.
        x_repr = x.mean(dim=2)  # (B, N, D)
        
        Q = self.q_proj(x_repr).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_repr).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_repr).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if coords is not None:
            distances = self._compute_distances(coords)
            distance_bias = self.distance_encoding(distances.unsqueeze(-1)).permute(0, 3, 1, 2)
            attn_scores = attn_scores + distance_bias
        
        if station_mask is not None:
            mask = station_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        output = self.out_proj(attn_output)
        
        return output.unsqueeze(2).expand_as(x)
    
    def _compute_distances(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute haversine distances between station coordinates - STABILIZED VERSION"""
        R = 6371.0  # Earth radius in km
        lat = torch.deg2rad(coords[..., 0])
        lon = torch.deg2rad(coords[..., 1])
        
        lat1, lon1 = lat.unsqueeze(2), lon.unsqueeze(2)
        lat2, lon2 = lat.unsqueeze(1), lon.unsqueeze(1)
        
        dlat, dlon = lat2 - lat1, lon2 - lon1
        
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        
        # CRITICAL FIX: Clamp to prevent sqrt of negative numbers from floating point errors
        a = torch.clamp(a, 0, 1)
        
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        return R * c

class WavePhysicsTransformerLayer(nn.Module):
    """
    Transformer layer that combines temporal self-attention with spatial attention
    using a learnable gate. This is a more powerful, balanced approach.
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8, 
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.spatial_attention = SpatialAttention(d_model, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Gated fusion for temporal and spatial information
        self.temporal_spatial_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                station_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, T, D = x.shape
        x_flat = x.view(B * N, T, D)
        
        # 1. Temporal Attention
        attn_temporal, _ = self.temporal_attention(x_flat, x_flat, x_flat)
        x_temporal = x + self.dropout(attn_temporal.view(B, N, T, D))
        x_temporal = self.norm1(x_temporal)

        # 2. Spatial Attention
        x_spatial = self.spatial_attention(x_temporal, coords, station_mask)
        
        # 3. Gated Fusion
        gate_input = torch.cat([x_temporal, x_spatial], dim=-1)
        gate = self.temporal_spatial_gate(gate_input)
        x_fused = self.norm2(x_temporal + gate * self.dropout(x_spatial))
        
        # 4. Feed-forward Network
        ff_output = self.feedforward(x_fused)
        x_out = self.norm3(x_fused + self.dropout(ff_output))
        
        return x_out

class PhysicsInformedWaveTransformer(nn.Module):
    """
    Complete physics-informed wave transformer with separate output heads
    to preserve domain-specific context.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_sequence_length: int = 168
    ):
        super().__init__()
        self.d_model = d_model
        self.feature_embedding = PhysicsInformedEmbedding(d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_sequence_length, d_model), requires_grad=False)
        self._generate_positional_encoding(max_sequence_length)

        self.transformer_layers = nn.ModuleList([
            WavePhysicsTransformerLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.spectral_head = nn.Linear(d_model, 10)
        self.directional_head = nn.Linear(d_model, 5)
        self.meteorological_head = nn.Linear(d_model, 3)
        self.validation_head = nn.Linear(d_model, 2)
        
    def _generate_positional_encoding(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoder.data[0, :, :] = pe
        
    def forward(self, sequence_features: torch.Tensor, station_coords: torch.Tensor,
                station_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        B, N, T, F = sequence_features.shape
        
        x = self.feature_embedding(sequence_features) * math.sqrt(self.d_model)
        x = x + self.pos_encoder[:, :T, :]
        
        for layer in self.transformer_layers:
            x = layer(x, station_coords, station_mask)
        
        final_features = x[:, :, -1, :]  # (B, N, d_model)
        
        spectral_pred = self.spectral_head(final_features)
        directional_pred = self.directional_head(final_features)
        meteorological_pred = self.meteorological_head(final_features)
        validation_pred = self.validation_head(final_features)
        
        combined_pred = torch.cat([spectral_pred, directional_pred, meteorological_pred, validation_pred], dim=-1)

        return {
            'spectral': spectral_pred,
            'directional': directional_pred,
            'meteorological': meteorological_pred,
            'validation': validation_pred,
            'combined': combined_pred
        }

class PhysicsInformedLoss(nn.Module):
    """
    Loss function that handles separate physics-domain outputs and enforces constraints.
    """
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.feature_weights = feature_weights or {
            'height': 3.0, 'spectral': 2.0, 'directional': 1.5,
            'meteorological': 1.0, 'validation': 0.5,
            'conservation': 0.5, 'fraction': 0.5
        }
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor, station_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        losses = {}
        total_loss = torch.tensor(0.0, device=targets.device)
        
        valid_mask = station_mask.bool()
        if not valid_mask.any():
            return {'total': total_loss, 'height_mae': torch.tensor(0.0)}

        # Spectral loss (features 0-9)
        pred_spec = predictions['spectral'][valid_mask]
        targ_spec = targets[valid_mask][:, :10]
        loss_spec = self.mse_loss(pred_spec, targ_spec)
        loss_height = self.mse_loss(pred_spec[:, 0], targ_spec[:, 0])
        total_loss += self.feature_weights['spectral'] * loss_spec
        total_loss += self.feature_weights['height'] * loss_height
        losses['spectral_loss'] = loss_spec + loss_height

        # Directional loss (features 10-14)
        pred_dir = predictions['directional'][valid_mask]
        targ_dir = targets[valid_mask][:, 10:15]
        # Circular for angles
        pred_angles = torch.deg2rad(pred_dir[:, [0, 2]])
        targ_angles = torch.deg2rad(targ_dir[:, [0, 2]])
        angle_diff = torch.atan2(torch.sin(pred_angles - targ_angles), torch.cos(pred_angles - targ_angles))
        loss_circ = (angle_diff**2).mean()
        loss_spread = self.mse_loss(pred_dir[:, [1, 3, 4]], targ_dir[:, [1, 3, 4]])
        loss_directional = loss_circ + loss_spread
        total_loss += self.feature_weights['directional'] * loss_directional
        losses['directional_loss'] = loss_directional

        # Meteorological loss (features 15-17)
        pred_met = predictions['meteorological'][valid_mask]
        targ_met = targets[valid_mask][:, 15:18]
        pred_wind_rad = torch.deg2rad(pred_met[:, 1])
        targ_wind_rad = torch.deg2rad(targ_met[:, 1])
        wind_diff = torch.atan2(torch.sin(pred_wind_rad - targ_wind_rad), torch.cos(pred_wind_rad - targ_wind_rad))
        loss_wind_circ = (wind_diff**2).mean()
        loss_wind_other = self.mse_loss(pred_met[:, [0, 2]], targ_met[:, [0, 2]])
        loss_met = loss_wind_circ + loss_wind_other
        total_loss += self.feature_weights['meteorological'] * loss_met
        losses['meteorological_loss'] = loss_met

        # Validation loss (features 18-19)
        pred_val = predictions['validation'][valid_mask]
        targ_val = targets[valid_mask][:, 18:20]
        loss_val = self.mse_loss(pred_val, targ_val)
        total_loss += self.feature_weights['validation'] * loss_val
        losses['validation_loss'] = loss_val

        # Physics constraints on predictions
        loss_cons = self.mse_loss(pred_spec[:, 4], pred_spec[:, 6] + pred_spec[:, 7])
        loss_frac = self.mse_loss(pred_spec[:, 8] + pred_spec[:, 9], torch.ones_like(pred_spec[:, 8]))
        total_loss += self.feature_weights['conservation'] * loss_cons
        total_loss += self.feature_weights['fraction'] * loss_frac
        losses['physics_constraints'] = loss_cons + loss_frac

        losses['total'] = total_loss
        losses['height_mae'] = torch.nn.functional.l1_loss(pred_spec[:, 0], targ_spec[:, 0])
        return losses


def create_physics_model(device: str = 'cuda'):
    """Create the complete physics-informed wave forecasting model"""
    model = PhysicsInformedWaveTransformer(
        d_model=256, num_heads=8, num_layers=6,
        dim_feedforward=1024, dropout=0.1
    ).to(device)
    
    criterion = PhysicsInformedLoss().to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.feature_embedding.parameters(), 'lr': 1e-4},
        {'params': model.transformer_layers.parameters(), 'lr': 5e-5},
        {'params': model.spectral_head.parameters(), 'lr': 1e-4},
        {'params': model.directional_head.parameters(), 'lr': 5e-5},
        {'params': model.meteorological_head.parameters(), 'lr': 5e-5},
        {'params': model.validation_head.parameters(), 'lr': 5e-5}
    ], weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=5, verbose=True
    )
    
    return model, criterion, optimizer, scheduler