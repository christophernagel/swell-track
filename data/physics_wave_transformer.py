"""
physics_wave_transformer.py
---------------------------
Enhanced transformer architecture specifically designed for your 20-dimensional
physics-informed wave features with spatiotemporal modeling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import math


class PhysicsInformedEmbedding(nn.Module):
    """
    Feature embedding layer that respects wave physics relationships.
    Treats different feature groups (spectral, directional, meteorological) appropriately.
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Separate embeddings for different physics domains
        self.spectral_proj = nn.Linear(10, d_model // 2)      # Features 0-9
        self.directional_proj = nn.Linear(5, d_model // 4)    # Features 10-14  
        self.meteorological_proj = nn.Linear(3, d_model // 8) # Features 15-17
        self.validation_proj = nn.Linear(2, d_model // 8)     # Features 18-19
        
        # Final projection to combine all physics domains
        self.combined_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_stations, seq_len, 20)
        returns: (batch_size, n_stations, seq_len, d_model)
        """
        # Split features by physics domain
        spectral = x[..., :10]          # Wave energy, periods, swell/wind
        directional = x[..., 10:15]     # Wave directions and spreading
        meteorological = x[..., 15:18]  # Wind conditions
        validation = x[..., 18:20]      # Cross-validation metrics
        
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
        
        # Final projection and normalization
        embedded = self.combined_proj(combined)
        return self.dropout(self.norm(embedded))


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for modeling wave propagation between stations.
    Uses geographic coordinates and wave physics principles.
    """
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model) 
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Geographic distance encoding
        self.distance_encoding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, num_heads)
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
        
        # Reshape for multi-head attention
        x_flat = x.view(B, N * T, D)
        
        Q = self.q_proj(x_flat).view(B, N * T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_flat).view(B, N * T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_flat).view(B, N * T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add geographic distance bias
        if coords is not None:
            distances = self._compute_distances(coords)  # (B, N, N)
            distance_bias = self.distance_encoding(distances.unsqueeze(-1))  # (B, N, N, num_heads)
            
            # Expand distance bias to match attention dimensions
            distance_bias = distance_bias.permute(0, 3, 1, 2)  # (B, num_heads, N, N)
            distance_bias = distance_bias.repeat_interleave(T, dim=2).repeat_interleave(T, dim=3)
            
            attn_scores = attn_scores + distance_bias
        
        # Apply station mask
        if station_mask is not None:
            # Create expanded mask for sequence dimension
            expanded_mask = station_mask.unsqueeze(1).repeat_interleave(T, dim=1)  # (B, N*T)
            mask_2d = expanded_mask.unsqueeze(1) & expanded_mask.unsqueeze(2)  # (B, N*T, N*T)
            mask_2d = mask_2d.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # (B, num_heads, N*T, N*T)
            
            attn_scores = attn_scores.masked_fill(~mask_2d, float('-inf'))
        
        # Compute attention weights and apply to values
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N * T, D)
        output = self.out_proj(attn_output)
        
        return output.view(B, N, T, D)
    
    def _compute_distances(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute haversine distances between station coordinates"""
        # coords: (B, N, 2) - lat/lon in degrees
        lat = coords[..., 0] * math.pi / 180  # Convert to radians
        lon = coords[..., 1] * math.pi / 180
        
        # Broadcast for pairwise distance computation
        lat1 = lat.unsqueeze(2)  # (B, N, 1)
        lon1 = lon.unsqueeze(2)
        lat2 = lat.unsqueeze(1)  # (B, 1, N)
        lon2 = lon.unsqueeze(1)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        
        return 6371.0 * c  # Earth radius in km


class WavePhysicsTransformerLayer(nn.Module):
    """
    Transformer layer enhanced with wave physics principles.
    Combines temporal self-attention with spatial wave propagation modeling.
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8, 
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        # Temporal self-attention
        self.temporal_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Spatial attention for wave propagation
        self.spatial_attention = SpatialAttention(d_model, num_heads)
        
        # Feed-forward network with wave physics constraints
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Gate for combining temporal and spatial information
        self.temporal_spatial_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                station_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch_size, n_stations, seq_len, d_model)
        coords: (batch_size, n_stations, 2)
        station_mask: (batch_size, n_stations)
        """
        B, N, T, D = x.shape
        
        # Temporal attention (process each station independently)
        x_temporal = x.view(B * N, T, D)
        attn_output, _ = self.temporal_attention(x_temporal, x_temporal, x_temporal)
        x_temporal = self.norm1(x + attn_output.view(B, N, T, D))
        
        # Spatial attention (model wave propagation between stations)
        x_spatial = self.spatial_attention(x_temporal, coords, station_mask)
        
        # Combine temporal and spatial information with learnable gate
        combined_input = torch.cat([x_temporal, x_spatial], dim=-1)
        gate = self.temporal_spatial_gate(combined_input)
        x_combined = self.norm2(x_temporal + gate * x_spatial)
        
        # Feed-forward network
        ff_output = self.feedforward(x_combined.view(B * N * T, D)).view(B, N, T, D)
        
        return self.norm3(x_combined + ff_output)


class PhysicsInformedWaveTransformer(nn.Module):
    """
    Complete physics-informed wave transformer for your 20-dimensional features.
    Designed specifically for multi-station wave forecasting with proper wave physics.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_sequence_length: int = 168  # 1 week at hourly resolution
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Physics-informed feature embedding
        self.feature_embedding = PhysicsInformedEmbedding(d_model)
        
        # Learnable positional encoding for temporal sequences
        self.pos_encoding = nn.Parameter(torch.randn(max_sequence_length, d_model))
        
        # Stack of transformer layers with wave physics
        self.transformer_layers = nn.ModuleList([
            WavePhysicsTransformerLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads for different prediction tasks
        self.spectral_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 10)  # Spectral features 0-9
        )
        
        self.directional_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 5)   # Directional features 10-14
        )
        
        self.meteorological_head = nn.Sequential(
            nn.Linear(d_model, d_model // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 8, 3)   # Meteorological features 15-17
        )
        
        self.validation_head = nn.Sequential(
            nn.Linear(d_model, d_model // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 8, 2)   # Validation features 18-19
        )
        
        # Combined prediction head
        self.combined_head = nn.Linear(d_model, 20)
        
    def forward(self, sequence_features: torch.Tensor, station_coords: torch.Tensor,
                station_mask: torch.Tensor, output_physics_separate: bool = False) -> Dict[str, torch.Tensor]:
        """
        sequence_features: (batch_size, n_stations, seq_len, 20)
        station_coords: (batch_size, n_stations, 2)
        station_mask: (batch_size, n_stations) - True for valid stations
        """
        B, N, T, F = sequence_features.shape
        
        # Physics-informed feature embedding
        x = self.feature_embedding(sequence_features)  # (B, N, T, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:T].unsqueeze(0).unsqueeze(0)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, station_coords, station_mask)
        
        # Use final timestep for prediction
        final_features = x[:, :, -1, :]  # (B, N, d_model)
        
        if output_physics_separate:
            # Separate outputs for different physics domains
            outputs = {
                'spectral': self.spectral_head(final_features),
                'directional': self.directional_head(final_features),
                'meteorological': self.meteorological_head(final_features),
                'validation': self.validation_head(final_features)
            }
            
            # Combine all outputs
            outputs['combined'] = torch.cat([
                outputs['spectral'],
                outputs['directional'], 
                outputs['meteorological'],
                outputs['validation']
            ], dim=-1)
            
        else:
            # Single combined output
            outputs = {
                'combined': self.combined_head(final_features)
            }
        
        return outputs


class PhysicsInformedLoss(nn.Module):
    """
    Multi-task loss function that respects wave physics principles.
    Includes energy conservation, dispersion relations, and cross-validation.
    """
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        # Default weights for different physics domains
        self.feature_weights = feature_weights or {
            'spectral': 2.0,        # Higher weight for core wave physics
            'directional': 1.0,     # Directional information
            'meteorological': 0.5,  # Supporting meteorological data
            'validation': 0.3       # Cross-validation metrics
        }
        
        # Loss functions for different feature types
        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor, station_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        predictions: Dict with keys matching physics domains
        targets: (batch_size, n_stations, 20) - target features
        station_mask: (batch_size, n_stations) - valid station mask
        """
        losses = {}
        total_loss = 0.0
        
        # Get combined predictions
        if 'combined' in predictions:
            pred_combined = predictions['combined']
        else:
            # Reconstruct from separate predictions
            pred_combined = torch.cat([
                predictions['spectral'],
                predictions['directional'],
                predictions['meteorological'], 
                predictions['validation']
            ], dim=-1)
        
        # Apply station mask
        valid_pred = pred_combined[station_mask]  # (n_valid_stations, 20)
        valid_target = targets[station_mask]      # (n_valid_stations, 20)
        
        if len(valid_pred) == 0:
            return {'total': torch.tensor(0.0, requires_grad=True)}
        
        # Spectral physics loss (features 0-9)
        spectral_pred = valid_pred[:, :10]
        spectral_target = valid_target[:, :10]
        
        # Standard MSE for most spectral features
        spectral_mse = self.mse_loss(spectral_pred, spectral_target).mean()
        
        # Special handling for wave height (feature 0) - most important
        height_loss = self.mse_loss(spectral_pred[:, 0], spectral_target[:, 0]).mean()
        
        # Energy conservation constraint (features 4, 6, 7)
        # total_energy should approximately equal swell_energy + windsea_energy
        total_energy_pred = spectral_pred[:, 4]
        swell_energy_pred = spectral_pred[:, 6] 
        windsea_energy_pred = spectral_pred[:, 7]
        energy_conservation_loss = self.mse_loss(
            total_energy_pred, 
            swell_energy_pred + windsea_energy_pred
        ).mean()
        
        # Fraction constraint (features 8, 9 should sum to ~1)
        fraction_sum = spectral_pred[:, 8] + spectral_pred[:, 9]
        fraction_constraint_loss = self.mse_loss(
            fraction_sum, 
            torch.ones_like(fraction_sum)
        ).mean()
        
        spectral_loss = (spectral_mse + 2.0 * height_loss + 
                        0.5 * energy_conservation_loss + 
                        0.3 * fraction_constraint_loss)
        
        losses['spectral'] = spectral_loss
        total_loss += self.feature_weights['spectral'] * spectral_loss
        
        # Directional physics loss (features 10-14)
        directional_pred = valid_pred[:, 10:15]
        directional_target = valid_target[:, 10:15]
        
        # Circular loss for directional features (10, 12)
        direction_indices = [0, 2]  # primary_direction, secondary_direction
        directional_loss = 0.0
        
        for i in direction_indices:
            if i < directional_pred.shape[1]:
                # Convert to radians for circular distance
                pred_rad = directional_pred[:, i] * torch.pi / 180
                target_rad = directional_target[:, i] * torch.pi / 180
                
                # Circular distance
                angle_diff = torch.atan2(torch.sin(pred_rad - target_rad), 
                                       torch.cos(pred_rad - target_rad))
                circular_loss = (angle_diff ** 2).mean()
                directional_loss += circular_loss
        
        # Standard MSE for spread features (11, 13, 14)
        spread_pred = directional_pred[:, [1, 3, 4]]  # primary_spread, bimodal_strength, directional_separation
        spread_target = directional_target[:, [1, 3, 4]]
        spread_loss = self.mse_loss(spread_pred, spread_target).mean()
        
        directional_loss += spread_loss
        losses['directional'] = directional_loss
        total_loss += self.feature_weights['directional'] * directional_loss
        
        # Meteorological physics loss (features 15-17)
        if valid_pred.shape[1] > 15:
            meteorological_pred = valid_pred[:, 15:18]
            meteorological_target = valid_target[:, 15:18]
            
            # Standard MSE for wind speed
            wind_speed_loss = self.mse_loss(meteorological_pred[:, 0], meteorological_target[:, 0]).mean()
            
            # Circular loss for wind direction (feature 16)
            wind_dir_pred_rad = meteorological_pred[:, 1] * torch.pi / 180
            wind_dir_target_rad = meteorological_target[:, 1] * torch.pi / 180
            wind_dir_loss = (torch.atan2(torch.sin(wind_dir_pred_rad - wind_dir_target_rad),
                                       torch.cos(wind_dir_pred_rad - wind_dir_target_rad)) ** 2).mean()
            
            # Standard MSE for wind-wave alignment
            alignment_loss = self.mse_loss(meteorological_pred[:, 2], meteorological_target[:, 2]).mean()
            
            meteorological_loss = wind_speed_loss + wind_dir_loss + alignment_loss
            losses['meteorological'] = meteorological_loss
            total_loss += self.feature_weights['meteorological'] * meteorological_loss
        
        # Validation loss (features 18-19)
        if valid_pred.shape[1] > 18:
            validation_pred = valid_pred[:, 18:20]
            validation_target = valid_target[:, 18:20]
            
            validation_loss = self.mse_loss(validation_pred, validation_target).mean()
            losses['validation'] = validation_loss
            total_loss += self.feature_weights['validation'] * validation_loss
        
        losses['total'] = total_loss
        return losses


# Example usage and training setup
def create_physics_model(device: str = 'cuda'):
    """Create the complete physics-informed wave forecasting model"""
    model = PhysicsInformedWaveTransformer(
        d_model=256,
        num_heads=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)
    
    # Physics-informed loss function
    criterion = PhysicsInformedLoss().to(device)
    
    # Optimizer with different learning rates for different components
    optimizer = torch.optim.AdamW([
        {'params': model.feature_embedding.parameters(), 'lr': 1e-4},
        {'params': model.transformer_layers.parameters(), 'lr': 5e-5},
        {'params': [model.spectral_head.parameters(), model.combined_head.parameters()], 'lr': 1e-4},
        {'params': [model.directional_head.parameters(), model.meteorological_head.parameters()], 'lr': 5e-5}
    ], weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    return model, criterion, optimizer, scheduler