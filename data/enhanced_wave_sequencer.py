"""
enhanced_wave_sequencer.py
--------------------------
Enhanced WaveSequenceDataset for 20-dimensional physics-informed features
with proper handling of multi-station spatiotemporal sequences.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path


class EnhancedWaveSequenceDataset(Dataset):
    """
    Enhanced dataset for 20-dimensional physics-informed wave features.
    
    Features:
    - Handles multi-station spatiotemporal sequences
    - Physics-informed normalization
    - Missing data interpolation
    - Station-aware batching
    """
    
    def __init__(
        self,
        features_file: str,
        sequence_length: int = 24,
        prediction_horizon: int = 1,
        min_stations: int = 3,
        normalize_features: bool = True,
        interpolate_missing: bool = True
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.min_stations = min_stations
        self.normalize_features = normalize_features
        self.interpolate_missing = interpolate_missing
        
        # Load enhanced features
        with open(features_file, 'r') as f:
            self.raw_data = json.load(f)
        
        # Feature names from your enhanced processor
        self.feature_names = [
            # Spectral physics (0-9)
            'sig_height', 'peak_period', 'mean_period', 'peak_freq', 'total_energy',
            'spectral_width', 'swell_energy', 'windsea_energy', 'swell_fraction', 'windsea_fraction',
            # Directional physics (10-14)  
            'primary_direction', 'primary_spread', 'secondary_direction', 'bimodal_strength', 'directional_separation',
            # Meteorological physics (15-17)
            'wind_speed', 'wind_direction', 'wind_wave_alignment',
            # Validation features (18-19)
            'spectral_hs_error', 'spectral_tp_error'
        ]
        
        # Process and prepare sequences
        self.processed_data = self._process_station_data()
        self.sequences = self._create_sequences()
        
        if self.normalize_features:
            self._normalize_features()
    
    def _process_station_data(self) -> Dict[str, pd.DataFrame]:
        """Convert raw data to station DataFrames with proper timestamp handling"""
        processed = {}
        
        for station_id, station_data in self.raw_data.items():
            if 'enhanced_features' not in station_data:
                continue
                
            # Convert to DataFrame
            features = np.array(station_data['enhanced_features'])
            timestamps = pd.to_datetime(station_data['timestamps'])
            
            df = pd.DataFrame(features, columns=self.feature_names, index=timestamps)
            df = df.sort_index()  # Ensure temporal order
            
            # Handle missing data if requested
            if self.interpolate_missing:
                df = self._interpolate_physics_aware(df)
            
            processed[station_id] = df
            
        return processed
    
    def _interpolate_physics_aware(self, df: pd.DataFrame) -> pd.DataFrame:
        """Physics-informed interpolation for missing values"""
        df_interp = df.copy()
        
        # Group features by physics type for appropriate interpolation
        spectral_features = self.feature_names[0:10]
        directional_features = self.feature_names[10:15]
        meteorological_features = self.feature_names[15:18]
        validation_features = self.feature_names[18:20]
        
        # Linear interpolation for continuous physics variables
        for feature_group in [spectral_features, meteorological_features, validation_features]:
            df_interp[feature_group] = df_interp[feature_group].interpolate(method='linear', limit=3)
        
        # Circular interpolation for directional data
        for feature in ['primary_direction', 'secondary_direction', 'wind_direction']:
            if feature in df_interp.columns:
                # Convert to radians, interpolate, convert back
                angles_rad = np.radians(df_interp[feature])
                x = np.cos(angles_rad)
                y = np.sin(angles_rad)
                x_interp = pd.Series(x).interpolate(method='linear', limit=2)
                y_interp = pd.Series(y).interpolate(method='linear', limit=2)
                angles_interp = np.degrees(np.arctan2(y_interp, x_interp)) % 360
                df_interp[feature] = angles_interp
        
        # Forward fill for remaining directional spread features
        remaining_directional = [f for f in directional_features if f not in ['primary_direction', 'secondary_direction']]
        df_interp[remaining_directional] = df_interp[remaining_directional].fillna(method='ffill', limit=2)
        
        return df_interp
    
    def _create_sequences(self) -> List[Dict]:
        """Create spatiotemporal sequences from multi-station data"""
        sequences = []
        
        # Find common time ranges across stations
        all_timestamps = set()
        for df in self.processed_data.values():
            all_timestamps.update(df.index)
        
        common_timestamps = sorted(list(all_timestamps))
        
        # Create sequences
        for i in range(len(common_timestamps) - self.sequence_length - self.prediction_horizon):
            start_time = common_timestamps[i]
            end_time = common_timestamps[i + self.sequence_length - 1]
            target_time = common_timestamps[i + self.sequence_length + self.prediction_horizon - 1]
            
            # Collect data from available stations
            station_features = []
            station_ids = []
            station_coords = []
            
            for station_id, df in self.processed_data.items():
                # Check if we have data for this time window
                sequence_data = []
                valid_count = 0
                
                for t in range(i, i + self.sequence_length):
                    timestamp = common_timestamps[t]
                    if timestamp in df.index:
                        features = df.loc[timestamp].values
                        if not np.all(np.isnan(features[:10])):  # At least spectral features present
                            sequence_data.append(features)
                            valid_count += 1
                        else:
                            sequence_data.append(np.full(20, np.nan))
                    else:
                        sequence_data.append(np.full(20, np.nan))
                
                # Include station if we have enough valid data
                if valid_count >= self.sequence_length * 0.7:  # 70% data availability
                    station_features.append(np.array(sequence_data))
                    station_ids.append(station_id)
                    
                    # Add station coordinates (from your enhanced stations data)
                    # You'll need to load this from your stations configuration
                    station_coords.append([0.0, 0.0])  # Placeholder - add real coords
            
            # Only create sequence if we have minimum number of stations
            if len(station_features) >= self.min_stations:
                # Get target data
                target_features = []
                for j, station_id in enumerate(station_ids):
                    df = self.processed_data[station_id]
                    if target_time in df.index:
                        target_features.append(df.loc[target_time].values)
                    else:
                        target_features.append(np.full(20, np.nan))
                
                sequences.append({
                    'sequence_features': np.array(station_features),  # (n_stations, seq_len, 20)
                    'target_features': np.array(target_features),     # (n_stations, 20)
                    'station_ids': station_ids,
                    'station_coords': np.array(station_coords),       # (n_stations, 2)
                    'start_time': start_time,
                    'end_time': end_time,
                    'target_time': target_time
                })
        
        return sequences
    
    def _normalize_features(self):
        """Physics-informed feature normalization"""
        # Collect all features for normalization statistics
        all_features = []
        for seq in self.sequences:
            features = seq['sequence_features']  # (n_stations, seq_len, 20)
            all_features.append(features.reshape(-1, 20))
        
        all_features = np.vstack(all_features)
        
        # Compute normalization parameters
        self.feature_means = np.nanmean(all_features, axis=0)
        self.feature_stds = np.nanstd(all_features, axis=0)
        
        # Prevent division by zero
        self.feature_stds[self.feature_stds == 0] = 1.0
        
        # Special handling for directional features (already in appropriate ranges)
        directional_indices = [10, 12, 16]  # primary_direction, secondary_direction, wind_direction
        for idx in directional_indices:
            self.feature_means[idx] = 0.0
            self.feature_stds[idx] = 180.0  # Normalize to [-1, 1] range
        
        # Fraction features are already in [0,1] range
        fraction_indices = [8, 9]  # swell_fraction, windsea_fraction
        for idx in fraction_indices:
            self.feature_means[idx] = 0.5
            self.feature_stds[idx] = 0.5
        
        # Apply normalization to all sequences
        for seq in self.sequences:
            seq['sequence_features'] = (seq['sequence_features'] - self.feature_means) / self.feature_stds
            seq['target_features'] = (seq['target_features'] - self.feature_means) / self.feature_stds
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        return {
            'sequence_features': torch.tensor(seq['sequence_features'], dtype=torch.float32),
            'target_features': torch.tensor(seq['target_features'], dtype=torch.float32),
            'station_coords': torch.tensor(seq['station_coords'], dtype=torch.float32),
            'n_stations': torch.tensor(len(seq['station_ids']), dtype=torch.long),
            'station_ids': seq['station_ids'],  # Keep as strings
            'timestamps': {
                'start': seq['start_time'],
                'end': seq['end_time'], 
                'target': seq['target_time']
            }
        }


def create_enhanced_dataloaders(
    features_file: str,
    sequence_length: int = 24,
    batch_size: int = 16,
    split_ratio: float = 0.8,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/validation dataloaders for enhanced wave data"""
    
    dataset = EnhancedWaveSequenceDataset(
        features_file=features_file,
        sequence_length=sequence_length,
        **dataset_kwargs
    )
    
    # Temporal split (not random) to preserve causality
    total_sequences = len(dataset)
    train_size = int(total_sequences * split_ratio)
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_sequences))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Custom collate function for variable number of stations
    def collate_fn(batch):
        # Find maximum number of stations in batch
        max_stations = max(item['n_stations'].item() for item in batch)
        batch_size = len(batch)
        seq_len = batch[0]['sequence_features'].shape[1]
        feature_dim = batch[0]['sequence_features'].shape[2]
        
        # Pad sequences to same number of stations
        padded_sequences = torch.zeros(batch_size, max_stations, seq_len, feature_dim)
        padded_targets = torch.zeros(batch_size, max_stations, feature_dim)
        padded_coords = torch.zeros(batch_size, max_stations, 2)
        station_masks = torch.zeros(batch_size, max_stations, dtype=torch.bool)
        
        for i, item in enumerate(batch):
            n_stations = item['n_stations'].item()
            padded_sequences[i, :n_stations] = item['sequence_features']
            padded_targets[i, :n_stations] = item['target_features']
            padded_coords[i, :n_stations] = item['station_coords']
            station_masks[i, :n_stations] = True
        
        return {
            'sequence_features': padded_sequences,
            'target_features': padded_targets,
            'station_coords': padded_coords,
            'station_masks': station_masks,
            'batch_timestamps': [item['timestamps'] for item in batch]
        }
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep temporal order
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    return train_loader, val_loader