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

# --- FIX: Added real station coordinates ---
# Source: NOAA National Data Buoy Center
STATION_COORDS = {
    "46001": [56.299, -147.893], # Gulf of Alaska
    "46013": [38.231, -123.327], # Bodega Bay, CA
    "46014": [37.755, -122.845], # Point Reyes, CA
    "46022": [40.738, -124.529], # Eureka, CA
    "46026": [35.750, -121.899], # Harvest, CA
    "46027": [34.250, -120.450], # Santa Maria, CA
    "46054": [33.743, -119.181], # W. Santa Barbara Channel
    "46059": [33.918, -120.762], # Offshore Santa Barbara
    "46060": [32.500, -118.000], # San Clemente Basin, CA
    "46086": [32.533, -118.336]  # San Diego, CA
}


# --- FIX: Moved collate_fn to top level to be picklable by multiprocessing workers ---
def wave_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for the WaveSequenceDataset.
    Pads sequences to the maximum number of stations in a batch.
    """
    # Find maximum number of stations in batch
    max_stations = max(item['n_stations'].item() for item in batch)
    batch_size = len(batch)
    
    # Check if batch is empty or has items without sequence features
    if not batch or 'sequence_features' not in batch[0]:
        return {}

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
            df_interp[feature_group] = df_interp[feature_group].interpolate(method='linear', limit=3, limit_direction='both')
        
        # Circular interpolation for directional data
        for feature in ['primary_direction', 'secondary_direction', 'wind_direction']:
            if feature in df_interp.columns:
                # Convert to radians, interpolate, convert back
                angles_rad = np.radians(df_interp[feature])
                x_comp = np.cos(angles_rad)
                y_comp = np.sin(angles_rad)
                x_interp = pd.Series(x_comp, index=df.index).interpolate(method='linear', limit=3, limit_direction='both')
                y_interp = pd.Series(y_comp, index=df.index).interpolate(method='linear', limit=3, limit_direction='both')
                angles_interp = np.degrees(np.arctan2(y_interp, x_interp)) % 360
                df_interp[feature] = angles_interp
        
        # --- FIX: Replaced deprecated fillna(method='ffill') with ffill() ---
        # Forward fill for remaining directional spread features
        remaining_directional = [f for f in directional_features if f not in ['primary_direction', 'secondary_direction']]
        df_interp[remaining_directional] = df_interp[remaining_directional].ffill(limit=3).bfill(limit=3)
        
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
        total_len = len(common_timestamps)
        for i in range(total_len - self.sequence_length - self.prediction_horizon + 1):
            start_time = common_timestamps[i]
            end_time = common_timestamps[i + self.sequence_length - 1]
            target_time = common_timestamps[i + self.sequence_length + self.prediction_horizon - 1]
            
            sequence_window = common_timestamps[i : i + self.sequence_length]
            
            # Collect data from available stations
            station_features = []
            station_ids = []
            station_coords_list = []
            
            for station_id, df in self.processed_data.items():
                # Check if we have data for this time window
                # Use reindex to get a complete sequence window with NaNs for missing times
                sequence_df = df.reindex(sequence_window)
                
                # Check for data availability (e.g., at least 70% non-NaN rows)
                if sequence_df.notna().all(axis=1).sum() >= self.sequence_length * 0.7:
                    # If there are still some NaNs after reindexing, interpolate them
                    if sequence_df.isna().any().any():
                       sequence_df = self._interpolate_physics_aware(sequence_df)

                    station_features.append(sequence_df.values)
                    station_ids.append(station_id)
                    
                    # --- FIX: Add real station coordinates from dictionary ---
                    station_coords_list.append(STATION_COORDS.get(station_id, [0.0, 0.0]))
            
            # Only create sequence if we have minimum number of stations
            if len(station_features) >= self.min_stations:
                # Get target data
                target_features = []
                for j, station_id in enumerate(station_ids):
                    df = self.processed_data[station_id]
                    if target_time in df.index:
                        target_features.append(df.loc[target_time].values)
                    else:
                        # If target is missing, use NaN placeholder
                        target_features.append(np.full(len(self.feature_names), np.nan))
                
                sequences.append({
                    'sequence_features': np.array(station_features),  # (n_stations, seq_len, 20)
                    'target_features': np.array(target_features),     # (n_stations, 20)
                    'station_ids': station_ids,
                    'station_coords': np.array(station_coords_list),  # (n_stations, 2)
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
            # Reshape, ignoring potential NaN values from padding
            all_features.append(features.reshape(-1, 20))
        
        if not all_features:
            self.feature_means = np.zeros(20)
            self.feature_stds = np.ones(20)
            return

        all_features = np.vstack(all_features)
        
        # Compute normalization parameters, ignoring NaNs
        self.feature_means = np.nanmean(all_features, axis=0)
        self.feature_stds = np.nanstd(all_features, axis=0)
        
        # Prevent division by zero
        self.feature_stds[self.feature_stds == 0] = 1.0
        
        # Special handling for directional features (normalize to sin/cos range)
        directional_indices = [10, 12, 16] # primary, secondary, wind direction
        for idx in directional_indices:
            self.feature_means[idx] = 180.0 # Center around 180
            self.feature_stds[idx] = 180.0  # Scale to [-1, 1] range after centering
        
        # Fraction features are already in [0,1] range
        fraction_indices = [8, 9]  # swell_fraction, windsea_fraction
        for idx in fraction_indices:
            self.feature_means[idx] = 0.5
            self.feature_stds[idx] = 0.5
        
        # Apply normalization to all sequences
        for seq in self.sequences:
            seq['sequence_features'] = (seq['sequence_features'] - self.feature_means) / self.feature_stds
            seq['target_features'] = (seq['target_features'] - self.feature_means) / self.feature_stds
            # NaN values created by normalization should be handled by the model or collate_fn
            np.nan_to_num(seq['sequence_features'], copy=False, nan=0.0)
            np.nan_to_num(seq['target_features'], copy=False, nan=0.0)

    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        return {
            'sequence_features': torch.tensor(seq['sequence_features'], dtype=torch.float32),
            'target_features': torch.tensor(seq['target_features'], dtype=torch.float32),
            'station_coords': torch.tensor(seq['station_coords'], dtype=torch.float32),
            'n_stations': torch.tensor(len(seq['station_ids']), dtype=torch.long),
            'station_ids': seq['station_ids'],  # Keep as strings for reference
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
    num_workers: int = 2,
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
    
    # --- FIX: Use top-level collate_fn for multiprocessing compatibility ---
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data for better generalization
        collate_fn=wave_collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        collate_fn=wave_collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader