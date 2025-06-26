# enhanced_wave_sequencer.py
# Fixed version with proper normalization (no data leakage) and robust data handling

import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json
from pathlib import Path

# Station coordinates dictionary
STATION_COORDS = {
    "46001": [56.3, -148.2], "46013": [38.2, -123.3], "46014": [37.8, -122.8],
    "46022": [40.7, -124.5], "46026": [35.7, -121.9], "46027": [34.5, -120.6],
    "46054": [33.7, -118.2], "46059": [32.9, -119.3], "46060": [32.5, -118.0],
    "46086": [30.0, -125.0]
}

FEATURE_NAMES = [
    'sig_height', 'peak_period', 'mean_period', 'peak_freq', 'total_energy',
    'spectral_width', 'swell_energy', 'windsea_energy', 'swell_fraction', 'windsea_fraction',
    'primary_direction', 'primary_spread', 'secondary_direction', 'bimodal_strength', 'directional_separation',
    'wind_speed', 'wind_direction', 'wind_wave_alignment',
    'spectral_hs_error', 'spectral_tp_error'
]

class WaveDataNormalizer:
    """
    Handles the fitting and transformation of wave features.
    This class is fitted ONLY on the training data to prevent leakage.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.means = None
        self.stds = None

    def fit(self, dataset: Subset):
        """
        Calculates mean and std from a dataset subset (training set).
        """
        all_features = []
        for i in range(len(dataset)):
            # Access the underlying dataset's __getitem__
            raw_item = dataset.dataset[dataset.indices[i]]
            # Reshape from (n_stations, seq_len, 20) to (n_stations * seq_len, 20)
            features = raw_item['sequence_features'].reshape(-1, len(self.feature_names))
            all_features.append(features)

        all_features = np.vstack(all_features)

        self.means = np.nanmean(all_features, axis=0)
        self.stds = np.nanstd(all_features, axis=0)
        self.stds[self.stds == 0] = 1.0  # Avoid division by zero

        # Special handling for physics-informed features
        directional_indices = [self.feature_names.index(f) for f in ['primary_direction', 'secondary_direction', 'wind_direction']]
        for idx in directional_indices:
            self.means[idx] = 0.0
            self.stds[idx] = 180.0  # Normalize to [-1, 1] range

        fraction_indices = [self.feature_names.index(f) for f in ['swell_fraction', 'windsea_fraction']]
        for idx in fraction_indices:
            self.means[idx] = 0.5
            self.stds[idx] = 0.5 # Normalize to [-1, 1] range

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise RuntimeError("Normalizer must be fitted before transforming data.")
        return (features - self.means) / self.stds

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise RuntimeError("Normalizer must be fitted before inverse transforming data.")
        return features * self.stds + self.means

class EnhancedWaveSequenceDataset(Dataset):
    """
    Loads and processes raw wave data into spatiotemporal sequences.
    This class DOES NOT perform normalization.
    """
    def __init__(self, features_file: str, sequence_length: int = 24, prediction_horizon: int = 1, min_stations: int = 3, interpolate_missing: bool = True):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.min_stations = min_stations
        self.interpolate_missing = interpolate_missing
        self.feature_names = FEATURE_NAMES

        with open(features_file, 'r') as f:
            self.raw_data = json.load(f)

        self.processed_data = self._process_station_data()
        self.sequences = self._create_sequences()

    def _process_station_data(self) -> Dict[str, pd.DataFrame]:
        processed = {}
        for station_id, station_data in self.raw_data.items():
            if 'enhanced_features' not in station_data:
                continue
            features = np.array(station_data['enhanced_features'])
            timestamps = pd.to_datetime(station_data['timestamps'])
            df = pd.DataFrame(features, columns=self.feature_names, index=timestamps).sort_index()

            if self.interpolate_missing:
                df = self._interpolate_physics_aware(df)
            processed[station_id] = df
        return processed

    def _interpolate_physics_aware(self, df: pd.DataFrame) -> pd.DataFrame:
        """Physics-informed interpolation for missing values - FIXED VERSION"""
        df_interp = df.copy()
        
        # 1. Linear interpolation for continuous physics variables
        continuous_features = self.feature_names[0:10] + self.feature_names[15:18] + self.feature_names[18:20]
        df_interp[continuous_features] = df_interp[continuous_features].interpolate(method='linear', limit=3, limit_direction='both')

        # 2. Circular interpolation for directional data
        for feature in ['primary_direction', 'secondary_direction', 'wind_direction']:
            if feature in df_interp.columns:
                angles_rad = np.radians(df_interp[feature])
                x, y = np.cos(angles_rad), np.sin(angles_rad)
                x_interp = x.interpolate(method='linear', limit=2, limit_direction='both')
                y_interp = y.interpolate(method='linear', limit=2, limit_direction='both')
                angles_interp = np.degrees(np.arctan2(y_interp, x_interp)) % 360
                df_interp[feature] = angles_interp

        # 3. Forward fill for remaining directional spread features
        remaining_directional = ['primary_spread', 'bimodal_strength', 'directional_separation']
        df_interp[remaining_directional] = df_interp[remaining_directional].ffill(limit=2).bfill(limit=2)
        
        # 4. CRITICAL FIX: Use robust forward/backward fill instead of fillna(0)
        df_interp = df_interp.ffill().bfill()
        
        # 5. Only as final failsafe for completely empty columns
        df_interp = df_interp.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df_interp

    def _create_sequences(self) -> List[Dict]:
        sequences = []
        all_timestamps = sorted(list(set(ts for df in self.processed_data.values() for ts in df.index)))
        
        for i in range(len(all_timestamps) - self.sequence_length - self.prediction_horizon + 1):
            sequence_timestamps = all_timestamps[i : i + self.sequence_length]
            target_time = all_timestamps[i + self.sequence_length + self.prediction_horizon - 1]

            station_features, station_ids, station_coords, target_features = [], [], [], []
            for station_id, df in self.processed_data.items():
                if all(t in df.index for t in sequence_timestamps) and target_time in df.index:
                    sequence_data = df.loc[sequence_timestamps].values
                    target_data = df.loc[target_time].values
                    
                    # CRITICAL: Check for NaN values after interpolation
                    if not np.any(np.isnan(sequence_data)) and not np.any(np.isnan(target_data)):
                        if not np.all(np.isnan(sequence_data[:, :10])): # Require spectral features
                            station_features.append(sequence_data)
                            station_ids.append(station_id)
                            station_coords.append(STATION_COORDS.get(station_id, [0.0, 0.0]))
                            target_features.append(target_data)

            if len(station_features) >= self.min_stations:
                sequences.append({
                    'sequence_features': np.array(station_features),
                    'target_features': np.array(target_features),
                    'station_ids': station_ids,
                    'station_coords': np.array(station_coords)
                })
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        return self.sequences[idx]

class NormalizedDatasetWrapper(Dataset):
    """
    A wrapper that applies a pre-fitted normalizer to a dataset subset.
    """
    def __init__(self, dataset_subset: Subset, normalizer: WaveDataNormalizer):
        self.dataset_subset = dataset_subset
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.dataset_subset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset_subset[idx]
        
        # Apply normalization
        seq_norm = self.normalizer.transform(item['sequence_features'])
        tgt_norm = self.normalizer.transform(item['target_features'])

        return {
            'sequence_features': torch.tensor(seq_norm, dtype=torch.float32),
            'target_features': torch.tensor(tgt_norm, dtype=torch.float32),
            'station_coords': torch.tensor(item['station_coords'], dtype=torch.float32),
            'n_stations': torch.tensor(len(item['station_ids']), dtype=torch.long)
        }

def wave_collate_fn(batch: List[Dict]) -> Dict:
    """Pads sequences in a batch to the maximum number of stations."""
    max_stations = max(item['n_stations'].item() for item in batch)
    batch_size = len(batch)
    seq_len = batch[0]['sequence_features'].shape[1]
    feature_dim = batch[0]['sequence_features'].shape[2]

    padded_sequences = torch.zeros(batch_size, max_stations, seq_len, feature_dim)
    padded_targets = torch.zeros(batch_size, max_stations, feature_dim)
    padded_coords = torch.zeros(batch_size, max_stations, 2)
    station_masks = torch.zeros(batch_size, max_stations, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item['n_stations'].item()
        padded_sequences[i, :n] = item['sequence_features']
        padded_targets[i, :n] = item['target_features']
        padded_coords[i, :n] = item['station_coords']
        station_masks[i, :n] = True

    return {
        'sequence_features': padded_sequences,
        'target_features': padded_targets,
        'station_coords': padded_coords,
        'station_masks': station_masks
    }

def create_enhanced_dataloaders(
    features_file: str,
    sequence_length: int = 24,
    batch_size: int = 16,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    num_workers: int = 2,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, WaveDataNormalizer]:
    """
    Creates train, validation, and test dataloaders with proper, leak-free normalization.
    """
    # Remove any split_ratio parameter that might cause conflicts
    dataset_kwargs.pop('split_ratio', None)
    
    # 1. Create the full, raw dataset
    full_dataset = EnhancedWaveSequenceDataset(
        features_file=features_file,
        sequence_length=sequence_length,
        **dataset_kwargs
    )

    # 2. Perform temporal split
    total_sequences = len(full_dataset)
    train_end = int(total_sequences * split_ratios[0])
    val_end = train_end + int(total_sequences * split_ratios[1])
    
    train_indices = list(range(train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, total_sequences))

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

    # 3. Fit normalizer ONLY on the training data
    normalizer = WaveDataNormalizer(feature_names=FEATURE_NAMES)
    normalizer.fit(train_subset)

    # 4. Wrap subsets with the fitted normalizer
    train_wrapped = NormalizedDatasetWrapper(train_subset, normalizer)
    val_wrapped = NormalizedDatasetWrapper(val_subset, normalizer)
    test_wrapped = NormalizedDatasetWrapper(test_subset, normalizer)

    # 5. Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_wrapped, batch_size=batch_size, shuffle=True, collate_fn=wave_collate_fn, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_wrapped, batch_size=batch_size, shuffle=False, collate_fn=wave_collate_fn, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_wrapped, batch_size=batch_size, shuffle=False, collate_fn=wave_collate_fn, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, normalizer