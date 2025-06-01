"""
wave_sequencer.py
------------------
Defines the WaveSequenceDataset for creating sliding-window sequences
from buoy time-series data, handling timestamps, missing data, and
target preparation.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime


class WaveSequenceDataset(Dataset):
    """
    Creates fixed-length sequences from buoy data. Each sequence can optionally
    include a prediction horizon and surf spot data.
    
    buoy_data: Dict of {buoy_id -> np.ndarray of shape (total_timesteps, feature_dim)}
    timestamps: List of datetime objects (length = total_timesteps)
    sequence_length: number of timesteps per sequence
    prediction_horizon: how far ahead to predict
    surf_spot_data: optional dict {spot_id -> np.ndarray of shape (total_timesteps, conditions_dim)}
    """
    def __init__(
        self,
        buoy_data: Dict[str, np.ndarray],
        timestamps: List[datetime],
        sequence_length: int = 24,
        prediction_horizon: int = 1,
        surf_spot_data: Optional[Dict[str, np.ndarray]] = None
    ):
        self.buoy_data = buoy_data
        self.timestamps = timestamps
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.surf_spot_data = surf_spot_data
        
        self.sequence_indices = self._create_sequence_indices()

    def _create_sequence_indices(self) -> List[int]:
        total_timesteps = len(self.timestamps)
        valid_starts = []
        for i in range(total_timesteps - self.sequence_length - self.prediction_horizon):
            # Check for NaNs in each buoy’s data window
            valid = True
            for _, data in self.buoy_data.items():
                if np.any(np.isnan(data[i:i + self.sequence_length])):
                    valid = False
                    break
            if valid:
                valid_starts.append(i)
        return valid_starts

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = self.sequence_indices[idx]
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.prediction_horizon
        
        sequence_data = []
        target_data = []
        missing_buoys = []

        # Collect data from each buoy
        for _, data in self.buoy_data.items():
            seq = data[start_idx:end_idx]  # shape: (sequence_length, feature_dim)
            tgt = data[target_idx]         # shape: (feature_dim,)
            sequence_data.append(seq)
            target_data.append(tgt)
            # Mark timesteps as missing if any feature is NaN
            missing = np.any(np.isnan(seq), axis=1).astype(np.float32)
            missing_buoys.append(missing)
        
        # Convert to (sequence_length, num_buoys, feature_dim)
        features = np.stack(sequence_data, axis=1)
        # Convert missing data to (sequence_length, num_buoys)
        missing_array = np.stack(missing_buoys, axis=1)

        # Timestamps for the sequence
        seq_timestamps = self.timestamps[start_idx:end_idx]

        batch = {
            'features': torch.tensor(features, dtype=torch.float32),
            'timestamps': seq_timestamps,
            'missing_buoys': torch.tensor(missing_array, dtype=torch.float32),
            'buoy_states_target': torch.tensor(np.stack(target_data, axis=0), dtype=torch.float32)
        }

        # Surf spot targets if provided
        if self.surf_spot_data is not None:
            spot_targets = []
            for _, spot_arr in self.surf_spot_data.items():
                spot_targets.append(spot_arr[target_idx])
            batch['surf_conditions_target'] = torch.tensor(np.stack(spot_targets, axis=0), dtype=torch.float32)

        return batch


def create_dataloaders(
    buoy_data: Dict[str, np.ndarray],
    timestamps: List[datetime],
    sequence_length: int = 24,
    batch_size: int = 32,
    split_ratio: float = 0.8,
    surf_spot_data: Optional[Dict[str, np.ndarray]] = None
):
    """
    Creates train/validation dataloaders for the wave sequence dataset.
    """
    dataset = WaveSequenceDataset(
        buoy_data=buoy_data,
        timestamps=timestamps,
        sequence_length=sequence_length,
        surf_spot_data=surf_spot_data
    )
    total = len(dataset)
    train_size = int(total * split_ratio)
    val_size = total - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    return train_loader, val_loader
