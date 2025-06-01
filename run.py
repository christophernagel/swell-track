"""
run.py
------
Example main script that ties everything together:
- Reads station metadata
- Builds a BuoyNetwork
- Creates a SwellAnalyzer for real-time ingestion
- Builds a WaveSequenceDataset for historical sequences
- Instantiates an EnhancedWaveTransformer
- Trains and evaluates the model via WaveModelTrainer
"""

import json
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict

from swell_tracker.core import BuoyNetwork, SwellAnalyzer, to_torch_sparse
from swell_tracker.wave_sequencer import create_dataloaders
from swell_tracker.wave_transformer import EnhancedWaveTransformer
from swell_tracker.model_trainer import WaveModelTrainer


def main():
    # 1) Load station metadata
    with open("stations.json", "r") as f:
        station_data = json.load(f)
    
    # 2) Build BuoyNetwork
    network = BuoyNetwork(station_data)
    
    # 3) Create a SwellAnalyzer for demonstration
    analyzer = SwellAnalyzer(network, history_size=100, feature_names=['height', 'sig_height', 'period'])
    
    # Populate the wave state with some synthetic data
    now = datetime.now()
    for t in range(20):
        timestamp = now + timedelta(minutes=5 * t)
        # Example measurements for each buoy
        measurements: Dict[str, Dict[str, float]] = {}
        for i, buoy in enumerate(network.buoys):
            # Synthetic pattern
            measurements[buoy['id']] = {
                'height': 1.0 + 0.1 * t + i * 0.05,
                'sig_height': 2.0 + 0.05 * t + i * 0.02,
                'period': 8.0 + 0.01 * t + i * 0.01
            }
        analyzer.state.add_observation(timestamp, measurements)
    
    # Optional: update propagation to incorporate physics constraints
    analyzer.update_propagation(decay=0.9, max_speed=30.0)
    
    # 4) Convert the ring-buffer into a small training batch (for demonstration)
    try:
        batch = analyzer.get_training_batch(window=5)
        print("Demo batch shapes:")
        print("  Inputs:", batch['inputs'].shape)
        print("  Targets:", batch['targets'].shape)
        print("  Mask:", batch['mask'].shape)
        print("  Positional encodings:", batch['positional_encodings'].shape)
    except ValueError as e:
        print("Error generating training batch:", e)
    
    # 5) Example: Convert the adjacency matrix to a sparse PyTorch tensor
    sparse_tensor = to_torch_sparse(analyzer.propagation)
    print("Sparse adjacency:", sparse_tensor)
    
    # 6) Build a dataset/dataloader from historical arrays
    # Suppose we have arrays for each buoy with shape (total_timesteps, feature_dim).
    # For demonstration, we'll reuse the ring-buffer data from analyzer.state.
    # Real usage: you'd have arrays from your ingestion pipeline.
    num_timesteps = analyzer.state.cursor
    buoy_data = {}
    for b_idx, buoy in enumerate(network.buoys):
        # Extract the portion of the ring-buffer that was filled
        buoy_data[buoy['id']] = analyzer.state.observations[:num_timesteps, b_idx, :]
    
    # Timestamps
    timestamps = list(analyzer.state.timestamps[:num_timesteps])
    # Convert np.datetime64 to python datetime
    timestamps = [ts.astype('datetime64[s]').tolist() for ts in timestamps]
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        buoy_data=buoy_data,
        timestamps=timestamps,
        sequence_length=5,   # matches window above
        batch_size=2
    )
    
    # 7) Instantiate the EnhancedWaveTransformer
    feature_dim = len(analyzer.state.feature_names)
    model = EnhancedWaveTransformer(
        num_buoys=len(network.buoys),
        feature_dim=feature_dim,
        d_model=64,
        nhead=4,
        num_encoder_layers=2
    )
    
    # 8) Set up trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = WaveModelTrainer(model, optimizer)
    
    # 9) Train for a couple of epochs (demo only)
    for epoch in range(2):
        metrics = trainer.train_epoch(train_loader, validation_loader=val_loader)
        print(f"Epoch {epoch+1} metrics:", metrics)


if __name__ == "__main__":
    main()
