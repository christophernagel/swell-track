"""
core.py
--------
Contains the foundational classes for buoy network representation,
wave state management, basic swell analysis, and enhanced physics integration.
"""

import numpy as np
from datetime import datetime
from scipy.sparse import csr_array
from typing import Dict, List, Optional

import torch


class OnlineNormalizer:
    """
    A simple online normalizer using Welford’s algorithm.
    Maintains a running mean and variance for streaming data.
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    def normalize(self, x: float) -> float:
        std = np.sqrt(self.variance) if self.variance > 0 else 1.0
        return (x - self.mean) / std


class BuoyNetwork:
    """
    Represents a static buoy network using adjacency information.
    Each buoy is a dict with:
      - 'id': str
      - 'lat', 'lon': float
      - 'neighbors': List[str]
      - optional 'distances', 'bearings', etc.
    """
    def __init__(self, buoys: List[dict]):
        # Sort to ensure stable indexing
        self.buoys = sorted(buoys, key=lambda b: b['id'])
        self.buoy_index = {b['id']: i for i, b in enumerate(self.buoys)}
        
        # Build sparse matrices for adjacency, distances, bearings, etc.
        self.adjacency = self._build_sparse_matrix('neighbors', binary=True)
        self.distances = self._build_sparse_matrix('distances', binary=False)
        self.bearings = self._build_sparse_matrix('bearings', binary=False)

    def _build_sparse_matrix(self, key: str, binary: bool = False) -> csr_array:
        """
        Constructs a CSR matrix from neighbor relationships.
        If binary=True, matrix entries are 1 where adjacency is present.
        Otherwise, uses buoy[key] to populate data values.
        """
        size = len(self.buoys)
        indptr = [0]
        indices = []
        data = []
        for buoy in self.buoys:
            neighbor_ids = buoy.get('neighbors', [])
            mapped = [self.buoy_index[nid] for nid in neighbor_ids]
            indices.extend(mapped)
            if binary:
                data.extend([1.0] * len(mapped))
            else:
                values = buoy.get(key, [])
                if len(values) != len(neighbor_ids):
                    raise ValueError(
                        f"Mismatch in lengths for buoy {buoy['id']}: "
                        f"{len(neighbor_ids)} neighbors vs {len(values)} values for '{key}'"
                    )
                data.extend(values)
            indptr.append(len(indices))
        
        return csr_array(
            (np.array(data, dtype=np.float32),
             np.array(indices, dtype=np.int32),
             np.array(indptr, dtype=np.int32)),
            shape=(size, size)
        )


class WaveState:
    """
    Maintains a ring-buffer of observations and computes delta matrices.
    Observations are normalized online (feature-wise) via OnlineNormalizer.
    """
    def __init__(self, network: BuoyNetwork, history_size: int = 1440,
                 feature_names: List[str] = None):
        if feature_names is None:
            feature_names = ['height', 'sig_height', 'period']
        self.network = network
        self.feature_names = feature_names
        self.history_size = history_size
        
        num_buoys = len(network.buoys)
        num_features = len(feature_names)
        
        self.observations = np.zeros((history_size, num_buoys, num_features), dtype=np.float32)
        self.timestamps = np.full(history_size, np.datetime64('1970-01-01'))
        
        # Deltas are stored in CSR matrices, one per feature.
        self.deltas = {
            feat: network.adjacency.copy().astype(np.float32)
            for feat in feature_names
        }
        self.normalizers = {feat: OnlineNormalizer() for feat in feature_names}
        self.cursor = 0

    def add_observation(self, timestamp: datetime, measurements: Dict[str, Dict[str, float]]):
        """
        Adds a new observation for each buoy at a given timestamp.
        Updates ring-buffer and normalizes new data using Welford’s algorithm.
        """
        idx = self.cursor % self.history_size
        self.timestamps[idx] = np.datetime64(timestamp)
        
        for buoy_id, feats in measurements.items():
            buoy_idx = self.network.buoy_index[buoy_id]
            for f, feature in enumerate(self.feature_names):
                raw_value = feats.get(feature, 0.0)
                self.normalizers[feature].update(raw_value)
                norm_value = self.normalizers[feature].normalize(raw_value)
                self.observations[idx, buoy_idx, f] = norm_value
        
        self._update_deltas(idx)
        self.cursor += 1

    def _update_deltas(self, idx: int):
        """
        Updates feature-wise delta matrices (neighbor_val - self_val).
        """
        current_obs = self.observations[idx]  # shape: (num_buoys, num_features)
        adj = self.network.adjacency
        counts = np.diff(adj.indptr)
        
        for f, feature in enumerate(self.feature_names):
            neighbor_vals = current_obs[adj.indices, f]
            self_vals = np.repeat(current_obs[:, f], counts)
            self.deltas[feature].data = neighbor_vals - self_vals


class WavePhysics:
    """
    Enhanced physics module for wave propagation modeling.
    Incorporates wind, bathymetry, and seasonal effects.
    """
    def __init__(self, bathymetry_data: Optional[Dict[str, float]] = None):
        self.bathymetry = bathymetry_data or {}
        self.seasonal_params = self._init_seasonal_parameters()
        
    def _init_seasonal_parameters(self) -> Dict[str, Dict[str, float]]:
        """Initialize season-specific parameters."""
        return {
            'winter': {'base_decay': 0.92, 'max_speed': 28.0},
            'spring': {'base_decay': 0.89, 'max_speed': 30.0},
            'summer': {'base_decay': 0.87, 'max_speed': 32.0},
            'fall': {'base_decay': 0.90, 'max_speed': 31.0}
        }
    
    def compute_propagation_parameters(
        self,
        timestamp: datetime,
        wind_speed: float,
        wind_direction: float,
        wave_direction: float
    ) -> Dict[str, float]:
        """
        Compute dynamic propagation parameters based on conditions.
        """
        season = self._get_season(timestamp)
        base_params = self.seasonal_params[season]
        
        # Adjust decay based on wind-wave alignment
        wind_wave_angle = abs(wind_direction - wave_direction)
        wind_factor = np.cos(np.radians(wind_wave_angle))
        
        # Increase decay when wind opposes wave direction
        adjusted_decay = base_params['base_decay'] * (1.0 - 0.1 * max(0, wind_factor))
        
        # Adjust max speed based on wind speed
        # Wind in same direction can increase max speed
        speed_adjustment = wind_speed * max(0, wind_factor) * 0.05
        adjusted_speed = base_params['max_speed'] + speed_adjustment
        
        return {
            'decay': adjusted_decay,
            'max_speed': adjusted_speed
        }
    
    def apply_bathymetry_effects(
        self,
        propagation: csr_array,
        buoy_network: BuoyNetwork
    ) -> csr_array:
        """
        Modify propagation based on bathymetry data.
        Shallow water affects wave speed and height.
        """
        modified = propagation.copy()
        
        for i, buoy in enumerate(buoy_network.buoys):
            depth = self.bathymetry.get(buoy['id'], float('inf'))
            if depth < float('inf'):
                # Calculate wavelength from period (if available)
                wave_period = buoy.get('period', 10.0)  # default 10s if not provided
                # Deep water wavelength (L₀ = gT²/2π)
                deep_water_wavelength = 9.81 * wave_period**2 / (2 * np.pi)
                
                # Iterative solution for wavelength in finite depth using Newton's method
                wavelength = deep_water_wavelength
                for _ in range(5):  # usually converges in a few iterations
                    tanh_term = np.tanh(2 * np.pi * depth / wavelength)
                    wavelength = (9.81 * wave_period**2 * tanh_term) / (2 * np.pi)
                
                # Shallow water wave speed modification
                depth_factor = np.tanh(2 * np.pi * depth / wavelength)
                speed_modifier = np.sqrt(depth_factor)
                
                # Modify relevant propagation entries
                row_start = modified.indptr[i]
                row_end = modified.indptr[i + 1]
                modified.data[row_start:row_end] *= speed_modifier
        
        return modified
    
    @staticmethod
    def _get_season(timestamp: datetime) -> str:
        """Determine season from timestamp."""
        month = timestamp.month
        if month in (12, 1, 2):
            return 'winter'
        elif month in (3, 4, 5):
            return 'spring'
        elif month in (6, 7, 8):
            return 'summer'
        else:
            return 'fall'


class SwellAnalyzer:
    """
    Combines the static buoy network and wave state to generate training sequences,
    apply physical propagation constraints, and provide data for downstream models.
    """
    def __init__(self, network: BuoyNetwork, history_size: int = 1440,
                 feature_names: List[str] = None,
                 bathymetry_data: Optional[Dict[str, float]] = None):
        self.network = network
        self.state = WaveState(network, history_size, feature_names)
        # Additional matrix to track wave propagation (height-based).
        self.propagation = network.adjacency.copy().astype(np.float32)
        
        # Integrate the enhanced physics module
        self.physics = WavePhysics(bathymetry_data)
    
    def update_propagation(self, wind_speed: float, wind_direction: float, wave_direction: float):
        """
        Enhanced propagation update using physics-informed parameters.
        """
        if self.state.cursor < 2:
            return
        
        # Compute updated propagation parameters dynamically
        idx_current = (self.state.cursor - 1) % self.state.history_size
        timestamp = self.state.timestamps[idx_current]
        physics_params = self.physics.compute_propagation_parameters(timestamp, wind_speed, wind_direction, wave_direction)
        decay = physics_params['decay']
        max_speed = physics_params['max_speed']
        
        idx_prev = (self.state.cursor - 2) % self.state.history_size
        t_curr = self.state.timestamps[idx_current]
        t_prev = self.state.timestamps[idx_prev]
        time_delta = (t_curr - t_prev).astype('timedelta64[s]').item()
        
        # Avoid division by zero or negative time deltas
        if time_delta.total_seconds() <= 0:
            return
        
        speeds = (self.network.distances.data * 1000) / time_delta  # m/s
        feasible = speeds < max_speed
        
        # Use 'height' deltas to drive propagation
        height_deltas = self.state.deltas['height'].data
        filtered_deltas = height_deltas * feasible
        
        # Update propagation matrix using dynamic decay
        self.propagation.data = decay * self.propagation.data + (1 - decay) * filtered_deltas
        
        # Apply bathymetry effects to adjust for shallow water conditions
        self.propagation = self.physics.apply_bathymetry_effects(self.propagation, self.network)


def to_torch_sparse(csr: csr_array) -> torch.Tensor:
    """
    Utility to convert a SciPy CSR matrix to a PyTorch sparse CSR tensor.
    """
    return torch.sparse_csr_tensor(
        torch.tensor(csr.indptr, dtype=torch.int32),
        torch.tensor(csr.indices, dtype=torch.int32),
        torch.tensor(csr.data, dtype=torch.float32),
        size=csr.shape
    )
