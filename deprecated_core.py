from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
from math import radians, degrees, sin, cos, atan2, sqrt, log
from collections import defaultdict
import logging

# Constants
PRECISION_MODE = 'mixed'
CALCULATION_DTYPE = np.float32 if PRECISION_MODE == 'mixed' else np.float16
STORAGE_DTYPE = np.float16 if PRECISION_MODE != 'float32' else np.float32
MIN_PERIOD = 1.0  # Minimum wave period in seconds
MIN_DEPTH = 0.1  # Minimum water depth in meters (to avoid division by zero)

if PRECISION_MODE == 'float32':
    CALCULATION_DTYPE = np.float32
    STORAGE_DTYPE = np.float32
else:
    CALCULATION_DTYPE = np.float32 if PRECISION_MODE == 'mixed' else np.float16
    STORAGE_DTYPE = np.float16

def _check_overflow(value: np.ndarray, dtype: np.dtype) -> bool:
    """Check for numerical overflow."""
    info = np.finfo(dtype)
    return np.any(value < info.min) or np.any(value > info.max)

def _safe_cast(value: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Safely cast array to target dtype, scaling to avoid overflow."""
    info = np.finfo(target_dtype)
    if _check_overflow(value, target_dtype):
        scale_factor = np.maximum(np.abs(value).max() / (info.max * 0.9), 1.0)
        value = value / scale_factor
    return value.astype(target_dtype)

def _validate_measurement(raw: Dict[str, float]) -> None:
    """Validate measurement data for physical consistency."""
    required_keys = {'height', 'sig_height', 'period', 'direction'}
    missing_keys = required_keys - raw.keys()
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")
    
    if any(raw[k] < 0 for k in ['height', 'sig_height', 'period']):
        raise ValueError("Measurement values must be non-negative")
    
    if not (0 <= raw['direction'] <= 360):
        raise ValueError("Direction must be between 0 and 360 degrees")
    
    if raw['period'] < MIN_PERIOD:
        raise ValueError(f"Wave period must be ≥{MIN_PERIOD} seconds")
    
    if raw['sig_height'] < 1.6 * raw['height']:
        raise ValueError("Sig. height should be ~1.6× average height in swell conditions")

@dataclass
class OnlineStats:
    """Online statistics for feature normalization."""
    count: int = 0
    mean: np.float32 = np.float32(0.0)
    m2: np.float32 = np.float32(0.0)  # Sum of squared differences
    min_val: np.float32 = np.float32(np.inf)
    max_val: np.float32 = np.float32(-np.inf)

    def update(self, value: np.float32):
        """Update statistics with a new value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / np.float32(self.count)
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = np.minimum(self.min_val, value)
        self.max_val = np.maximum(self.max_val, value)

    @property
    def variance(self) -> np.float32:
        """Compute variance."""
        return self.m2 / np.float32(self.count) if self.count > 1 else np.float32(0.0)

    @property
    def std(self) -> np.float32:
        """Compute standard deviation."""
        return np.sqrt(self.variance).astype(np.float32)

@dataclass
class FeatureNormalizer:
    """Normalize features using online statistics."""
    stats: Dict[str, OnlineStats] = field(default_factory=lambda: defaultdict(OnlineStats))
    
    def update(self, feature: str, value: float):
        """Update statistics for a feature."""
        if self.stats[feature].count > 1 and self.stats[feature].std < 1e-6:
            raise ValueError(f"Zero variance detected in {feature}")
        self.stats[feature].update(np.float32(value))
    
    def normalize(self, feature: str, value: float) -> np.float32:
        """Normalize a feature value."""
        stats = self.stats[feature]
        return (np.float32(value) - stats.mean) / (stats.std or np.float32(1.0))
    
    def denormalize(self, feature: str, normalized_value: float) -> np.float32:
        """Denormalize a feature value."""
        stats = self.stats[feature]
        return np.float32(normalized_value) * stats.std + stats.mean

@dataclass
class EnhancedEncodedMeasurement:
    """Encoded measurement with temporal/spatial uncertainty."""
    timestamp: datetime
    raw_values: Dict[str, float]
    encoded_values: np.ndarray
    temporal_uncertainty: np.float32
    spatial_uncertainty: np.float32
    gradient_scale: np.float32 = field(init=False)
    depth: Optional[np.float32] = None  # Added for shallow-water physics

    def __post_init__(self):
        """Compute gradient scale for mixed precision training."""
        self.gradient_scale = np.float32(1.0 / (self.temporal_uncertainty + self.spatial_uncertainty))

    @classmethod
    def from_raw(cls, timestamp: datetime, raw_values: Dict[str, float],
                 reference_time: datetime, normalizer: FeatureNormalizer,
                 time_gap: float, neighbor_count: int, depth: Optional[float] = None) -> 'EnhancedEncodedMeasurement':
        """Create an encoded measurement from raw data."""
        _validate_measurement(raw_values)
        
        time_delta = np.float32((timestamp - reference_time).total_seconds() / 86400)
        dir_rad = np.float32(radians(raw_values['direction']))
        
        encoded = np.array([
            time_delta,
            normalizer.normalize('height', raw_values['height']),
            normalizer.normalize('sig_height', raw_values['sig_height']),
            normalizer.normalize('period', raw_values['period']),
            np.float32(sin(dir_rad)),
            np.float32(cos(dir_rad)),
            log(np.nextafter(np.float32(raw_values['period']), np.inf)),  # Avoid underflow
        ], dtype=CALCULATION_DTYPE)

        temp_unc = np.clip(time_gap / 1800, 0.1, 4.0)
        spatial_unc = 1.0 - (min(neighbor_count, 3) / 3.0)
        
        return cls(
            timestamp=timestamp,
            raw_values=raw_values,
            encoded_values=_safe_cast(encoded, STORAGE_DTYPE),
            temporal_uncertainty=np.float32(temp_unc),
            spatial_uncertainty=np.float32(spatial_unc),
            depth=np.float32(depth) if depth else None
        )

@dataclass
class BuoyNode:
    """Buoy node with geospatial and measurement data."""
    id: str
    lat: np.float32
    lon: np.float32
    measurements: List[EnhancedEncodedMeasurement] = field(default_factory=list)
    _nearest_neighbors: Dict[str, Tuple[np.float32, np.float32]] = field(default_factory=dict)
    encoded_position: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.update_encoded_position()
    
    def update_encoded_position(self):
        """Encode lat/lon as trigonometric features."""
        lat_rad = np.float32(radians(self.lat))
        lon_rad = np.float32(radians(self.lon % 360))
        self.encoded_position = np.array([
            sin(lat_rad),
            cos(lat_rad),
            sin(lon_rad),
            cos(lon_rad)
        ], dtype=STORAGE_DTYPE)
    
    def get_vector_to(self, other: 'BuoyNode') -> Tuple[np.float32, np.float32]:
        """Compute distance and bearing to another buoy."""
        lat1 = np.float32(radians(self.lat))
        lon1 = np.float32(radians(self.lon % 360))
        lat2 = np.float32(radians(other.lat))
        lon2 = np.float32(radians(other.lon % 360))
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.float32((sin(dlat/2)**2) + cos(lat1) * cos(lat2) * (sin(dlon/2)**2))
        distance = np.float32(6371 * 2 * np.arcsin(np.sqrt(a)))
        
        x = sin(dlon) * cos(lat2)
        y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        bearing = np.float32(degrees(atan2(x, y)) % 360)
        
        return distance, bearing

class SpatialIndex:
    """Maintains spatial relationships between buoys using a grid-based index."""
    def __init__(self, grid_size: float = 5.0):
        """
        Initialize the spatial index.
        
        Args:
            grid_size (float): Size of each grid cell in degrees (default: 5.0°).
        """
        self.grid_size = grid_size
        self.grid = defaultdict(set)  # Maps grid cells to buoy IDs
        self.total_lon_cells = int(360 / grid_size)  # Number of longitude cells

    def _get_grid_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Get the grid cell for a given latitude and longitude.
        
        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
        
        Returns:
            Tuple[int, int]: Grid cell coordinates (lat_index, lon_index).
        """
        return (
            int(np.floor(lat / self.grid_size)),
            int(np.floor((lon % 360) / self.grid_size))
        )

    def _handle_polar_buoy(self, buoy: BuoyNode):
        """
        Special handling for buoys near the poles.
        
        Args:
            buoy (BuoyNode): Buoy near the pole.
        """
        lat_cell = int(np.floor(buoy.lat / self.grid_size))
        for lon_cell in range(self.total_lon_cells):
            self.grid[(lat_cell, lon_cell)].add(buoy.id)

    def add_buoy(self, buoy: BuoyNode) -> None:
        """
        Add a buoy to the spatial index.
        
        Args:
            buoy (BuoyNode): Buoy to add.
        """
        lat, lon = buoy.lat, buoy.lon % 360
        
        # Special handling for polar buoys
        if abs(lat) >= 89.9:
            self._handle_polar_buoy(buoy)
            return
            
        # Add to primary grid cell
        cell = self._get_grid_cell(lat, lon)
        self.grid[cell].add(buoy.id)
        
        # Add to adjacent cells for high-latitude buoys
        if abs(lat) >= 80:
            for dlat, dlon in [(-1,-1), (-1,0), (-1,1),
                              (0,-1), (0,0), (0,1),
                              (1,-1), (1,0), (1,1)]:
                adj_cell = (
                    cell[0] + dlat,
                    (cell[1] + dlon) % self.total_lon_cells
                )
                self.grid[adj_cell].add(buoy.id)

    def get_nearby_buoys(self, target: BuoyNode, max_distance: float = 1000.0) -> Set[str]:
        """
        Get IDs of buoys within max_distance km of target.
        
        Args:
            target (BuoyNode): Target buoy.
            max_distance (float): Maximum distance in kilometers (default: 1000.0 km).
        
        Returns:
            Set[str]: IDs of nearby buoys.
        """
        nearby = set()
        cell = self._get_grid_cell(target.lat, target.lon % 360)
        
        # Search adjacent grid cells
        for dlat, dlon in [(-1,-1), (-1,0), (-1,1),
                          (0,-1), (0,0), (0,1),
                          (1,-1), (1,0), (1,1)]:
            lat_cell = cell[0] + dlat
            lon_cell = (cell[1] + dlon) % self.total_lon_cells
            for buoy_id in self.grid.get((lat_cell, lon_cell), set()):
                if buoy_id != target.id:
                    buoy = self.tracker.buoys[buoy_id]
                    distance, _ = target.get_vector_to(buoy)
                    if distance <= max_distance:
                        nearby.add(buoy_id)
        
        return nearby

@dataclass
class SwellTracker:
    """Core swell tracking and analysis system."""
    buoys: Dict[str, BuoyNode] = field(default_factory=dict)
    buoy_indices: Dict[str, int] = field(default_factory=dict)
    normalizer: FeatureNormalizer = field(default_factory=FeatureNormalizer)
    reference_time: datetime = field(default_factory=lambda: datetime(2000, 1, 1))
    _spatial_index: Optional[SpatialIndex] = None
    propagation_graph: Dict[Tuple[str, str], List[Tuple[int, int, float, float]]] = field(default_factory=dict)
    _gradient_scale: np.float32 = np.float32(1024.0)
    _dirty_neighbors: Set[str] = field(default_factory=set)
    data_buffer: Optional[np.memmap] = field(default=None)

    def get_nearby_buoys(self, target_buoy):
        return self._spatial_index.get_nearby_buoys(
            target_buoy, self.buoys, max_distance=1000.0
        )

    @property
    def spatial_index(self) -> SpatialIndex:
        """Access the spatial index."""
        return self._spatial_index

    def add_buoy(self, buoy: BuoyNode) -> None:
        """Add a buoy to the tracking system."""
        self.buoys[buoy.id] = buoy
        self.buoy_indices[buoy.id] = len(self.buoy_indices)
        self._spatial_index.add_buoy(buoy)
        self._dirty_neighbors.add(buoy.id)

    def add_measurements(self, measurements: Dict[str, List[Tuple]], bathymetry: Dict[str, float] = None) -> None:
        """Add measurements for multiple buoys.
        
        Args:
            measurements: Dict mapping buoy IDs to lists of (timestamp, height, sig_height, period, direction) tuples
            bathymetry: Optional dict mapping buoy IDs to water depth in meters
        """
        for buoy_id, measurement_list in measurements.items():
            if buoy_id not in self.buoys:
                raise ValueError(f"Unknown buoy ID: {buoy_id}")
                
            buoy = self.buoys[buoy_id]
            depth = bathymetry.get(buoy_id) if bathymetry else None
            
            for timestamp, height, sig_height, period, direction in measurement_list:
                raw_values = {
                    'height': height,
                    'sig_height': sig_height,
                    'period': period,
                    'direction': direction
                }
                
                # Update normalizer statistics
                for key, value in raw_values.items():
                    self.normalizer.update(key, value)
                
                # Create encoded measurement
                measurement = EnhancedEncodedMeasurement.from_raw(
                    timestamp=timestamp,
                    raw_values=raw_values,
                    reference_time=self.reference_time,
                    normalizer=self.normalizer,
                    time_gap=0.0,  # Will be updated in refresh_all_neighbors
                    neighbor_count=len(buoy._nearest_neighbors),
                    depth=depth
                )
                
                buoy.measurements.append(measurement)

    def refresh_all_neighbors(self) -> None:
        """Update neighbor relationships for all buoys."""
        if not self._dirty_neighbors:
            return
            
        for buoy_id in self._dirty_neighbors:
            buoy = self.buoys[buoy_id]
            nearby = self._spatial_index.get_nearby_buoys(buoy)
            
            buoy._nearest_neighbors.clear()
            for other_id in nearby:
                other = self.buoys[other_id]
                distance, bearing = buoy.get_vector_to(other)
                buoy._nearest_neighbors[other_id] = (distance, bearing)
        
        self._dirty_neighbors.clear()

    def initialize_data_buffer(self, max_seq_len: int, max_buoys: int, buffer_path: str = "swell_data.bin"):
        """Initialize a memory-mapped buffer for large datasets."""
        self.data_buffer = np.memmap(
            buffer_path,
            dtype=STORAGE_DTYPE,
            mode="w+",
            shape=(max_seq_len, max_buoys, 7)
        )

    def get_batch(self, start_idx: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve a batch of sequences for transformer training."""
        if self.data_buffer is None:
            raise ValueError("Data buffer not initialized. Call `initialize_data_buffer` first.")
        batch_sequences = self.data_buffer[start_idx:start_idx + batch_size]
        batch_masks = (batch_sequences[:, :, 0] != 0).astype(np.float32)
        return batch_sequences, batch_masks

    def save_preprocessed(self, path: str):
        """Save preprocessed data to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "seq": self.data_buffer,
                "mask": (self.data_buffer[:, :, 0] != 0).astype(np.float32),
                "gradient_scale": self._gradient_scale,
                "propagation_graph": self.propagation_graph
            }, f, protocol=4)

    def load_preprocessed(self, path: str):
        """Load preprocessed data from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.data_buffer = data["seq"]
            self._gradient_scale = data["gradient_scale"]
            self.propagation_graph = data["propagation_graph"]

    def get_sequence(self, start: datetime, end: datetime, length: int = 24) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Get a sequence of encoded measurements for transformer training."""
        if not self.buoys:
            raise ValueError("No buoys available to generate sequence")
            
        if start >= end:
            raise ValueError("Start time must be before end time")
            
        if self._dirty_neighbors:
            self.refresh_all_neighbors()
            
        if not self.reference_time:
            self.reference_time = start
            self._recompute_encodings()
        
        # Pad length to next power of 2 for transformer efficiency
        padded_length = 2 ** int(np.ceil(np.log2(length)))
        seq = np.zeros((padded_length, len(self.buoys), 7), dtype=STORAGE_DTYPE)
        mask = np.zeros((padded_length, len(self.buoys)), dtype=bool)
        time_step = (end - start) / length
        
        for t in range(padded_length):
            current = start + time_step * t
            for bid, buoy in self.buoys.items():
                idx = self.buoy_indices[bid]
                window = timedelta(minutes=30)
                measurements = [
                    m for m in buoy.measurements
                    if current - window <= m.timestamp <= current + window
                ]
                
                if measurements:
                    values = np.array([m.encoded_values for m in measurements], dtype=CALCULATION_DTYPE)
                    seq[t, idx] = _safe_cast(np.mean(values, axis=0), STORAGE_DTYPE)
                    mask[t, idx] = True
                else:
                    seq[t, idx], mask[t, idx] = self._fill_missing(t, idx, seq, mask)
        
        # Precompute attention mask for transformer
        attention_mask = (mask.astype(np.float32) * self._gradient_scale).astype(STORAGE_DTYPE)
        
        self.build_propagation_graph(seq, mask)
        return seq, attention_mask, self.propagation_graph
        
        
    def _fill_missing(self, t: int, idx: int, seq: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, bool]:
            buoy = list(self.buoys.values())[idx]
            values, weights = [], []
            
            for n_id, (dist, _) in buoy._nearest_neighbors.items():
                n_idx = self.buoy_indices[n_id]
                if mask[t, n_idx]:
                    weight = np.float32(1.0 / (dist + 1e-6))
                    weights.append(weight)
                    values.append(seq[t, n_idx].astype(CALCULATION_DTYPE))
            
            if values:
                weighted_avg = np.average(values, axis=0, weights=weights)
                return weighted_avg.astype(STORAGE_DTYPE), True
            
            return np.array([
                0.0,
                self.normalizer.normalize('height', self.normalizer.stats['height'].mean),
                self.normalizer.normalize('sig_height', self.normalizer.stats['sig_height'].mean),
                self.normalizer.normalize('period', self.normalizer.stats['period'].mean),
                np.float32(0.0),
                np.float32(1.0),
                np.log(self.normalizer.stats['period'].mean + 1e-6)
            ], dtype=STORAGE_DTYPE), True
    
    def analyze_propagation(self, seq: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> Dict:
            analysis = {'paths': [], 'max_precision_error': 0.0}
            
            # Precision validation
            cast_seq = seq.astype(STORAGE_DTYPE).astype(CALCULATION_DTYPE)
            analysis['max_precision_error'] = np.max(np.abs(seq.astype(CALCULATION_DTYPE) - cast_seq))
            
            # Propagation analysis
            for bid in self.buoys:
                idx = self.buoy_indices[bid]
                buoy = self.buoys[bid]
                
                for n_id, (dist, bearing) in buoy._nearest_neighbors.items():
                    n_idx = self.buoy_indices.get(n_id, -1)
                    if n_idx == -1:
                        continue
                    
                    for t in range(seq.shape[0] - 1):
                        if mask[t, idx] and mask[t+1, n_idx]:
                            h1 = self.normalizer.denormalize('height', seq[t, idx, 1])
                            h2 = self.normalizer.denormalize('height', seq[t+1, n_idx, 1])
                            
                            if abs(h2 - h1) > threshold:
                                analysis['paths'].append({
                                    'from': bid,
                                    'to': n_id,
                                    'distance': dist,
                                    'bearing': bearing,
                                    'Δt': t,
                                    'Δh': np.float32(h2 - h1)
                                })
            
            return analysis
    
    def _recompute_encodings(self):
            """Update temporal encodings after reference time change"""
            if not self.reference_time:
                return
            
            for buoy in self.buoys.values():
                for m in buoy.measurements:
                    time_delta = np.float32(
                        (m.timestamp - self.reference_time).total_seconds() / 86400
                    )
                    m.encoded_values[0] = time_delta.astype(STORAGE_DTYPE)
    
        # Mixed precision training utilities ---------------------------------------
    def get_gradient_scale(self) -> np.float32:
            """Get current gradient scale for mixed precision training"""
            return self._gradient_scale
        
    def apply_gradient_scaling(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
            """Apply gradient scaling for mixed precision training"""
            return [g * self._gradient_scale for g in gradients]
        
    def update_gradient_scale(self, has_overflow: bool):
            """Dynamic loss scaling from NVIDIA's mixed precision guidelines"""
            if has_overflow:
                self._gradient_scale = max(self._gradient_scale / 2, 1.0)
            else:
                self._gradient_scale *= min(self._gradient_scale * 2, 2**24)