# SwellTracker: Physics-Informed Wave Forecasting with Spatiotemporal Transformers

**A deep learning system for predicting ocean wave conditions using multi-station buoy networks and physics-informed neural networks.**

---

## Overview

SwellTracker is an advanced wave forecasting system that combines real-time oceanographic data collection with physics-informed transformer models to predict wave conditions across buoy networks. The system captures multi-dimensional wave data from NOAA's National Data Buoy Center (NDBC), processes it into physics-aware features, and uses spatiotemporal attention mechanisms to model wave propagation patterns between stations.

Unlike traditional wave models that rely on numerical weather prediction, SwellTracker learns directly from observational data while respecting fundamental wave physics constraints like energy conservation and directional consistency.

## Core Concept: Network State Modeling

The fundamental innovation of SwellTracker is treating ocean wave systems as **dynamic networks** where each buoy represents a node with complex spatiotemporal relationships. The system models:

- **Network State**: The collective wave conditions across all active buoy stations at any given time
- **Propagation Patterns**: How wave energy travels between stations based on distance, bathymetry, and wave physics
- **Temporal Evolution**: How network states evolve over time due to storm systems, swell propagation, and local generation

### Time Series Transformer Architecture

SwellTracker employs a specialized transformer architecture that:

1. **Encodes Station Data**: Each buoy's wave measurements are embedded as tokens containing spectral features, directional information, and meteorological conditions
2. **Spatial Attention**: Models wave propagation between stations using geographic coordinates and haversine distances
3. **Temporal Attention**: Captures how wave conditions evolve over 24-hour sequences
4. **Physics Constraints**: Enforces energy conservation and directional consistency during training

---

## Current Performance

*Latest training run on California Coast stations:*
- **Test Wave Height MAE**: 0.479m 
- **Test Peak Period MAE**: 0.368s
- **Energy Conservation Error**: 0.1152
- **Model Parameters**: 7.16M parameters
- **Training Data**: Multi-station 24-hour sequences with 20-dimensional physics features

---

## Pipeline Components

### 1. Data Collection (`production_buoy_collector.py`)

**Purpose**: Collection of real-time NDBC data with full historical retention

**Process**:
- Fetches 7 data streams per station: wave parameters, raw spectral density, and complete directional suite (α₁, α₂, r₁, r₂)
- Implements atomic writes and deduplication to maintain data integrity
- Supports continuous operation with graceful shutdown handling
- Maintains gzipped historical files with no data expiration

**Key Features**:
- Parallel collection across multiple stations
- Robust error handling for network failures
- Temporal deduplication based on timestamp parsing
- Configurable collection intervals (default: 30 minutes)

### 2. Feature Engineering (`full_spectrum_processor.py`)

**Purpose**: Transform raw NDBC data into physics-informed feature vectors

**Input Processing**:
- **Raw Spectral Data**: S(f) energy density across frequency bands
- **Directional Data**: Mean wave direction (α₁), secondary direction (α₂), and directional spreads (r₁, r₂)
- **Meteorological Data**: Wind speed, direction, and atmospheric conditions

**Feature Extraction** (20-dimensional vectors):
- **Spectral Physics** (10 features): Significant height, peak/mean periods, energy distribution, swell/wind-sea separation
- **Directional Physics** (5 features): Primary/secondary directions, directional spread, bimodal strength
- **Meteorological** (3 features): Wind speed/direction, wind-wave alignment
- **Validation** (2 features): Cross-validation with NDBC processed parameters

**Physics-Aware Processing**:
- Circular interpolation for directional data (handles 359°→1° transitions)
- Energy conservation validation between spectral components
- Temporal alignment between different data streams (30-60 minute windows)

### 3. Sequence Generation (`enhanced_wave_sequencer.py`)

**Purpose**: Create spatiotemporal training sequences from multi-station data

**Token Encoding Process**:
1. **Station Encoding**: Each buoy becomes a spatial node with lat/lon coordinates
2. **Feature Embedding**: 20D physics features → 256D learned embeddings respecting physics domains
3. **Temporal Sequencing**: 24-hour sequences (hourly observations) 
4. **Spatial Grouping**: Simultaneous observations across multiple stations (minimum 3 stations per sequence)

**Data Architecture**:
- **Input Shape**: `(batch_size, n_stations, sequence_length, features)`
- **Normalization**: Physics-aware normalization (fitted only on training data)
- **Temporal Split**: 70% train, 15% validation, 15% test (chronological split)

### 4. Physics-Informed Model (`physics_wave_transformer.py`)

**Purpose**: Spatiotemporal transformer with wave physics constraints

**Architecture Components**:

- **Physics-Informed Embedding**: Separate projections for spectral, directional, meteorological, and validation features
- **Spatial Attention**: Geographic distance-biased attention using haversine distances
- **Temporal Attention**: Standard self-attention over 24-hour sequences  
- **Gated Fusion**: Learnable combination of spatial and temporal information flows

**Physics Constraints**:
- **Energy Conservation**: `total_energy = swell_energy + windsea_energy`
- **Fraction Consistency**: `swell_fraction + windsea_fraction = 1.0`
- **Circular Loss**: Proper handling of directional variables (0° = 360°)

**Training Features**:
- Multi-domain loss weighting (spectral, directional, meteorological)
- Early stopping with validation monitoring
- Learning rate scheduling with plateau detection

### 5. Training Pipeline (`train_wave_model.py`)

**Purpose**: End-to-end model training with comprehensive monitoring

**Training Process**:
1. Load enhanced features from JSON
2. Create temporal data splits with proper normalization
3. Initialize physics-informed model and loss functions
4. Train with early stopping (patience: 15 epochs)
5. Evaluate on held-out test set

**Monitoring & Logging**:
- Weights & Biases integration for experiment tracking
- Real-time training metrics (loss, MAE, physics constraint violations)
- Automatic model checkpointing and artifact saving
- Training curve visualization

---

## Quick Start

```bash
# Install dependencies
pip install torch numpy pandas scipy requests wandb

# Collect data for California coast stations
python production_buoy_collector.py --stations "46022,46026,46027" --data-dir [directory]

# Process raw data into physics features  
python full_spectrum_processor.py --stations "46022,46026,46027" --output enhanced_features.json

# Train the model
python train_wave_model.py --features enhanced_features.json --epochs 100
```

---

## Technical Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+ (CUDA optional)
- **Core Dependencies**: NumPy, Pandas, SciPy, Requests
- **Optional**: Weights & Biases for experiment tracking

---

## Data Sources

- **NOAA National Data Buoy Center (NDBC)**: Real-time wave, meteorological, and spectral data
- **Station Coverage**: Configurable buoy networks (default: California Coast priority stations)
- **Update Frequency**: 30-minute collection intervals with historical retention

---

