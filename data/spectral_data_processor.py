#!/usr/bin/env python3
"""
spectral_data_processor.py
--------------------------
Complete processor for NDBC spectral data integration with your wave transformer.
Updated to handle raw spectral density (.data_spec) files for true frequency-domain data.
Location: data/spectral_data_processor.py (or swell_tracker/spectral_data_processor.py)
"""

import gzip
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging
from scipy import signal
from scipy.interpolate import interp1d
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpectralDataProcessor:
    """Process NDBC spectral data for transformer training"""
    
    def __init__(self, data_dir: str = "data/buoy_data"):
        self.data_dir = Path(data_dir)
        
        # Standard NDBC frequency bins (Hz) - common across most stations
        self.freq_bins = np.array([
            0.0200, 0.0325, 0.0375, 0.0425, 0.0475, 0.0525, 0.0575, 0.0625,
            0.0675, 0.0725, 0.0775, 0.0825, 0.0875, 0.0925, 0.1000, 0.1100,
            0.1200, 0.1300, 0.1400, 0.1500, 0.1600, 0.1700, 0.1800, 0.1900,
            0.2000, 0.2100, 0.2200, 0.2300, 0.2400, 0.2500, 0.2600, 0.2700,
            0.2800, 0.2900, 0.3000, 0.3100, 0.3200, 0.3300, 0.3400, 0.3500,
            0.3650, 0.3850, 0.4050, 0.4250, 0.4450, 0.4650, 0.4850
        ])
        
        # Wave frequency thresholds for classification
        self.swell_threshold = 0.15  # Hz - waves below this are swell
        self.windsea_threshold = 0.25  # Hz - waves above this are wind sea
        
    def parse_raw_spectral_file(self, filepath: Path) -> List[Dict]:
        """Parse a .data_spec file containing true spectral density (m²/Hz)"""
        records = []
        
        try:
            with gzip.open(filepath, 'rt') as f:
                content = f.read().strip()
        
            lines = content.split('\n')
            
            # The .data_spec format has frequency bins in the header
            # Format: YY MM DD hh mm (freq1) (freq2) ... (freq47)
            #         2024 12 01 14 00 0.033 0.033 0.033 ...
            
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.split()
                if len(parts) < 52:  # 5 time fields + 47 frequency bins
                    continue
                
                try:
                    # Parse timestamp
                    year = int(parts[0])
                    if year < 100:
                        year += 2000
                    month = int(parts[1])
                    day = int(parts[2])
                    hour = int(parts[3])
                    minute = int(parts[4])
                    
                    timestamp = datetime(year, month, day, hour, minute)
                    
                    # Parse spectral density values (m²/Hz)
                    spectral_values = []
                    for i in range(5, min(52, len(parts))):  # 47 frequency bins
                        val = parts[i]
                        if val in ['MM', '999.00', '-999.00', '99.00']:
                            spectral_values.append(np.nan)
                        else:
                            spectral_values.append(float(val))
                    
                    records.append({
                        'timestamp': timestamp,
                        'spectral_density': np.array(spectral_values),
                        'n_frequencies': len(spectral_values),
                        'is_raw_spectral': True  # Flag to indicate this is true spectral density
                    })
                    
                except Exception as e:
                    logger.debug(f"Error parsing raw spectral line: {line[:50]}... - {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
        
        return records
        
    def parse_directional_file(self, filepath: Path) -> List[Dict]:
        """Parse directional spectral data files"""
        records = []
        
        try:
            with gzip.open(filepath, 'rt') as f:
                content = f.read().strip()
                
            lines = content.split('\n')
            
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.split()
                if len(parts) < 5:
                    continue
                    
                try:
                    # Parse timestamp
                    year = int(parts[0])
                    if year < 100:
                        year += 2000
                    month = int(parts[1])
                    day = int(parts[2])
                    hour = int(parts[3])
                    minute = int(parts[4])
                    
                    timestamp = datetime(year, month, day, hour, minute)
                    
                    # Parse directional values (mean wave direction for each frequency)
                    directional_values = []
                    for val in parts[5:]:
                        if val in ['MM', '999.0', '-999.0', '999']:
                            directional_values.append(np.nan)
                        else:
                            try:
                                dir_val = float(val)
                                # Convert negative directions to 0-360 range
                                if dir_val < 0:
                                    dir_val += 360
                                directional_values.append(dir_val)
                            except ValueError:
                                directional_values.append(np.nan)
                    
                    records.append({
                        'timestamp': timestamp,
                        'mean_directions': np.array(directional_values),
                        'n_frequencies': len(directional_values)
                    })
                    
                except Exception as e:
                    logger.debug(f"Error parsing directional line: {line[:50]}... - {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            
        return records
    
    def parse_spectral_file(self, filepath: Path) -> List[Dict]:
        """Parse a single compressed spectral file (legacy wave summary data)"""
        records = []
        
        try:
            with gzip.open(filepath, 'rt') as f:
                content = f.read().strip()
                
            lines = [line for line in content.split('\n') 
                    if line.strip() and not line.startswith('#')]
            
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                    
                try:
                    year, month, day, hour, minute = map(int, parts[:5])
                    timestamp = datetime(year, month, day, hour, minute)
                    
                    # Parse spectral values
                    spectral_values = []
                    for val in parts[5:]:
                        if val == 'MM' or val == '999.0' or val == '-999.0':
                            spectral_values.append(np.nan)
                        else:
                            spectral_values.append(float(val))
                    
                    records.append({
                        'timestamp': timestamp,
                        'spectral_density': np.array(spectral_values),
                        'n_frequencies': len(spectral_values),
                        'is_raw_spectral': False  # This is summary data
                    })
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing line: {line[:50]}... - {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            
        return records
    
    def parse_wave_file(self, filepath: Path) -> List[Dict]:
        """Parse a single compressed wave parameter file"""
        records = []
        
        try:
            with gzip.open(filepath, 'rt') as f:
                content = f.read().strip()
                
            lines = content.split('\n')
            if len(lines) < 2:
                return records
                
            # Parse header - look for standard NDBC columns
            header_candidates = []
            for line in lines[:3]:
                if any(col in line.upper() for col in ['WVHT', 'DPD', 'APD', 'MWD']):
                    header_candidates.append(line.replace('#', '').split())
            
            if not header_candidates:
                logger.warning(f"No valid header found in {filepath}")
                return records
                
            headers = header_candidates[0]
            data_start = next(i for i, line in enumerate(lines) 
                            if not line.startswith('#') and line.strip())
            
            for line in lines[data_start:]:
                parts = line.split()
                if len(parts) < len(headers):
                    continue
                    
                try:
                    # Create record dict
                    record = {'timestamp': None}
                    
                    # Parse timestamp
                    year_col = next((i for i, h in enumerate(headers) if h.upper() in ['YY', 'YYYY']), None)
                    mm_col = next((i for i, h in enumerate(headers) if h.upper() == 'MM'), None)
                    dd_col = next((i for i, h in enumerate(headers) if h.upper() == 'DD'), None)
                    hh_col = next((i for i, h in enumerate(headers) if h.upper() == 'HH'), None)
                    min_col = next((i for i, h in enumerate(headers) if h.upper() in ['MM', 'MIN']), None)
                    
                    if all(col is not None for col in [year_col, mm_col, dd_col, hh_col]):
                        year = int(parts[year_col])
                        if year < 100:  # 2-digit year
                            year += 2000
                        month = int(parts[mm_col])
                        day = int(parts[dd_col])
                        hour = int(parts[hh_col])
                        minute = int(parts[min_col]) if min_col is not None else 0
                        
                        record['timestamp'] = datetime(year, month, day, hour, minute)
                    
                    # Parse wave parameters
                    for i, header in enumerate(headers):
                        if i < len(parts) and parts[i] not in ['MM', '999.0', '-999.0']:
                            try:
                                record[header.lower()] = float(parts[i])
                            except ValueError:
                                record[header.lower()] = np.nan
                        else:
                            record[header.lower()] = np.nan
                    
                    if record['timestamp'] is not None:
                        records.append(record)
                        
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing wave line: {line[:50]}... - {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            
        return records
    
    def compute_bulk_parameters(self, spectral_density: np.ndarray, 
                              frequencies: np.ndarray) -> Dict[str, float]:
        """Compute bulk wave parameters from spectral density"""
        # Remove invalid values
        valid_mask = ~np.isnan(spectral_density) & (spectral_density >= 0)
        if not np.any(valid_mask):
            return {k: np.nan for k in [
                'significant_height', 'peak_period', 'mean_period', 
                'spectral_width', 'peak_frequency', 'energy_density'
            ]}
        
        freq = frequencies[valid_mask]
        spec = spectral_density[valid_mask]
        
        # Ensure frequencies are positive and sorted
        if len(freq) == 0 or np.any(freq <= 0):
            return {k: np.nan for k in [
                'significant_height', 'peak_period', 'mean_period', 
                'spectral_width', 'peak_frequency', 'energy_density'
            ]}
        
        # Moments of the spectrum
        m0 = np.trapz(spec, freq)  # 0th moment (total energy)
        
        if m0 <= 0:
            return {k: np.nan for k in [
                'significant_height', 'peak_period', 'mean_period', 
                'spectral_width', 'peak_frequency', 'energy_density'
            ]}
        
        m1 = np.trapz(spec * freq, freq)  # 1st moment
        m2 = np.trapz(spec * freq**2, freq)  # 2nd moment
        
        # Significant wave height (H_s = 4 * sqrt(m0))
        significant_height = 4 * np.sqrt(m0)
        
        # Peak frequency and period
        peak_idx = np.argmax(spec)
        peak_frequency = freq[peak_idx]
        peak_period = 1.0 / peak_frequency
        
        # Mean period (Tm01 = m0/m1)
        mean_period = m0 / m1 if m1 > 0 else np.nan
        
        # Spectral width (measure of spectrum spreading)
        if m1 > 0 and m2 > 0:
            spectral_width = np.sqrt(m0 * m2 - m1**2) / m1
        else:
            spectral_width = np.nan
        
        return {
            'significant_height': significant_height,
            'peak_period': peak_period,
            'mean_period': mean_period,
            'spectral_width': spectral_width,
            'peak_frequency': peak_frequency,
            'energy_density': m0
        }
    
    def extract_swell_components(self, spectral_density: np.ndarray,
                               frequencies: np.ndarray) -> Dict[str, np.ndarray]:
        """Separate swell from wind waves using frequency thresholds"""
        swell_mask = frequencies < self.swell_threshold
        windsea_mask = frequencies > self.windsea_threshold
        transition_mask = (frequencies >= self.swell_threshold) & (frequencies <= self.windsea_threshold)
        
        swell_spec = np.where(swell_mask, spectral_density, 0)
        windsea_spec = np.where(windsea_mask, spectral_density, 0)
        transition_spec = np.where(transition_mask, spectral_density, 0)
        
        # Compute energy in each component
        valid_freq = ~np.isnan(frequencies) & (frequencies > 0)
        
        if np.any(valid_freq):
            swell_energy = np.trapz(swell_spec[valid_freq], frequencies[valid_freq])
            windsea_energy = np.trapz(windsea_spec[valid_freq], frequencies[valid_freq])
            transition_energy = np.trapz(transition_spec[valid_freq], frequencies[valid_freq])
        else:
            swell_energy = windsea_energy = transition_energy = 0
        
        total_energy = swell_energy + windsea_energy + transition_energy
        
        return {
            'swell_spectrum': swell_spec,
            'windsea_spectrum': windsea_spec,
            'transition_spectrum': transition_spec,
            'swell_energy': swell_energy,
            'windsea_energy': windsea_energy,
            'transition_energy': transition_energy,
            'swell_fraction': swell_energy / (total_energy + 1e-8),
            'windsea_fraction': windsea_energy / (total_energy + 1e-8)
        }
    
    def create_transformer_features(self, station_id: str, 
                                   spectral_records: List[Dict],
                                   directional_records: List[Dict],
                                   wave_records: List[Dict]) -> np.ndarray:
        """Convert spectral, directional, and wave data to transformer features"""
        
        # Combine timestamps from all data sources
        all_timestamps = set()
        spectral_dict = {}
        directional_dict = {}
        wave_dict = {}
        
        # Index spectral records by timestamp
        for record in spectral_records:
            ts = record['timestamp']
            all_timestamps.add(ts)
            spectral_dict[ts] = record
        
        # Index directional records by timestamp
        for record in directional_records:
            ts = record['timestamp']
            all_timestamps.add(ts)
            directional_dict[ts] = record
        
        # Index wave records by timestamp
        for record in wave_records:
            ts = record['timestamp']
            all_timestamps.add(ts)
            wave_dict[ts] = record
        
        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)
        
        # Create feature array
        features = []
        
        for ts in sorted_timestamps:
            feature_vec = np.full(18, np.nan)  # 18 features total (added directional features)
            
            # Get spectral data if available
            if ts in spectral_dict:
                spec_record = spectral_dict[ts]
                spec_values = spec_record['spectral_density']
                
                # Use appropriate frequency bins
                if len(spec_values) == len(self.freq_bins):
                    frequencies = self.freq_bins
                    spectral_density = spec_values
                else:
                    # Interpolate to standard frequency grid
                    freq_interp = np.linspace(0.02, 0.485, len(spec_values))
                    f_interp = interp1d(freq_interp, spec_values, 
                                      bounds_error=False, fill_value=np.nan)
                    spectral_density = f_interp(self.freq_bins)
                    frequencies = self.freq_bins
                
                # Compute bulk parameters from spectral data
                bulk_params = self.compute_bulk_parameters(spectral_density, frequencies)
                swell_components = self.extract_swell_components(spectral_density, frequencies)
                
                # Features 0-5: Basic spectral parameters
                feature_vec[0] = bulk_params['significant_height']
                feature_vec[1] = bulk_params['peak_period']
                feature_vec[2] = bulk_params['mean_period']
                feature_vec[3] = bulk_params['spectral_width']
                feature_vec[4] = bulk_params['peak_frequency']
                feature_vec[5] = bulk_params['energy_density']
                
                # Features 6-9: Swell/wind sea separation
                feature_vec[6] = swell_components['swell_energy']
                feature_vec[7] = swell_components['windsea_energy']
                feature_vec[8] = swell_components['swell_fraction']
                feature_vec[9] = swell_components['windsea_fraction']
                
                # Features 10-11: Derived parameters
                feature_vec[10] = np.log1p(bulk_params['energy_density'])  # Log energy
                feature_vec[11] = bulk_params['peak_period'] * bulk_params['significant_height']  # Steepness proxy
                
                # Feature 12: Raw spectral flag
                feature_vec[12] = 1.0 if spec_record.get('is_raw_spectral', False) else 0.0
            
            # Get directional data if available (features 13-15)
            if ts in directional_dict:
                dir_record = directional_dict[ts]
                mean_directions = dir_record['mean_directions']
                
                # Compute directional statistics
                valid_dirs = mean_directions[~np.isnan(mean_directions)]
                if len(valid_dirs) > 0:
                    # Convert to radians for circular statistics
                    dirs_rad = np.deg2rad(valid_dirs)
                    
                    # Mean direction using circular statistics
                    mean_dir = np.rad2deg(np.arctan2(np.mean(np.sin(dirs_rad)), 
                                                   np.mean(np.cos(dirs_rad))))
                    if mean_dir < 0:
                        mean_dir += 360
                    
                    # Directional spread (circular standard deviation)
                    R = np.sqrt(np.mean(np.sin(dirs_rad))**2 + np.mean(np.cos(dirs_rad))**2)
                    dir_spread = np.rad2deg(np.sqrt(-2 * np.log(R))) if R > 0 else np.nan
                    
                    # Peak directional frequency (most common direction)
                    hist, bins = np.histogram(valid_dirs, bins=36, range=(0, 360))
                    peak_dir = bins[np.argmax(hist)] + 5  # Center of bin
                    
                    feature_vec[13] = mean_dir
                    feature_vec[14] = dir_spread
                    feature_vec[15] = peak_dir
            
            # Get wave parameter data if available (features 16-17)
            if ts in wave_dict:
                wave_record = wave_dict[ts]
                feature_vec[15] = wave_record.get('wvht', np.nan)  # Wave height
                feature_vec[16] = wave_record.get('dpd', np.nan)   # Dominant period
                feature_vec[17] = wave_record.get('mwd', np.nan)   # Mean wave direction
            
            features.append(feature_vec)
        
        return np.array(features), sorted_timestamps
    
    def load_station_data(self, station_id: str, max_files: int = 100) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load raw spectral, directional, and wave data for a station"""
        raw_dir = self.data_dir / "raw"
        
        # Updated to look for raw_spectral files!
        raw_spectral_files = sorted(raw_dir.glob(f"{station_id}_raw_spectral_*.gz"))[-max_files:]
        directional_files = sorted(raw_dir.glob(f"{station_id}_directional_*.gz"))[-max_files:]
        wave_files = sorted(raw_dir.glob(f"{station_id}_wave_*.gz"))[-max_files:]
        
        # Fallback to legacy spectral files if no raw spectral files found
        if not raw_spectral_files:
            legacy_spectral_files = sorted(raw_dir.glob(f"{station_id}_spectral_*.gz"))[-max_files:]
            logger.info(f"No raw spectral files found for {station_id}, using {len(legacy_spectral_files)} legacy spectral files")
        else:
            legacy_spectral_files = []
        
        logger.info(f"Loading {len(raw_spectral_files)} raw spectral, {len(directional_files)} directional, and {len(wave_files)} wave files for {station_id}")
        
        # Parse raw spectral files (true frequency-domain data!)
        all_spectral_records = []
        for filepath in raw_spectral_files:
            records = self.parse_raw_spectral_file(filepath)
            all_spectral_records.extend(records)
        
        # Parse legacy spectral files if needed
        for filepath in legacy_spectral_files:
            records = self.parse_spectral_file(filepath)
            all_spectral_records.extend(records)
        
        # Parse directional files
        all_directional_records = []
        for filepath in directional_files:
            records = self.parse_directional_file(filepath)
            all_directional_records.extend(records)
        
        # Parse wave files
        all_wave_records = []
        for filepath in wave_files:
            records = self.parse_wave_file(filepath)
            all_wave_records.extend(records)
        
        # Remove duplicates and sort by timestamp
        spectral_dict = {r['timestamp']: r for r in all_spectral_records}
        directional_dict = {r['timestamp']: r for r in all_directional_records}
        wave_dict = {r['timestamp']: r for r in all_wave_records}
        
        spectral_records = sorted(spectral_dict.values(), key=lambda x: x['timestamp'])
        directional_records = sorted(directional_dict.values(), key=lambda x: x['timestamp'])
        wave_records = sorted(wave_dict.values(), key=lambda x: x['timestamp'])
        
        logger.info(f"Loaded {len(spectral_records)} spectral, {len(directional_records)} directional, and {len(wave_records)} wave records for {station_id}")
        
        return spectral_records, directional_records, wave_records
    
    def process_all_stations(self, station_ids: List[str]) -> Dict[str, Tuple[np.ndarray, List[datetime]]]:
        """Process all stations and return transformer-ready features"""
        
        station_features = {}
        
        for station_id in station_ids:
            try:
                logger.info(f"Processing station {station_id}...")
                
                # Load raw data
                spectral_records, directional_records, wave_records = self.load_station_data(station_id)
                
                if not spectral_records and not directional_records and not wave_records:
                    logger.warning(f"No data found for station {station_id}")
                    continue
                
                # Create features
                features, timestamps = self.create_transformer_features(
                    station_id, spectral_records, directional_records, wave_records
                )
                
                if len(features) > 0:
                    station_features[station_id] = (features, timestamps)
                    logger.info(f"Created {features.shape[0]} feature vectors for {station_id}")
                else:
                    logger.warning(f"No valid features created for {station_id}")
                    
            except Exception as e:
                logger.error(f"Error processing station {station_id}: {e}")
                continue
        
        return station_features
    
    def save_processed_data(self, station_features: Dict, output_path: str = "data/processed_features.json"):
        """Save processed features for later use"""
        
        # Convert to serializable format
        serializable_data = {}
        
        for station_id, (features, timestamps) in station_features.items():
            serializable_data[station_id] = {
                'features': features.tolist(),
                'timestamps': [ts.isoformat() for ts in timestamps],
                'feature_names': [
                    'significant_height', 'peak_period', 'mean_period', 'spectral_width',
                    'peak_frequency', 'energy_density', 'swell_energy', 'windsea_energy',
                    'swell_fraction', 'windsea_fraction', 'log_energy', 'steepness_proxy',
                    'is_raw_spectral', 'mean_direction', 'directional_spread', 'peak_direction',
                    'wave_height', 'dominant_period', 'mean_wave_direction'
                ]
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Saved processed features to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process NDBC spectral data")
    parser.add_argument("--data-dir", default="data/buoy_data", help="Data directory")
    parser.add_argument("--stations", help="Comma-separated station IDs")
    parser.add_argument("--output", default="data/processed_features.json", help="Output file")
    parser.add_argument("--max-files", type=int, default=100, help="Max files per station")
    
    args = parser.parse_args()
    
    processor = SpectralDataProcessor(args.data_dir)
    
    # Get station list
    if args.stations:
        station_ids = [s.strip() for s in args.stations.split(",")]
    else:
        # Auto-detect stations from collected data
        raw_dir = Path(args.data_dir) / "raw"
        if raw_dir.exists():
            # Prioritize raw spectral files, fall back to legacy spectral
            station_ids = list(set(
                f.name.split('_')[0] for f in raw_dir.glob("*_raw_spectral_*.gz")
            ))
            if not station_ids:
                station_ids = list(set(
                    f.name.split('_')[0] for f in raw_dir.glob("*_spectral_*.gz")
                ))
            logger.info(f"Auto-detected {len(station_ids)} stations: {', '.join(sorted(station_ids))}")
        else:
            logger.error(f"No data directory found at {raw_dir}")
            return
    
    # Process stations
    station_features = processor.process_all_stations(station_ids)
    
    if station_features:
        processor.save_processed_data(station_features, args.output)
        
        # Print summary
        total_features = sum(len(features) for features, _ in station_features.values())
        print(f"\nProcessing Summary:")
        print(f"Stations processed: {len(station_features)}")
        print(f"Total feature vectors: {total_features}")
        print(f"Features per vector: 18")
        print(f"Output saved to: {args.output}")
        
        # Show sample data
        if station_features:
            sample_station = next(iter(station_features))
            sample_features, sample_timestamps = station_features[sample_station]
            print(f"\nSample from station {sample_station}:")
            print(f"Time range: {sample_timestamps[0]} to {sample_timestamps[-1]}")
            print(f"Feature shape: {sample_features.shape}")
            
            # Show data quality info
            raw_spectral_count = np.sum(sample_features[:, 12] == 1.0)
            print(f"Raw spectral records: {raw_spectral_count}/{len(sample_features)} ({100*raw_spectral_count/len(sample_features):.1f}%)")
    else:
        logger.error("No station data was successfully processed")

if __name__ == "__main__":
    main()