#!/usr/bin/env python3
"""
Enhanced Multi-Stream Spectral Processor - COMPLETE REFACTORED VERSION

Processes all collected NDBC data types:
- Raw spectral density (S(f))
- Complete directional suite (alpha1, alpha2, r1, r2) 
- Processed spectral parameters (validation)
- Meteorological data (wave generation physics)

Outputs 20-dimensional physics-informed feature vectors optimized for transformer training.
"""

import gzip
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging
from scipy.interpolate import interp1d
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class EnhancedMultiStreamProcessor:
    """
    Enhanced processor for complete NDBC data integration.
    
    Combines:
    - Raw spectral density S(f) for true wave physics
    - Complete directional data for propagation modeling
    - Meteorological data for generation mechanisms
    - Processed parameters for validation
    """

    def __init__(self, data_dir: str = "data/buoy_data"):
        self.data_dir = Path(data_dir)
        
        # Extended frequency grid (0.033-0.485 Hz)
        self.target_freq_bins = np.array([
            0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080,
            0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180,
            0.190, 0.200, 0.220, 0.240, 0.260, 0.280, 0.300, 0.320, 0.350, 0.400, 
            0.450, 0.485
        ])
        
        # Wave physics parameters
        self.swell_threshold = 0.125    # Hz - swell/wind separation
        self.windsea_threshold = 0.25   # Hz - wind wave domain
        
        # Data validation limits
        self.max_reasonable_energy = 100.0   # m²/Hz
        self.min_reasonable_freq = 0.020     # Hz
        self.max_reasonable_freq = 0.500     # Hz
        
        # Enhanced 20-dimensional feature set
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

    def parse_raw_spectral_line(self, line: str) -> Optional[Dict]:
        """Parse raw spectral density line"""
        if line.startswith('#') or not line.strip():
            return None
        
        parts = line.split()
        if len(parts) < 8:
            return None

        try:
            # Parse timestamp
            year = int(parts[0])
            if year < 100:
                year += 2000
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            
            # Validate date components
            if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
                return None
                
            timestamp = datetime(year, month, day, hour, minute)
            sep_freq = float(parts[5])
            
            # Parse spectral density values
            frequencies = []
            spectral_densities = []
            
            i = 6
            while i < len(parts) - 1:
                try:
                    spec_str = parts[i]
                    if i + 1 < len(parts) and '(' in parts[i + 1] and ')' in parts[i + 1]:
                        freq_str = parts[i + 1].strip('()')
                        
                        if spec_str not in ['MM', '999.00', '-999.00', '99.00']:
                            spec_val = float(spec_str)
                            freq = float(freq_str)
                            
                            if (self.min_reasonable_freq <= freq <= self.max_reasonable_freq and 
                                0 <= spec_val <= self.max_reasonable_energy):
                                frequencies.append(freq)
                                spectral_densities.append(spec_val)
                        
                        i += 2
                    else:
                        i += 1
                        
                except (ValueError, IndexError):
                    i += 1
                    continue
            
            if len(frequencies) < 5:
                return None
            
            # Sort by frequency
            freq_array = np.array(frequencies)
            spec_array = np.array(spectral_densities)
            sort_idx = np.argsort(freq_array)
            
            return {
                'timestamp': timestamp,
                'frequencies': freq_array[sort_idx],
                'spectral_density': spec_array[sort_idx],
                'separation_frequency': sep_freq,
                'n_frequencies': len(frequencies)
            }
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing raw spectral line: {e}")
            return None

    def parse_directional_line(self, line: str, data_type: str) -> Optional[Dict]:
        """Parse directional data line (alpha1, alpha2, r1, r2)"""
        if line.startswith('#') or not line.strip():
            return None
        
        parts = line.split()
        if len(parts) < 7:
            return None

        try:
            # Parse timestamp
            year = int(parts[0])
            if year < 100:
                year += 2000
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            
            if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
                return None
                
            timestamp = datetime(year, month, day, hour, minute)
            
            # Parse directional values with frequencies
            frequencies = []
            values = []
            
            i = 5
            while i < len(parts) - 1:
                try:
                    val_str = parts[i]
                    if i + 1 < len(parts) and '(' in parts[i + 1] and ')' in parts[i + 1]:
                        freq_str = parts[i + 1].strip('()')
                        
                        # Check for missing data indicators (999.0 or 999.00 for missing)
                        if val_str not in ['999.0', '999.00', 'MM']:
                            val = float(val_str)
                            freq = float(freq_str)
                            
                            if self.min_reasonable_freq <= freq <= self.max_reasonable_freq:
                                # Handle directional data validation
                                if data_type in ['alpha1', 'alpha2']:
                                    # Direction: 0-360 degrees
                                    if 0 <= val <= 360:
                                        values.append(val)
                                        frequencies.append(freq)
                                elif data_type in ['r1', 'r2']:
                                    # Directional spread: 0-1
                                    if 0 <= val <= 1:
                                        values.append(val)
                                        frequencies.append(freq)
                        
                        i += 2
                    else:
                        i += 1
                        
                except (ValueError, IndexError):
                    i += 1
                    continue
            
            if len(frequencies) < 3:
                return None
            
            # Sort by frequency
            freq_array = np.array(frequencies)
            val_array = np.array(values)
            sort_idx = np.argsort(freq_array)
            
            return {
                'timestamp': timestamp,
                'frequencies': freq_array[sort_idx],
                'values': val_array[sort_idx],
                'data_type': data_type,
                'n_frequencies': len(frequencies)
            }
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing directional line: {e}")
            return None

    def parse_wave_line(self, line: str, headers: List[str]) -> Optional[Dict]:
        """Parse meteorological/wave data line - FIXED VERSION"""
        if line.startswith('#') or not line.strip():
            return None
        
        parts = line.split()
        if len(parts) < len(headers):
            return None
        
        try:
            # Create simple header-to-value mapping
            data = {}
            for i, header in enumerate(headers):
                if i < len(parts):
                    data[header.upper()] = parts[i]  # Store as uppercase for consistency
            
            # Parse timestamp - handle the cleaned headers
            year = int(data.get('YY', data.get('YYYY', '0')))
            if year < 100:
                year += 2000
            month = int(data.get('MM', '0'))
            day = int(data.get('DD', '0'))
            hour = int(data.get('HH', '0'))
            minute = int(data.get('MN', data.get('MIN', '0')))
            
            if not (1900 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
                return None
            
            timestamp = datetime(year, month, day, hour, minute)
            parsed_rec = {'timestamp': timestamp}
            
            # Parse parameters using exact header names from your data
            param_mappings = {
                'wvht': 'WVHT',     # Wave height
                'dpd': 'DPD',       # Dominant period
                'mwd': 'MWD',       # Mean wave direction
                'wspd': 'WSPD',     # Wind speed  
                'wdir': 'WDIR',     # Wind direction
                'pres': 'PRES',     # Pressure
                'atmp': 'ATMP',     # Air temperature
                'wtmp': 'WTMP'      # Water temperature
            }
            
            for param_name, header_name in param_mappings.items():
                value = np.nan
                if header_name in data and data[header_name] not in ['MM', '999.0', '999', '-999.0', '99.0']:
                    try:
                        value = float(data[header_name])
                        
                        # Validation
                        if param_name == 'wvht' and (value < 0 or value > 30):
                            value = np.nan
                        elif param_name == 'dpd' and (value < 0 or value > 30):
                            value = np.nan
                        elif param_name in ['mwd', 'wdir'] and (value < 0 or value > 360):
                            value = np.nan
                        elif param_name == 'wspd' and (value < 0 or value > 100):
                            value = np.nan
                            
                    except ValueError:
                        value = np.nan
                
                parsed_rec[param_name] = value
            
            return parsed_rec
            
        except (ValueError, KeyError, IndexError) as e:
            logger.debug(f"Wave parsing error: {e}")
            return None

    def interpolate_to_standard_grid(self, frequencies: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Interpolate data to standard frequency grid"""
        if len(frequencies) != len(values) or len(frequencies) < 3:
            return np.full(len(self.target_freq_bins), np.nan)
        
        valid_mask = (~np.isnan(values) & (~np.isnan(frequencies)) & (values >= 0))
        if np.sum(valid_mask) < 3:
            return np.full(len(self.target_freq_bins), np.nan)
        
        freq_valid = frequencies[valid_mask]
        vals_valid = values[valid_mask]
        
        # Remove duplicates
        unique_freqs, unique_indices = np.unique(freq_valid, return_index=True)
        if len(unique_freqs) < 3:
            return np.full(len(self.target_freq_bins), np.nan)
        
        vals_unique = vals_valid[unique_indices]
        
        try:
            f_interp = interp1d(unique_freqs, vals_unique, 
                              kind='linear', bounds_error=False, fill_value=np.nan)
            interpolated = f_interp(self.target_freq_bins)
            return interpolated
        except ValueError:
            return np.full(len(self.target_freq_bins), np.nan)

    def compute_bulk_parameters(self, frequencies: np.ndarray, spectral_density: np.ndarray) -> Dict[str, float]:
        """Compute bulk wave parameters from spectral density"""
        params = {
            'significant_height': np.nan, 'peak_period': np.nan, 'mean_period': np.nan,
            'spectral_width': np.nan, 'peak_frequency': np.nan, 'total_energy': np.nan
        }
        
        valid_mask = (~np.isnan(spectral_density) & (spectral_density >= 0) & 
                     (~np.isnan(frequencies)) & (frequencies > 0))
        
        if np.sum(valid_mask) < 3:
            return params
        
        freq = frequencies[valid_mask]
        spec = spectral_density[valid_mask]
        
        try:
            # Spectral moments - use scipy.integrate.trapezoid instead of deprecated trapz
            from scipy.integrate import trapezoid
            m0 = trapezoid(spec, freq)
            if m0 <= 0:
                return params
            
            m1 = trapezoid(spec * freq, freq)
            m2 = trapezoid(spec * freq**2, freq)
            
            # Bulk parameters
            params['total_energy'] = m0
            params['significant_height'] = 4 * np.sqrt(m0)
            
            # Peak frequency and period
            peak_idx = np.argmax(spec)
            params['peak_frequency'] = freq[peak_idx]
            params['peak_period'] = 1.0 / params['peak_frequency']
            
            # Mean period
            if m1 > 0:
                params['mean_period'] = m0 / m1
            
            # Spectral width
            if m1 > 0 and m2 > 0:
                params['spectral_width'] = np.sqrt(abs(m0 * m2 - m1**2) / m1**2)
            
            return params
            
        except Exception as e:
            logger.debug(f"Error computing bulk parameters: {e}")
            return params

    def separate_wave_components(self, frequencies: np.ndarray, spectral_density: np.ndarray) -> Dict[str, float]:
        """Separate swell and wind wave components"""
        result = {
            'swell_energy': np.nan, 'windsea_energy': np.nan,
            'swell_fraction': np.nan, 'windsea_fraction': np.nan
        }
        
        valid_mask = (~np.isnan(spectral_density) & (spectral_density >= 0) & 
                     (~np.isnan(frequencies)) & (frequencies > 0))
        
        if np.sum(valid_mask) < 3:
            return result
        
        freq = frequencies[valid_mask]
        spec = spectral_density[valid_mask]
        
        try:
            from scipy.integrate import trapezoid
            swell_mask = freq < self.swell_threshold
            windsea_mask = freq > self.windsea_threshold
            
            swell_energy = trapezoid(spec[swell_mask], freq[swell_mask]) if np.any(swell_mask) else 0
            windsea_energy = trapezoid(spec[windsea_mask], freq[windsea_mask]) if np.any(windsea_mask) else 0
            total_energy = swell_energy + windsea_energy
            
            if total_energy > 0:
                result.update({
                    'swell_energy': swell_energy,
                    'windsea_energy': windsea_energy,
                    'swell_fraction': swell_energy / total_energy,
                    'windsea_fraction': windsea_energy / total_energy
                })
            
            return result
            
        except Exception as e:
            logger.debug(f"Error separating wave components: {e}")
            return result

    def compute_directional_features(self, alpha1_data, alpha2_data, 
                                   r1_data, r2_data, 
                                   spectral_density: np.ndarray) -> Dict[str, float]:
        """Compute enhanced directional features from complete directional suite - FIXED VERSION"""
        features = {
            'primary_direction': np.nan, 'primary_spread': np.nan, 
            'secondary_direction': np.nan, 'bimodal_strength': np.nan, 
            'directional_separation': np.nan
        }
        
        try:
            # Primary directional parameters
            # FIX: Check if data exists using 'is not None' instead of boolean evaluation
            if alpha1_data is not None and r1_data is not None:
                alpha1_interp = self.interpolate_to_standard_grid(alpha1_data['frequencies'], alpha1_data['values'])
                r1_interp = self.interpolate_to_standard_grid(r1_data['frequencies'], r1_data['values'])
                
                valid_mask = ~np.isnan(alpha1_interp) & ~np.isnan(r1_interp) & ~np.isnan(spectral_density)
                
                if np.any(valid_mask):
                    # Energy-weighted primary direction
                    weights = spectral_density[valid_mask]
                    if np.sum(weights) > 0:
                        # Handle circular mean for directions
                        alpha1_rad = np.radians(alpha1_interp[valid_mask])
                        x_mean = np.average(np.cos(alpha1_rad), weights=weights)
                        y_mean = np.average(np.sin(alpha1_rad), weights=weights)
                        features['primary_direction'] = np.degrees(np.arctan2(y_mean, x_mean)) % 360
                        
                        # Average primary spread (clamp r1 values to [0,1] range)
                        r1_valid = np.clip(r1_interp[valid_mask], 0.0, 1.0)
                        avg_r1 = np.average(r1_valid, weights=weights)
                        features['primary_spread'] = np.degrees(np.arccos(avg_r1))
            
            # Secondary directional parameters
            # FIX: Check if data exists using 'is not None' instead of boolean evaluation
            if alpha2_data is not None and r2_data is not None:
                alpha2_interp = self.interpolate_to_standard_grid(alpha2_data['frequencies'], alpha2_data['values'])
                r2_interp = self.interpolate_to_standard_grid(r2_data['frequencies'], r2_data['values'])
                
                valid_mask = ~np.isnan(alpha2_interp) & ~np.isnan(r2_interp) & ~np.isnan(spectral_density)
                
                if np.any(valid_mask):
                    weights = spectral_density[valid_mask]
                    r2_valid = np.clip(r2_interp[valid_mask], 0.0, 1.0)
                    
                    # Check if secondary system is significant
                    avg_r2 = np.average(r2_valid, weights=weights) if np.sum(weights) > 0 else 0
                    
                    if avg_r2 > 0.3:  # Significant secondary system
                        # Energy-weighted secondary direction
                        alpha2_rad = np.radians(alpha2_interp[valid_mask])
                        x_mean = np.average(np.cos(alpha2_rad), weights=weights)
                        y_mean = np.average(np.sin(alpha2_rad), weights=weights)
                        features['secondary_direction'] = np.degrees(np.arctan2(y_mean, x_mean)) % 360
                        
                        # Bimodal strength
                        features['bimodal_strength'] = avg_r2
                        
                        # Directional separation
                        if not np.isnan(features['primary_direction']):
                            angular_diff = abs(features['secondary_direction'] - features['primary_direction'])
                            features['directional_separation'] = min(angular_diff, 360 - angular_diff)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error computing directional features: {e}")
            return features

    def load_station_files(self, station_id: str) -> Tuple[List[Dict], Dict[str, List[Dict]], List[Dict]]:
        """Load all data types for a station - FIXED VERSION"""
        current_dir = self.data_dir / "current"
        
        # Load raw spectral data
        spectral_records = []
        spectral_file = current_dir / f"{station_id}_raw_spectral.gz"
        if spectral_file.exists():
            try:
                with gzip.open(spectral_file, 'rt') as f:
                    for line in f:
                        record = self.parse_raw_spectral_line(line)
                        if record:
                            spectral_records.append(record)
            except Exception as e:
                logger.error(f"Error reading {spectral_file}: {e}")
        
        # Load directional data
        directional_records = {}
        directional_types = ['alpha1', 'alpha2', 'r1', 'r2']
        
        for dtype in directional_types:
            directional_records[dtype] = []
            dir_file = current_dir / f"{station_id}_directional_{dtype}.gz"
            if dir_file.exists():
                try:
                    with gzip.open(dir_file, 'rt') as f:
                        for line in f:
                            record = self.parse_directional_line(line, dtype)
                            if record:
                                directional_records[dtype].append(record)
                except Exception as e:
                    logger.error(f"Error reading {dir_file}: {e}")
        
        # Load wave data - FIXED VERSION
        wave_records = []
        wave_file = current_dir / f"{station_id}_wave.gz"
        if wave_file.exists():
            try:
                with gzip.open(wave_file, 'rt') as f:
                    lines = list(f)
                    
                # Find headers with better parsing
                headers = []
                header_line_num = None
                for i, line in enumerate(lines[:10]):
                    if any(indicator in line.upper() for indicator in ['WVHT', 'DPD', 'MWD', 'WSPD', 'WDIR']):
                        # CRITICAL FIX: Properly clean header line
                        header_line = line.strip()
                        if header_line.startswith('#'):
                            header_line = header_line[1:]  # Remove leading #
                        headers = header_line.split()
                        header_line_num = i
                        logger.debug(f"Found headers at line {i}: {headers}")
                        break
                
                if not headers:
                    logger.warning(f"No headers found in {wave_file}")
                else:
                    # Parse data lines
                    parsed_count = 0
                    for line_num, line in enumerate(lines):
                        if line_num <= header_line_num or line.startswith('#'):  # Skip header lines
                            continue
                            
                        record = self.parse_wave_line(line, headers)
                        if record:
                            wave_records.append(record)
                            parsed_count += 1
                            
                            # Debug first few successful records
                            if parsed_count <= 3:
                                logger.debug(f"Wave record {parsed_count}: wspd={record.get('wspd')}, wdir={record.get('wdir')}")
                    
                    logger.debug(f"Wave file parsing: {parsed_count} records from {len(lines)} lines")
                
            except Exception as e:
                logger.error(f"Error reading {wave_file}: {e}")
        
        logger.info(f"Station {station_id}: {len(spectral_records)} spectral, "
                   f"{sum(len(records) for records in directional_records.values())} directional, "
                   f"{len(wave_records)} wave records")
        
        return spectral_records, directional_records, wave_records

    def create_enhanced_features(self, station_id: str, verbose: bool = False) -> Tuple[np.ndarray, List[datetime], Dict]:
        """Create enhanced 20-dimensional feature vectors - FIXED VERSION"""
        spectral_recs, dir_recs, wave_recs = self.load_station_files(station_id)
        
        if not spectral_recs:
            return np.array([]), [], {
                'error': 'No raw spectral data found',
                'total_records': 0,
                'valid_features': 0
            }
        
        # Convert to DataFrames for temporal alignment
        spec_df = pd.DataFrame(spectral_recs).set_index('timestamp')
        wave_df = pd.DataFrame(wave_recs).set_index('timestamp') if wave_recs else None
        
        # Convert directional records to DataFrames
        dir_dfs = {}
        for dtype, records in dir_recs.items():
            if records:
                try:
                    dir_dfs[dtype] = pd.DataFrame(records).set_index('timestamp')
                    logger.debug(f"Created {dtype} DataFrame with {len(dir_dfs[dtype])} records")
                except Exception as e:
                    logger.debug(f"Error creating {dtype} DataFrame: {e}")
        
        logger.debug(f"Available directional DataFrames: {list(dir_dfs.keys())}")
        
        features = []
        timestamps = []
        processing_errors = 0
        
        for timestamp, row in spec_df.iterrows():
            try:
                # Initialize 20-dimensional feature vector
                feature_vector = np.full(20, np.nan)
                
                # Interpolate spectral data
                interpolated_spec = self.interpolate_to_standard_grid(
                    row['frequencies'], row['spectral_density']
                )
                
                if np.all(np.isnan(interpolated_spec)):
                    continue
                
                # Compute spectral physics features (0-9)
                bulk_params = self.compute_bulk_parameters(self.target_freq_bins, interpolated_spec)
                wave_components = self.separate_wave_components(self.target_freq_bins, interpolated_spec)
                
                feature_vector[0] = bulk_params['significant_height']
                feature_vector[1] = bulk_params['peak_period']
                feature_vector[2] = bulk_params['mean_period']
                feature_vector[3] = bulk_params['peak_frequency']
                feature_vector[4] = bulk_params['total_energy']
                feature_vector[5] = bulk_params['spectral_width']
                feature_vector[6] = wave_components['swell_energy']
                feature_vector[7] = wave_components['windsea_energy']
                feature_vector[8] = wave_components['swell_fraction']
                feature_vector[9] = wave_components['windsea_fraction']
                
                # Compute directional physics features (10-14) - FIXED VERSION
                dir_data = {}
                for dtype in ['alpha1', 'alpha2', 'r1', 'r2']:
                    if dtype in dir_dfs and len(dir_dfs[dtype]) > 0:
                        # Find closest directional record within 30 minutes
                        time_diffs = abs(dir_dfs[dtype].index - timestamp)
                        min_diff_idx = time_diffs.argmin()
                        min_diff = time_diffs[min_diff_idx]
                        if min_diff <= pd.Timedelta(minutes=30):
                            closest_timestamp = dir_dfs[dtype].index[min_diff_idx]
                            # FIX: Convert pandas Series to dict to avoid boolean evaluation issues
                            series_data = dir_dfs[dtype].loc[closest_timestamp]
                            dir_data[dtype] = {
                                'frequencies': series_data['frequencies'],
                                'values': series_data['values'],
                                'data_type': series_data['data_type'],
                                'n_frequencies': series_data['n_frequencies']
                            }
                
                if len(dir_data) >= 2:  # Need at least alpha1 and r1
                    directional_features = self.compute_directional_features(
                        dir_data.get('alpha1'), dir_data.get('alpha2'),
                        dir_data.get('r1'), dir_data.get('r2'),
                        interpolated_spec
                    )
                    
                    feature_vector[10] = directional_features['primary_direction']
                    feature_vector[11] = directional_features['primary_spread']
                    feature_vector[12] = directional_features['secondary_direction']
                    feature_vector[13] = directional_features['bimodal_strength']
                    feature_vector[14] = directional_features['directional_separation']
                
                # Compute meteorological physics features (15-17) - FIXED VERSION
                if wave_df is not None and len(wave_df) > 0:
                    time_diffs = abs(wave_df.index - timestamp)
                    min_diff_idx = time_diffs.argmin()
                    min_diff = time_diffs[min_diff_idx]
                    if min_diff <= pd.Timedelta(minutes=30):
                        closest_timestamp = wave_df.index[min_diff_idx]
                        closest_wave = wave_df.loc[closest_timestamp]
                        
                        feature_vector[15] = closest_wave.get('wspd', np.nan)  # Wind speed
                        feature_vector[16] = closest_wave.get('wdir', np.nan)  # Wind direction
                        
                        # Wind-wave alignment
                        if (not np.isnan(feature_vector[16]) and not np.isnan(feature_vector[10])):
                            wind_dir = feature_vector[16]
                            wave_dir = feature_vector[10]
                            alignment = abs(wind_dir - wave_dir)
                            feature_vector[17] = min(alignment, 360 - alignment)  # Minimum angular difference
                        
                        # Validation features (18-19)
                        reported_hs = closest_wave.get('wvht', np.nan)
                        reported_tp = closest_wave.get('dpd', np.nan)
                        
                        if not np.isnan(reported_hs) and not np.isnan(feature_vector[0]):
                            feature_vector[18] = abs(feature_vector[0] - reported_hs) / max(feature_vector[0], reported_hs) if max(feature_vector[0], reported_hs) > 0 else 0
                        
                        if not np.isnan(reported_tp) and not np.isnan(feature_vector[1]):
                            feature_vector[19] = abs(feature_vector[1] - reported_tp) / max(feature_vector[1], reported_tp) if max(feature_vector[1], reported_tp) > 0 else 0
                
                # Only keep records with valid significant wave height
                if not np.isnan(feature_vector[0]):
                    features.append(feature_vector)
                    timestamps.append(timestamp)
                
            except Exception as e:
                processing_errors += 1
                logger.debug(f"Error processing record at {timestamp}: {e}")
                # FIX: Use the 'verbose' argument instead of 'args.verbose'
                if verbose:
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        # Create results
        features_array = np.array(features) if features else np.array([]).reshape(0, 20)
        
        stats = {
            'total_records': len(spectral_recs),
            'valid_features': len(features),
            'processing_errors': processing_errors,
            'completeness': len(features) / len(spectral_recs) if spectral_recs else 0
        }
        
        if features:
            stats['time_span_days'] = (max(timestamps) - min(timestamps)).days if len(timestamps) > 1 else 0
            # Calculate nan_percentage only if features_array is not empty
            if features_array.size > 0:
                stats['nan_percentage_by_feature'] = np.isnan(features_array).mean(axis=0).tolist() * 100
            else:
                stats['nan_percentage_by_feature'] = [100.0] * 20

            # Physics validation
            valid_heights = features_array[:, 0][~np.isnan(features_array[:, 0])]
            if len(valid_heights) > 0:
                stats['wave_height_range'] = [float(np.min(valid_heights)), float(np.max(valid_heights))]
                stats['mean_wave_height'] = float(np.mean(valid_heights))
                
            # Directional data availability
            valid_directions = features_array[:, 10][~np.isnan(features_array[:, 10])]
            stats['directional_coverage'] = len(valid_directions) / len(features) if features else 0
            
            # Meteorological data availability  
            valid_wind = features_array[:, 15][~np.isnan(features_array[:, 15])]
            stats['meteorological_coverage'] = len(valid_wind) / len(features) if features else 0
        
        return features_array, timestamps, stats


def main():
    """Enhanced main function for multi-stream processing"""
    parser = argparse.ArgumentParser(description="Enhanced Multi-Stream NDBC Processor")
    parser.add_argument("--data-dir", default="data/buoy_data", 
                       help="Data directory containing station files")
    parser.add_argument("--test-single", help="Test with single station ID")
    parser.add_argument("--stations", help="Comma-separated station IDs")
    parser.add_argument("--output", default="enhanced_features.json", 
                       help="Output file for enhanced features")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    processor = EnhancedMultiStreamProcessor(args.data_dir)
    
    # Single station testing mode
    if args.test_single:
        station_id = args.test_single
        logger.info(f"=== Enhanced Processing for Station: {station_id} ===")
        
        features, timestamps, stats = processor.create_enhanced_features(station_id, verbose=args.verbose)        
        print(f"\n🌊 Enhanced Multi-Stream Processing Results for Station {station_id}")
        print("=" * 80)
        
        if 'error' in stats or not stats.get('valid_features', 0):
            print("❌ Processing failed - no valid enhanced features extracted")
            print(f"Error: {stats.get('error', 'Unknown processing error')}")
            print(f"Total records found: {stats.get('total_records', 0)}")
            return
        
        print(f"✅ Enhanced features extracted: {stats['valid_features']:,}")
        print(f"📊 Processing completeness: {stats['completeness']:.1%}")
        print(f"📅 Temporal coverage: {stats.get('time_span_days', 0)} days")
        print(f"⚠️  Processing errors: {stats.get('processing_errors', 0)}")
        print(f"🧭 Directional coverage: {stats.get('directional_coverage', 0):.1%}")
        print(f"🌬️  Meteorological coverage: {stats.get('meteorological_coverage', 0):.1%}")
        
        if 'wave_height_range' in stats:
            print(f"🌊 Wave height range: {stats['wave_height_range'][0]:.2f} - {stats['wave_height_range'][1]:.2f} m")
            print(f"📈 Mean wave height: {stats['mean_wave_height']:.2f} m")
        
        if features.shape[0] > 0:
            print(f"\n🔬 Enhanced Feature Analysis (First Record):")
            print("-" * 70)
            for i, (name, value) in enumerate(zip(processor.feature_names, features[0])):
                if not np.isnan(value):
                    if 'energy' in name.lower():
                        print(f"  {i:2d}. {name:22}: {value:12.4e}")
                    elif 'direction' in name.lower():
                        print(f"  {i:2d}. {name:22}: {value:12.1f}°")
                    elif 'error' in name.lower() or 'fraction' in name.lower():
                        print(f"  {i:2d}. {name:22}: {value:12.3f}")
                    else:
                        print(f"  {i:2d}. {name:22}: {value:12.3f}")
                else:
                    print(f"  {i:2d}. {name:22}: {'NaN':>12}")
            
            print(f"\n📊 Feature Completeness Analysis:")
            print("-" * 50)
            for i, (name, pct) in enumerate(zip(processor.feature_names, stats['nan_percentage_by_feature'])):
                if pct < 5:
                    indicator = "🟢 Excellent"
                elif pct < 20:
                    indicator = "🟡 Good"
                elif pct < 50:
                    indicator = "🟠 Fair"
                else:
                    indicator = "🔴 Poor"
                print(f"  {i:2d}. {name:22}: {pct:5.1f}% missing {indicator}")
            
            # Enhanced validation
            print(f"\n🔬 Multi-Stream Data Validation:")
            print("-" * 40)
            
            # Spectral physics validation
            heights = features[:, 0][~np.isnan(features[:, 0])]
            periods = features[:, 1][~np.isnan(features[:, 1])]
            
            if len(heights) > 0:
                print(f"  Spectral Physics:")
                print(f"    Wave Heights: {np.min(heights):.2f} - {np.max(heights):.2f} m")
                print(f"    Peak Periods: {np.min(periods):.1f} - {np.max(periods):.1f} s" if len(periods) > 0 else "    Peak Periods: No data")
            
            # Directional validation
            directions = features[:, 10][~np.isnan(features[:, 10])]
            spreads = features[:, 11][~np.isnan(features[:, 11])]
            
            if len(directions) > 0:
                print(f"  Directional Physics:")
                print(f"    Wave Directions: {np.min(directions):.0f}° - {np.max(directions):.0f}°")
                print(f"    Directional Spreads: {np.min(spreads):.1f}° - {np.max(spreads):.1f}°" if len(spreads) > 0 else "    Spreads: No data")
            
            # Meteorological validation
            wind_speeds = features[:, 15][~np.isnan(features[:, 15])]
            wind_dirs = features[:, 16][~np.isnan(features[:, 16])]
            
            if len(wind_speeds) > 0:
                print(f"  Meteorological Physics:")
                print(f"    Wind Speeds: {np.min(wind_speeds):.1f} - {np.max(wind_speeds):.1f} m/s")
                print(f"    Wind Directions: {np.min(wind_dirs):.0f}° - {np.max(wind_dirs):.0f}°" if len(wind_dirs) > 0 else "    Wind Directions: No data")
            
            # Cross-validation
            hs_errors = features[:, 18][~np.isnan(features[:, 18])]
            tp_errors = features[:, 19][~np.isnan(features[:, 19])]
            
            if len(hs_errors) > 0:
                print(f"  Cross-Validation:")
                print(f"    Hs Error (mean): {np.mean(hs_errors):.3f} ({np.mean(hs_errors)*100:.1f}%)")
                print(f"    Tp Error (mean): {np.mean(tp_errors):.3f} ({np.mean(tp_errors)*100:.1f}%)" if len(tp_errors) > 0 else "    Tp Error: No data")
        
        return
    
    # Multi-station processing mode
    if args.stations:
        station_ids = [s.strip() for s in args.stations.split(",")]
    else:
        # Auto-detect stations from raw spectral files
        current_dir = Path(args.data_dir) / "current"
        if not current_dir.exists():
            logger.error(f"Data directory not found: {current_dir}")
            return
        
        spectral_files = list(current_dir.glob("*_raw_spectral.gz"))
        if not spectral_files:
            logger.error("No raw spectral files found")
            return
            
        station_ids = sorted([f.name.split('_')[0] for f in spectral_files])
        logger.info(f"Auto-detected {len(station_ids)} stations: {station_ids}")
    
    # Process all stations
    all_results = {}
    success_count = 0
    
    for i, station_id in enumerate(station_ids, 1):
        logger.info(f"Processing station {i}/{len(station_ids)}: {station_id}")
        
        try:
            features, timestamps, stats = processor.create_enhanced_features(station_id, verbose=args.verbose)
            
            if len(features) > 0:
                all_results[station_id] = {
                    'enhanced_features': features.tolist(),
                    'timestamps': [ts.isoformat() for ts in timestamps],
                    'statistics': stats,
                    'feature_names': processor.feature_names,
                    'frequency_grid': processor.target_freq_bins.tolist(),
                    'processing_metadata': {
                        'swell_threshold': processor.swell_threshold,
                        'windsea_threshold': processor.windsea_threshold,
                        'feature_count': 20,
                        'data_streams': ['raw_spectral', 'directional_alpha1', 'directional_alpha2', 
                                       'directional_r1', 'directional_r2', 'wave', 'spectral']
                    }
                }
                success_count += 1
                logger.info(f"✅ {station_id}: {len(features)} enhanced features "
                           f"({stats['completeness']:.1%} complete, "
                           f"{stats.get('directional_coverage', 0):.1%} directional, "
                           f"{stats.get('meteorological_coverage', 0):.1%} meteorological)")
            else:
                logger.warning(f"❌ {station_id}: No valid enhanced features extracted")
                
        except Exception as e:
            logger.error(f"💥 Station {station_id} failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Save results
    if all_results:
        try:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"💾 Enhanced features saved to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return
    
    # Enhanced summary
    print(f"\n🌊 Enhanced Multi-Stream Processing Summary")
    print("=" * 60)
    print(f"🎯 Stations processed: {success_count}/{len(station_ids)}")
    
    if all_results:
        total_features = sum(len(data['enhanced_features']) for data in all_results.values())
        total_records = sum(data['statistics']['total_records'] for data in all_results.values())
        avg_completeness = np.mean([data['statistics']['completeness'] for data in all_results.values()])
        avg_directional = np.mean([data['statistics'].get('directional_coverage', 0) for data in all_results.values()])
        avg_meteorological = np.mean([data['statistics'].get('meteorological_coverage', 0) for data in all_results.values()])
        
        print(f"📊 Total enhanced features: {total_features:,}")
        print(f"📈 Total spectral records: {total_records:,}")
        print(f"✅ Average completeness: {avg_completeness:.1%}")
        print(f"🧭 Average directional coverage: {avg_directional:.1%}")
        print(f"🌬️  Average meteorological coverage: {avg_meteorological:.1%}")
        print(f"💾 Output: {args.output}")
        
        print(f"\n🔬 Enhanced Feature Summary:")
        print("- 20 physics-informed features per observation")
        print("- Multi-stream data integration (spectral + directional + meteorological)")
        print("- Cross-validation with NDBC processed parameters")
        print("- Enhanced directional physics (bimodality, separation, spread)")
        print("- Wind-wave generation mechanisms")
        print("- Ready for advanced transformer spatiotemporal modeling")
        
    print(f"\n🌊 Enhanced multi-stream processing complete!")


if __name__ == "__main__":
    main()