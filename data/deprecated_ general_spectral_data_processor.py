#!/usr/bin/env python3
"""
Fixed Spectral Data Processor - Debug Version

Addresses specific issues:
- Invalid month values in wave data
- Duplicate timestamp handling
- Enhanced error reporting
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

class FixedSpectralDataProcessor:
    """Fixed processor with enhanced error handling"""

    def __init__(self, data_dir: str = "data/buoy_data"):
        self.data_dir = Path(data_dir)
        
        # Extended frequency grid to cover full NDBC range (0.033-0.485 Hz)
        self.target_freq_bins = np.array([
            0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080,
            0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180,
            0.190, 0.200, 0.220, 0.240, 0.260, 0.280, 0.300, 0.320, 0.350, 0.400, 
            0.450, 0.485
        ])
        
        # Wave classification thresholds
        self.swell_threshold = 0.125  # Hz
        self.windsea_threshold = 0.25  # Hz
        
        # Enhanced validation limits
        self.max_reasonable_energy = 100.0  # m²/Hz
        self.min_reasonable_freq = 0.020   # Hz (covers 0.033 minimum)
        self.max_reasonable_freq = 0.500   # Hz (covers 0.485 maximum)
        
        # Feature names for consistent indexing
        self.feature_names = [
            'sig_height', 'peak_period', 'mean_period', 'peak_freq', 'total_energy',
            'spectral_width', 'swell_energy', 'windsea_energy', 'swell_fraction', 'windsea_fraction',
            'log_energy', 'steepness_proxy', 'sep_freq', 'n_freq_bins', 'data_quality',
            'wave_height_reported', 'dom_period_reported', 'wave_dir_reported', 'mean_dir_calc', 'dir_spread_calc'
        ]

    def _parse_line(self, line: str, is_directional: bool = False) -> Optional[Dict]:
        """Enhanced parser with better error handling"""
        if line.startswith('#') or not line.strip():
            return None
        
        parts = line.split()
        if len(parts) < 7: 
            return None

        try:
            # Handle both 2-digit and 4-digit years
            year = int(parts[0])
            if year < 100:
                year += 2000
            elif year < 1900:
                year += 2000
                
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            
            # Validate date components before creating datetime
            if not (1 <= month <= 12):
                logger.debug(f"Invalid month: {month} in line: {line[:50]}...")
                return None
            if not (1 <= day <= 31):
                logger.debug(f"Invalid day: {day} in line: {line[:50]}...")
                return None
            if not (0 <= hour <= 23):
                logger.debug(f"Invalid hour: {hour} in line: {line[:50]}...")
                return None
            if not (0 <= minute <= 59):
                logger.debug(f"Invalid minute: {minute} in line: {line[:50]}...")
                return None
                
            timestamp = datetime(year, month, day, hour, minute)
            
            # Validate we have paired data (value + frequency)
            data_start_index = 6 if not is_directional else 5
            remaining_fields = len(parts) - data_start_index
            expected_pairs = remaining_fields // 2
            
            if expected_pairs < 5:  # Need minimum data points
                logger.debug(f"Insufficient data pairs: {expected_pairs}")
                return None
            
            values, freqs = [], []
            i = data_start_index
            
            while i < len(parts) - 1:
                val_str, freq_str = parts[i], parts[i+1]
                
                # Check for valid frequency format: (0.xxx)
                if '(' in freq_str and ')' in freq_str and val_str not in ['MM', '999.00', '-999.00', '99.00']:
                    try:
                        val, freq = float(val_str), float(freq_str.strip('()'))
                        
                        # Enhanced frequency range validation
                        if self.min_reasonable_freq <= freq <= self.max_reasonable_freq:
                            # For directional data, handle angle wrapping
                            if is_directional:
                                if val < 0:
                                    val += 360
                                # Validate direction range
                                if 0 <= val <= 360:
                                    values.append(val)
                                    freqs.append(freq)
                            else:
                                # For spectral data, validate energy values
                                if 0 <= val <= self.max_reasonable_energy:
                                    values.append(val)
                                    freqs.append(freq)
                    except (ValueError, IndexError):
                        pass
                i += 2

            if len(freqs) < 5: 
                logger.debug(f"Insufficient valid frequencies: {len(freqs)}")
                return None
            
            # Sort by frequency for consistency
            sort_idx = np.argsort(freqs)
            
            result = {
                'timestamp': timestamp, 
                'frequencies': np.array(freqs)[sort_idx],
                'n_frequencies': len(freqs)
            }
            
            if is_directional:
                result['mean_directions'] = np.array(values)[sort_idx]
            else:
                result['spectral_density'] = np.array(values)[sort_idx]
                result['separation_frequency'] = float(parts[5])
                result['data_quality'] = 'good' if len(freqs) > 25 else 'partial'
                
            return result
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Parse error: {e} in line: {line[:50]}...")
            return None

    def parse_raw_spectral_line(self, line: str) -> Optional[Dict]:
        """Parse raw spectral data line"""
        return self._parse_line(line, is_directional=False)

    def parse_directional_line(self, line: str) -> Optional[Dict]:
        """Parse directional data line"""
        return self._parse_line(line, is_directional=True)

    def parse_wave_line(self, line: str, headers: List[str]) -> Optional[Dict]:
        """Enhanced wave data parser with strict validation"""
        if line.startswith('#') or not line.strip(): 
            return None
        
        parts = line.split()
        if len(parts) < len(headers): 
            return None
        
        try:
            # Create mapping from headers to values (case-insensitive)
            header_map = {h.lower(): val for h, val in zip(headers, parts)}
            
            # Find timestamp fields with multiple possible header names
            year_key = next((k for k in header_map if k in ['yy', 'yyyy', '#yy', '#yyyy']), None)
            mm_key = next((k for k in header_map if k in ['mm', '#mm']), None)
            dd_key = next((k for k in header_map if k in ['dd', '#dd']), None)
            hh_key = next((k for k in header_map if k in ['hh', '#hh']), None)
            min_key = next((k for k in header_map if k in ['mn', 'min', '#mn', '#min']), None)
            
            if not all([year_key, mm_key, dd_key, hh_key]):
                logger.debug("Missing required timestamp fields")
                return None
            
            # Parse and validate timestamp components
            year = int(header_map[year_key])
            if year < 100:
                year += 2000
            elif year < 1900:
                year += 2000
                
            month = int(header_map[mm_key])
            day = int(header_map[dd_key])
            hour = int(header_map[hh_key])
            minute = int(header_map[min_key]) if min_key and header_map[min_key] not in ['MM', '99'] else 0
            
            # Strict validation of date components
            if not (1900 <= year <= 2030):
                logger.debug(f"Invalid year: {year}")
                return None
            if not (1 <= month <= 12):
                logger.debug(f"Invalid month: {month}")
                return None
            if not (1 <= day <= 31):
                logger.debug(f"Invalid day: {day}")
                return None
            if not (0 <= hour <= 23):
                logger.debug(f"Invalid hour: {hour}")
                return None
            if not (0 <= minute <= 59):
                logger.debug(f"Invalid minute: {minute}")
                return None
            
            timestamp = datetime(year, month, day, hour, minute)
            
            # Parse wave parameters with enhanced validation
            parsed_rec = {'timestamp': timestamp}
            wave_params = {
                'wvht': ['wvht', 'hs'],  # Significant wave height
                'dpd': ['dpd', 'tp'],    # Dominant wave period
                'mwd': ['mwd', 'dp'],    # Mean wave direction
                'apd': ['apd'],          # Average wave period
                'gst': ['gst'],          # Gust speed
                'wdir': ['wdir'],        # Wind direction
                'wspd': ['wspd']         # Wind speed
            }
            
            for param, possible_keys in wave_params.items():
                value = np.nan
                for key in possible_keys:
                    if key in header_map and header_map[key] not in ['MM', '999.0', '999', '-999.0']:
                        try:
                            value = float(header_map[key])
                            # Enhanced validation
                            if param == 'wvht' and (value < 0 or value > 30):
                                value = np.nan
                            elif param in ['dpd', 'apd'] and (value < 0 or value > 30):
                                value = np.nan
                            elif param in ['mwd', 'wdir'] and (value < 0 or value > 360):
                                value = np.nan
                            elif param in ['wspd', 'gst'] and (value < 0 or value > 100):
                                value = np.nan
                            break
                        except ValueError:
                            continue
                parsed_rec[param] = value
            
            return parsed_rec
            
        except (ValueError, KeyError, IndexError) as e:
            logger.debug(f"Wave parsing error: {e}")
            return None

    def interpolate_to_standard_grid(self, freqs: np.ndarray, vals: np.ndarray) -> np.ndarray:
        """Enhanced interpolation with better error handling"""
        if len(freqs) < 3 or len(freqs) != len(vals): 
            return np.full(self.target_freq_bins.shape, np.nan)
        
        # Remove NaN and invalid values
        valid_mask = ~np.isnan(vals) & ~np.isnan(freqs) & (vals >= 0)
        if np.sum(valid_mask) < 3:
            return np.full(self.target_freq_bins.shape, np.nan)
        
        freqs_valid = freqs[valid_mask]
        vals_valid = vals[valid_mask]
        
        # Remove duplicates and ensure monotonic frequency
        unique_freqs, unique_indices = np.unique(freqs_valid, return_index=True)
        if len(unique_freqs) < 3:
            return np.full(self.target_freq_bins.shape, np.nan)
        
        vals_unique = vals_valid[unique_indices]
        
        try:
            f_interp = interp1d(unique_freqs, vals_unique, 
                              kind='linear', bounds_error=False, fill_value=0.0)
            interpolated = f_interp(self.target_freq_bins)
            return interpolated
        except ValueError as e:
            logger.debug(f"Interpolation error: {e}")
            return np.full(self.target_freq_bins.shape, np.nan)

    def compute_bulk_parameters(self, freqs: np.ndarray, spec: np.ndarray) -> Dict[str, float]:
        """Enhanced bulk parameter computation"""
        params = {
            'significant_height': np.nan, 'peak_period': np.nan, 'mean_period': np.nan, 
            'spectral_width': np.nan, 'peak_frequency': np.nan, 'total_energy': np.nan
        }
        
        valid_mask = ~np.isnan(spec) & (spec >= 0) & ~np.isnan(freqs) & (freqs > 0)
        if np.sum(valid_mask) < 3:
            return params
        
        freqs_valid = freqs[valid_mask]
        spec_valid = spec[valid_mask]
        
        try:
            # Spectral moments
            m0 = np.trapz(spec_valid, freqs_valid)
            if m0 <= 0:
                return params
            
            params['total_energy'] = m0
            params['significant_height'] = 4 * np.sqrt(m0)
            
            # Peak frequency and period
            peak_idx = np.argmax(spec_valid)
            params['peak_frequency'] = freqs_valid[peak_idx]
            params['peak_period'] = 1.0 / params['peak_frequency']

            # Mean period and spectral width
            m1 = np.trapz(spec_valid * freqs_valid, freqs_valid)
            if m1 > 0:
                params['mean_period'] = m0 / m1
                
                m2 = np.trapz(spec_valid * freqs_valid**2, freqs_valid)
                if m2 > 0:
                    params['spectral_width'] = np.sqrt(abs(m0 * m2 - m1**2) / m1**2)
                
        except Exception as e:
            logger.debug(f"Error computing bulk parameters: {e}")
        
        return params
        
    def separate_wave_components(self, freqs: np.ndarray, spec: np.ndarray) -> Dict[str, float]:
        """Enhanced wave component separation"""
        result = {
            'swell_energy': np.nan, 'windsea_energy': np.nan, 
            'swell_fraction': np.nan, 'windsea_fraction': np.nan
        }
        
        valid_mask = ~np.isnan(spec) & (spec >= 0) & ~np.isnan(freqs) & (freqs > 0)
        if np.sum(valid_mask) < 3:
            return result
        
        freqs_valid = freqs[valid_mask]
        spec_valid = spec[valid_mask]
        
        try:
            swell_mask = freqs_valid < self.swell_threshold
            windsea_mask = freqs_valid > self.windsea_threshold
            
            swell_energy = np.trapz(spec_valid[swell_mask], freqs_valid[swell_mask]) if np.any(swell_mask) else 0
            windsea_energy = np.trapz(spec_valid[windsea_mask], freqs_valid[windsea_mask]) if np.any(windsea_mask) else 0
            total_energy = swell_energy + windsea_energy
            
            if total_energy > 0:
                result.update({
                    'swell_energy': swell_energy,
                    'windsea_energy': windsea_energy,
                    'swell_fraction': swell_energy / total_energy,
                    'windsea_fraction': windsea_energy / total_energy
                })
            
        except Exception as e:
            logger.debug(f"Error separating wave components: {e}")
        
        return result

    def load_station_files(self, station_id: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Enhanced file loading with better error handling"""
        current_dir = self.data_dir / "current"

        def _read_generic(file_path: Path, parser_func):
            records = []
            if not file_path.exists(): 
                logger.warning(f"File not found: {file_path}")
                return records
            
            try:
                with gzip.open(file_path, 'rt') as f:
                    line_count = 0
                    success_count = 0
                    for line in f:
                        line_count += 1
                        record = parser_func(line)
                        if record: 
                            records.append(record)
                            success_count += 1
                    logger.debug(f"Processed {line_count} lines, extracted {success_count} records from {file_path.name}")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
            return records

        def _read_wave_file(file_path: Path):
            """Enhanced wave file reader with better header detection"""
            records, headers = [], []
            if not file_path.exists(): 
                logger.warning(f"Wave file not found: {file_path}")
                return records
            
            try:
                with gzip.open(file_path, 'rt') as f:
                    lines = list(f)
                    
                # Find header line with multiple possible indicators
                header_indicators = ['WVHT', 'DPD', 'MWD', 'APD', 'YY', 'MM', 'DD', 'HH']
                for line in lines[:10]:  # Check first 10 lines
                    if any(indicator in line.upper() for indicator in header_indicators):
                        headers = line.strip('#\n ').split()
                        logger.debug(f"Found headers: {headers}")
                        break
                    
                if not headers: 
                    logger.warning(f"No valid headers found in {file_path.name}")
                    return []
                    
                # Parse all data lines with enhanced error handling
                success_count = 0
                for line_num, line in enumerate(lines, 1):
                    try:
                        record = self.parse_wave_line(line, headers)
                        if record: 
                            records.append(record)
                            success_count += 1
                    except Exception as e:
                        logger.debug(f"Error parsing wave line {line_num}: {e}")
                        
                logger.debug(f"Wave file: {success_count} valid records from {len(lines)} lines")
                        
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
            return records

        # Load all data types
        spectral_records = _read_generic(current_dir / f"{station_id}_raw_spectral.gz", self.parse_raw_spectral_line)
        directional_records = _read_generic(current_dir / f"{station_id}_directional.gz", self.parse_directional_line)
        wave_records = _read_wave_file(current_dir / f"{station_id}_wave.gz")
        
        logger.info(f"Station {station_id}: {len(spectral_records)} spectral, {len(directional_records)} directional, {len(wave_records)} wave records")
        return spectral_records, directional_records, wave_records

    def create_feature_vectors(self, station_id: str) -> Tuple[np.ndarray, List[datetime], Dict]:
        """Enhanced feature vector creation with duplicate handling"""
        spectral_recs, dir_recs, wave_recs = self.load_station_files(station_id)
        
        if not spectral_recs: 
            return np.array([]), [], {'error': 'No spectral data', 'total_spectral_records': 0, 'valid_feature_vectors': 0}

        # Handle duplicates by keeping the last occurrence
        def deduplicate_records(records):
            if not records:
                return records
            # Create a dictionary with timestamp as key, keeping last occurrence
            dedup_dict = {}
            for record in records:
                dedup_dict[record['timestamp']] = record
            return list(dedup_dict.values())

        # Deduplicate all record types
        spectral_recs = deduplicate_records(spectral_recs)
        dir_recs = deduplicate_records(dir_recs)
        wave_recs = deduplicate_records(wave_recs)
        
        logger.debug(f"After deduplication: {len(spectral_recs)} spectral, {len(dir_recs)} directional, {len(wave_recs)} wave")

        # Create DataFrames for efficient temporal lookups
        try:
            spec_df = pd.DataFrame(spectral_recs).set_index('timestamp')
            dir_df = pd.DataFrame(dir_recs).set_index('timestamp') if dir_recs else None
            wave_df = pd.DataFrame(wave_recs).set_index('timestamp') if wave_recs else None
        except Exception as e:
            logger.error(f"Error creating DataFrames: {e}")
            return np.array([]), [], {'error': f'DataFrame creation failed: {e}', 'total_spectral_records': len(spectral_recs), 'valid_feature_vectors': 0}

        features, timestamps = [], []
        processing_errors = 0
        
        for timestamp, row in spec_df.iterrows():
            try:
                # Initialize feature dictionary
                f = {name: np.nan for name in self.feature_names}

                # Interpolate spectral data to standard grid
                interpolated_spec = self.interpolate_to_standard_grid(row['frequencies'], row['spectral_density'])
                if np.all(np.isnan(interpolated_spec)): 
                    continue

                # Compute bulk parameters and wave components
                bulk = self.compute_bulk_parameters(self.target_freq_bins, interpolated_spec)
                comp = self.separate_wave_components(self.target_freq_bins, interpolated_spec)
                
                # Fill spectral features
                for key in ['significant_height', 'peak_period', 'mean_period', 'peak_frequency', 'total_energy', 'spectral_width']:
                    f[key.replace('_frequency', '_freq')] = bulk.get(key, np.nan)
                
                for key in ['swell_energy', 'windsea_energy', 'swell_fraction', 'windsea_fraction']:
                    f[key] = comp.get(key, np.nan)
                
                # Derived features
                f['log_energy'] = np.log1p(f['total_energy']) if not np.isnan(f['total_energy']) else np.nan
                if not np.isnan(f['peak_period']) and not np.isnan(f['sig_height']):
                    f['steepness_proxy'] = f['peak_period'] * f['sig_height']
                
                f['sep_freq'] = row.get('separation_frequency', np.nan)
                f['n_freq_bins'] = float(row.get('n_frequencies', len(row['frequencies'])))
                f['data_quality'] = 1.0 if f['n_freq_bins'] > 25 else 0.5

                # Enhanced temporal matching for wave data (simple nearest neighbor)
                if wave_df is not None and len(wave_df) > 0:
                    try:
                        # Find closest wave record within 30 minutes
                        time_diffs = abs(wave_df.index - timestamp)
                        min_diff_idx = time_diffs.argmin()
                        if time_diffs.iloc[min_diff_idx] <= pd.Timedelta(minutes=30):
                            closest_wave = wave_df.iloc[min_diff_idx]
                            f['wave_height_reported'] = closest_wave.get('wvht', np.nan)
                            f['dom_period_reported'] = closest_wave.get('dpd', np.nan)
                            f['wave_dir_reported'] = closest_wave.get('mwd', np.nan)
                    except Exception as e:
                        logger.debug(f"Wave matching error: {e}")

                # Enhanced temporal matching for directional data
                if dir_df is not None and len(dir_df) > 0:
                    try:
                        # Find closest directional record within 30 minutes
                        time_diffs = abs(dir_df.index - timestamp)
                        min_diff_idx = time_diffs.argmin()
                        if time_diffs.iloc[min_diff_idx] <= pd.Timedelta(minutes=30):
                            closest_dir = dir_df.iloc[min_diff_idx]
                            interp_dirs = self.interpolate_to_standard_grid(closest_dir['frequencies'], closest_dir['mean_directions'])
                            valid_dirs = interp_dirs[~np.isnan(interp_dirs)]
                            if len(valid_dirs) > 0:
                                f['mean_dir_calc'] = np.mean(valid_dirs)
                                f['dir_spread_calc'] = np.std(valid_dirs) if len(valid_dirs) > 1 else 0.0
                    except Exception as e:
                        logger.debug(f"Directional matching error: {e}")

                # Convert to ordered vector and validate
                feature_vector = [f[name] for name in self.feature_names]
                if not np.isnan(feature_vector[0]):  # sig_height is required
                    features.append(feature_vector)
                    timestamps.append(timestamp)
                
            except Exception as e:
                processing_errors += 1
                logger.debug(f"Error processing timestamp {timestamp}: {e}")

        # Create results array and statistics
        features_array = np.array(features) if features else np.array([]).reshape(0, len(self.feature_names))
        
        stats = {
            'total_spectral_records': len(spectral_recs), 
            'valid_feature_vectors': len(features),
            'processing_errors': processing_errors,
            'completeness': len(features) / len(spectral_recs) if spectral_recs else 0
        }
        
        if features:
            stats['time_span_days'] = (max(timestamps) - min(timestamps)).days if len(timestamps) > 1 else 0
            stats['nan_percentage_by_feature'] = np.isnan(features_array).mean(axis=0) * 100
            
            # Data quality metrics
            valid_sig_heights = features_array[:, 0][~np.isnan(features_array[:, 0])]
            if len(valid_sig_heights) > 0:
                stats['sig_height_range'] = [float(np.min(valid_sig_heights)), float(np.max(valid_sig_heights))]
                stats['mean_sig_height'] = float(np.mean(valid_sig_heights))
        else:
            logger.error("No valid feature vectors created - check data quality")
        
        return features_array, timestamps, stats

def main():
    """Enhanced main function with better error reporting"""
    parser = argparse.ArgumentParser(description="Fixed NDBC spectral data processor")
    parser.add_argument("--data-dir", default="data/buoy_data", help="Data directory")
    parser.add_argument("--test-single", help="Test with single station ID")
    parser.add_argument("--stations", help="Comma-separated station IDs")
    parser.add_argument("--output", default="fixed_features.json", help="Output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    processor = FixedSpectralDataProcessor(args.data_dir)
    
    # Test mode - single station
    if args.test_single:
        station_id = args.test_single
        logger.info(f"=== Testing Fixed Processor with Station: {station_id} ===")
        
        features, timestamps, stats = processor.create_feature_vectors(station_id)
        
        print(f"\n🔧 Fixed Test Results for Station {station_id}")
        print("=" * 60)
        
        if 'error' in stats or not stats.get('valid_feature_vectors', 0):
            print("❌ Processing failed. No valid feature vectors created.")
            print(f"Details: {stats.get('error', 'Unknown error')}")
            print(f"Total spectral records found: {stats.get('total_spectral_records', 0)}")
            print(f"Processing errors: {stats.get('processing_errors', 0)}")
            
            # Additional debugging info
            if args.verbose:
                print("\n🔍 Debug Information:")
                print("- Check if data files exist in correct location")
                print("- Verify data format matches expected NDBC format")
                print("- Check for data quality issues (invalid timestamps, corrupted values)")
            return
            
        print(f"✅ Valid feature vectors: {stats['valid_feature_vectors']:,}")
        print(f"📊 Completeness: {stats['completeness']:.1%}")
        print(f"📅 Time span: {stats.get('time_span_days', 0)} days")
        print(f"⚠️  Processing errors: {stats.get('processing_errors', 0)}")
        
        if 'sig_height_range' in stats:
            print(f"🌊 Wave height range: {stats['sig_height_range'][0]:.2f} - {stats['sig_height_range'][1]:.2f} m")
            print(f"📈 Mean wave height: {stats['mean_sig_height']:.2f} m")
        
        if features.shape[0] > 0:
            print(f"\n🔍 Sample Feature Vector (First Record):")
            print("-" * 50)
            for name, value in zip(processor.feature_names, features[0]):
                if not np.isnan(value):
                    if 'energy' in name.lower():
                        print(f"  {name:22}: {value:10.3e}")
                    elif 'fraction' in name.lower():
                        print(f"  {name:22}: {value:10.3f}")
                    else:
                        print(f"  {name:22}: {value:10.3f}")
                else:
                    print(f"  {name:22}: {'NaN':>10}")

            print(f"\n📊 Data Completeness by Feature:")
            print("-" * 50)
            for name, pct in zip(processor.feature_names, stats['nan_percentage_by_feature']):
                if pct < 10:
                    indicator = "🟢"
                elif pct < 50:
                    indicator = "🟡"
                else:
                    indicator = "🔴"
                print(f"  {name:22}: {pct:5.1f}% {indicator}")
            
            # Enhanced physics validation
            print(f"\n🔬 Physics Validation:")
            print("-" * 30)
            sig_heights = features[:, 0]
            valid_heights = sig_heights[~np.isnan(sig_heights)]
            
            if len(valid_heights) > 0:
                min_h, max_h = np.min(valid_heights), np.max(valid_heights)
                mean_h = np.mean(valid_heights)
                
                print(f"  Range: {min_h:.2f} - {max_h:.2f} m")
                print(f"  Mean:  {mean_h:.2f} m")
                
                if np.any((valid_heights < 0.1) | (valid_heights > 20)):
                    print("  ⚠️  Some heights outside typical range (0.1-20m)")
                else:
                    print("  ✅ All heights within reasonable range")
                
                # Check for data quality indicators
                good_quality = np.sum(features[:, 14] > 0.9)  # data_quality feature
                print(f"  📊 High-quality records: {good_quality}/{len(features)} ({100*good_quality/len(features):.1f}%)")
        
        return
    
    # Full processing mode
    if args.stations:
        station_ids = [s.strip() for s in args.stations.split(",")]
    else:
        # Auto-detect stations
        current_dir = Path(args.data_dir) / "current"
        if not current_dir.exists():
            logger.error(f"Data directory not found: {current_dir}")
            return
            
        spectral_files = list(current_dir.glob("*_raw_spectral.gz"))
        station_ids = sorted(list(set(f.name.split('_')[0] for f in spectral_files)))
        logger.info(f"Auto-detected {len(station_ids)} stations: {station_ids}")
    
    if not station_ids:
        logger.error("No stations found to process")
        return
    
    # Process all stations
    all_results = {}
    success_count = 0
    
    for i, station_id in enumerate(station_ids, 1):
        logger.info(f"Processing station {i}/{len(station_ids)}: {station_id}")
        
        try:
            features, timestamps, stats = processor.create_feature_vectors(station_id)
            
            if len(features) > 0:
                all_results[station_id] = {
                    'features': features.tolist(),
                    'timestamps': [ts.isoformat() for ts in timestamps],
                    'statistics': stats,
                    'feature_names': processor.feature_names,
                    'frequency_grid': processor.target_freq_bins.tolist()
                }
                success_count += 1
                logger.info(f"✅ {station_id}: {len(features)} vectors ({stats['completeness']:.1%} complete)")
            else:
                logger.warning(f"❌ {station_id}: No valid feature vectors created")
                
        except Exception as e:
            logger.error(f"💥 Failed to process station {station_id}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Save results
    if all_results:
        try:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"💾 Results saved to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return
    
    # Enhanced summary statistics
    print(f"\n🎯 Fixed Processing Summary")
    print("=" * 50)
    print(f"🎯 Stations processed: {success_count}/{len(station_ids)}")
    
    if all_results:
        total_features = sum(len(data['features']) for data in all_results.values())
        total_records = sum(data['statistics']['total_spectral_records'] for data in all_results.values())
        avg_completeness = np.mean([data['statistics']['completeness'] for data in all_results.values()])
        
        print(f"📊 Total feature vectors: {total_features:,}")
        print(f"📈 Total spectral records: {total_records:,}")
        print(f"✅ Average completeness: {avg_completeness:.1%}")
        print(f"💾 Output file: {args.output}")
        
        # Best and worst performing stations
        if len(all_results) > 1:
            completeness_by_station = {
                station: data['statistics']['completeness'] 
                for station, data in all_results.items()
            }
            
            best_station = max(completeness_by_station, key=completeness_by_station.get)
            worst_station = min(completeness_by_station, key=completeness_by_station.get)
            
            print(f"\n🏆 Best completeness: {best_station} ({completeness_by_station[best_station]:.1%})")
            print(f"📉 Worst completeness: {worst_station} ({completeness_by_station[worst_station]:.1%})")
        
        # Feature quality summary
        all_features = np.concatenate([np.array(data['features']) for data in all_results.values()])
        if len(all_features) > 0:
            overall_nan_pct = np.isnan(all_features).mean(axis=0) * 100
            
            print(f"\n📊 Overall Feature Quality:")
            print("-" * 30)
            critical_features = ['sig_height', 'peak_period', 'total_energy']
            for i, name in enumerate(processor.feature_names):
                if name in critical_features:
                    indicator = "🔴" if overall_nan_pct[i] > 50 else "🟡" if overall_nan_pct[i] > 10 else "🟢"
                    print(f"  {name:18}: {overall_nan_pct[i]:5.1f}% {indicator}")
    else:
        print("❌ No stations successfully processed")
        
    print(f"\n🔧 Fixed processing complete!")

if __name__ == "__main__":
    main()