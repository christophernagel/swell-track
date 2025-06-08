#!/usr/bin/env python3
"""
Focused Raw Spectral Data Processor

Designed specifically for NDBC raw spectral density data (.data_spec equivalent)
with custom file naming: {station_id}_raw_spectral.gz

Physics-based feature extraction for wave forecasting system competing with LOTUS.
Processes frequency-domain wave energy S(f) for transformer-based spatiotemporal modeling.
"""

import gzip
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import logging
from scipy.interpolate import interp1d
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class RawSpectralProcessor:
    """
    Focused processor for NDBC raw spectral density data.
    
    Designed for physics-based wave forecasting system with:
    - Frequency-domain energy analysis S(f)
    - Physics-informed feature extraction (15 features)
    - Transformer-ready spatiotemporal data preparation
    """

    def __init__(self, data_dir: str = "data/buoy_data"):
        self.data_dir = Path(data_dir)
        
        # Extended frequency grid for NDBC spectral data (0.033-0.485 Hz)
        # Covers typical NDBC range with good resolution for swell/wind wave separation
        self.target_freq_bins = np.array([
            0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080,
            0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180,
            0.190, 0.200, 0.220, 0.240, 0.260, 0.280, 0.300, 0.320, 0.350, 0.400, 
            0.450, 0.485
        ])
        
        # Wave physics parameters for swell/wind wave separation
        self.swell_threshold = 0.125    # Hz - frequency separating swell from wind waves
        self.windsea_threshold = 0.25   # Hz - frequency defining wind sea domain
        
        # Data validation limits for quality control
        self.max_reasonable_energy = 100.0   # m²/Hz - maximum realistic spectral density
        self.min_reasonable_freq = 0.020     # Hz - minimum NDBC frequency
        self.max_reasonable_freq = 0.500     # Hz - maximum NDBC frequency
        
        # Physics-informed feature names for transformer input
        self.feature_names = [
            # Bulk wave parameters (0-5)
            'sig_height', 'peak_period', 'mean_period', 'peak_freq', 'total_energy', 'spectral_width',
            # Wave component separation (6-9) 
            'swell_energy', 'windsea_energy', 'swell_fraction', 'windsea_fraction',
            # Derived physics features (10-12)
            'log_energy', 'steepness_proxy', 'sep_freq',
            # Data quality indicators (13-14)
            'n_freq_bins', 'data_quality'
        ]

    def parse_raw_spectral_line(self, line: str) -> Optional[Dict]:
        """
        Parse NDBC raw spectral density line.
        
        Format: YY MM DD hh mm Sep_Freq spec_1 (freq_1) spec_2 (freq_2) ...
        
        Returns dict with:
        - timestamp: datetime object
        - frequencies: numpy array of frequency bins (Hz)
        - spectral_density: numpy array of energy density values (m²/Hz)
        - separation_frequency: swell/wind wave separation frequency
        - metadata: quality and completeness information
        """
        # Skip comment lines and empty lines
        if line.startswith('#') or not line.strip():
            return None
        
        parts = line.split()
        if len(parts) < 8:  # Need timestamp + sep_freq + at least one spec/freq pair
            return None

        try:
            # Parse timestamp - handle both 2-digit and 4-digit years
            year = int(parts[0])
            if year < 100:
                year += 2000
            elif year < 1900:
                year += 2000
                
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            
            # Validate date components
            if not (1 <= month <= 12):
                logger.debug(f"Invalid month {month} in line: {line[:50]}...")
                return None
            if not (1 <= day <= 31):
                logger.debug(f"Invalid day {day} in line: {line[:50]}...")
                return None
            if not (0 <= hour <= 23):
                logger.debug(f"Invalid hour {hour} in line: {line[:50]}...")
                return None
            if not (0 <= minute <= 59):
                logger.debug(f"Invalid minute {minute} in line: {line[:50]}...")
                return None
                
            timestamp = datetime(year, month, day, hour, minute)
            
            # Parse separation frequency
            sep_freq = float(parts[5])
            
            # Parse spectral density values with frequencies
            # Format: spec_1 (freq_1) spec_2 (freq_2) ...
            frequencies = []
            spectral_densities = []
            
            i = 6  # Start after separation frequency
            while i < len(parts) - 1:
                try:
                    # Current part should be spectral density value
                    spec_str = parts[i]
                    
                    # Next part should be frequency in parentheses
                    if i + 1 < len(parts) and '(' in parts[i + 1] and ')' in parts[i + 1]:
                        freq_str = parts[i + 1].strip('()')
                        
                        # Skip missing data indicators
                        if spec_str not in ['MM', '999.00', '-999.00', '99.00']:
                            spec_val = float(spec_str)
                            freq = float(freq_str)
                            
                            # Validate frequency and energy ranges
                            if (self.min_reasonable_freq <= freq <= self.max_reasonable_freq and 
                                0 <= spec_val <= self.max_reasonable_energy):
                                frequencies.append(freq)
                                spectral_densities.append(spec_val)
                        
                        i += 2  # Move to next spec/freq pair
                    else:
                        # Malformed data, skip
                        i += 1
                        
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing spec/freq pair at position {i}: {e}")
                    i += 1
                    continue
            
            # Require minimum number of frequency bins for valid spectrum
            if len(frequencies) < 5:
                logger.debug(f"Insufficient spectral data: {len(frequencies)} bins")
                return None
            
            # Sort by frequency to ensure proper ordering
            freq_array = np.array(frequencies)
            spec_array = np.array(spectral_densities)
            sort_idx = np.argsort(freq_array)
            
            return {
                'timestamp': timestamp,
                'frequencies': freq_array[sort_idx],
                'spectral_density': spec_array[sort_idx],
                'separation_frequency': sep_freq,
                'n_frequencies': len(frequencies),
                'data_quality': 'good' if len(frequencies) > 25 else 'partial'
            }
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing raw spectral line: {line[:50]}... - {e}")
            return None

    def interpolate_to_standard_grid(self, frequencies: np.ndarray, 
                                   spectral_density: np.ndarray) -> np.ndarray:
        """
        Interpolate spectral data to standardized frequency grid.
        
        Required for consistent feature extraction across different stations
        and time periods with varying frequency resolutions.
        """
        if len(frequencies) != len(spectral_density) or len(frequencies) < 3:
            return np.full(len(self.target_freq_bins), np.nan)
        
        # Remove invalid data points
        valid_mask = (~np.isnan(spectral_density) & (~np.isnan(frequencies)) & 
                     (spectral_density >= 0) & (frequencies > 0))
        
        if np.sum(valid_mask) < 3:
            return np.full(len(self.target_freq_bins), np.nan)
        
        freq_valid = frequencies[valid_mask]
        spec_valid = spectral_density[valid_mask]
        
        # Remove duplicate frequencies and ensure monotonic ordering
        unique_freqs, unique_indices = np.unique(freq_valid, return_index=True)
        if len(unique_freqs) < 3:
            return np.full(len(self.target_freq_bins), np.nan)
        
        spec_unique = spec_valid[unique_indices]
        
        try:
            # Linear interpolation to standard grid
            f_interp = interp1d(unique_freqs, spec_unique, 
                              kind='linear', bounds_error=False, 
                              fill_value=0.0)  # Use 0 for out-of-bounds (physics appropriate)
            interpolated = f_interp(self.target_freq_bins)
            
            return interpolated
            
        except ValueError as e:
            logger.debug(f"Interpolation failed: {e}")
            return np.full(len(self.target_freq_bins), np.nan)

    def compute_bulk_parameters(self, frequencies: np.ndarray, 
                              spectral_density: np.ndarray) -> Dict[str, float]:
        """
        Compute bulk wave parameters from spectral density using wave physics.
        
        Based on spectral moments:
        - m₀ = ∫ S(f) df  (total energy)
        - m₁ = ∫ f S(f) df  (first moment)  
        - m₂ = ∫ f² S(f) df  (second moment)
        """
        # Initialize with NaN values
        params = {
            'significant_height': np.nan, 'peak_period': np.nan, 'mean_period': np.nan,
            'spectral_width': np.nan, 'peak_frequency': np.nan, 'total_energy': np.nan
        }
        
        # Validate input data
        valid_mask = (~np.isnan(spectral_density) & (spectral_density >= 0) & 
                     (~np.isnan(frequencies)) & (frequencies > 0))
        
        if np.sum(valid_mask) < 3:
            return params
        
        freq = frequencies[valid_mask]
        spec = spectral_density[valid_mask]
        
        try:
            # Compute spectral moments using trapezoidal integration
            m0 = np.trapz(spec, freq)  # Total wave energy
            if m0 <= 0:
                return params
            
            m1 = np.trapz(spec * freq, freq)
            m2 = np.trapz(spec * freq**2, freq)
            
            # Bulk wave parameters from spectral moments
            params['total_energy'] = m0
            params['significant_height'] = 4 * np.sqrt(m0)  # Standard relationship Hs = 4√m₀
            
            # Peak frequency and period
            peak_idx = np.argmax(spec)
            params['peak_frequency'] = freq[peak_idx]
            params['peak_period'] = 1.0 / params['peak_frequency']
            
            # Mean period from spectral moments
            if m1 > 0:
                params['mean_period'] = m0 / m1
            
            # Spectral width (measure of frequency spreading)
            if m1 > 0 and m2 > 0:
                params['spectral_width'] = np.sqrt(abs(m0 * m2 - m1**2) / m1**2)
            
            return params
            
        except Exception as e:
            logger.debug(f"Error computing bulk parameters: {e}")
            return params

    def separate_wave_components(self, frequencies: np.ndarray,
                               spectral_density: np.ndarray) -> Dict[str, float]:
        """
        Separate swell and wind wave components using frequency domain analysis.
        
        Physical basis:
        - Swell: f < 0.125 Hz (periods > 8s) - distantly generated waves
        - Wind waves: f > 0.25 Hz (periods < 4s) - locally generated waves
        """
        # Initialize with NaN values
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
            # Define frequency masks for wave components
            swell_mask = freq < self.swell_threshold
            windsea_mask = freq > self.windsea_threshold
            
            # Integrate energy in each frequency band
            swell_energy = np.trapz(spec[swell_mask], freq[swell_mask]) if np.any(swell_mask) else 0
            windsea_energy = np.trapz(spec[windsea_mask], freq[windsea_mask]) if np.any(windsea_mask) else 0
            total_energy = swell_energy + windsea_energy
            
            # Compute energy fractions
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

    def load_raw_spectral_file(self, station_id: str) -> List[Dict]:
        """
        Load raw spectral data file for a station.
        
        File naming convention: {station_id}_raw_spectral.gz
        """
        current_dir = self.data_dir / "current"
        spectral_file = current_dir / f"{station_id}_raw_spectral.gz"
        
        if not spectral_file.exists():
            logger.warning(f"Raw spectral file not found: {spectral_file}")
            return []
        
        records = []
        try:
            with gzip.open(spectral_file, 'rt') as f:
                line_count = 0
                success_count = 0
                
                for line in f:
                    line_count += 1
                    record = self.parse_raw_spectral_line(line)
                    if record:
                        records.append(record)
                        success_count += 1
                
                logger.info(f"Station {station_id}: Processed {line_count} lines, "
                           f"extracted {success_count} valid spectral records")
                
        except Exception as e:
            logger.error(f"Error reading {spectral_file}: {e}")
        
        return records

    def create_physics_features(self, station_id: str) -> Tuple[np.ndarray, List[datetime], Dict]:
        """
        Create physics-informed feature vectors for transformer training.
        
        Returns:
        - features: (n_timestamps, 15) array of physics features
        - timestamps: list of corresponding datetime objects  
        - stats: processing statistics and data quality metrics
        """
        # Load raw spectral data
        spectral_records = self.load_raw_spectral_file(station_id)
        
        if not spectral_records:
            return np.array([]), [], {
                'error': 'No raw spectral data found',
                'total_records': 0,
                'valid_features': 0
            }
        
        # Process each spectral record
        features = []
        timestamps = []
        processing_errors = 0
        
        for record in spectral_records:
            try:
                # Interpolate to standard frequency grid
                interpolated_spec = self.interpolate_to_standard_grid(
                    record['frequencies'], 
                    record['spectral_density']
                )
                
                # Skip if interpolation failed
                if np.all(np.isnan(interpolated_spec)):
                    continue
                
                # Compute physics-based features
                bulk_params = self.compute_bulk_parameters(
                    self.target_freq_bins, interpolated_spec
                )
                wave_components = self.separate_wave_components(
                    self.target_freq_bins, interpolated_spec
                )
                
                # Create 15-dimensional feature vector
                feature_vector = np.full(15, np.nan)
                
                # Bulk parameters (0-5)
                feature_vector[0] = bulk_params['significant_height']
                feature_vector[1] = bulk_params['peak_period']
                feature_vector[2] = bulk_params['mean_period']
                feature_vector[3] = bulk_params['peak_frequency']
                feature_vector[4] = bulk_params['total_energy']
                feature_vector[5] = bulk_params['spectral_width']
                
                # Wave components (6-9)
                feature_vector[6] = wave_components['swell_energy']
                feature_vector[7] = wave_components['windsea_energy']
                feature_vector[8] = wave_components['swell_fraction']
                feature_vector[9] = wave_components['windsea_fraction']
                
                # Derived features (10-12)
                feature_vector[10] = (np.log1p(bulk_params['total_energy']) 
                                    if not np.isnan(bulk_params['total_energy']) else np.nan)
                
                if (not np.isnan(bulk_params['peak_period']) and 
                    not np.isnan(bulk_params['significant_height'])):
                    feature_vector[11] = bulk_params['peak_period'] * bulk_params['significant_height']
                
                feature_vector[12] = record['separation_frequency']
                
                # Data quality (13-14)
                feature_vector[13] = float(record['n_frequencies'])
                feature_vector[14] = 1.0 if record['data_quality'] == 'good' else 0.5
                
                # Only keep records with valid significant wave height
                if not np.isnan(feature_vector[0]):
                    features.append(feature_vector)
                    timestamps.append(record['timestamp'])
                
            except Exception as e:
                processing_errors += 1
                logger.debug(f"Error processing record at {record.get('timestamp', 'unknown')}: {e}")
        
        # Create results
        features_array = np.array(features) if features else np.array([]).reshape(0, 15)
        
        # Compute statistics
        stats = {
            'total_records': len(spectral_records),
            'valid_features': len(features),
            'processing_errors': processing_errors,
            'completeness': len(features) / len(spectral_records) if spectral_records else 0
        }
        
        if features:
            stats['time_span_days'] = (max(timestamps) - min(timestamps)).days if len(timestamps) > 1 else 0
            stats['nan_percentage_by_feature'] = np.isnan(features_array).mean(axis=0) * 100
            
            # Physics validation
            valid_heights = features_array[:, 0][~np.isnan(features_array[:, 0])]
            if len(valid_heights) > 0:
                stats['wave_height_range'] = [float(np.min(valid_heights)), float(np.max(valid_heights))]
                stats['mean_wave_height'] = float(np.mean(valid_heights))
        
        return features_array, timestamps, stats

def main():
    """Main function for raw spectral data processing and feature extraction."""
    parser = argparse.ArgumentParser(description="NDBC Raw Spectral Data Processor")
    parser.add_argument("--data-dir", default="data/buoy_data", 
                       help="Data directory containing station files")
    parser.add_argument("--test-single", help="Test with single station ID")
    parser.add_argument("--stations", help="Comma-separated station IDs")
    parser.add_argument("--output", default="spectral_features.json", 
                       help="Output file for physics features")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    processor = RawSpectralProcessor(args.data_dir)
    
    # Single station testing mode
    if args.test_single:
        station_id = args.test_single
        logger.info(f"=== Processing Raw Spectral Data for Station: {station_id} ===")
        
        features, timestamps, stats = processor.create_physics_features(station_id)
        
        print(f"\n🌊 Raw Spectral Processing Results for Station {station_id}")
        print("=" * 70)
        
        if 'error' in stats or not stats.get('valid_features', 0):
            print("❌ Processing failed - no valid physics features extracted")
            print(f"Error: {stats.get('error', 'Unknown processing error')}")
            print(f"Total records found: {stats.get('total_records', 0)}")
            return
        
        print(f"✅ Physics features extracted: {stats['valid_features']:,}")
        print(f"📊 Processing completeness: {stats['completeness']:.1%}")
        print(f"📅 Temporal coverage: {stats.get('time_span_days', 0)} days")
        print(f"⚠️  Processing errors: {stats.get('processing_errors', 0)}")
        
        if 'wave_height_range' in stats:
            print(f"🌊 Wave height range: {stats['wave_height_range'][0]:.2f} - {stats['wave_height_range'][1]:.2f} m")
            print(f"📈 Mean wave height: {stats['mean_wave_height']:.2f} m")
        
        if features.shape[0] > 0:
            print(f"\n🔬 Physics Feature Analysis (First Record):")
            print("-" * 60)
            for i, (name, value) in enumerate(zip(processor.feature_names, features[0])):
                if not np.isnan(value):
                    if 'energy' in name.lower():
                        print(f"  {i:2d}. {name:18}: {value:12.4e}")
                    else:
                        print(f"  {i:2d}. {name:18}: {value:12.4f}")
                else:
                    print(f"  {i:2d}. {name:18}: {'NaN':>12}")
            
            print(f"\n📊 Feature Completeness Analysis:")
            print("-" * 45)
            for i, (name, pct) in enumerate(zip(processor.feature_names, stats['nan_percentage_by_feature'])):
                if pct < 5:
                    indicator = "🟢 Excellent"
                elif pct < 20:
                    indicator = "🟡 Good"
                elif pct < 50:
                    indicator = "🟠 Fair"
                else:
                    indicator = "🔴 Poor"
                print(f"  {i:2d}. {name:18}: {pct:5.1f}% missing {indicator}")
            
            # Physics validation
            print(f"\n🔬 Wave Physics Validation:")
            print("-" * 35)
            heights = features[:, 0][~np.isnan(features[:, 0])]
            periods = features[:, 1][~np.isnan(features[:, 1])]
            energies = features[:, 4][~np.isnan(features[:, 4])]
            
            if len(heights) > 0:
                print(f"  Wave Heights: {np.min(heights):.2f} - {np.max(heights):.2f} m")
                print(f"  Peak Periods: {np.min(periods):.1f} - {np.max(periods):.1f} s" if len(periods) > 0 else "  Peak Periods: No data")
                print(f"  Energy Range: {np.min(energies):.2e} - {np.max(energies):.2e} m²" if len(energies) > 0 else "  Energy Range: No data")
                
                # Physics checks
                if np.any((heights < 0.1) | (heights > 20)):
                    print("  ⚠️  Some wave heights outside typical range (0.1-20m)")
                else:
                    print("  ✅ All wave heights within realistic range")
                
                # Data quality summary
                high_quality = np.sum(features[:, 14] > 0.9)
                print(f"  📊 High-quality records: {high_quality}/{len(features)} ({100*high_quality/len(features):.1f}%)")
        
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
            logger.error("No raw spectral files found - check file naming convention")
            return
            
        station_ids = sorted([f.name.split('_')[0] for f in spectral_files])
        logger.info(f"Auto-detected {len(station_ids)} stations: {station_ids}")
    
    # Process all stations
    all_results = {}
    success_count = 0
    
    for i, station_id in enumerate(station_ids, 1):
        logger.info(f"Processing station {i}/{len(station_ids)}: {station_id}")
        
        try:
            features, timestamps, stats = processor.create_physics_features(station_id)
            
            if len(features) > 0:
                all_results[station_id] = {
                    'physics_features': features.tolist(),
                    'timestamps': [ts.isoformat() for ts in timestamps],
                    'statistics': stats,
                    'feature_names': processor.feature_names,
                    'frequency_grid': processor.target_freq_bins.tolist(),
                    'wave_physics_params': {
                        'swell_threshold': processor.swell_threshold,
                        'windsea_threshold': processor.windsea_threshold
                    }
                }
                success_count += 1
                logger.info(f"✅ {station_id}: {len(features)} physics features ({stats['completeness']:.1%})")
            else:
                logger.warning(f"❌ {station_id}: No valid physics features extracted")
                
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
            logger.info(f"💾 Physics features saved to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return
    
    # Summary
    print(f"\n🌊 Raw Spectral Processing Summary")
    print("=" * 50)
    print(f"🎯 Stations processed: {success_count}/{len(station_ids)}")
    
    if all_results:
        total_features = sum(len(data['physics_features']) for data in all_results.values())
        total_records = sum(data['statistics']['total_records'] for data in all_results.values())
        avg_completeness = np.mean([data['statistics']['completeness'] for data in all_results.values()])
        
        print(f"📊 Total physics features: {total_features:,}")
        print(f"📈 Total spectral records: {total_records:,}")
        print(f"✅ Average completeness: {avg_completeness:.1%}")
        print(f"💾 Output: {args.output}")
        
        print(f"\n🔬 Physics Feature Summary:")
        print("- 15 physics-informed features per observation")
        print("- Frequency-domain energy analysis S(f)")
        print("- Swell/wind wave component separation") 
        print("- Ready for transformer spatiotemporal modeling")
        
    print(f"\n🌊 Raw spectral processing complete!")

if __name__ == "__main__":
    main()