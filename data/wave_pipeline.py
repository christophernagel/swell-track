#!/usr/bin/env python3
"""
Integrated NDBC Wave Physics Pipeline
Combines optimized data collection with enhanced multi-stream processing
"""

import os
import gzip
import json
import time
import signal
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import argparse
from scipy.interpolate import interp1d
import hashlib
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("wave_pipeline.log")]
)
logger = logging.getLogger('ndbc_pipeline')

# ==================== DATA COLLECTOR ====================
@dataclass
class CollectionRecord:
    station_id: str
    timestamp: datetime
    data_type: str
    status: str
    records_collected: int
    records_appended: int
    file_size: int
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    data_hash: Optional[str] = None

class OptimizedBuoyCollector:
    """Efficient NDBC data collector with full historical retention"""
    
    DATA_TYPES = [
        "wave", "spectral", "raw_spectral",
        "directional_alpha1", "directional_alpha2",
        "directional_r1", "directional_r2"
    ]
    
    URL_MAP = {
        "wave": "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt",
        "spectral": "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec",
        "raw_spectral": "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.data_spec",
        "directional_alpha1": "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swdir",
        "directional_alpha2": "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swdir2",
        "directional_r1": "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swr1",
        "directional_r2": "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swr2"
    }
    
    def __init__(self, data_dir: str = "data/buoy_data", config_file: str = "enhanced_stations.json"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup directory structure
        (self.data_dir / "current").mkdir(exist_ok=True)
        
        # Load station configuration
        with open(config_file, 'r') as f:
            self.stations = json.load(f)
        
        # HTTP session configuration
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WavePhysicsBot/2.0 (Scientific Research; +https://github.com/yourrepo/wave-physics)'
        })
        self.session.timeout = 20
        
        # Graceful shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Shutdown signal {signum} received")
        self.shutdown_requested = True
    
    def _get_data_filepath(self, station_id: str, data_type: str) -> Path:
        """Generate standardized file path for data type"""
        return self.data_dir / "current" / f"{station_id}_{data_type}.gz"
    
    def _parse_timestamp(self, parts: list) -> Optional[datetime]:
        """Robust timestamp parser with validation"""
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            
            # Validate components
            if not (2000 <= year <= 2100 and 1 <= month <= 12 and 
                    1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
                return None
                
            return datetime(year, month, day, hour, minute)
        except (ValueError, IndexError):
            return None
    
    def _load_existing_data(self, filepath: Path) -> Tuple[List[str], Optional[datetime], Optional[datetime]]:
        """Load existing data with error handling"""
        if not filepath.exists():
            return [], None, None
        
        try:
            with gzip.open(filepath, 'rt') as f:
                lines = f.read().strip().split('\n')
            
            data_lines = [line for line in lines if line.strip() and not line.startswith('#')]
            if not data_lines:
                return lines, None, None
            
            timestamps = []
            for line in data_lines[:1000]:  # Sample first 1000 lines for efficiency
                ts = self._parse_timestamp(line.split())
                if ts:
                    timestamps.append(ts)
            
            return lines, min(timestamps), max(timestamps) if timestamps else (None, None)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return [], None, None
    
    def _deduplicate_data(self, existing: List[str], new_content: str) -> Tuple[List[str], int]:
        """Efficient deduplication with timestamp-based merging"""
        # Create timestamp-indexed dictionary
        ts_map = {}
        for line in existing:
            if line.strip() and not line.startswith('#'):
                ts = self._parse_timestamp(line.split())
                if ts:
                    ts_map[ts] = line
        
        # Process new content
        new_records = 0
        for line in new_content.strip().split('\n'):
            if line.strip() and not line.startswith('#'):
                ts = self._parse_timestamp(line.split())
                if ts:
                    if ts not in ts_map:
                        new_records += 1
                    ts_map[ts] = line
        
        # Reconstruct sorted data
        sorted_ts = sorted(ts_map.keys())
        sorted_lines = [ts_map[ts] for ts in sorted_ts]
        
        # Preserve headers
        headers = [line for line in existing if line.startswith('#')]
        return headers + sorted_lines, new_records
    
    def _atomic_write(self, filepath: Path, content: str) -> None:
        """Safe atomic write with temp file"""
        temp_path = filepath.parent / f".{filepath.name}.tmp"
        try:
            with gzip.open(temp_path, 'wt') as f:
                f.write(content)
            shutil.move(temp_path, filepath)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def collect_station_data(self, station_id: str, data_type: str) -> CollectionRecord:
        """Collect single data type for a station"""
        start_time = time.time()
        url = self.URL_MAP[data_type].format(station_id=station_id)
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            new_content = response.text.strip()
            
            if not new_content or len(new_content) < 100:
                return CollectionRecord(
                    station_id=station_id,
                    timestamp=datetime.now(),
                    data_type=data_type,
                    status="empty",
                    records_collected=0,
                    records_appended=0,
                    file_size=0,
                    response_time=time.time()-start_time
                )
            
            filepath = self._get_data_filepath(station_id, data_type)
            existing, first_ts, last_ts = self._load_existing_data(filepath)
            
            # Deduplicate and merge
            final_content, new_records = self._deduplicate_data(existing, new_content)
            self._atomic_write(filepath, "\n".join(final_content))
            
            return CollectionRecord(
                station_id=station_id,
                timestamp=datetime.now(),
                data_type=data_type,
                status="success",
                records_collected=len(new_content.split('\n')),
                records_appended=new_records,
                file_size=filepath.stat().st_size,
                response_time=time.time()-start_time,
                data_hash=hashlib.sha256(new_content.encode()).hexdigest()[:16]
            )
            
        except requests.RequestException as e:
            return CollectionRecord(
                station_id=station_id,
                timestamp=datetime.now(),
                data_type=data_type,
                status="network_error",
                records_collected=0,
                records_appended=0,
                file_size=0,
                error_message=str(e),
                response_time=time.time()-start_time
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return CollectionRecord(
                station_id=station_id,
                timestamp=datetime.now(),
                data_type=data_type,
                status="error",
                records_collected=0,
                records_appended=0,
                file_size=0,
                error_message=str(e),
                response_time=time.time()-start_time
            )
    
    def collect_batch(self, station_ids: List[str], max_workers: int = 8) -> Dict[str, List[CollectionRecord]]:
        """Parallel collection for multiple stations"""
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for sid in station_ids:
                for dtype in self.DATA_TYPES:
                    if self.shutdown_requested:
                        break
                    futures[executor.submit(self.collect_station_data, sid, dtype)] = (sid, dtype)
            
            for future in as_completed(futures):
                sid, dtype = futures[future]
                try:
                    record = future.result()
                    if sid not in results:
                        results[sid] = []
                    results[sid].append(record)
                    status = "✓" if record.status == "success" else "✗"
                    logger.info(f"{status} {sid}-{dtype}: {record.records_appended} new records")
                except Exception as e:
                    logger.error(f"Processing failed: {e}")
        
        return results

# ==================== DATA PROCESSOR ====================
class WavePhysicsProcessor:
    """Enhanced physics-aware feature extractor"""
    
    FEATURE_NAMES = [
        'sig_height', 'peak_period', 'mean_period', 'peak_freq', 'total_energy',
        'spectral_width', 'swell_energy', 'windsea_energy', 'swell_fraction', 'windsea_fraction',
        'primary_direction', 'primary_spread', 'secondary_direction', 'bimodal_strength', 'directional_separation',
        'wind_speed', 'wind_direction', 'wind_wave_alignment',
        'spectral_hs_error', 'spectral_tp_error'
    ]
    
    def __init__(self, data_dir: str = "data/buoy_data"):
        self.data_dir = Path(data_dir)
        self.freq_bins = np.array([
            0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080,
            0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180,
            0.190, 0.200, 0.220, 0.240, 0.260, 0.280, 0.300, 0.320, 0.350, 0.400, 
            0.450, 0.485
        ])
        
        # Physics parameters
        self.swell_threshold = 0.125
        self.windsea_threshold = 0.25
        self.max_energy = 100.0

    def _parse_line(self, line: str, headers: List[str] = None) -> Optional[Dict]:
        """Unified line parser with robust validation"""
        if not line.strip() or line.startswith('#'):
            return None
            
        parts = line.split()
        if not parts or len(parts) < 5:
            return None
            
        try:
            # Handle different formats
            if headers and len(parts) >= len(headers):
                data = dict(zip(headers, parts))
                time_fields = ['YY', 'MM', 'DD', 'hh', 'mm']
                ts_parts = [data.get(f, '0') for f in time_fields]
            else:
                ts_parts = parts[:5]
                data = {f"col_{i}": p for i, p in enumerate(parts)}
            
            ts = self._parse_timestamp(ts_parts)
            if not ts:
                return None
                
            return {'timestamp': ts, 'data': data}
        except Exception:
            return None

    def _parse_timestamp(self, parts: List[str]) -> Optional[datetime]:
        """Robust timestamp parsing with validation"""
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            
            # Fix 2-digit years
            if year < 100:
                year += 2000
                
            # Validate components
            if not (2020 <= year <= 2030 and 1 <= month <= 12 and 
                    1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
                return None
                
            return datetime(year, month, day, hour, minute)
        except (ValueError, IndexError):
            return None

    def _load_data_file(self, station_id: str, data_type: str) -> List[Dict]:
        """Load and parse data file with automatic header detection"""
        filepath = self.data_dir / "current" / f"{station_id}_{data_type}.gz"
        if not filepath.exists():
            return []
            
        try:
            with gzip.open(filepath, 'rt') as f:
                lines = f.readlines()
                
            # Detect headers
            headers = []
            for line in lines:
                if any(x in line for x in ['YY', 'WVHT', 'WSPD']):
                    clean_line = line.strip().lstrip('#')
                    headers = clean_line.split()
                    break
                    
            # Parse records
            return [rec for rec in (self._parse_line(line, headers) for line in lines) if rec]
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return []

    def _interpolate_spectrum(self, freqs: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Interpolate to standard frequency grid"""
        valid_mask = (~np.isnan(values)) & (freqs > 0) & (values >= 0)
        if np.sum(valid_mask) < 3:
            return np.full(len(self.freq_bins), np.nan)
            
        try:
            interp_fn = interp1d(
                freqs[valid_mask], 
                values[valid_mask], 
                kind='linear', 
                bounds_error=False, 
                fill_value=np.nan
            )
            return interp_fn(self.freq_bins)
        except Exception:
            return np.full(len(self.freq_bins), np.nan)

    def _compute_spectral_features(self, freqs: np.ndarray, spec: np.ndarray) -> Dict:
        """Calculate spectral physics features"""
        valid_mask = (~np.isnan(spec)) & (freqs > 0) & (spec >= 0)
        if np.sum(valid_mask) < 3:
            return {}
            
        try:
            # Numerical integration
            df = np.diff(freqs, prepend=freqs[0])
            m0 = np.sum(spec[valid_mask] * df[valid_mask])
            m1 = np.sum(freqs[valid_mask] * spec[valid_mask] * df[valid_mask])
            m2 = np.sum(freqs[valid_mask]**2 * spec[valid_mask] * df[valid_mask])
            
            # Basic parameters
            hs = 4 * np.sqrt(max(m0, 0))
            peak_idx = np.nanargmax(spec[valid_mask])
            tp = 1 / freqs[valid_mask][peak_idx] if freqs[valid_mask][peak_idx] > 0 else np.nan
            mean_period = m0 / m1 if m1 > 0 else np.nan
            
            # Wave component separation
            swell_mask = valid_mask & (freqs < self.swell_threshold)
            windsea_mask = valid_mask & (freqs > self.windsea_threshold)
            swell_energy = np.sum(spec[swell_mask] * df[swell_mask])
            windsea_energy = np.sum(spec[windsea_mask] * df[windsea_mask])
            total_energy = swell_energy + windsea_energy
            
            return {
                'sig_height': hs,
                'peak_period': tp,
                'mean_period': mean_period,
                'peak_freq': freqs[valid_mask][peak_idx],
                'total_energy': m0,
                'spectral_width': np.sqrt(max(m0*m2 - m1**2, 0)) / m1 if m1 > 0 else np.nan,
                'swell_energy': swell_energy,
                'windsea_energy': windsea_energy,
                'swell_fraction': swell_energy / total_energy if total_energy > 0 else np.nan,
                'windsea_fraction': windsea_energy / total_energy if total_energy > 0 else np.nan
            }
        except Exception:
            return {}

    def process_station(self, station_id: str) -> Tuple[np.ndarray, List[datetime], Dict]:
        """Process all data for a single station"""
        # Load raw spectral data
        spectral_data = self._load_data_file(station_id, "raw_spectral")
        if not spectral_data:
            return np.array([]), [], {'error': 'No spectral data'}
        
        # Load directional data
        dir_data = {dtype: self._load_data_file(station_id, f"directional_{dtype}") 
                   for dtype in ['alpha1', 'alpha2', 'r1', 'r2']}
        
        # Load wave (meteorological) data
        wave_data = self._load_data_file(station_id, "wave")
        
        # Convert to DataFrames
        spec_df = pd.DataFrame(spectral_data).set_index('timestamp')
        wave_df = pd.DataFrame(wave_data).set_index('timestamp') if wave_data else pd.DataFrame()
        dir_dfs = {dtype: pd.DataFrame(data).set_index('timestamp') for dtype, data in dir_data.items() if data}
        
        features = []
        timestamps = []
        stats = {'total': len(spectral_data), 'processed': 0}
        
        for ts, spec_row in spec_df.iterrows():
            try:
                # Initialize feature vector
                feat_vec = np.full(len(self.FEATURE_NAMES), np.nan)
                
                # Process spectral data
                freqs = np.array(spec_row['data'].get('frequencies', []))
                spec = np.array(spec_row['data'].get('spectral_density', []))
                spec_interp = self._interpolate_spectrum(freqs, spec)
                
                if np.all(np.isnan(spec_interp)):
                    continue
                    
                spec_features = self._compute_spectral_features(self.freq_bins, spec_interp)
                for i, key in enumerate([
                    'sig_height', 'peak_period', 'mean_period', 'peak_freq', 
                    'total_energy', 'spectral_width', 'swell_energy', 
                    'windsea_energy', 'swell_fraction', 'windsea_fraction'
                ]):
                    feat_vec[i] = spec_features.get(key, np.nan)
                
                # Process directional data
                dir_features = {}
                for dtype, df in dir_dfs.items():
                    if not df.empty:
                        time_diff = (df.index - ts).total_seconds().abs()
                        closest_idx = time_diff.idxmin()
                        if time_diff[closest_idx] < 1800:  # 30 minutes
                            dir_row = df.loc[closest_idx]
                            dir_features[dtype] = {
                                'freqs': np.array(dir_row['data'].get('frequencies', [])),
                                'values': np.array(dir_row['data'].get('values', []))
                            }
                
                # Calculate directional features
                if dir_features:
                    # ... [directional feature calculations] ...
                    pass
                
                # Add meteorological data
                if not wave_df.empty:
                    time_diff = (wave_df.index - ts).total_seconds().abs()
                    closest_idx = time_diff.idxmin()
                    if time_diff[closest_idx] < 3600:  # 1 hour
                        wave_row = wave_df.loc[closest_idx]['data']
                        # ... [meteorological feature calculations] ...
                
                # Validation features
                # ... [validation calculations] ...
                
                features.append(feat_vec)
                timestamps.append(ts)
                stats['processed'] += 1
                
            except Exception as e:
                logger.debug(f"Processing error at {ts}: {e}")
        
        return np.array(features), timestamps, stats

# ==================== PIPELINE CONTROLLER ====================
class WavePipeline:
    """Integrated controller for end-to-end processing"""
    
    def __init__(self, config_file: str = "enhanced_stations.json"):
        self.config_file = config_file
        self.collector = OptimizedBuoyCollector(config_file=config_file)
        self.processor = WavePhysicsProcessor()
        
    def run_collection(self, station_ids: List[str]):
        """Run data collection for specified stations"""
        logger.info(f"Starting data collection for {len(station_ids)} stations")
        results = self.collector.collect_batch(station_ids)
        
        # Summary report
        for sid, records in results.items():
            success = sum(1 for r in records if r.status == "success")
            total = len(records)
            new_recs = sum(r.records_appended for r in records)
            logger.info(f"Station {sid}: {success}/{total} successful, {new_recs} new records")
        
    def run_processing(self, station_ids: List[str], output_file: str):
        """Process data and save features"""
        all_results = {}
        for sid in station_ids:
            logger.info(f"Processing {sid}")
            features, timestamps, stats = self.processor.process_station(sid)
            if features.size > 0:
                all_results[sid] = {
                    'features': features.tolist(),
                    'timestamps': [ts.isoformat() for ts in timestamps],
                    'stats': stats,
                    'feature_names': self.processor.FEATURE_NAMES
                }
        
        if all_results:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Saved features to {output_file}")
        return all_results

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDBC Wave Physics Pipeline")
    parser.add_argument("--collect", action="store_true", help="Run data collection")
    parser.add_argument("--process", action="store_true", help="Run data processing")
    parser.add_argument("--stations", help="Comma-separated station IDs")
    parser.add_argument("--output", default="wave_features.json", help="Output file for features")
    parser.add_argument("--config", default="enhanced_stations.json", help="Station configuration file")
    
    args = parser.parse_args()
    station_ids = [s.strip() for s in args.stations.split(",")] if args.stations else ["46022"]
    
    pipeline = WavePipeline(config_file=args.config)
    
    if args.collect:
        pipeline.run_collection(station_ids)
    
    if args.process:
        results = pipeline.run_processing(station_ids, args.output)
        
        # Print summary
        if results:
            print("\nWave Physics Processing Summary")
            print("=" * 60)
            for sid, data in results.items():
                stats = data['stats']
                print(f"Station {sid}:")
                print(f"  Records: {stats['processed']}/{stats['total']} processed "
                      f"({stats['processed']/stats['total']:.1%})")
                if data['features']:
                    sample = np.array(data['features'][0])
                    print("  Sample Features:")
                    for i, (name, val) in enumerate(zip(data['feature_names'], sample)):
                        if not np.isnan(val):
                            unit = "°" if "direction" in name else "m" if "height" in name else "s" if "period" in name else ""
                            print(f"    {i:2d}. {name:20}: {val:8.3f}{unit}")
