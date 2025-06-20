#!/usr/bin/env python3
"""
Optimized NDBC Data Collector
Maintains single files per station/data_type, appending new records and managing 45-day windows
"""

import os
import gzip
import json
import time
import signal
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import sqlite3
from contextlib import contextmanager
import hashlib
import tempfile
import shutil

@dataclass
class CollectionRecord:
    """Record of a data collection attempt"""
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
    """
    Optimized NDBC data collector that maintains single files per station/data_type
    and manages 45-day rolling windows
    """
    
    def __init__(self, data_dir: str = "data/buoy_data", config_file: str = "enhanced_stations.json"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimized directory structure
        (self.data_dir / "current").mkdir(exist_ok=True)  # Current rolling files
        (self.data_dir / "archive").mkdir(exist_ok=True)  # Long-term storage
        (self.data_dir / "temp").mkdir(exist_ok=True)     # Temporary processing
        
        # Load station configuration
        with open(config_file, 'r') as f:
            self.stations = json.load(f)
        
        self.logger = logging.getLogger(__name__)
        
        # Collection settings
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SciWaveBot/1.0 (Wave Research; +https://github.com/yourrepo/wave-forecasting)'
        })
        
        # No data retention - maintain full historical logs
        
        # Graceful shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Shutdown signal {signum} received, finishing current operations...")
        self.shutdown_requested = True
    
    def _get_data_filepath(self, station_id: str, data_type: str) -> Path:
        """Get the path to the historical data file for a station/data_type"""
        filename = f"{station_id}_{data_type}.gz"
        return self.data_dir / "current" / filename
    
    def _parse_timestamp_from_line(self, line: str, data_type: str) -> Optional[datetime]:
        """Parse timestamp from a data line"""
        if not line.strip() or line.startswith('#'):
            return None
        
        parts = line.split()
        if len(parts) < 5:
            return None
        
        try:
            year = int(parts[0])
            if year < 100:  # 2-digit year
                year += 2000
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            
            return datetime(year, month, day, hour, minute)
        except (ValueError, IndexError):
            return None
    
    def _load_existing_data(self, filepath: Path) -> Tuple[List[str], Optional[datetime], Optional[datetime]]:
        """Load existing data file and return lines with first/last timestamps"""
        if not filepath.exists():
            return [], None, None
        
        try:
            with gzip.open(filepath, 'rt') as f:
                lines = f.read().strip().split('\n')
            
            # Find data lines (non-comment, non-empty)
            data_lines = [line for line in lines if line.strip() and not line.startswith('#')]
            
            if not data_lines:
                return lines, None, None
            
            # Get first and last timestamps
            data_type = filepath.stem.split('_')[1]  # Extract data_type from filename
            first_ts = self._parse_timestamp_from_line(data_lines[0], data_type)
            last_ts = self._parse_timestamp_from_line(data_lines[-1], data_type)
            
            return lines, first_ts, last_ts
            
        except Exception as e:
            self.logger.error(f"Error loading existing data from {filepath}: {e}")
            return [], None, None
    
    def _clean_old_data(self, lines: List[str], data_type: str) -> List[str]:
        """No cleaning - maintain full historical record"""
        return lines
    
    def _deduplicate_and_sort(self, lines: List[str], new_content: str, data_type: str) -> Tuple[List[str], int]:
        """Merge new content with existing data, deduplicating and sorting by timestamp"""
        # Parse new content
        new_lines = new_content.strip().split('\n')
        new_data_lines = [line for line in new_lines if line.strip() and not line.startswith('#')]
        
        # Get existing data lines
        existing_data = [line for line in lines if line.strip() and not line.startswith('#')]
        
        # Combine and deduplicate by timestamp
        timestamp_to_line = {}
        
        # Add existing data
        for line in existing_data:
            ts = self._parse_timestamp_from_line(line, data_type)
            if ts:
                timestamp_to_line[ts] = line
        
        # Add new data (will overwrite duplicates)
        new_records_count = 0
        for line in new_data_lines:
            ts = self._parse_timestamp_from_line(line, data_type)
            if ts:
                if ts not in timestamp_to_line:
                    new_records_count += 1
                timestamp_to_line[ts] = line
        
        # Sort by timestamp and recreate lines
        sorted_timestamps = sorted(timestamp_to_line.keys())
        sorted_data_lines = [timestamp_to_line[ts] for ts in sorted_timestamps]
        
        # Keep original header structure
        header_lines = [line for line in lines if line.startswith('#') or not line.strip()]
        if not header_lines and new_lines:
            # Use headers from new content if no existing headers
            header_lines = [line for line in new_lines if line.startswith('#') or not line.strip()]
        
        final_lines = header_lines + sorted_data_lines
        return final_lines, new_records_count
    
    def _atomic_write(self, filepath: Path, content: str) -> None:
        """Atomically write content to file using temporary file"""
        temp_path = filepath.parent / f".{filepath.name}.tmp"
        
        try:
            with gzip.open(temp_path, 'wt') as f:
                f.write(content)
            
            # Atomic move
            shutil.move(temp_path, filepath)
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def collect_and_update_station_data(self, station_id: str, data_type: str) -> CollectionRecord:
        """Collect new data and update the station's data file"""
        start_time = time.time()
        
        url_map = {
            "wave": f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt",
            "spectral": f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec",
            "raw_spectral": f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.data_spec",
            
            # Complete directional suite
            "directional_alpha1": f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swdir",   # Mean direction
            "directional_alpha2": f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swdir2",  # Secondary direction
            "directional_r1": f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swr1",        # Primary spread
            "directional_r2": f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swr2"         # Secondary spread
        }
        
        url = url_map.get(data_type)
        if not url:
            raise ValueError(f"Unknown data type: {data_type}")
        
        try:
            # Fetch new data
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            response_time = time.time() - start_time
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
                    response_time=response_time
                )
            
            # Get filepath and load existing data
            filepath = self._get_data_filepath(station_id, data_type)
            existing_lines, first_ts, last_ts = self._load_existing_data(filepath)
            
            # Merge and deduplicate (no cleaning - maintain full history)
            final_lines, new_records = self._deduplicate_and_sort(existing_lines, new_content, data_type)
            
            # Write updated file atomically
            final_content = '\n'.join(final_lines)
            self._atomic_write(filepath, final_content)
            
            file_size = filepath.stat().st_size
            content_hash = hashlib.sha256(new_content.encode('utf-8')).hexdigest()[:16]
            
            # Count total records in new content
            new_lines = [line for line in new_content.split('\n') if line.strip() and not line.startswith('#')]
            
            self.logger.debug(f"Updated {station_id}_{data_type}: +{new_records} new records")
            
            return CollectionRecord(
                station_id=station_id,
                timestamp=datetime.now(),
                data_type=data_type,
                status="success",
                records_collected=len(new_lines),
                records_appended=new_records,
                file_size=file_size,
                response_time=response_time,
                data_hash=content_hash
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
                response_time=time.time() - start_time
            )
        
        except Exception as e:
            self.logger.error(f"Unexpected error collecting {station_id} ({data_type}): {e}")
            return CollectionRecord(
                station_id=station_id,
                timestamp=datetime.now(),
                data_type=data_type,
                status="error",
                records_collected=0,
                records_appended=0,
                file_size=0,
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    def get_priority_stations(self, max_stations: int = 30) -> List[str]:
        """Get prioritized station list for collection"""
        candidates = []
        
        for station_id, station in self.stations.items():
            priority = station.get('collection_priority', 5)
            has_spectral = station.get('has_spectral', False)
            
            if priority <= 3 and has_spectral:
                candidates.append((station_id, priority))
        
        # Sort by priority
        candidates.sort(key=lambda x: x[1])
        return [station_id for station_id, _ in candidates[:max_stations]]
    
    def collect_batch_optimized(self, station_ids: List[str], max_workers: int = 8) -> Dict[str, List[CollectionRecord]]:
        """Collect data for multiple stations using optimized approach"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all collection tasks
            futures = {}
            
            for station_id in station_ids:
                if self.shutdown_requested:
                    break
                
                # Collect all data types for every station (let NDBC decide what's available)
                futures[executor.submit(self.collect_and_update_station_data, station_id, "raw_spectral")] = (station_id, "raw_spectral")
                futures[executor.submit(self.collect_and_update_station_data, station_id, "directional_alpha1")] = (station_id, "directional_alpha1")
                futures[executor.submit(self.collect_and_update_station_data, station_id, "directional_alpha2")] = (station_id, "directional_alpha2")
                futures[executor.submit(self.collect_and_update_station_data, station_id, "directional_r1")] = (station_id, "directional_r1")
                futures[executor.submit(self.collect_and_update_station_data, station_id, "directional_r2")] = (station_id, "directional_r2")
                futures[executor.submit(self.collect_and_update_station_data, station_id, "spectral")] = (station_id, "spectral")
                futures[executor.submit(self.collect_and_update_station_data, station_id, "wave")] = (station_id, "wave")
            
            # Collect results
            for future in as_completed(futures):
                if self.shutdown_requested:
                    break
                
                station_id, data_type = futures[future]
                
                try:
                    record = future.result()
                    
                    if station_id not in results:
                        results[station_id] = []
                    results[station_id].append(record)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {station_id} ({data_type}): {e}")
        
        return results
    
    def archive_old_data(self) -> None:
        """Archive system disabled - maintaining full historical logs"""
        self.logger.info("Archive system disabled - maintaining full historical logs")
        pass
    
    def run_continuous_collection(self, interval_minutes: int = 30, max_workers: int = 8):
        """Run continuous optimized data collection"""
        station_list = self.get_priority_stations()
        self.logger.info(f"Starting continuous collection for {len(station_list)} stations")
        self.logger.info(f"Collection interval: {interval_minutes} minutes")
        self.logger.info(f"Maintaining full historical logs (no data expiration)")
        
        cycle_count = 0
        
        while not self.shutdown_requested:
            cycle_start = time.time()
            cycle_count += 1
            
            try:
                self.logger.info(f"Starting collection cycle {cycle_count}...")
                results = self.collect_batch_optimized(station_list, max_workers)
                
                # Calculate statistics
                total_collections = sum(len(records) for records in results.values())
                successful = sum(1 for records in results.values() 
                               for record in records if record.status == "success")
                new_records = sum(record.records_appended for records in results.values() 
                                for record in records if record.status == "success")
                
                cycle_time = time.time() - cycle_start
                self.logger.info(f"Cycle {cycle_count} complete: {successful}/{total_collections} successful, "
                               f"{new_records} new records in {cycle_time:.1f}s")
                
                # Optional backup every 24 cycles (~12 hours at 30min intervals)
                if cycle_count % 24 == 0:
                    self.logger.info(f"Completed {cycle_count} cycles - logs growing continuously")
                
                # Wait for next cycle
                sleep_time = max(0, (interval_minutes * 60) - cycle_time)
                if sleep_time > 0 and not self.shutdown_requested:
                    self.logger.info(f"Sleeping for {sleep_time:.1f} seconds until next cycle...")
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in collection cycle {cycle_count}: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
        
        self.logger.info("Collection stopped gracefully")
    
    def get_data_summary(self) -> Dict:
        """Get summary of current data files"""
        current_dir = self.data_dir / "current"
        summary = {}
        
        for filepath in current_dir.glob("*.gz"):
            try:
                lines, first_ts, last_ts = self._load_existing_data(filepath)
                data_lines = [line for line in lines if line.strip() and not line.startswith('#')]
                
                # Parse filename: handle both old and new naming conventions
                stem = filepath.stem  # Remove .gz extension
                if '_directional_' in stem:
                    # New format: 46022_directional_alpha1
                    parts = stem.split('_directional_')
                    station_id = parts[0]
                    data_type = f"directional_{parts[1]}"
                elif '_raw_spectral' in stem:
                    # New format: 46022_raw_spectral  
                    station_id = stem.replace('_raw_spectral', '')
                    data_type = 'raw_spectral'
                else:
                    # Standard format: 46022_wave, 46022_spectral
                    parts = stem.split('_', 1)  # Split only on first underscore
                    station_id = parts[0]
                    data_type = parts[1] if len(parts) > 1 else 'unknown'
                
                if station_id not in summary:
                    summary[station_id] = {}
                
                # Calculate days of data
                days_of_data = (last_ts - first_ts).days if (first_ts and last_ts) else 0
                
                summary[station_id][data_type] = {
                    'records': len(data_lines),
                    'first_timestamp': first_ts.isoformat() if first_ts else None,
                    'last_timestamp': last_ts.isoformat() if last_ts else None,
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                    'days_of_data': days_of_data
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing {filepath}: {e}")
        
        return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized NDBC Data Collector")
    parser.add_argument("--config", default="enhanced_stations.json", 
                       help="Enhanced stations configuration file")
    parser.add_argument("--data-dir", default="data/buoy_data", 
                       help="Base directory for data storage")
    parser.add_argument("--stations", 
                       help="Comma-separated station IDs (default: use priority list)")
    parser.add_argument("--continuous", action="store_true", 
                       help="Run continuous collection")
    parser.add_argument("--interval", type=int, default=30, 
                       help="Collection interval in minutes (default: 30)")
    parser.add_argument("--max-workers", type=int, default=8, 
                       help="Maximum parallel workers (default: 8)")
    parser.add_argument("--summary", action="store_true",
                       help="Show current data summary")
    parser.add_argument("--archive", action="store_true",
                       help="Run data archival process")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize collector
    try:
        collector = OptimizedBuoyCollector(args.data_dir, args.config)
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        print(f"Failed to initialize collector: {e}")
        return 1
    
    # Handle different modes
    if args.summary:
        summary = collector.get_data_summary()
        print("\nData Summary:")
        print("=" * 80)
        for station_id, data_types in summary.items():
            print(f"\nStation {station_id}:")
            for data_type, stats in data_types.items():
                print(f"  {data_type:12}: {stats['records']:5} records, "
                      f"{stats['file_size_mb']:6.1f}MB, "
                      f"{stats['days_of_data']:2} days "
                      f"({stats['first_timestamp'][:10] if stats['first_timestamp'] else 'N/A'} to "
                      f"{stats['last_timestamp'][:10] if stats['last_timestamp'] else 'N/A'})")
        return 0
    
    if args.archive:
        collector.archive_old_data()
        return 0
    
    # Determine stations to collect
    if args.stations:
        station_list = [s.strip() for s in args.stations.split(",")]
    else:
        station_list = collector.get_priority_stations()
    
    # Run collection
    try:
        if args.continuous:
            collector.run_continuous_collection(args.interval, args.max_workers)
        else:
            results = collector.collect_batch_optimized(station_list)
            
            # Print results
            print(f"\nCollection Results:")
            print(f"{'Station':<8} {'Type':<12} {'Status':<12} {'New':<5} {'Total':<7} {'Size (KB)':<10}")
            print("-" * 70)
            
            for station_id, records in results.items():
                for record in records:
                    status_symbol = "✓" if record.status == "success" else "✗"
                    size_kb = record.file_size / 1024 if record.file_size else 0
                    print(f"{station_id:<8} {record.data_type:<12} {status_symbol} {record.status:<11} "
                          f"{record.records_appended:<5} {record.records_collected:<7} {size_kb:<10.1f}")
            
            # Summary stats
            total_new = sum(record.records_appended for records in results.values() 
                           for record in records if record.status == "success")
            successful_collections = sum(1 for records in results.values() 
                                       for record in records if record.status == "success")
            total_collections = sum(len(records) for records in results.values())
            
            print(f"\nSummary: {successful_collections}/{total_collections} successful, "
                  f"{total_new} new records added")
    
    except KeyboardInterrupt:
        print("Collection interrupted by user")
        return 0
    except Exception as e:
        print(f"Collection failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())