#!/usr/bin/env python3
"""
Production NDBC Data Collector
High-performance, fault-tolerant data collection system for NDBC buoy stations
"""

import os
import sys
import json
import time
import gzip
import shutil
import signal
import logging
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import sqlite3
from contextlib import contextmanager
import hashlib

# Configure logging
def setup_logging(log_dir: str = "logs"):
    """Setup comprehensive logging"""
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(f"{log_dir}/collector.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Error file handler
    error_handler = logging.FileHandler(f"{log_dir}/errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(error_handler)
    
    return root_logger

@dataclass
class CollectionRecord:
    """Record of a data collection attempt"""
    station_id: str
    timestamp: datetime
    data_type: str
    status: str
    records_collected: int
    file_size: int
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    data_hash: Optional[str] = None

class CollectionDatabase:
    """SQLite database for tracking collection metadata"""
    
    def __init__(self, db_path: str = "data/collection_metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    records_collected INTEGER,
                    file_size INTEGER,
                    error_message TEXT,
                    response_time REAL,
                    data_hash TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_station_timestamp 
                ON collections(station_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_created 
                ON collections(status, created_at)
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def record_collection(self, record: CollectionRecord):
        """Record a collection attempt"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO collections 
                (station_id, timestamp, data_type, status, records_collected, 
                 file_size, error_message, response_time, data_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.station_id,
                record.timestamp.isoformat(),
                record.data_type,
                record.status,
                record.records_collected,
                record.file_size,
                record.error_message,
                record.response_time,
                record.data_hash
            ))
    
    def get_last_collection(self, station_id: str, data_type: str) -> Optional[datetime]:
        """Get timestamp of last successful collection"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT timestamp FROM collections 
                WHERE station_id = ? AND data_type = ? AND status = 'success'
                ORDER BY timestamp DESC LIMIT 1
            """, (station_id, data_type))
            
            result = cursor.fetchone()
            return datetime.fromisoformat(result[0]) if result else None
    
    def get_statistics(self, hours: int = 24) -> Dict:
        """Get collection statistics for last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    status,
                    data_type,
                    COUNT(*) as count,
                    AVG(response_time) as avg_response_time,
                    SUM(records_collected) as total_records
                FROM collections 
                WHERE datetime(created_at) > datetime(?)
                GROUP BY status, data_type
            """, (cutoff.isoformat(),))
            
            return {
                f"{row[1]}_{row[0]}": {
                    'count': row[2],
                    'avg_response_time': row[3],
                    'total_records': row[4]
                }
                for row in cursor.fetchall()
            }

class ProductionBuoyCollector:
    """Production-grade NDBC data collector"""
    
    def __init__(self, data_dir: str = "data/buoy_data", config_file: str = "enhanced_stations.json"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "archive").mkdir(exist_ok=True)
        
        # Load station configuration
        with open(config_file, 'r') as f:
            self.stations = json.load(f)
        
        self.db = CollectionDatabase()
        self.logger = logging.getLogger(__name__)
        
        # Collection settings
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SciWaveBot/1.0 (Wave Research; +https://github.com/yourrepo/wave-forecasting)'
        })
        
        # Graceful shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Shutdown signal {signum} received, finishing current operations...")
        self.shutdown_requested = True
    
    def _calculate_file_hash(self, content: str) -> str:
        """Calculate SHA256 hash of content for deduplication"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def collect_station_data(self, station_id: str, data_type: str) -> CollectionRecord:
        """Collect data for a single station"""
        start_time = time.time()
        
        # Determine URL based on data type
        url_map = {
            "wave": f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt",
            "spectral": f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec",
            "raw_spectral": f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.data_spec",
            "directional": f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swdir"
        }
        
        url = url_map.get(data_type)
        if not url:
            raise ValueError(f"Unknown data type: {data_type}")
        
        try:
            # Make request
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            response_time = time.time() - start_time
            content = response.text.strip()
            
            if not content or len(content) < 100:
                return CollectionRecord(
                    station_id=station_id,
                    timestamp=datetime.now(),
                    data_type=data_type,
                    status="empty",
                    records_collected=0,
                    file_size=0,
                    response_time=response_time
                )
            
            # Calculate hash for deduplication
            content_hash = self._calculate_file_hash(content)
            
            # Count records (lines that don't start with # and aren't empty)
            lines = [line for line in content.split('\n') if line.strip() and not line.startswith('#')]
            records_count = len(lines)
            
            # Save raw data
            timestamp = datetime.now()
            filename = self._get_filename(station_id, data_type, timestamp)
            filepath = self.data_dir / "raw" / filename
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Compress and save (add .gz extension here)
            compressed_filepath = filepath.with_suffix('.gz')
            with gzip.open(compressed_filepath, 'wt') as f:
                f.write(content)
            
            file_size = compressed_filepath.stat().st_size
            
            self.logger.debug(f"Collected {records_count} records for {station_id} ({data_type})")
            
            return CollectionRecord(
                station_id=station_id,
                timestamp=timestamp,
                data_type=data_type,
                status="success",
                records_collected=records_count,
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
                file_size=0,
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    def _get_filename(self, station_id: str, data_type: str, timestamp: datetime) -> str:
        """Generate filename for raw data (without extension)"""
        date_str = timestamp.strftime("%Y%m%d")
        time_str = timestamp.strftime("%H%M")
        return f"{station_id}_{data_type}_{date_str}_{time_str}"
    
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
    
    def collect_batch(self, station_ids: List[str], max_workers: int = 8) -> Dict[str, List[CollectionRecord]]:
        """Collect data for multiple stations in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all collection tasks
            futures = {}
            
            for station_id in station_ids:
                if self.shutdown_requested:
                    break
                
                # Collect multiple data types for stations with raw spectral capability
                if self.stations.get(station_id, {}).get('has_spectral', False):
                    # Collect raw spectral density (the real treasure!)
                    futures[executor.submit(self.collect_station_data, station_id, "raw_spectral")] = (station_id, "raw_spectral")
                    # Collect directional spectral data  
                    futures[executor.submit(self.collect_station_data, station_id, "directional")] = (station_id, "directional")
                    # Still collect summary for comparison
                    futures[executor.submit(self.collect_station_data, station_id, "spectral")] = (station_id, "spectral")
                
                # Always collect standard wave data
                futures[executor.submit(self.collect_station_data, station_id, "wave")] = (station_id, "wave")
            
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
                    
                    # Record in database
                    self.db.record_collection(record)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {station_id} ({data_type}): {e}")
        
        return results
    
    def run_continuous_collection(self, interval_minutes: int = 60, max_workers: int = 8):
        """Run continuous data collection"""
        station_list = self.get_priority_stations()
        self.logger.info(f"Starting continuous collection for {len(station_list)} stations")
        self.logger.info(f"Collection interval: {interval_minutes} minutes")
        self.logger.info(f"Priority stations: {', '.join(station_list[:10])}{'...' if len(station_list) > 10 else ''}")
        
        while not self.shutdown_requested:
            cycle_start = time.time()
            
            try:
                self.logger.info("Starting collection cycle...")
                results = self.collect_batch(station_list, max_workers)
                
                # Log cycle summary
                total_collections = sum(len(records) for records in results.values())
                successful = sum(1 for records in results.values() 
                               for record in records if record.status == "success")
                
                cycle_time = time.time() - cycle_start
                self.logger.info(f"Cycle complete: {successful}/{total_collections} successful in {cycle_time:.1f}s")
                
                # Print detailed stats every few cycles
                if datetime.now().minute % 30 == 0:  # Every 30 minutes
                    stats = self.db.get_statistics(hours=6)
                    self.logger.info(f"6-hour statistics: {stats}")
                
                # Wait for next cycle
                sleep_time = max(0, (interval_minutes * 60) - cycle_time)
                if sleep_time > 0 and not self.shutdown_requested:
                    self.logger.info(f"Sleeping for {sleep_time:.1f} seconds until next cycle...")
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in collection cycle: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
        
        self.logger.info("Collection stopped gracefully")
    
    def run_single_collection(self, station_ids: Optional[List[str]] = None):
        """Run a single collection cycle"""
        if station_ids is None:
            station_ids = self.get_priority_stations()
        
        self.logger.info(f"Running single collection for {len(station_ids)} stations")
        results = self.collect_batch(station_ids)
        
        # Print results
        print(f"\nCollection Results:")
        print(f"{'Station':<8} {'Type':<8} {'Status':<12} {'Records':<8} {'Size (KB)':<10} {'Time (s)':<8}")
        print("-" * 70)
        
        for station_id, records in results.items():
            for record in records:
                status_symbol = "✓" if record.status == "success" else "✗"
                size_kb = record.file_size / 1024 if record.file_size else 0
                print(f"{station_id:<8} {record.data_type:<8} {status_symbol} {record.status:<11} "
                      f"{record.records_collected:<8} {size_kb:<10.1f} {record.response_time or 0:<8.2f}")
        
        # Summary stats
        total_records = sum(r.records_collected for records in results.values() for r in records)
        successful_collections = sum(1 for records in results.values() 
                                   for r in records if r.status == "success")
        total_collections = sum(len(records) for records in results.values())
        
        print(f"\nSummary: {successful_collections}/{total_collections} successful, "
              f"{total_records} total records collected")

def main():
    parser = argparse.ArgumentParser(description="Production NDBC Data Collector")
    parser.add_argument("--config", default="enhanced_stations.json", 
                       help="Enhanced stations configuration file")
    parser.add_argument("--data-dir", default="data/buoy_data", 
                       help="Base directory for data storage")
    parser.add_argument("--stations", 
                       help="Comma-separated station IDs (default: use priority list)")
    parser.add_argument("--continuous", action="store_true", 
                       help="Run continuous collection")
    parser.add_argument("--interval", type=int, default=60, 
                       help="Collection interval in minutes (default: 60)")
    parser.add_argument("--max-workers", type=int, default=8, 
                       help="Maximum parallel workers (default: 8)")
    parser.add_argument("--log-dir", default="logs", 
                       help="Directory for log files")
    parser.add_argument("--quiet", action="store_true", 
                       help="Reduce console output")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    if args.quiet:
        logging.getLogger().handlers[1].setLevel(logging.WARNING)
    
    # Initialize collector
    try:
        collector = ProductionBuoyCollector(args.data_dir, args.config)
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        logger.info("Run the station enhancer first: python station_enhancer.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize collector: {e}")
        sys.exit(1)
    
    # Determine stations to collect
    if args.stations:
        station_list = [s.strip() for s in args.stations.split(",")]
        logger.info(f"Using specified stations: {station_list}")
    else:
        station_list = collector.get_priority_stations()
        logger.info(f"Using {len(station_list)} priority stations")
    
    # Run collection
    try:
        if args.continuous:
            collector.run_continuous_collection(args.interval, args.max_workers)
        else:
            collector.run_single_collection(station_list)
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()