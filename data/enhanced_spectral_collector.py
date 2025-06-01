#!/usr/bin/env python3
"""
Enhanced Spectral Data Collector
Collects multiple types of NDBC data including raw spectral density
"""

from pathlib import Path
import requests
from datetime import datetime
import gzip
import logging

logger = logging.getLogger(__name__)

class EnhancedSpectralCollector:
    """Collect multiple types of NDBC spectral data"""
    
    def __init__(self, data_dir: str = "data/buoy_data"):
        self.data_dir = Path(data_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SciWaveBot/1.0 (Wave Research)'
        })
        
        # Define data types and their URLs
        self.data_types = {
            'wave': '.txt',           # Standard meteorological data
            'wave_summary': '.spec',  # Processed wave parameters (what you have now)
            'raw_spectral': '.data_spec',  # Raw spectral density (what you want!)
            'directional': '.swdir',  # Directional spectral data
            'directional2': '.swdir2', # Alternative directional format
            'alpha1': '.swdir',       # Directional parameters (alpha1)
            'alpha2': '.swdir2',      # Directional parameters (alpha2)
            'r1': '.swdir',           # Directional spread (r1)
            'r2': '.swdir2'           # Directional spread (r2)
        }
    
    def test_data_availability(self, station_id: str) -> dict:
        """Test which data types are available for a station"""
        availability = {}
        
        for data_type, extension in self.data_types.items():
            url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}{extension}"
            
            try:
                response = self.session.head(url, timeout=10)
                availability[data_type] = {
                    'available': response.status_code == 200,
                    'url': url,
                    'status_code': response.status_code
                }
            except Exception as e:
                availability[data_type] = {
                    'available': False,
                    'url': url,
                    'error': str(e)
                }
        
        return availability
    
    def collect_raw_spectral_data(self, station_id: str) -> dict:
        """Collect raw spectral density data (.data_spec)"""
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.data_spec"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            content = response.text.strip()
            if not content or len(content) < 100:
                return {'status': 'empty', 'records': 0}
            
            # Save raw spectral data
            timestamp = datetime.now()
            filename = f"{station_id}_raw_spectral_{timestamp.strftime('%Y%m%d_%H%M')}.gz"
            filepath = self.data_dir / "raw" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with gzip.open(filepath, 'wt') as f:
                f.write(content)
            
            # Count data lines
            lines = [line for line in content.split('\n') 
                    if line.strip() and not line.startswith('#')]
            
            return {
                'status': 'success',
                'records': len(lines),
                'file_size': filepath.stat().st_size,
                'filepath': filepath
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def collect_directional_data(self, station_id: str) -> dict:
        """Collect directional spectral data (.swdir)"""
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.swdir"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            content = response.text.strip()
            if not content or len(content) < 100:
                return {'status': 'empty', 'records': 0}
            
            # Save directional data
            timestamp = datetime.now()
            filename = f"{station_id}_directional_{timestamp.strftime('%Y%m%d_%H%M')}.gz"
            filepath = self.data_dir / "raw" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with gzip.open(filepath, 'wt') as f:
                f.write(content)
            
            lines = [line for line in content.split('\n') 
                    if line.strip() and not line.startswith('#')]
            
            return {
                'status': 'success', 
                'records': len(lines),
                'file_size': filepath.stat().st_size,
                'filepath': filepath
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

def test_station_spectral_capabilities(station_ids: list):
    """Test spectral data availability for multiple stations"""
    collector = EnhancedSpectralCollector()
    
    print("Testing Raw Spectral Data Availability")
    print("=" * 60)
    print(f"{'Station':<8} {'Wave':<6} {'Summary':<8} {'Raw Spec':<9} {'Direction':<10}")
    print("-" * 60)
    
    results = {}
    
    for station_id in station_ids:
        availability = collector.test_data_availability(station_id)
        
        # Simple status indicators
        wave = "✓" if availability['wave']['available'] else "✗"
        summary = "✓" if availability['wave_summary']['available'] else "✗"
        raw_spec = "✓" if availability['raw_spectral']['available'] else "✗"
        direction = "✓" if availability['directional']['available'] else "✗"
        
        print(f"{station_id:<8} {wave:<6} {summary:<8} {raw_spec:<9} {direction:<10}")
        
        results[station_id] = availability
    
    return results

def collect_raw_spectral_sample(station_ids: list):
    """Collect sample raw spectral data to examine format"""
    collector = EnhancedSpectralCollector()
    
    print(f"\nCollecting Raw Spectral Samples")
    print("=" * 40)
    
    for station_id in station_ids[:5]:  # Test first 5 stations
        print(f"\nTesting {station_id}:")
        
        # Test raw spectral
        raw_result = collector.collect_raw_spectral_data(station_id)
        print(f"  Raw spectral: {raw_result['status']}")
        if raw_result['status'] == 'success':
            print(f"    Records: {raw_result['records']}")
            print(f"    File size: {raw_result['file_size']} bytes")
        
        # Test directional
        dir_result = collector.collect_directional_data(station_id)
        print(f"  Directional: {dir_result['status']}")
        if dir_result['status'] == 'success':
            print(f"    Records: {dir_result['records']}")
            print(f"    File size: {dir_result['file_size']} bytes")

if __name__ == "__main__":
    # Test with your priority stations
    priority_stations = [
        '46022', '46026', '46027', '46054', '46013', '46014',
        '46059', '46063', '46042', '46086', '46221', '46224',
        '51001', '51002', '51101', '51201', '44013', '44014'
    ]
    
    # Test availability
    results = test_station_spectral_capabilities(priority_stations)
    
    # Collect samples
    collect_raw_spectral_sample(priority_stations)
    
    # Summary
    raw_spectral_available = [
        station for station, data in results.items() 
        if data['raw_spectral']['available']
    ]
    
    print(f"\n" + "=" * 60)
    print(f"Summary: {len(raw_spectral_available)}/{len(priority_stations)} stations have raw spectral data")
    
    if raw_spectral_available:
        print(f"Stations with raw spectral data:")
        for station in raw_spectral_available:
            print(f"  {station}")
        
        print(f"\nTo collect raw spectral data, update your collector URLs to:")
        print(f"  Raw spectral: https://www.ndbc.noaa.gov/data/realtime2/{{station_id}}.data_spec")
        print(f"  Directional: https://www.ndbc.noaa.gov/data/realtime2/{{station_id}}.swdir")
    else:
        print("No stations have raw spectral data available.")
        print("Your current wave summary data is still valuable for forecasting!")