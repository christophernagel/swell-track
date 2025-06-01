#!/usr/bin/env python3
"""
spectral_debug_tool.py
---------------------
Debug tool to examine spectral file contents and identify parsing issues
"""

import gzip
from pathlib import Path
import argparse

def examine_spectral_file(filepath: Path, max_lines: int = 10):
    """Examine the contents of a spectral file"""
    print(f"\n=== Examining {filepath.name} ===")
    
    try:
        with gzip.open(filepath, 'rt') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        print(f"Total lines: {len(lines)}")
        print(f"File size: {filepath.stat().st_size} bytes")
        
        print(f"\nFirst {max_lines} lines:")
        for i, line in enumerate(lines[:max_lines]):
            print(f"{i+1:2d}: {repr(line)}")
        
        # Check for data lines (non-comment, non-empty)
        data_lines = [line for line in lines if line.strip() and not line.startswith('#')]
        print(f"\nData lines found: {len(data_lines)}")
        
        if data_lines:
            print(f"\nFirst data line: {repr(data_lines[0])}")
            parts = data_lines[0].split()
            print(f"Parts in first data line: {len(parts)}")
            print(f"First 10 parts: {parts[:10]}")
            
            # Try to parse timestamp
            if len(parts) >= 5:
                try:
                    year, month, day, hour, minute = map(int, parts[:5])
                    print(f"Parsed timestamp: {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}")
                    
                    # Check spectral values
                    spectral_parts = parts[5:]
                    print(f"Spectral values: {len(spectral_parts)} values")
                    if spectral_parts:
                        print(f"First 5 spectral values: {spectral_parts[:5]}")
                        print(f"Last 5 spectral values: {spectral_parts[-5:]}")
                        
                        # Count non-missing values
                        valid_count = sum(1 for val in spectral_parts 
                                        if val not in ['MM', '999.0', '-999.0'])
                        print(f"Non-missing spectral values: {valid_count}/{len(spectral_parts)}")
                        
                except ValueError as e:
                    print(f"Error parsing timestamp: {e}")
            else:
                print("Not enough parts for timestamp parsing")
        
        # Look for any error indicators
        error_indicators = ['Error', 'ERROR', 'error', 'Not Found', '404']
        for indicator in error_indicators:
            if indicator in content:
                print(f"\n⚠️  WARNING: Found '{indicator}' in file content!")
                break
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def examine_wave_file(filepath: Path, max_lines: int = 10):
    """Examine the contents of a wave file for comparison"""
    print(f"\n=== Examining {filepath.name} (wave file for comparison) ===")
    
    try:
        with gzip.open(filepath, 'rt') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        print(f"Total lines: {len(lines)}")
        
        print(f"\nFirst {max_lines} lines:")
        for i, line in enumerate(lines[:max_lines]):
            print(f"{i+1:2d}: {repr(line)}")
        
        # Find header line
        header_candidates = []
        for i, line in enumerate(lines[:5]):
            if any(col in line.upper() for col in ['WVHT', 'DPD', 'APD', 'MWD']):
                header_candidates.append((i, line.replace('#', '').split()))
        
        if header_candidates:
            header_idx, headers = header_candidates[0]
            print(f"\nFound headers at line {header_idx + 1}: {headers}")
        
        # Check data lines
        data_lines = [line for line in lines if not line.startswith('#') and line.strip()]
        print(f"Data lines found: {len(data_lines)}")
        
        if data_lines:
            print(f"First data line: {repr(data_lines[0])}")
            
    except Exception as e:
        print(f"❌ Error reading wave file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Debug spectral data files")
    parser.add_argument("--data-dir", default="data/buoy_data", help="Data directory")
    parser.add_argument("--station", default="46022", help="Station to examine")
    parser.add_argument("--max-lines", type=int, default=15, help="Max lines to show")
    
    args = parser.parse_args()
    
    raw_dir = Path(args.data_dir) / "raw"
    
    if not raw_dir.exists():
        print(f"❌ Raw data directory not found: {raw_dir}")
        return
    
    # Find files for the station
    spectral_files = list(raw_dir.glob(f"{args.station}_spectral_*.gz"))
    wave_files = list(raw_dir.glob(f"{args.station}_wave_*.gz"))
    
    print(f"Found {len(spectral_files)} spectral files and {len(wave_files)} wave files for station {args.station}")
    
    if not spectral_files:
        print(f"❌ No spectral files found for station {args.station}")
        available_stations = set()
        for f in raw_dir.glob("*_spectral_*.gz"):
            available_stations.add(f.name.split('_')[0])
        if available_stations:
            print(f"Available stations with spectral data: {sorted(available_stations)}")
        return
    
    # Examine first spectral file
    examine_spectral_file(spectral_files[0], args.max_lines)
    
    # Compare with wave file if available
    if wave_files:
        examine_wave_file(wave_files[0], args.max_lines)
    
    # Check if spectral files are actually empty or contain error messages
    print(f"\n=== File Size Analysis ===")
    for filepath in spectral_files[:3]:
        size = filepath.stat().st_size
        print(f"{filepath.name}: {size} bytes")
        
        if size < 100:  # Very small files are suspicious
            print(f"⚠️  {filepath.name} is very small ({size} bytes) - might be an error response")

if __name__ == "__main__":
    main()