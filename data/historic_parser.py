#!/usr/bin/env python3
import argparse
import os
import re
import gzip
from datetime import datetime
import numpy as np
from urllib.request import urlopen
from urllib.parse import urlparse, parse_qs
import json

# Import helper functions from your modules.
from swell_tracker.data.xml_parser import extract_station_id_from_filename  # adjust as needed
from swell_tracker.core import sinusoidal_positional_encoding

def build_ndbc_url(station_id, year, directory="data/historical/stdmet/"):
    """
    Builds a standardized NDBC URL for a given station id and year.
    Example output:
      https://www.ndbc.noaa.gov/view_text_file.php?filename=46232h2024.txt.gz&dir=data/historical/stdmet/
    """
    return f"https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h{year}.txt.gz&dir={directory}"

def read_and_encode_historic_measurements(link, station_dict, features=None, add_positional_encoding=True):
    """
    Reads a historic measurement file from a URL and returns a dictionary with:
      - station_id and metadata (from station_dict)
      - a list of timestamps
      - raw measurements as a numpy array (shape: [num_records, num_features])
      - measurements with positional encoding applied (if add_positional_encoding is True)
    
    The URL is expected to include a query parameter "filename" (e.g.,
      https://www.ndbc.noaa.gov/view_text_file.php?filename=46232h2024.txt.gz&dir=...).
    If the file is gzip-compressed, it will be decompressed.
    """
    # Set default features if not provided.
    if features is None:
        features = ["WVHT", "DPD", "APD"]
    
    # Mapping of feature names to column indices.
    col_mapping = {
        "WVHT": 8,
        "DPD": 9,
        "APD": 10,
    }
    
    # Parse the URL to extract the filename.
    parsed_url = urlparse(link)
    qs = parse_qs(parsed_url.query)
    if "filename" in qs:
        filename = qs["filename"][0]
    else:
        filename = parsed_url.path.split("/")[-1]
    
    # Determine if the file is gzipped.
    gzipped = False
    if filename.endswith(".gz"):
        gzipped = True
        filename = filename[:-3]  # Remove .gz for station id extraction.
    
    # Extract the station id using the helper.
    station_id = extract_station_id_from_filename(filename)
    if station_id not in station_dict:
        raise ValueError(f"Station id {station_id} not found in station dictionary.")
    station_metadata = station_dict[station_id]
    
    timestamps = []
    measurements = []
    
    # Fetch file content from the URL.
    try:
        with urlopen(link) as response:
            content = response.read()
    except Exception as e:
        raise RuntimeError(f"Error fetching file from URL: {e}")
    
    # If the file is expected to be gzipped, attempt to decompress.
    if gzipped:
        try:
            content = gzip.decompress(content)
        except OSError as e:
            if "Not a gzipped file" in str(e):
                print("Warning: File is not gzipped; using raw content.")
                # Leave content as-is.
            else:
                raise RuntimeError(f"Error decompressing gzip content: {e}")
    
    # Decode the content into a string.
    try:
        content = content.decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error decoding content: {e}")
    
    # Process each line in the file.
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        # Parse the timestamp from the first five columns (YY, MM, DD, hh, mm).
        try:
            year_val = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            timestamp = datetime(year_val, month, day, hour, minute)
        except Exception as e:
            print(f"Skipping line due to timestamp parsing error: {line}\nError: {e}")
            continue
        
        record = []
        for feat in features:
            try:
                col_index = col_mapping[feat]
                value = float(parts[col_index])
                # Filter out unreliable placeholder values.
                if value in [99.0, 999.0, 9999.0]:
                    value = np.nan
            except (KeyError, ValueError):
                value = np.nan
            record.append(value)
        
        timestamps.append(timestamp)
        measurements.append(record)
    
    measurements = np.array(measurements, dtype=np.float32)
    
    # Optionally add sinusoidal positional encoding.
    if add_positional_encoding:
        seq_len, d_model = measurements.shape
        pos_enc = sinusoidal_positional_encoding(seq_len, d_model)
        measurements_encoded = measurements + pos_enc
    else:
        measurements_encoded = measurements
    
    return {
        "station_id": station_id,
        "station_metadata": station_metadata,
        "timestamps": timestamps,
        "raw_measurements": measurements,
        "encoded_measurements": measurements_encoded
    }

def store_station_record(result, year, output_dir="data/records/historic"):
    """
    Stores a station's historic measurement record as a CSV-like text file.
    The file is named using the convention "{station_id},{year}record.txt" and stored under output_dir.
    
    The file will include a header row and each subsequent row will contain the ISO timestamp and the raw measurement values.
    """
    station_id = result["station_id"]
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{station_id},{year}record.txt"
    filepath = os.path.join(output_dir, filename)
    
    header = "Timestamp,WVHT,DPD,APD\n"
    with open(filepath, "w") as f:
        f.write(header)
        for ts, row in zip(result["timestamps"], result["raw_measurements"]):
            ts_str = ts.isoformat()
            row_str = ",".join(f"{val:.2f}" for val in row)
            f.write(f"{ts_str},{row_str}\n")
    
    print(f"Historic record saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(
        description="Fetch, process, and store NDBC historic measurement files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--link",
        help="Full URL to a historic measurement file (e.g., ...?filename=46232h2024.txt.gz&dir=...)."
    )
    group.add_argument(
        "--stations",
        help="Comma-separated list of station IDs. The script will build the URL(s) for a given year."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now().year,
        help="Year for which to fetch historic records (default: current year)."
    )
    parser.add_argument(
        "--output",
        help="Optional output file to write the combined results as JSON."
    )
    args = parser.parse_args()
    
    try:
        with open("stations.json", "r") as f:
            station_dict = json.load(f)
    except Exception as e:
        parser.error(f"Error loading station dictionary: {e}")
    
    results = {}
    
    if args.link:
        try:
            result = read_and_encode_historic_measurements(args.link, station_dict)
            results[result["station_id"]] = result
            store_station_record(result, args.year)
        except Exception as e:
            print(f"Error processing station: {e}")
    else:
        station_ids = [s.strip() for s in args.stations.split(",") if s.strip()]
        for sid in station_ids:
            url = build_ndbc_url(sid, args.year)
            try:
                result = read_and_encode_historic_measurements(url, station_dict)
                results[sid] = result
                store_station_record(result, args.year)
            except Exception as e:
                print(f"Error processing station {sid}: {e}")
    
    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(results, f, default=str, indent=2)
            print(f"\nCombined results written to {args.output}")
        except Exception as e:
            print(f"Error writing output: {e}")

if __name__ == "__main__":
    main()
