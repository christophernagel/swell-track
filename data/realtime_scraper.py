#!/usr/bin/env python3
import argparse
import os
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import csv

def build_station_url(station_id):
    """
    Build the standardized URL for a given station ID.
    For example: https://www.ndbc.noaa.gov/station_page.php?station=46232
    """
    return f"https://www.ndbc.noaa.gov/station_page.php?station={station_id}"

def scrape_previous_observations(station_id):
    """
    Fetches the page for station_id and locates the table within <section id="wavedata">
    having class="dataTable" and <caption>Previous observations</caption>.
    
    Returns:
      header: List of column names (from the thead).
      data_rows: List of rows (each a list of cell text), from the tbody.
    """
    url = build_station_url(station_id)
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching page for station {station_id}: {response.status_code}")
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    # 1) Locate the <section id="wavedata">.
    wave_section = soup.find("section", id="wavedata")
    if not wave_section:
        raise Exception(f"Could not find <section id='wavedata'> for station {station_id}")
    
    # 2) Within that section, find all tables with class="dataTable".
    data_tables = wave_section.find_all("table", class_="dataTable")
    if not data_tables:
        raise Exception(f"No <table class='dataTable'> found in <section id='wavedata'> for station {station_id}")
    
    # 3) Identify the table whose <caption> contains "Previous observations".
    target_table = None
    for table in data_tables:
        cap = table.find("caption")
        if cap and "Previous observations" in cap.get_text():
            target_table = table
            break
    
    if not target_table:
        raise Exception(f"Could not find a table with caption 'Previous observations' for station {station_id}")
    
    # 4) Parse the <thead> for column headers, and <tbody> for data rows.
    #    Adjust this logic if the table structure differs.
    thead = target_table.find("thead")
    tbody = target_table.find("tbody")
    if not thead or not tbody:
        raise Exception(f"Missing thead/tbody in the 'Previous observations' table for station {station_id}")
    
    # Extract header from thead.
    header_row = thead.find("tr")
    if not header_row:
        raise Exception(f"No <tr> in thead for station {station_id}")
    header_cells = header_row.find_all(["th", "td"])
    header = [cell.get_text(strip=True) for cell in header_cells]
    
    # Extract data rows from tbody.
    data_rows = []
    for tr in tbody.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in tr.find_all(["th", "td"])]
        if cells:
            data_rows.append(cells)
    
    return header, data_rows

def store_station_record(station_id, header, data_rows, base_output_dir="data/records/realtime"):
    """
    Stores the scraped realtime data into a CSV file under a dynamic directory structure.
    The file is stored in: <base_output_dir>/<year>/<month>/<day>/<station_id>_<YYYYMMDD_HHMM>.csv
    """
    # Get current date components.
    now = datetime.now()
    year_str = str(now.year)
    month_str = f"{now.month:02d}"
    day_str = f"{now.day:02d}"
    
    # Build the dynamic directory path.
    output_dir = os.path.join(base_output_dir, year_str, month_str, day_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the filename using station_id and current timestamp.
    timestamp_str = now.strftime("%Y%m%d_%H%M")
    filename = f"{station_id}_{timestamp_str}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Write the CSV file.
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data_rows:
            writer.writerow(row)
    print(f"Saved station {station_id} realtime record to {filepath}")

def process_stations(station_ids):
    """
    For each station, scrape the 'Previous observations' table within
    <section id="wavedata"> and store the data.
    """
    for sid in station_ids:
        try:
            print(f"Processing station {sid}...")
            header, data_rows = scrape_previous_observations(sid)
            store_station_record(sid, header, data_rows)
        except Exception as e:
            print(f"Error processing station {sid}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Scrape 'Previous observations' table from NDBC for a list of stations."
    )
    parser.add_argument(
        "--stations",
        required=True,
        help="Comma-separated list of station IDs (e.g., 46232,46025)"
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="If specified, the scraper will run every 23.5 hours."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=23.5,
        help="Polling interval in hours (default 23.5 hours)."
    )
    args = parser.parse_args()
    
    station_ids = [s.strip() for s in args.stations.split(",") if s.strip()]
    
    if args.repeat:
        while True:
            print(f"\n{datetime.now()}: Starting scrape cycle...")
            process_stations(station_ids)
            print(f"{datetime.now()}: Cycle complete. Sleeping for {args.interval} hours...\n")
            time.sleep(args.interval * 3600)
    else:
        process_stations(station_ids)

if __name__ == "__main__":
    main()
