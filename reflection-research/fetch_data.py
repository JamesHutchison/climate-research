from pathlib import Path
import requests
import os
from datetime import datetime, date

# example URLs:
# newest file - https://gml.noaa.gov/aftp/data/radiation/baseline/brw/2025/brw25110.dat
# oldest file - https://gml.noaa.gov/aftp/data/radiation/baseline/brw/1998/brw98001.dat
# BRW = barrow
# 2025 = year
# 25 = year (again)
# 110 = day of year

# leap years have 366 files, non-leap years have 365 files

BASE_URL = "https://gml.noaa.gov/aftp/data/radiation/baseline"
LOCATIONS = ["smo"]  # ["brw", "mlo", "spo"]
START_YEAR = 1998
END_YEAR = datetime.now().year

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir(Path(__file__).parent)


def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_file(url, save_path):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"Skipping existing file: {save_path}")
        return

    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def main():
    current_year = datetime.now().year
    current_doy = date.today().timetuple().tm_yday

    for location in LOCATIONS:
        for year in range(START_YEAR, END_YEAR + 1):
            days_in_year = 366 if is_leap_year(year) else 365
            year_short = str(year)[-2:]

            # Create directory structure
            year_dir = f"{location}/{year}"
            ensure_dir(year_dir)

            # Calculate max day to download
            max_day = current_doy if year == current_year else days_in_year

            for day in range(1, max_day + 1):
                url = f"{BASE_URL}/{location}/{year}/{location}{year_short}{day:03d}.dat"
                save_path = f"{year_dir}/{location}{year_short}{day:03d}.dat"
                download_file(url, save_path)


if __name__ == "__main__":
    main()
