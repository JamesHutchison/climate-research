import os
from pathlib import Path
import numpy as np
import pandas as pd

from glob import glob

current_dir = Path(__file__).parent
os.chdir(current_dir)


def parse_noaa_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        return pd.DataFrame()

    station_line = lines[0].strip()
    location_line = lines[1].strip()

    column_names = [
        'year', 'jday', 'month', 'day', 'hour', 'min', 'dt', 'zen',
        'dw_solar', 'uw_solar', 'direct_n', 'diffuse',
        'dw_ir', 'dw_casetemp', 'dw_dometemp', 'uw_ir', 'uw_casetemp', 'uw_dometemp',
        'uvb', 'par', 'netsolar', 'netir', 'totalnet', 'temp', 'rh', 'windspd',
        'winddir', 'pressure'
    ]

    data = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < len(column_names):
            continue
        row = parts[:len(column_names)]
        data.append(row)

    df = pd.DataFrame(data, columns=column_names)
    df = df.astype({
        'year': int, 'jday': int, 'month': int, 'day': int,
        'dw_solar': float, 'uw_solar': float, 'direct_n': float
    })

    return df[['year', 'month', 'day', 'dw_solar', 'uw_solar', 'direct_n']]


def compute_daily_percentiles(df):
    grouped = df.groupby(['year', 'month', 'day'])
    result = []

    for (year, month, day), group in grouped:
        result.append({
            'year': year,
            'month': month,
            'day': day,
            'dw_solar_50': np.percentile(group['dw_solar'], 50),
            'dw_solar_90': np.percentile(group['dw_solar'], 90),
            'dw_solar_99': np.percentile(group['dw_solar'], 99),
            'uw_solar_50': np.percentile(group['uw_solar'], 50),
            'uw_solar_90': np.percentile(group['uw_solar'], 90),
            'uw_solar_99': np.percentile(group['uw_solar'], 99),
            'direct_n_50': np.percentile(group['direct_n'], 50),
            'direct_n_90': np.percentile(group['direct_n'], 90),
            'direct_n_99': np.percentile(group['direct_n'], 99),
        })

    return pd.DataFrame(result)


def compute_daily_sums(df):
    # Filter values > -10 for summing
    df_filtered = df.copy()
    for col in ['dw_solar', 'uw_solar', 'direct_n']:
        df_filtered.loc[df_filtered[col] <= -10, col] = 0

    grouped = df_filtered.groupby(['year', 'month', 'day'])
    result = grouped.agg({
        'dw_solar': 'sum',
        'uw_solar': 'sum',
        'direct_n': 'sum'
    }).reset_index()

    # Rename columns to indicate they are sums
    result.columns = ['year', 'month', 'day',
                      'dw_solar_sum', 'uw_solar_sum', 'direct_n_sum']
    return result


def process_directory(base_dir, year):
    all_files = glob(os.path.join(
        base_dir, str(year), '**/*.dat'), recursive=True)
    results_percentiles = []
    results_sums = []

    for filepath in all_files:
        df = parse_noaa_file(filepath)
        if df.empty:
            continue
        daily_stats = compute_daily_percentiles(df)
        daily_sums = compute_daily_sums(df)
        results_percentiles.append(daily_stats)
        results_sums.append(daily_sums)

    if results_percentiles:
        percentiles_df = pd.concat(results_percentiles, ignore_index=True)
        sums_df = pd.concat(results_sums, ignore_index=True)
        # Merge the percentiles and sums on date columns
        return pd.merge(percentiles_df, sums_df,
                        on=['year', 'month', 'day'],
                        how='outer')
    else:
        print(f'No valid data found for year {year}.')
        return pd.DataFrame()


def main():
    stations = ['brw']  # [ 'mlo', 'spo']
    for station in stations:
        base_dir = current_dir / station
        all_results = []

        for year in range(1998, 2025):
            print(f'Processing year {year}...')
            year_df = process_directory(base_dir, year)
            if not year_df.empty:
                all_results.append(year_df)

        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            # Sort by date (year, month, day)
            final_df = final_df.sort_values(['year', 'month', 'day'])
            filename = f'output_{station}.csv'
            final_df.to_csv(filename, index=False)
            print(f'Output written to {filename}')
        else:
            print('No valid data found for any year.')


if __name__ == '__main__':
    main()
