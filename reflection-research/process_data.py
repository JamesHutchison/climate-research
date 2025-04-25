import os
from pathlib import Path
import numpy as np
import pandas as pd

from glob import glob

current_dir = Path(__file__).parent
os.chdir(current_dir)


def parse_noaa_file(filepath, is_polar: bool):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        return pd.DataFrame()

    station_line = lines[0].strip()
    location_line = lines[1].strip()

    column_names = [
        'year', 'jday', 'month', 'day', 'hour', 'min', 'dt', 'zen',
        'dw_solar', 'dw_solar_valid', 'uw_solar', 'uw_solar_valid', 'direct_n', 'direct_n_valid',
        'dw_ir', 'dw_ir_valid', 'dw_casetemp', 'dw_casetemp_valid', 'dw_dometemp', 'dw_dometemp_valid',
        'uw_ir', 'uw_ir_valid', 'uw_casetemp', 'uw_casetemp_valid', 'uw_dometemp', 'uw_dometemp_valid',
        'uvb', 'uvb_valid', 'par', 'par_valid', 'netsolar', 'netsolar_valid', 'netir', 'netir_valid',
        'totalnet', 'totalnet_valid', 'temp', 'temp_valid', 'rh', 'rh_valid', 'windspd', 'windspd_valid',
        'winddir', 'winddir_valid', 'pressure', 'pressure_valid'
    ]
    if not is_polar:
        column_names.remove("direct_n")
        column_names.remove("direct_n_valid")

    data = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < len(column_names):
            continue
        row = parts[:len(column_names)]
        data.append(row)

    df = pd.DataFrame(data, columns=column_names)
    cols_to_examine = {
        'year': int, 'jday': int, 'month': int, 'day': int,
        'dw_solar': float, 'uw_solar': float, 'direct_n': float
    }
    if not is_polar:
        cols_to_examine.pop('direct_n')
    df = df.astype(cols_to_examine)

    cols = list(cols_to_examine.keys())
    return df[cols]


def compute_daily_percentiles(df, is_polar: bool):
    grouped = df.groupby(['year', 'month', 'day'])
    result = []

    for (year, month, day), group in grouped:
        results = {
            'year': year,
            'month': month,
            'day': day,
            'dw_solar_50': np.percentile(group['dw_solar'], 50),
            'dw_solar_75': np.percentile(group['dw_solar'], 75),
            'dw_solar_90': np.percentile(group['dw_solar'], 90),
            'dw_solar_99': np.percentile(group['dw_solar'], 99),
            'uw_solar_50': np.percentile(group['uw_solar'], 50),
            'uw_solar_75': np.percentile(group['uw_solar'], 75),
            'uw_solar_90': np.percentile(group['uw_solar'], 90),
            'uw_solar_99': np.percentile(group['uw_solar'], 99),
        }
        if is_polar:
            results['direct_n_50'] = np.percentile(group['direct_n'], 50)
            results['direct_n_75'] = np.percentile(group['direct_n'], 75)
            results['direct_n_90'] = np.percentile(group['direct_n'], 90)
            results['direct_n_99'] = np.percentile(group['direct_n'], 99)
        result.append(results)

    return pd.DataFrame(result)


def compute_daily_sums(df, is_polar):
    # Filter values > -10 for summing
    df_filtered = df.copy()
    if is_polar:
        col_list = ['dw_solar', 'uw_solar', 'direct_n']
    else:
        col_list = ['dw_solar', 'uw_solar']
    for col in col_list:
        mask = df_filtered[col] < 1
        if mask.any():
            df_filtered.loc[mask, col] = 0

    grouped = df_filtered.groupby(['year', 'month', 'day'])
    agg_info = {
        'dw_solar': 'sum',
        'uw_solar': 'sum',
        'direct_n': 'sum'
    }
    if not is_polar:
        agg_info.pop('direct_n')
    result = grouped.agg(agg_info).reset_index()

    result_columns = ['year', 'month', 'day',
                      'dw_solar_sum', 'uw_solar_sum', 'direct_n_sum']

    if not is_polar:
        result_columns.remove('direct_n_sum')

    # Rename columns to indicate they are sums
    result.columns = result_columns
    return result


def process_directory(base_dir, year, is_polar: bool):
    all_files = glob(os.path.join(
        base_dir, str(year), '**/*.dat'), recursive=True)
    results_percentiles = []
    results_sums = []

    for filepath in all_files:
        df = parse_noaa_file(filepath, is_polar)
        if df.empty:
            continue
        daily_stats = compute_daily_percentiles(df, is_polar)
        daily_sums = compute_daily_sums(df, is_polar)
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
    stations = [(True, 'brw'), (False, 'mlo'), (True, 'spo'), (False, 'smo')]
    for is_polar, station in stations:
        base_dir = current_dir / station
        all_results = []

        for year in range(1998, 2025):
            print(f'Processing year {year}...')
            year_df = process_directory(base_dir, year, is_polar)
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
