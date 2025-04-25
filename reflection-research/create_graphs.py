import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap


def create_graphs(site_code: str, max_year=2024):
    # Get the script's directory and construct CSV path
    script_dir = Path(__file__).parent
    csv_path = script_dir / f'output_{site_code}.csv'

    # Load the CSV data
    df = pd.read_csv(csv_path)

    # Filter out data beyond max_year
    df = df[df['year'] <= max_year]

    # Combine the year, month, and day into a single date column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.set_index('date', inplace=True)

    # Add day of year column
    df['day_of_year'] = df.index.dayofyear

    # Create the output directory if it doesn't exist
    graphs_dir = script_dir / f'{site_code}_graphs'
    graphs_dir.mkdir(exist_ok=True)

    # List of columns to plot
    value_columns = df.columns.difference(
        ['year', 'month', 'day', 'day_of_year'])

    # Get first and last 4 years
    all_years = sorted(df['year'].unique())
    first_years = all_years[:4]
    last_years = all_years[-4:]
    focus_years = first_years + last_years

    # Define quarters
    quarters = {
        'q1': (1, 90),   # Jan-Mar
        'q2': (91, 181),  # Apr-Jun
        'q3': (182, 273),  # Jul-Sep
        'q4': (274, 366)  # Oct-Dec
    }

    colors = plt.colormaps['coolwarm']

    # Generate quarterly plots for focus years
    for column in value_columns:
        for q_name, (start_day, end_day) in quarters.items():
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Create color map from blue to green
            year_colors = LinearSegmentedColormap.from_list(
                '', ['darkblue', 'green'])

            # Create normalized color values (0 to 1) for each year
            year_positions = {year: i/(len(focus_years)-1)
                              for i, year in enumerate(focus_years)}

            for period, years in [("First 4 years", first_years), ("Last 4 years", last_years)]:
                # Initialize lists to store all detrended data for this period
                all_rel_stds = []
                all_days = []

                for year in years:
                    yearly_data = df[df['year'] == year]
                    quarter_data = yearly_data[
                        (yearly_data['day_of_year'] >= start_day) &
                        (yearly_data['day_of_year'] <= end_day)
                    ]

                    if len(quarter_data) < 5:  # Skip if not enough data points
                        continue

                    normalized_days = quarter_data['day_of_year'] - start_day

                    # Plot original data with period-specific color
                    color = 'darkblue' if period == "First 4 years" else 'darkgreen'
                    ax1.plot(normalized_days, quarter_data[column],
                             label=f"{year}",
                             color=color,
                             alpha=0.3)

                    # Filter spikes using rolling median
                    rolling_med = quarter_data[column].rolling(
                        window=5, center=True).median()
                    spike_threshold = rolling_med.std() * 3  # 3 sigma threshold
                    is_spike = abs(
                        quarter_data[column] - rolling_med) > spike_threshold
                    clean_data = quarter_data[column].copy()
                    clean_data[is_spike] = rolling_med[is_spike]

                    # Calculate detrended data with cleaned values
                    normalized_days_safe = normalized_days.copy()
                    valid_mask = normalized_days_safe > 10
                    normalized_days_safe = normalized_days_safe[valid_mask]
                    quarter_data_safe = clean_data[valid_mask]

                    if len(quarter_data_safe) < 5:  # Skip if not enough valid points
                        continue

                    coeffs = np.polyfit(
                        np.log(normalized_days_safe), quarter_data_safe, deg=2)
                    fitted_trend = coeffs[0] * \
                        np.log(normalized_days_safe) + coeffs[1]
                    detrended_data = clean_data - fitted_trend

                    # Calculate relative standard deviation with error handling
                    rolling_mean = detrended_data.rolling(
                        window=5, min_periods=5).mean()
                    rolling_std = detrended_data.rolling(
                        window=5, min_periods=5).std()

                    # Avoid division by zero or very small values
                    rolling_rel_std = pd.Series(np.zeros_like(
                        rolling_mean), index=rolling_mean.index)
                    valid_mean = abs(rolling_mean) > 1e-6
                    rolling_rel_std[valid_mean] = (
                        rolling_std[valid_mean] / rolling_mean[valid_mean].abs()) * 100

                    # Filter out extreme values
                    rolling_rel_std = rolling_rel_std.clip(
                        0, 100)  # Cap at 100%

                    all_rel_stds.append(rolling_rel_std)
                    all_days.append(normalized_days)

                # Only proceed if we have valid data
                if not all_rel_stds:
                    continue

                # Calculate mean standard deviation for this period
                common_days = np.arange(start_day, end_day + 1) - start_day
                resampled_stds = []

                for days, rel_std in zip(all_days, all_rel_stds):
                    valid = ~np.isnan(rel_std) & (
                        rel_std > 0) & (rel_std < 100)
                    if valid.sum() >= 5:  # Only include if enough valid points
                        resampled = np.interp(common_days,
                                              days[valid],
                                              rel_std[valid],
                                              left=np.nan, right=np.nan)
                        resampled_stds.append(resampled)

                if resampled_stds:
                    mean_std = np.nanmean(resampled_stds, axis=0)
                    # Additional smoothing of the final result
                    smooth_window = 5
                    mean_std = pd.Series(mean_std).rolling(
                        smooth_window, center=True, min_periods=3).mean()

                    color = 'blue' if period == "First 4 years" else 'green'
                    ax2.plot(common_days, mean_std,
                             label=f"Average {period}",
                             color=color,
                             linewidth=2)

            ax1.set_title(
                f'{site_code.upper()} - {column} ({q_name.upper()})\nFirst/Last 4 Years')
            ax1.set_xlabel(f'Days (from start of {q_name.upper()})')
            ax1.set_ylabel(column)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(bottom=0)

            ax2.set_title(
                '5-Day Rolling Relative Std-Dev vs 5 Day Rolling Mean')
            ax2.set_xlabel(f'Days (from start of {q_name.upper()})')
            ax2.set_ylabel(f'Relative Standard Deviation of {column} (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(bottom=0)

            plt.tight_layout()

            plt.savefig(
                graphs_dir / f'{site_code}_{column}_focused_{q_name}.png')
            plt.close()

    # Create a separate plot for each column
    for column in value_columns:
        # Daily graphs
        plt.figure()
        # Create alternating order of years (new/old)
        sorted_years = sorted(all_years)
        colors = LinearSegmentedColormap.from_list(
            'custom', ['blue', 'yellow', 'green'])
        alternating_years = []
        while sorted_years:
            alternating_years.append(sorted_years.pop())  # newest
            if sorted_years:
                alternating_years.append(sorted_years.pop(0))  # oldest

        for i, year in enumerate(alternating_years):
            yearly_data = df[df['year'] == year]
            plt.plot(yearly_data['day_of_year'], yearly_data[column],
                     color=colors(i/len(all_years)), alpha=0.35)
        plt.title(f'{site_code.upper()} - {column}')
        plt.xlabel('Day of Year')
        plt.ylabel(column)
        plt.ylim(bottom=0)
        plt.tight_layout()

        plt.savefig(graphs_dir / f'{site_code}_{column}.png')
        plt.close()

        # Yearly summary graphs
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Filter out values less than 1
        filtered_df = df[df[column] >= 1]
        yearly_sums = filtered_df.groupby('year')[column].sum()
        # Count valid days per year
        valid_days = filtered_df.groupby('year')[column].count()
        yearly_means = yearly_sums / valid_days

        if len(yearly_sums) > 0:
            # Plot sums
            ax1.bar(yearly_sums.index, yearly_sums.values)

            # Calculate trendline without outliers
            sorted_indices = np.argsort(yearly_sums.values)
            # Remove highest and lowest
            filtered_indices = sorted_indices[1:-1]
            filtered_years = yearly_sums.index[filtered_indices]
            filtered_sums = yearly_sums.values[filtered_indices]
            z = np.polyfit(filtered_years, filtered_sums, 1)
            p = np.poly1d(z)
            ax1.plot(yearly_sums.index, p(yearly_sums.index),
                     "r--", alpha=0.8, label='Trend (excluding outliers)')
            ax1.legend()
            ax1.set_title(f'{site_code.upper()} - {column} (Yearly Sum)')
            ax1.set_xlabel('Year')
            ax1.set_ylabel(f'Total {column}')

            # Plot means
            ax2.bar(yearly_means.index, yearly_means.values)

            # Calculate mean trendline without outliers
            sorted_mean_indices = np.argsort(yearly_means.values)
            # Remove highest and lowest
            filtered_mean_indices = sorted_mean_indices[1:-1]
            filtered_mean_years = yearly_means.index[filtered_mean_indices]
            filtered_means = yearly_means.values[filtered_mean_indices]
            z_mean = np.polyfit(filtered_mean_years, filtered_means, 1)
            p_mean = np.poly1d(z_mean)
            ax2.plot(yearly_means.index, p_mean(yearly_means.index),
                     "r--", alpha=0.8, label='Trend (excluding outliers)')
            ax2.legend()
            ax2.set_title(
                f'{site_code.upper()} - {column} (Daily Mean per Year)')
            ax2.set_xlabel('Year')
            ax2.set_ylabel(f'Mean {column} per day')

            plt.tight_layout()
            plt.savefig(
                graphs_dir / f'{site_code}_{column}_yearly_summary.png')
            plt.close()


if __name__ == '__main__':
    create_graphs('brw', 2024)
    create_graphs('spo', 2023)
    create_graphs('mlo', 2024)
    create_graphs('smo', 2024)
