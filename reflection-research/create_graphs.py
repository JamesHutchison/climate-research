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

    # Since we only have data for 1998, the loop will only iterate once
    years = df['year'].unique()
    colors = plt.colormaps['coolwarm']

    # Create a separate plot for each column
    for column in value_columns:
        # Daily graphs
        plt.figure()
        # Create alternating order of years (new/old)
        sorted_years = sorted(years)
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
                     color=colors(i/len(years)), alpha=0.35)
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
