import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap


def create_graphs(site_code: str):
    # Get the script's directory and construct CSV path
    script_dir = Path(__file__).parent
    csv_path = script_dir / f'output_{site_code}.csv'

    # Load the CSV data
    df = pd.read_csv(csv_path)

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
            'custom', ['blue', 'yellow'])
        alternating_years = []
        while sorted_years:
            if (len(sorted_years) > 0):
                alternating_years.append(sorted_years.pop())  # newest
            if (len(sorted_years) > 0):
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

        plt.savefig(graphs_dir / f'{column}.png')
        plt.close()

        # Yearly summary graphs
        plt.figure()
        # Filter out values less than -10 before grouping and summing
        filtered_df = df[df[column] >= -10]
        yearly_sums = filtered_df.groupby('year')[column].sum()
        plt.bar(yearly_sums.index, yearly_sums.values)

        # Add trend line for yearly data
        years_numeric = np.arange(len(yearly_sums))
        z = np.polyfit(years_numeric, yearly_sums.values, 1)
        p = np.poly1d(z)
        plt.plot(yearly_sums.index, p(years_numeric),
                 "r--", alpha=0.8, label='Trend')
        plt.legend()

        plt.title(
            f'{site_code.upper()} - {column} (Yearly Sum)')
        plt.xlabel('Year')
        plt.ylabel(f'Total {column}')
        plt.tight_layout()

        plt.savefig(graphs_dir / f'{column}_yearly_sum.png')
        plt.close()


if __name__ == '__main__':
    create_graphs('brw')
