import pandas as pd
import numpy as np


def bin(df,bin_col_name,no_of_bins):

    # number of bins and calculate quantiles
    num_bins = no_of_bins + 1
    quantiles = np.percentile(df[bin_col_name], np.linspace(0, 100, num_bins))

    # Compute bin ranges
    bin_ranges = [(quantiles[i], quantiles[i+1]) for i in range(num_bins - 1)]

    # bin ranges
    for i, bin_range in enumerate(bin_ranges):
        print(f'Bin {i + 1} Range: {bin_range[0]:.2f} to {bin_range[1]:.2f}')

