import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def equi_width_bin(data, bin_col_name, no_of_bins):

    # saving the original data
    df = data

    # get column names from df
    column_names = data.columns.to_list()
    col_index = column_names.index(bin_col_name)


    # get data in numpy array format for the column you are binning on
    data = data.to_numpy()[:, col_index]

    # getting bin width
    bin_width = (np.max(data) - np.min(data))/no_of_bins

    # Initialize an array to store bin edges
    bin_edges = [np.min(data) + i * bin_width for i in range(no_of_bins + 1)]
    # print("bin",bin_edges)
    
    # Assign data points to bins
    bin_assignments = np.digitize(data, bin_edges, right = True)

     # adding a new attribute called Bins --> which saves the bin assignments
    df["Bin_number"] = bin_assignments

    # writing back the bin assignemnts to a csv file called "US_binned_data.csv"
    df.to_csv("binned_data.csv", index = False)


    # Calculate the frequency of data in each bin
    hist, _ = np.histogram(data, bins = bin_edges)

    bin_freq = []
    bin_starts = []
    
    for i in range(len(hist)):
        bin_freq.append(hist[i])
        bin_starts.append(bin_edges[i])


    # gathering data to plot a bar graph displaying various bins and frequency of data point in each bin
    bin_categories = ["Bin " + str(i) for i in range(1, no_of_bins + 1)]
    bar_widths = [bin_width] * no_of_bins

    # Plotting Bar chart for Bins where each bin indicated the frequency
    plt.bar(bin_starts, bin_freq, width = bar_widths, edgecolor = "black")

    # Add value annotations inside each bar
    for i in range(len(bin_starts)):
        plt.annotate(str(bin_freq[i]), xy=(bin_starts[i], bin_freq[i]), ha='center', va='bottom')

    # Add labels and title
    plt.xlabel('Bins')
    plt.ylabel(bin_col_name + " frequency")
    plt.title("equi-width binning")

    # Set x-axis labels to be the bin_categories
    plt.xticks(bin_starts, bin_categories)

    # Save the plot as a PNG image
    plt.savefig('binning.png')

    # Display the chart
    plt.show()
