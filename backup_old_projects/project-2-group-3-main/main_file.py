import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from separating_data import seperating_US_Japan_data
from handle_missing_data.US_missing_data import US_regression
from handle_missing_data.Japan_missing_data import Japan_regression
from binning import equi_depth_bin # importing bining python methods
from clustering import partition_k_means # importing k-means method
from clustering import density_db_scan # importing db_scan
from clustering import statistical_GMM # GMM



def main(us_data,japan_data):


    # Step1 - Solving missing values --> using regression and mean_imputations-------------------

    # ------------------>> Solving missing values for US----------
    us_output_col = "gdp_pc_us" # Column on which we are performing regression
    us_drop_col = "govt_debt_us" # Dropping the column because it has too many missing values and mean impuations wont be useful for this

    # ----Regression no. 1---------
    data_fill_1 = US_regression(us_data, us_output_col, us_drop_col) # getting a df, but we still need to fill out the missing values present in our dropped column

    # saving the data for the column that was dropped from the US data_set earlier
    dropped_col_data = pd.DataFrame(us_data, columns = ['date', us_drop_col])
    merged_df = pd.merge(data_fill_1, dropped_col_data, on='date', how='inner') # merging regression_1 data and dropped column data


    # ----Regression no. 2----- Performing Regression again to fill the dropped column (i.e govt_debt_us)
    merged_debt_df = US_regression(merged_df, us_drop_col, None) # In regresssion 2, the missing data for govt_debt was filled


    # Generating a CSV file after filling all the missing data
    merged_debt_df.to_csv("us_solved_missing_data.csv", index = False)

    print("---- Missing data for US was filled (filename: us_solved_missing_data.csv)----")
    # ---------------------------------------------------------------------------------------


    # --->> Solving missing values for Japan------------------------
    japan_output_col = "govt_debt_j" # Column on which we are performing regression
    japan_drop_col = None # Not dropping any columns

    # ----Regression no. 1---------
    merged_japan_df = Japan_regression(japan_data, japan_output_col, japan_drop_col)

    # Generating a CSV file after filling all the missing data
    merged_japan_df.to_csv("japan_solved_missing_data.csv", index = False)

    # -----------------------------------------------------------------------------



    # getting csv files in which missing data issue was solved
    US_file_name = "us_solved_missing_data.csv"
    Japan_file_name = "japan_solved_missing_data.csv"

    US_file = os.path.join(current_directory, US_file_name)
    Japan_file = os.path.join(current_directory, Japan_file_name)

    US_df = pd.read_csv(US_file)
    Japan_df = pd.read_csv(Japan_file)

    # --------------------------------------------------------------------------------


    # Outlier and outlier detection code in present in the scripts folder


    current_dir = os.getcwd()
    combined_file = "new_combined_clean.csv"
    new_combined_file = os.path.join(current_dir,"datasets",combined_file)

    new_combined_data = pd.read_csv(new_combined_file)


    # ----------Step n - Binning for any one attribute (US_dataset in which attribute "attribute_name" was chosen for equi-width Binning)
    # need = no.of bins, binning col name, most updated cleaned dataframe
    no_of_bins = 5
    bin_col_name = "govt_debt_j"
    data_file = new_combined_data
    equi_depth_bin.bin(data_file,bin_col_name,no_of_bins)
    

    # --------------- Step n+1 - Clustering---------
    # All plots are stored in the folder Plots, they are not displayed
    
    # --->>> Clustering method 1: Partition Clustering :: k-means
    # need = dataframe, no.of clusters, list_of_col_names on which you want to perform clustering
    data_file = new_combined_data
    no_of_clusters = 2
    attributes_to_cluster = ["govt_debt_j", "interest_r_j", "gdp_pc_j"]
    partition_k_means.k_means(data_file, no_of_clusters, attributes_to_cluster)



    # --->>> Clustering method 2: Density Clustering :: db scan
    # need = eps_val, min_no_samples, most updated cleaned dataframe, list_of_col_names on which you want to perform clustering
    data_file = new_combined_data
    eps_val = 12
    min_points = 60
    attributes_to_cluster = ["govt_debt_j", "interest_r_j", "gdp_pc_j"]
    density_db_scan.db_scan(data_file, attributes_to_cluster, eps_val, min_points)


    # --->>> Clustering method 3: Statistical Clustering :: GMM
    # need = leatest data file, attributes to cluster, and no of clusters
    data_file = new_combined_data
    no_of_clusters = 2
    attributes_to_cluster = ["govt_debt_j", "interest_r_j", "gdp_pc_j"]
    statistical_GMM.gmm(data_file, attributes_to_cluster, no_of_clusters)





if __name__ == "__main__":

    # -----This space is getting all the unclean datasets and then sending it to the main function------

    current_directory = os.getcwd()

    # combined US_Japan data file --> so our goal is to separate them into different csv files
    combined_file_name = os.path.join(current_directory, "datasets","us_jap_data.csv")
    #  = "/datasets/us_jap_data.csv"
    combined_data_file = os.path.join(current_directory, combined_file_name)


    # Step0 - Separating the US_Japan combined data_set into seperate data_sets
    seperating_US_Japan_data.separate_data(combined_data_file)

    # # step1 - get the seperated data files and send them to the main method
    us_data_path = os.path.join(current_directory, "us_data.csv")
    japan_data_path = os.path.join(current_directory, "japan_data.csv")

    us_data = pd.read_csv(us_data_path)
    japan_data = pd.read_csv(japan_data_path)



    # Step2 - Call main function which performs all other steps of the project
    main(us_data,japan_data) # rest all steps are present in the main method

    






