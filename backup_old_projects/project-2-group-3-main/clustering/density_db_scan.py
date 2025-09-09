import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score
# from mpl_toolkits.mplot3d import Axes3D
# import pandas as pd


def db_scan(df, attributes_to_cluster, eps_val, min_no_samples):
    data = df[attributes_to_cluster].to_numpy()

    # Initialize DBSCAN model
    dbscan = DBSCAN(eps = eps_val, min_samples = min_no_samples)

    # Fit the model to the data and predict the clusters
    cluster_labels = dbscan.fit_predict(data)


    # Count the occurrences of each label
    unique_labels, counts = np.unique(cluster_labels, return_counts = True)

    # Print the count of data points in each cluster
    for label, count in zip(unique_labels, counts):
        print(f"DB_scan - Cluster {label}: {count} data points")


    silhouette_avg = silhouette_score(data, cluster_labels)
    print("----silhouette_score for db_scan------",silhouette_avg)


    # Visualize the data and clusters in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = cluster_labels, cmap = cm.rainbow)
    ax.set_xlabel(attributes_to_cluster[0])
    ax.set_ylabel(attributes_to_cluster[1])
    ax.set_zlabel(attributes_to_cluster[2])
    plt.title('DBSCAN Clustering in 3D')

    # saving the plot
    plt.savefig("plot/db_scan.png")

    # plt.show()