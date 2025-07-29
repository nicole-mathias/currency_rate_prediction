import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score


# from mpl_toolkits.mplot3d import Axes3D
# import pandas as pd


def k_means(df, no_of_clusters, attributes_to_cluster):
    
    # getting the numpy array for the columns we want to cluster
    data = df[attributes_to_cluster].to_numpy()
    
    # Initialize K-means model
    kmeans = KMeans(n_clusters = no_of_clusters)

    # Fit the model to the data
    kmeans.fit(data)
    cluster_labels = kmeans.predict(data)

    # Get cluster centers and labels for each data point
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Count the occurrences of each label
    unique_labels, counts = np.unique(cluster_labels, return_counts = True)

    # Print the count of data points in each cluster
    for label, count in zip(unique_labels, counts):
        print(f"K_means - Cluster {label}: {count} data points")

    silhouette_avg = silhouette_score(data, cluster_labels)
    print("----silhouette_score for k-means------",silhouette_avg)


    # Visualize the data and cluster centers in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = labels, cmap = cm.rainbow)
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker = 'x', color = 'red', label = 'Cluster Centers')
    ax.set_xlabel(attributes_to_cluster[0])
    ax.set_ylabel(attributes_to_cluster[1])
    ax.set_zlabel(attributes_to_cluster[2])
    plt.legend()
    plt.title('K-means Clustering in 3D')
    
    # saving the plot as k_means_clustering.png
    plt.savefig("plot/k_means_clustering.png")

    # plt.show()
