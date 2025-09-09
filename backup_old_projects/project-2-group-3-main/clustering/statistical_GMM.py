# import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score
# from mpl_toolkits.mplot3d import Axes3D
# import pandas as pd


def gmm(df, attributes_to_cluster, num_clusters):

    data = df[attributes_to_cluster].to_numpy()

    # Initialize GMM model
    gmm = GaussianMixture(n_components = num_clusters)

    # Fit the model to the data
    gmm.fit(data)

    # Predict the clusters for each data point
    cluster_labels = gmm.predict(data)

    # Access probabilities for each data point belonging to each cluster
    probabilities = gmm.predict_proba(data)


    silhouette_avg = silhouette_score(data, cluster_labels)
    print("----silhouette_score for GMM------",silhouette_avg)

    # Visualize the data and clusters in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = cluster_labels, cmap = cm.rainbow)
    ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], gmm.means_[:, 2], marker = 'x', color = 'red', label = 'Cluster Centers')
    ax.set_xlabel(attributes_to_cluster[0])
    ax.set_ylabel(attributes_to_cluster[1])
    ax.set_zlabel(attributes_to_cluster[2])
    plt.legend()
    plt.title('GMM Clustering in 3D')

    # saving the plot
    plt.savefig("plot/GMM.png")

    # plt.show()