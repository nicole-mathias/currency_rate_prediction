# Analysis Done by Dhiraj Saharia

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor # For outlier detection using LOF
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
plt.rcParams["figure.figsize"] = (12,8)
plt.style.use('ggplot')

# Note - All the images are saved and not displayed.

def cluster_and_plot(X_combined):
	# result = combined_df.dropna()
	labels_all = []
	X = X_combined.loc[:, ["govt_debt_j", "interest_r_j", "gdp_pc_j"]]
	for k in [2, 4, 5, 6]:
		hierarchical_cluster = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
		labels = hierarchical_cluster.fit_predict(X)
		labels_all.append(labels)
		silhouette_avg = silhouette_score(X, labels)
		print(f"Silhouette Score for no. of cluster = {k}: {silhouette_avg}")

	# Scatter plot for the CPI features with cluster labels for 2 clusters
	plt.scatter(X['govt_debt_j'], X['interest_r_j'], c=labels_all[0])
	plt.xlabel('Govt. Debt JP')
	plt.ylabel('Interest rate JP')
	plt.savefig('../plot/hierarchical-new.png', dpi=1200, bbox_inches='tight')
	print("*** Saved Cluster image ***")

def perform_LOF(X):
    plt.rcParams["figure.figsize"] = (9,9)
    i = 1
    outlier_factor_score = []
    X_clean = X.drop('date', axis=1)
    for K in [5, 10, 15]:
        clf = LocalOutlierFactor(n_neighbors=K, contamination='auto')
        y_pred = clf.fit(X_clean)
        # n_errors = (y_pred != ground_truth).sum()
        X_scores = clf.negative_outlier_factor_
        outlier_factor_score.append(X_scores)
        plt.subplot(3, 1, i)
        i += 1
        plt.plot(X['date'], X_scores, label=f"k={K}")
        plt.xlabel('Date')
        plt.ylabel('LOF Score')
        plt.legend()
        plt.savefig(f"../plot/lof-{K}.png", dpi=1200, bbox_inches='tight')
        print(f"*** Saving LOF (k={K}) Image ***")
        # plt.show()
    # print(outlier_factor_score)
    return outlier_factor_score

def read_and_merge_data():
	dataframe_US = pd.read_csv('../datasets/us_solved_missing_data.csv')
	dataframe_JP = pd.read_csv('../datasets/japan_solved_missing_data.csv')

	# Date column had '\n' chars - Remove them for further processing
	dataframe_JP = dataframe_JP.replace('\n', '', regex=True)
	dataframe_US = dataframe_US.replace('\n', '', regex=True)
	# Convert to datetime object
	dataframe_JP['date'] = pd.to_datetime(dataframe_JP['date'])
	dataframe_US['date'] = pd.to_datetime(dataframe_US['date'])
	# Set index to date for analysis
	dataframe_US.set_index('date')
	dataframe_JP.set_index('date')
	# Sort the index
	dataframe_US.sort_index(ascending=True, inplace=True)
	dataframe_JP.sort_index(ascending=True, inplace=True)
	# Combine the dataset
	combined_df = pd.merge(dataframe_US, dataframe_JP, on='date')
	combined_df.set_index('date')
	combined_df.sort_index(ascending=True, inplace=True)
	combined_df.to_csv('../datasets/combined.csv', encoding='utf-8', index=False)
	print(f"Combined Columns: {combined_df.columns}")
	return combined_df

def draw_boxplots(X):
	# Plot 1
	X[['govt_debt_us', 'govt_debt_j']].plot(kind='box', title='Govt. Debt Boxplots')
	plt.savefig('../plot/boxplot-debt.png', dpi=1200, bbox_inches='tight')
	plt.ylabel('Govt. Debt')
	print("*** Saved Box Plot #1 ***")
	# Plot 2
	X[['inflation_us', 'inflation_j']].plot(kind='box', title='Inflation Boxplots')
	plt.savefig('../plot/boxplot.png', dpi=1200, bbox_inches='tight')
	plt.ylabel('Inflation')
	print("*** Saved Box Plot #2 ***")

def main():
	# Read and Merge different data into one dataset for further processing
	combined_df = read_and_merge_data()
	# Outlier Detection
	scores = perform_LOF(combined_df)
	# After getting the scores, create a new dataset with outliers removed using the threshold value
	new_combined = combined_df[scores[1] <= -0.98]
	new_combined.to_csv('../datasets/new_combined_clean.csv', encoding='utf-8', index=False)
	print("*** Saving the clean dataset for further processing ***")
	cluster_and_plot(new_combined)
	# Draw Box-plots and IQR
	draw_boxplots(new_combined)
	

if __name__ == '__main__':
	main()
