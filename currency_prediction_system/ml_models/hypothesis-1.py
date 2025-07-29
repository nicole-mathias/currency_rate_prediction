import pandas as pd
from scipy import stats

# Hypothesis
"""
H0 - Null Hypothesis - There is no significant difference between means of the two features - CPI US and CPI JP
H1 - Alternative Hypothesis - There is significant difference between the features

"""

ALPHA = 0.05


def perform_t_test(feature_1, feature_2):
	'''
	This function performs the t-test and using the p-value accepts/reject the hypothesis
	'''
	print("**** Performing t-test ****")
	ttest_resp = stats.ttest_ind(feature_1, feature_2)
	print(ttest_resp)
	if ttest_resp.pvalue < ALPHA:
		print("Reject NULL Hypothesis with p-value: {}. There is significant difference.".format(ttest_resp.pvalue))
	else:
		print("Accept the Null Hypothesis with p-value:{}. There is no significant difference.".format(ttest_resp))

def main():
	# Import the dataset
	df = pd.read_csv("datasets/new_combined_clean.csv")
	cpi_us = df["cpi_us"]
	cpi_j = df["cpi_j"]
	exchange_rate = df["exchange_rate_USD_JY_x"]
	perform_t_test(cpi_us, cpi_j)


if __name__ == '__main__':
	main()