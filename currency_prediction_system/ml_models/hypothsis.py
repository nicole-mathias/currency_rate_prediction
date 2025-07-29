import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind
# Hypothesis
"""
H0 - Null Hypothesis - There is no significant difference between means of the two features - government gross debt - Japan and government gross debt - US
H1 - Alternative Hypothesis - There is a significant difference between the features

"""
path = os.getcwd() + '\\datasets\\new_combined_clean.csv'


# Government gross debt - Japan
govt_debt_j = pd.read_csv(path, usecols=['govt_debt_j'])
# government gross debt - US
govt_debt_us = pd.read_csv(path, usecols=['govt_debt_us'])

# two sampled t-test

t_stat, pvalue = ttest_ind(govt_debt_j, govt_debt_us)

print('t-statistics:', t_stat, '\np-value: ', pvalue)
