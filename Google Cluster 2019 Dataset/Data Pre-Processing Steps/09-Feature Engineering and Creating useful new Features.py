import pandas as pd
import numpy as np
import ast
import statistics



# Load your dataset into a pandas DataFrame
df = pd.read_csv('Google Borg Primary Dataset.csv')



# Convert string representations to lists, skipping NaN values
df['tail_cpu_usage_distribution_modified_listed_final'] = df['tail_cpu_usage_distribution_modified_brackets'].apply(lambda x: ast.literal_eval(x) if not pd.isna(x) else np.nan)








# Define a function to calculate the mean of each list, handling NaN values
def calculate_mean(lst):
    if isinstance(lst, list):
        return np.mean(lst)
    else:
        return np.nan
# Define a function to find the maximum value of each list, handling NaN values
def find_max(lst):
    if isinstance(lst, list):
        return max(lst)
    else:
        return np.nan

# Define a function to find the minimum value of each list, handling NaN values
def find_min(lst):
    if isinstance(lst, list):
        return min(lst)
    else:
        return np.nan
# Define a function to find the variance value of each list, handling NaN values
def find_variance(lst):
    if isinstance(lst, list):
        return np.var(lst)
    else:
        return np.nan


# Define a function to find the mode value of each list, handling NaN values
def find_mode(lst):
    if isinstance(lst, list):
        return statistics.mode(lst)
    else:
        return np.nan
# Define a function to find the median value of each list, handling NaN values
def find_median(lst):
    if isinstance(lst, list):
        return np.median(lst)
    else:
        return np.nan
# Define a function to find the difference between the maximum and minimum values of each list, handling NaN values
def find_max_min_diff(lst):
    if isinstance(lst, list):
        return max(lst) - min(lst)
    else:
        return np.nan







# Calculate the mean of each list and store the results in a new column
df['mean_cpu_usage_tail_distribution'] = df['tail_cpu_usage_distribution_modified_listed_final'].apply(calculate_mean)

# Find the maximum value of each list and store the results in a new column
df['max_cpu_usage_tail_distribution'] = df['tail_cpu_usage_distribution_modified_listed_final'].apply(find_max)

# Find the minimum value of each list and store the results in a new column
df['min_cpu_usage_tail_distribution'] = df['tail_cpu_usage_distribution_modified_listed_final'].apply(find_min)

# Find the variance value of each list and store the results in a new column
df['variance_cpu_usage_tail_distribution'] = df['tail_cpu_usage_distribution_modified_listed_final'].apply(find_variance)


# Find the mode value of each list and store the results in a new column
df['mode_cpu_usage_tail_distribution'] = df['tail_cpu_usage_distribution_modified_listed_final'].apply(find_mode)


# Find the median value of each list and store the results in a new column
df['median_cpu_usage_tail_distribution'] = df['tail_cpu_usage_distribution_modified_listed_final'].apply(find_median)


# Find the difference between the maximum and minimum values of each list and store the results in a new column
df['max_min_diff_cpu_usage_tail_distribution'] = df['tail_cpu_usage_distribution_modified_listed_final'].apply(find_max_min_diff)




















