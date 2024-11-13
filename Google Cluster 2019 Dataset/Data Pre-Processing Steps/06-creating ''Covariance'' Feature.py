import pandas as pd
import numpy as np
import ast
import statistics



# Load your dataset into a pandas DataFrame
df = pd.read_csv('Google Borg Primary Dataset.csv')



# Convert string representations to lists, skipping NaN values
df['cpu_usage_distribution_modified_9_listed_final'] = df['cpu_usage_distribution_modified_9_listed'].apply(lambda x: ast.literal_eval(x) if not pd.isna(x) else np.nan)



# Convert string representations to lists, skipping NaN values
df['tail_cpu_usage_distribution_modified_listed_final'] = df['tail_cpu_usage_distribution_modified_brackets'].apply(lambda x: ast.literal_eval(x) if not pd.isna(x) else np.nan)





# Function to calculate covariance while handling NaN
def cov_with_nan(a, b):
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if not mask.any():
        return np.nan
    a = np.array(a)[mask]
    b = np.array(b)[mask]
    return np.mean((a - np.mean(a)) * (b - np.mean(b)))

# Apply the function row-wise to calculate covariance
df['covariance'] = df.apply(lambda row: cov_with_nan(row['cpu_usage_distribution_modified_9_listed_final'],
                                                     row['tail_cpu_usage_distribution_modified_listed_final']),
                            axis=1)

