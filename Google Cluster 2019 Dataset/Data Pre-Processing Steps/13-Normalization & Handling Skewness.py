import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler




# Load your dataset into a pandas DataFrame
df = pd.read_csv('Google Borg Primary Dataset.csv')


a= ["time", "scheduling_class", "priority", "vertical_scaling", "scheduler", "start_time", "end_time", "assigned_memory", "page_cache_memory", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_rate", "resource_request_cpus", "resource_request_memory", "average_usage_cpus", "average_usage_memory", 
    "maximum_usage_cpus", "maximum_usage_memory", "random_sample_usage_cpus", "mean_cpu_usage", "max_cpu_usage", "min_cpu_usage", "variance_cpu_usage",
    "mode_cpu_usage", "median_cpu_usage", "max_min_diff_cpu_usage", "mean_cpu_usage_tail_distribution", "max_cpu_usage_tail_distribution", 
    "min_cpu_usage_tail_distribution", "variance_cpu_usage_tail_distribution", "mode_cpu_usage_tail_distribution", "median_cpu_usage_tail_distribution",
    "max_min_diff_cpu_usage_tail_distribution", "covariance"]
# 34 features just in case of normalization



# Replace 'numeric_cols' with your actual list of numeric column names
scaler = MinMaxScaler()
df[a] = scaler.fit_transform(df[a])



