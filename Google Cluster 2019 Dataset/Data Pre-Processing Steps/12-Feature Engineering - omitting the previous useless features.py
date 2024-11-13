import pandas as pd
import numpy as np
import ast
import statistics




# Load your dataset into a pandas DataFrame
df = pd.read_csv('Google Borg Primary Dataset.csv')




df['time_range'] = df['end_time'] - df['start_time']






df['random_sample_usage_memory'] = df['random_sample_usage_memory'].fillna(0)

df['mean_cpu_usage'] = df['mean_cpu_usage'].fillna(0)

df['max_cpu_usage'] = df['max_cpu_usage'].fillna(0)

df['min_cpu_usage'] = df['min_cpu_usage'].fillna(0)

df['variance_cpu_usage'] = df['variance_cpu_usage'].fillna(0)

df['mode_cpu_usage'] = df['mode_cpu_usage'].fillna(0)

df['median_cpu_usage'] = df['median_cpu_usage'].fillna(0)

df['max_min_diff_cpu_usage'] = df['max_min_diff_cpu_usage'].fillna(0)

df['mean_cpu_usage_tail_distribution'] = df['mean_cpu_usage_tail_distribution'].fillna(0)

df['max_cpu_usage_tail_distribution'] = df['max_cpu_usage_tail_distribution'].fillna(0)

df['min_cpu_usage_tail_distribution'] = df['min_cpu_usage_tail_distribution'].fillna(0)

df['variance_cpu_usage_tail_distribution'] = df['variance_cpu_usage_tail_distribution'].fillna(0)

df['mode_cpu_usage_tail_distribution'] = df['mode_cpu_usage_tail_distribution'].fillna(0)

df['median_cpu_usage_tail_distribution'] = df['median_cpu_usage_tail_distribution'].fillna(0)

df['max_min_diff_cpu_usage_tail_distribution'] = df['max_min_diff_cpu_usage_tail_distribution'].fillna(0)

df['covariance'] = df['covariance'].fillna(0)




df.drop(columns=['constraint'], inplace=True)
df.drop(columns=['start_after_collection_ids'], inplace=True)



