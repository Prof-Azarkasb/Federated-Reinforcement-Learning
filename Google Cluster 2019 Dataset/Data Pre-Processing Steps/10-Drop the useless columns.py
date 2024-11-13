import pandas as pd
import numpy as np
import ast
import statistics



# Load your dataset into a pandas DataFrame
df = pd.read_csv('Google Borg Primary Dataset.csv')






df.drop(columns=['tail_cpu_usage_distribution'], inplace=True)
df.drop(columns=['tail_cpu_usage_distribution_modified'], inplace=True)
df.drop(columns=['tail_cpu_usage_distribution_modified_brackets'], inplace=True)
df.drop(columns=['tail_cpu_usage_distribution_modified_listed'], inplace=True)
df.drop(columns=['tail_cpu_usage_distribution_modified_listed_final'], inplace=True)





