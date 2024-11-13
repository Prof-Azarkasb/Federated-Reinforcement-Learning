import pandas as pd
import numpy as np
import ast
import statistics



# Load the datasets
df = pd.read_csv('Google Borg Primary Dataset.csv')
df_Cp_final= pd.read_csv('pre_Final.csv')

# You can change it according to your actual column name
column_to_add = df_Cp_final['covariance']

# Add the column 
df['covariance'] = column_to_add




