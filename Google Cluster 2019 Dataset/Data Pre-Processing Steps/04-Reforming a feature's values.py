import pandas as pd
import numpy as np
import ast



# Load your dataset into a pandas DataFrame
df = pd.read_csv('Google Borg Primary Dataset.csv')



# Function to convert string to list
def convert_to_list(string):
    if isinstance(string, str):
        return [float(x) for x in string.split(',')]
    else:
        return np.nan
    

# Apply the function to convert the column
df['cpu_usage_distribution_modified'] = df['cpu_usage_distribution_modified'].apply(convert_to_list)

# Create a new column with the modified values
df['cpu_usage_distribution_modified_brackets'] = df['cpu_usage_distribution_modified'].apply(lambda x: str(x) if isinstance(x, list) else 'NaN')






