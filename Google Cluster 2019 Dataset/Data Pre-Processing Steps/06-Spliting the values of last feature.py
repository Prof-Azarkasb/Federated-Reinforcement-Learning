import pandas as pd
import numpy as np
import ast
import statistics



# Load your dataset into a pandas DataFrame
df = pd.read_csv('Google Borg Primary Dataset.csv')




# Function to process the string and convert it to the desired format
def process_string(s):
    # Remove brackets and split the string by whitespace
    numbers = s.strip('[]').split()
    # Join the numbers with commas
    return ', '.join(numbers)

# Apply the function to the column
df['tail_cpu_usage_distribution_modified'] = df['tail_cpu_usage_distribution'].apply(process_string)







