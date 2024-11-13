import ast
import pandas as pd



# Load your dataset into a pandas DataFrame
df = pd.read_csv('Google Borg Primary Dataset.csv')


# Fill NaN values in the "resource_request" column with a default dictionary
df['resource_request'].fillna("{'cpus': 0.0, 'memory': 0.0}", inplace=True)



# Replace 'Nane' with NaN if it's supposed to be NaN, otherwise replace it with 'None'
# Function to convert string to dictionary
def string_to_dict(string):
    if pd.isna(string):
        return None
    else:
        return ast.literal_eval(string)
    

# Convert the column
df['resource_request'] = df['resource_request'].apply(string_to_dict)





# Split 'resource_request' column into two separate columns
df['resource_request_cpus'] = df['resource_request'].apply(lambda x: x['cpus'])
df['resource_request_memory'] = df['resource_request'].apply(lambda x: x['memory'])





# Drop the original 'resource_request' column if needed
df.drop(columns=['resource_request'], inplace=True)






