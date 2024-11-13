import pandas as pd
import numpy as np
import ast



# Load your dataset into a pandas DataFrame
df = pd.read_csv('Google Borg Primary Dataset.csv')




# Function to process the string and convert it to the desired format
def process_string(s):
    # Remove brackets and split the string by whitespace
    numbers = s.strip('[]').split()
    # Join the numbers with commas
    return ', '.join(numbers)

# Apply the function to the column
df['cpu_usage_distribution_modified'] = df['cpu_usage_distribution'].apply(process_string)

























# # Convert string representations to lists
# df['cpu_usage_distribution'] = df['cpu_usage_distribution'].apply(ast.literal_eval)



# for value in df['cpu_usage_distribution']:
#     # Check if the value is a dictionary
#     if isinstance(value, dict):
#         print("Dictionary:", value)
#     # Check if the value is a string
#     elif isinstance(value, str):
#         print("String:", value)
#     else:
#         print("Other type:", value)



# # # Print the result
# # print(cpu_usage_distribution_list)



















# # Function to check if a value is a list with exactly 9 numbers
# def is_valid_distribution(value):
#     if isinstance(value, list) and len(value) == 9:
#         try:
#             float_values = [float(x) for x in value]
#             return all(isinstance(x, float) for x in float_values)
#         except ValueError:
#             return False
#     return False

# # Apply the function to filter rows
# df['cpu_usage_distribution'] = df['cpu_usage_distribution'].apply(lambda x: x if is_valid_distribution(x) else np.nan)

# # # Output the cleaned DataFrame
# # print(df)


# # Convert DataFrame to CSV
# df.to_csv('CPU5.csv', index=False)



























# # Splitting the 'cpu_usage_distribution' column by space and expanding it into separate columns
# df[['cpu_usage_distribution_1', 'cpu_usage_distribution_2', 'cpu_usage_distribution_3', 'cpu_usage_distribution_4', 'cpu_usage_distribution_5', 
#     'cpu_usage_distribution_6', 'cpu_usage_distribution_7', 'cpu_usage_distribution_8', 'cpu_usage_distribution_9']] = df['cpu_usage_distribution'].str.split(expand=True)



# # Converting the new columns to float type
# df[['cpu_usage_distribution_1', 'cpu_usage_distribution_2', 'cpu_usage_distribution_3', 'cpu_usage_distribution_4', 'cpu_usage_distribution_5', 
#     'cpu_usage_distribution_6', 'cpu_usage_distribution_7', 'cpu_usage_distribution_8', 'cpu_usage_distribution_9']]= df[[  'cpu_usage_distribution_1', 'cpu_usage_distribution_2', 
#                                                                                                                             'cpu_usage_distribution_3', 'cpu_usage_distribution_4', 
#                                                                                                                             'cpu_usage_distribution_5', 'cpu_usage_distribution_6', 
#                                                                                                                             'cpu_usage_distribution_7', 'cpu_usage_distribution_8', 
#                                                                                                                             'cpu_usage_distribution_9']].astype(float)




# # Optionally, you can drop the original 'cpu_usage_distribution' column
# # df.drop('cpu_usage_distribution', axis=1, inplace=True)
# print(df.head())
