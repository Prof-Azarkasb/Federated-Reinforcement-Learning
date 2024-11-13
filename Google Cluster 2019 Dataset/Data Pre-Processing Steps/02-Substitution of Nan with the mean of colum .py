import pandas as pd


# Load your dataset into a pandas DataFrame
df = pd.read_csv('Google Borg Primary Dataset.csv')


# print( df["resource_request_cpus"].isna().sum() )

df['scheduler'].fillna(0.0, inplace=True)
df['vertical_scaling'].fillna(1.0, inplace=True)



# Calculate the mean of the non-NaN values in the 'cycles_per_instruction' column
mean_cycles_per_instruction = df['cycles_per_instruction'].mean()
mean_memory_accesses_per_instruction = df['memory_accesses_per_instruction'].mean()


# Fill NaN values with the calculated mean
df['cycles_per_instruction'].fillna(mean_cycles_per_instruction, inplace=True)
df['memory_accesses_per_instruction'].fillna(mean_memory_accesses_per_instruction, inplace=True)



# If you want to confirm that there are no more NaN values in the column
print(df['memory_accesses_per_instruction'].isnull().sum())

