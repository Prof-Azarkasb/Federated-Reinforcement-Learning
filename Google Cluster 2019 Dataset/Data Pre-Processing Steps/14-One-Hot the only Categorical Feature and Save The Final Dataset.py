import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew





# Load your dataset into a pandas DataFrame
df = pd.read_csv('Google Borg Primary Dataset.csv')



# One-hot encode the 'event' column
df_encoded = pd.get_dummies(df['event'], prefix='event')

# Concatenate the one-hot encoded columns with the original DataFrame
df_final = pd.concat([df, df_encoded], axis=1)

# Drop the original 'event' column if needed
df_final.drop('event', axis=1, inplace=True)




df_final.to_csv('Final Dataset.csv', index=False)
