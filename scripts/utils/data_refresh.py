import pandas as pd
from utils.data_check import data_check
from utils.data_preparation_refresh import data_preparation_refresh
from utils.data_preparation_modelling import data_preparation_modelling

df = pd.read_csv('data/raw_data.csv')

# Check data
if data_check(df):
    print("Data check passed.")

# Prepare data
preprocessed_data = data_preparation_refresh(df)

# Prepare data for modelling
cleaned_data = data_preparation_modelling(preprocessed_data)

# Save data
preprocessed_data.to_csv('data/automation/preprocessed_data.csv', index=False)
cleaned_data.to_csv('data/automation/cleaned_data.csv', index=False)