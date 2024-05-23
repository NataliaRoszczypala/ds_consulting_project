# %%
import pandas as pd
from data_preprocessor import DataPreprocessor
# %%
# Load data
data = pd.read_csv('shopping_trends.csv', index_col=0)

print(data.head())
print(data.info())
print(data.describe())
print(data.describe(include='object'))
# %%
# Data Preprocessing
preprocessor = DataPreprocessor(data)

# Handle missing values
print(data.isnull().any())

# Encode categorical variables
preprocessor.one_hot_encode()

# Detect outliers
outliers = preprocessor.detect_outliers(method='iqr')
print("Outliers detected using IQR method:\n", outliers.sum())

# Get processed data
preprocessed_data = preprocessor.get_preprocessed_data()
print(preprocessed_data.head())