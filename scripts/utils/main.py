# %%
import pandas as pd
from data_preprocessor import DataPreprocessor
from eda import EDA
# %%
# Load data
data = pd.read_csv('data/shopping_trends.csv', index_col=0)

print(data.head())
print(data.info())
# %%
# Data Preprocessing
preprocessor = DataPreprocessor(data)

# Handle missing values
print(data.isnull().any())

# Detect outliers
outliers = preprocessor.detect_outliers(method='iqr')
print("Outliers detected using IQR method:\n", outliers.sum())

# Get preprocessed data
preprocessed_data = preprocessor.get_preprocessed_data()
print(preprocessed_data.head())
# %%
# EDA
eda = EDA(preprocessed_data)
eda.plot_categorical()
eda.plot_numerical()

print("Summary Statistics:")
print(eda.summary_statistics())

eda.correlation_matrix()
print("Correlation Statistics:")
print(eda.correlation_statistics())

print("Categorical Statistics:")
print(eda.categorical_statistics())

print("Categorical Correlations:")
print(eda.categorical_correlations())

#Based on the correlation analysis it can be observed that variables 'Discount Applied' and 'Promo Code Used' as well as 'Item Purchased' and 'Category' are highly correlated, thus we decided to remove variables 'Promo Code Used' and 'Item Purchased' not to include collinearity in the estimation.
preprocessed_data = preprocessed_data.drop(columns=['Promo Code Used', 'Item Purchased'])
print(preprocessed_data.head())