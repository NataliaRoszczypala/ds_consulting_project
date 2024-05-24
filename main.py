# %%
import pandas as pd
from data_preprocessor import DataPreprocessor
from eda import EDA
from regression import RegressionAnalysis
# %%
# Load data
data = pd.read_csv('shopping_trends.csv', index_col=0)

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
# %%
# Regression Analysis of Customer Satisfaction

# Encode categorical variables
preprocessor.one_hot_encode()
preprocessed_data = preprocessor.get_preprocessed_data()
print(preprocessed_data.head())

satisfaction_regression = RegressionAnalysis(preprocessed_data, 'Review Rating') # to be changed for clustered data !
satisfaction_regression.prepare_data()

# Fit regression models
satisfaction_regression.fit_linear_regression()

best_ridge = satisfaction_regression.tune_ridge_regression(param_grid={'alpha': [0.1, 1.0, 10.0, 100.0]})
print("Best Ridge parameters:", best_ridge)

best_lasso = satisfaction_regression.tune_lasso_regression(param_grid={'alpha': [0.1, 1.0, 10.0, 100.0]})
print("Best Lasso parameters:", best_lasso)

best_rf = satisfaction_regression.tune_random_forest(param_grid={'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]})
print("Best Random Forest parameters:", best_rf)

best_xgb = satisfaction_regression.tune_xgboost(param_grid={'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300]})
print("Best XGBoost parameters:", best_xgb)

# Evaluate regression models
evaluation = satisfaction_regression.evaluate_models()
print("Regression Model Evaluation:")
print(evaluation)

# Get coefficients or feature importances for the best model
best_model = evaluation.loc[evaluation['R2'].idxmax(), 'Model']
print(f"Best Model: {best_model}")
coefficients = satisfaction_regression.get_coefficients(best_model)
print(coefficients)