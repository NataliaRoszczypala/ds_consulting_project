import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from scipy.stats import chi2_contingency

class EDA:
    def __init__(self, data):
        self.data = data

    def plot_categorical(self):
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            plt.figure(figsize=(10, 5))
            sns.countplot(data = self.data, x = col)
            plt.title(f'{col} Distribution')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.show()

    def plot_numerical(self):
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(self.data[col], bins=30, kde=True)
            plt.title(f'{col} Distribution')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()

    def correlation_matrix(self):
        numerical_data = self.data.select_dtypes(include=['float64', 'int64'])
        plt.figure(figsize=(15, 10))
        sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def summary_statistics(self):
        return self.data.describe()

    def correlation_statistics(self):
        numerical_data = self.data.select_dtypes(include=['float64', 'int64'])
        return numerical_data.corr()

    def categorical_statistics(self):
        return self.data.describe(include='object')
    
    def chi_squared_test(self, col1, col2):
        contingency_table = pd.crosstab(self.data[col1], self.data[col2])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        return {
            'chi2_statistic': chi2,
            'p_value': p,
            'degrees_of_freedom': dof
        }

    def cramers_v(self, col1, col2):
        contingency_table = pd.crosstab(self.data[col1], self.data[col2])
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

    def categorical_correlations(self):
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        correlations = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
        for col1 in categorical_columns:
            for col2 in categorical_columns:
                if col1 == col2:
                    correlations.loc[col1, col2] = 1.0
                else:
                    correlations.loc[col1, col2] = self.cramers_v(col1, col2)
        return correlations

    def normality_test(self):
        normality_results = {}
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            k2, p = stats.normaltest(self.data[col])
            alpha = 1e-3
            normality_results[col] = {
                'k2': k2,
                'p-value': p,
                'normal': p >= alpha
            }
        return normality_results
