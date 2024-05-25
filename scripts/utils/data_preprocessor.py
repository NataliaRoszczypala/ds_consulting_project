import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class DataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def handle_missings(self, columns=None):
        if columns is None:
            columns = self.data.columns
        
        for col in columns:
            if self.data[col].isnull().any() == True and self.data[col].dtype in ['int64', 'float64']:
                self.data[col].fillna(self.data[col].mean())
            elif self.data[col].isnull().any() == True and self.data[col].dtype == 'object':
                self.data[col].fillna(self.data[col].mode()[0])
            
    def encode_categorical(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=['object']).columns
        
        for col in columns:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le

    def one_hot_encode(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=['object']).columns
        
        self.data = pd.get_dummies(self.data, columns=columns, drop_first=True)
    
    def standardization(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        self.data[columns] = self.scaler.fit_transform(self.data[columns])

    def normalization(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        scaler = MinMaxScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])

    def detect_outliers(self, columns=None, method='iqr'):
        if columns is None:
            columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        outliers = pd.DataFrame()
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound))
            elif method == 'z_score':
                mean = self.data[col].mean()
                std = self.data[col].std()
                z_scores = (self.data[col] - mean) / std
                outliers[col] = (abs(z_scores) > 3)
            else:
                raise ValueError("Invalid method. Use 'iqr' or 'z_score'.")
        
        return outliers

    def remove_outliers(self, columns=None, method='iqr'):
        outliers = self.detect_outliers(columns, method)
        
        for col in outliers.columns:
            self.data = self.data[~outliers[col]]
        
    def get_preprocessed_data(self):
        return self.data