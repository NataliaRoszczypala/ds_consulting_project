import pandas as pd

df = pd.read_csv('../data/input/shopping_trends.csv')

def data_check(df):
    
    required_cols = ['Customer ID', 'Age', 'Gender', 'Item Purchased', 'Category',
       'Purchase Amount (USD)', 'Location', 'Size', 'Color', 'Season',
       'Review Rating', 'Subscription Status', 'Payment Method',
       'Shipping Type', 'Discount Applied', 'Promo Code Used',
       'Previous Purchases', 'Preferred Payment Method',
       'Frequency of Purchases']
    
    required_data_types = {'Customer ID': 'int64',
        'Age': 'int64',
        'Gender': 'object',
        'Item Purchased': 'object',
        'Category': 'object',
        'Purchase Amount (USD)': 'int64',
        'Location': 'object',
        'Size': 'object',
        'Color': 'object',
        'Season': 'object',
        'Review Rating': 'float64',
        'Subscription Status': 'object',
        'Payment Method': 'object',
        'Shipping Type': 'object',
        'Discount Applied': 'object',
        'Promo Code Used': 'object',
        'Previous Purchases': 'int64',
        'Preferred Payment Method': 'object',
        'Frequency of Purchases': 'object'}

    # Check column names
    if required_cols!= list(df.columns):
        raise ValueError("Column names do not match.")
    
    # Check data types
    if required_data_types != df.dtypes.to_dict():
        return ValueError("Data types do not match.")
    
    # Check for non-null values - there shouldn't be any
    if df.isna().sum().any():
        return ValueError("There are missing values in the dataset.")
    
    return True

