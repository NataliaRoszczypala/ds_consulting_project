import pandas as pd

def data_preparation_refresh(df):
    preprocessed_data = df.drop(columns=['Promo Code Used', 'Item Purchased', 'Customer ID'])

    return preprocessed_data