import pandas as pd
from utils.data_preprocessor import DataPreprocessor

def data_preparation(df):

    df = df.drop(columns=['Promo Code Used', 'Item Purchased'])

    ordered_categorical_cols = ['Size', 'Frequency of Purchases']
    categorical_cols = ['Gender', 'Category', 'Location',
                        'Color', 'Season', 'Subscription Status',
                        'Payment Method', 'Shipping Type', 'Discount Applied',
                        'Preferred Payment Method']
    numerical_cols = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']

    cols = ordered_categorical_cols + categorical_cols + numerical_cols

    df = df[cols]

    # Define the order of the sizes
    size_order = ['S', 'M', 'L', 'XL']

    # Create a dictionary to map each size to an integer
    size_mapping = {size: index + 1 for index, size in enumerate(size_order)}

    # Map the size column using the defined order
    df['Size'] = df['Size'].map(size_mapping)

    frequency_mapping = {
        'Weekly': 6,
        'Fortnightly': 5,
        'Bi-Weekly': 4,
        'Monthly': 3,
        'Quarterly': 2,             # combine 'Quarterly' and 'Every 3 Months'       
        'Every 3 Months': 2,        # combine 'Quarterly' and 'Every 3 Months'
        'Annually': 1
    }

    df['Frequency of Purchases'] = df['Frequency of Purchases'].map(frequency_mapping)

    preprocessor = DataPreprocessor(df)

    preprocessor.one_hot_encode(columns=categorical_cols)

    preprocessor.normalization(columns=numerical_cols)
    cleaned_data = preprocessor.get_preprocessed_data()

    return cleaned_data