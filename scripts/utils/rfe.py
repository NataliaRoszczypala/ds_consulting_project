import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def apply_rfe(df, target_column, n_features_to_select=9, random_state=42):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost classifier
    model = RandomForestRegressor()

    # Apply Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=model, n_features_to_select=10)  # Modify n_features_to_select as needed
    rfe.fit(X_train, y_train)

    # Select top features
    selected_features = X.columns[rfe.support_].tolist()
    selected_features.append(target_column)

    return selected_features