import pandas as pd
from data_check import data_check
from data_preparation_refresh import data_preparation_refresh
from data_preparation_modelling import data_preparation_modelling
from sklearn.metrics import pairwise_distances_argmin_min
import pickle
from sklearn.decomposition import PCA
import os

df = pd.read_csv('../../data/input/shopping_trends.csv')

# Load centroids from the saved file
centroids = pd.read_csv('../../data/automation/automation_input/centroids.csv')

# Check data
if data_check(df):
    print("Data check passed.")

# Prepare data
preprocessed_data = data_preparation_refresh(df)

# Prepare data for modelling
cleaned_data = data_preparation_modelling(preprocessed_data)

# Save data
preprocessed_data.to_csv('../../data/automation/preprocessed_data.csv', index=False)
cleaned_data.to_csv('../../data/automation/cleaned_data.csv', index=False)

# Load the saved PCA model
with open('../../data/automation/automation_input/pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

# Apply the PCA transformation to the new data
cleaned_data_transformed = pca.transform(cleaned_data)

# Find the nearest centroid for each data point in the new data
closest_centroids, _ = pairwise_distances_argmin_min(cleaned_data_transformed, centroids)

# Assign the cluster to each data point
preprocessed_data['cluster'] = closest_centroids
preprocessed_data.to_csv('../../data/automation/preprocessed_data_cluster.csv', index=False)