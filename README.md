# Consumer Behaviour Analytics for Online Clothing Store

## Overview

This project aims to analyze consumer behavior in the online clothing retail sector to provide actionable insights for increasing customer satisfaction and retention. Given the significant growth and intense competition in the e-commerce market, particularly in the fashion segment, understanding consumer behavior is crucial for optimizing marketing efforts and enhancing profitability.

The main goal of this project was not only to perform the analysis but also to develop an easily and reusable process of updating the dashboard accorign to updated data.

The project is developed for the "Data Science - Consulting Approach" course conducted by Grzegorz Krochmal at the Faculty of Economic Sciences, University of Warsaw.

## Authors

* Natalia Roszczypała
* Kacper Gruca

## Analytics Objectives

1. Customer Segmentation: Identify categories of customers based on spending habits.
2. Satisfaction Analysis: Determine factors affecting customer satisfaction.
3. Retention Strategies: Analyze factors influencing the time gap since the last purchase and recommend strategies to improve customer retention.

## Data

The data is downloaded from Kaggle, you can find it under [this](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset?select=shopping_trends.csv) link.

## Analysis Process

1. **Data Extraction:** Collect relevant data from various sources.
2. **Data Preparation:** Clean and prepare data for analysis.
3. **Principal Component Analysis (PCA):** Reduce dimensionality of data for better analysis.
4. **Clustering:** Group customers into segments based on PCA results.
5. **XGBoost (XGB):** Apply machine learning models to both clusters for predictive analysis.
6. **Dashboard Integration (Power BI):** Load data into a dashboard for visualization and further insights.

## Structure of the procect

- data
  - automation - folder with inputs and outputs used for reusable process (refreshing data)
  - input - folder with input csv used for analysis
- results - results of the performed analysis 
  - classification_pca_one_hot_encode
  - clustering
  - regression
- scripts
  - utils - folder with .py files (mainly class and functions used in the analysis) but also with the .py files to refresh the data

