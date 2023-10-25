# Interactive Data Analysis and Causation Tool

## Overview

This project is an interactive web application built using Streamlit, designed for data analysis and causation analysis. Users can upload datasets, perform data preprocessing, explore data visually, conduct regression and classification analysis, perform extensive data analysis, and explore causal relationships in their data.

## Key Features

### Data Uploading and Preprocessing

- **CSV File Upload**: Easily upload your CSV datasets.
### Preprocessing - Feature Enginerring

- **Data Preview**: Get a quick overview of the uploaded data.
- **Handling Missing Values**: Deal with missing data using various techniques.
- **Currency and Percentage Changes**: Convert currency and percentage columns into numeric formats.
- **Column Removal**: Drop unnecessary columns from the dataset.
- **Data Transformation**: Apply a range of data transformations.
- **Data Encoding**: Encode categorical variables using one-hot encoding.
- **Time Series Features**: Convert date/time columns and create new features.
- **Binning/Bucketing**: Discretize numeric variables into bins.
- **Custom Feature Engineering**: Create new features based on custom logic.
- **Aggregate Columns**: Perform aggregations and generate new columns.
- **Create Binary Flags**: Generate binary flags based on specified conditions.

- **If you want to test all features, use listings.csv.**

### Data Exploration

- **Boxplot Visualization**: Visualize the distribution of numeric data with boxplots.
- **Binary Distribution**: Examine binary distribution using count plots.
- **Feature Comparison Graphs**: Compare two variables with scatter and line plots.
- **Feature Importance**: Assess feature importance for regression and classification tasks.

- **If you want to explore before using the regression model use df_selected1.csv**

### Regression Analysis

- **Regression Models**: Train and compare various regression models, including Gradient Boosting Regressor, Random Forest Regressor, Linear Regression, and Decision Tree Regressor.
- **Model Evaluation**: Compare models using R-squared scores and Mean Absolute Error (MAE).
- **Feature Importance**: Visualize feature importance.
- **Heat Map**: Display a heatmap to visualize correlations.
- **Prediction vs. Actual**: Compare predictions against actual values.
- **Residual Plot**: Examine residuals to assess model performance.

- **df_selcted1.csv works out of the box if you target price.**

### Classification Analysis

- **Classification Models**: Train and compare various classification models.
- **Model Evaluation**: Evaluate models using accuracy, precision, recall, and F1-score.
- **Feature Importance**: Visualize feature importance.
- **Heat Map**: Display a heatmap to visualize correlations.
- **Prediction vs. Actual**: Compare predictions against actual labels.

- **Use churn.csv.  You will have to do some feature engineering for best results.  Target churn as the feature of interest**

### Extensive Data Analysis

- **Dataset**: Explore the Iris dataset for clustering, dimensionality reduction, and feature selection.
- **Clustering**: Visualize clusters in 3D space using K-Means clustering.
- **Dimensionality Reduction**: Reduce dimensionality using PCA and visualize in 2D space.
- **Feature Selection**: Select top features based on F-values and p-values.

- **This was designed to bring in the hello world of data, the iris data set.  For out-of-the-box use, use iris.csv**

### Time Serires ARIMA Analysis

- **Time Series Analysis with ARIMA**: Conduct time series analysis, including ACF and PACF plots, ARIMA model fitting, model diagnostics, forecasting, and model evaluation.
- **Works well with the ParkData_5years.csv file.**
### Causality Analysis

- **Define Relationships**: Define causal relationships between variables using a graphical interface or upload a DOT file.
- **Create Causal Model**: Create a causal model based on defined relationships and estimate causal effects.
- **Estimation**: Estimate causal effects and assess their significance.
- **Run Refutation Tests**: Perform refutation tests to test the reliability of causal estimates.

-**Works out of the box if you bring in df_selcted1.csv and bring in causal_graph_complete.dot in the causal graph section.  Target price as the outcome and accommodates as the treatment**

### Save your model
- **Model Saving**:  After all your hard work of feature engineering exploring the data and the models you can save your current csv file.
