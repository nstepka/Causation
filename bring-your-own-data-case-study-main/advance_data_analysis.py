import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import plotly.express as px
import base64
from sklearn.ensemble import RandomForestClassifier


def advanced_data_analysis():
    if 'data' in st.session_state and st.session_state.data is not None:
    
        task = st.sidebar.radio(
            "Choose an Advanced Data Analysis task:",
            ["Classification", "Clustering", "Dimensionality Reduction", "Feature Selection"]
        )

        if task == "Classification":
            st.subheader("Classification")
            perform_classification(st.session_state.data)
        elif task == "Clustering":
            st.subheader("Clustering")
            perform_clustering(st.session_state.data)
        elif task == "Dimensionality Reduction":
            st.subheader("Dimensionality Reduction")
            perform_dimensionality_reduction(st.session_state.data)
        elif task == "Feature Selection":
            st.subheader("Feature Selection")
            perform_feature_selection(st.session_state.data)
    else:
        st.warning("Please upload data first.")    

def perform_classification(data):
    # Display the dropdown to select the target column
    target_column = st.selectbox("Select Target Column", data.columns, key="selectbox1")

    # If user selects a new target column, update the session state
    st.session_state.target_column = target_column
        
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    if X.select_dtypes(include=['object']).any().any():
        st.error("Please select a valid target column. One of the feature columns has non-numeric values.")
        return
    
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    st.write(f"Accuracy: {accuracy*100:.2f}%")

def perform_clustering(data):
    non_numeric_cols = data.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        st.write(f"Found non-numeric columns: {', '.join(non_numeric_cols)}")
        drop_non_numeric = st.checkbox("Drop non-numeric columns for clustering?", value=True)
        if drop_non_numeric:
            data = data.select_dtypes(exclude=['object'])
        else:
            st.warning("Using non-numeric columns may result in errors during clustering.")
    
    n_clusters = st.slider("Select number of clusters", 1, 10)
    kmeans = KMeans(n_clusters=n_clusters)
    data['Cluster'] = kmeans.fit_predict(data)
    fig = px.scatter_3d(data, x=data.columns[0], y=data.columns[1], z=data.columns[2], color='Cluster')
    st.plotly_chart(fig)


def perform_dimensionality_reduction(data):
    if 'target_column' not in st.session_state:
        st.warning("Please select a target column in the Classification page first.")
        return
        
    target_column = st.session_state.target_column
        
    pca = PCA(2)
    data_pca = pca.fit_transform(data.drop(target_column, axis=1))
    fig = px.scatter(data_pca, x=0, y=1, color=data[target_column])
    st.plotly_chart(fig)


def perform_feature_selection(data):
    if 'target_column' not in st.session_state:
        st.warning("Please select a target column in the Classification page first.")
        return
        
    target_column = st.session_state.target_column
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    selector = SelectKBest(f_classif, k=2)
    X_new = selector.fit_transform(X, y)
    
    top_features = X.columns[selector.get_support()]
    f_scores = selector.scores_[selector.get_support()]
    p_values = selector.pvalues_[selector.get_support()]
    
    st.write("Top Features with F-values and p-values:")
    for feat, score, p_value in zip(top_features, f_scores, p_values):
        st.write(f"{feat}: F-value = {score:.2f}, p-value = {p_value:.4f}")

