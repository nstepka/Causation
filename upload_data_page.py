import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import base64


import streamlit as st
import pandas as pd

def upload_data():
    st.markdown("""
    ## About this App
    This app is designed to assist users in quick modeling and provide insights into business data science projects.
    With a user-friendly interface, it facilitates various tasks from data upload to advanced analysis.
    I hope this tool proves beneficial for your data science endeavors!

    If you want to look at this app's code or see the datasets it works with out of the box, click the link below

    [App github page](https://github.com/nstepka/Causation)
    """)

    # Button to load specific datasets
    datasets = {
        'Load IRIS Dataset (Clustering & Feature Selection)': 'Data/IRIS.csv',
        'Load Churn Dataset (Classification Analysis)': 'Data/churn.csv',
        'Load Decision Tree Classifier Dataset (Will Buy)': 'Data/DTClassiferWillBuy.csv',
        'Load Decision Tree Regression Dataset (Loan Amount)': 'Data/DTRegressionLoan.csv',
        'Load Park Data (Time Series ARIMA Analysis)': 'Data/ParkData_5years.csv',
        'Load Airbnb Dataset (Price Regression & Data Exploration)': 'Data/df_selected1.csv',
        'Load Uncleaned Airbnb File (Data Cleaning & Exploration)': 'Data/df_selected1.csv'  # Assuming this is intentional duplicate for illustration
    }

    for button_label, file_path in datasets.items():
        if st.button(button_label):
            st.session_state.data = pd.read_csv(file_path)
            st.write(f"{button_label.split('(')[0].strip()} loaded:")
            st.write(st.session_state.data.head())

    # Option to clear the preloaded or uploaded file
    if st.button('Clear Data'):
        st.session_state.data = None
        st.write("Data cleared. You can now upload your own dataset or load one of the default datasets.")

    # Step 1: Upload CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.write(st.session_state.data.head())

    # Provide a button for users to download the dataframe
    if 'data' in st.session_state and st.session_state.data is not None:
        if st.button('Download Dataframe as CSV'):
            tmp_download_link = download_link(st.session_state.data, 'your_data.csv', 'Click here to download the data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

    # About the author section remains unchanged
    st.markdown("""
    ## About the Author
    Hello! I'm **Nicholas Stepka**, a recent graduate in Computer Science from Tennessee State University, where I specialized in Data Science. ...
    [Connect with me on LinkedIn](https://www.linkedin.com/in/nstepka/)
    """)

