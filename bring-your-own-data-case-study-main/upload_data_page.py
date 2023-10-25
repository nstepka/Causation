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


def upload_data():
    
    st.markdown("""
    ## About the Author
    Hello! I'm **Nicholas Stepka**, a Computer Data Scientist at Tennessee State University. 
    Currently, I'm pursuing my degree in Computer Science with a concentration in data science. 
    My research interests lie in causal inference and understanding tabular data to solve real-world problems.
    
    [Connect with me on LinkedIn](https://www.linkedin.com/in/nstepka/)""")

    st.markdown("""    
    ## About this App
    This app is designed to assist users in quick modeling and provide insights into business data science projects. 
    With a user-friendly interface, it facilitates various tasks from data upload to advanced analysis. 
    I hope this tool proves beneficial for your data science endeavors!
    """)

    # Step 1: Upload CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        st.session_state.original_data = pd.read_csv(uploaded_file)
        st.session_state.data = st.session_state.original_data.copy() 

        # Check if data is not None before displaying the preview
        st.write("Uploaded data preview:")
        st.write(st.session_state.data.head())

        # Provide a button for users to download the dataframe
        if st.button('Download Dataframe as CSV'):
            tmp_download_link = download_link(st.session_state.data, 'your_data.csv', 'Click here to download the data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)