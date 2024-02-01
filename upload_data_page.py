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
    ## About this App
    This app is designed to assist users in quick modeling and provide insights into business data science projects. 
    With a user-friendly interface, it facilitates various tasks from data upload to advanced analysis. 
    I hope this tool proves beneficial for your data science endeavors!

    If you want to look at this app's code or see the datasets it works with out of the box,click the link below
    
    [App github page](https://github.com/nstepka/Causation)
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

    
    st.markdown("""
    ## About the Author
    Hello! I'm **Nicholas Stepka**, a recent graduate in Computer Science from Tennessee State University, where I specialized in Data Science. I hold a strong passion for leveraging data to drive business solutions and innovation. My academic journey and professional experience have fostered a deep interest in business intelligence, machine learning, and statistical modeling.

    My expertise lies in analyzing and interpreting complex datasets to uncover actionable insights. Through my senior project, "Airbnb Rent Prediction Model Based on Large Data," I have developed a keen understanding of how machine learning algorithms can be applied to real-world scenarios, especially in predictive analytics.

    I am enthusiastic about exploring the intersection of technology and business, particularly how data-driven strategies can optimize performance and drive growth. My goal is to utilize my skills in Python, machine learning, and statistical modeling to contribute to the field of business intelligence and data science, creating impactful solutions for challenging problems.

    Check out my first Tableau Dashboard I created on a cold Sunday morning! https://public.tableau.com/app/profile/nstepka/viz/AirbnbDataSet_17052522761620/Dashboard1
    
    [Connect with me on LinkedIn](https://www.linkedin.com/in/nstepka/)""")

    
