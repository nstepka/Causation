from upload_data_page import upload_data
from feature_engineering import feature_engineering, display_data_preview, display_handle_missing_values, display_process_currency_percentage
from feature_engineering import display_drop_columns, display_data_transformation, display_encode_categorical, display_time_series_features, display_convert_to_datetime
from explore_data import explore_data, display_boxplot, display_binary_distribution, feature_comparison_graph_page
from regression import evaluate_model_page, display_model_performance_comparison, prepare_data, create_models, fit_models, evaluate_models, plot_model_performance
from regression import display_select_target_features_and_train, display_feature_importance, display_prediction_vs_actual, display_residuals_plot
from regression import display_correlation_heatmap, evaluate_model_page
from advance_data_analysis import advanced_data_analysis, perform_classification, perform_clustering, perform_dimensionality_reduction
from time_series_analysis import time_series_analysis, visualize_time_series_data,display_acf_pacf, fit_arima_model
from causation_page import causality_page

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
from prophet import Prophet


# Set page config
st.set_option('deprecation.showPyplotGlobalUse', False)





def sanitize_string(s):
    """Removes non-UTF-8 characters from a string."""
    return s.encode('utf-8', 'ignore').decode('utf-8')


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'





def save_data():
    """
    Provides a button to save the current dataframe to CSV.
    """
    # Check if there's data in the session state
    if 'data' in st.session_state and st.session_state.data is not None:
        # Provide a button for users to download the dataframe
        if st.button('Download Dataframe as CSV'):
            tmp_download_link = download_link(st.session_state.data, 'your_data.csv', 'Click here to download the data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
    else:
        st.warning("No data available to save. Please upload data first.")
        

def main_updated():
    st.title("Interactive Model Builder")
    
    

    # Sidebar for primary task selection
    primary_task = st.sidebar.radio(
    "Choose a primary task:",
    ["Data Upload", "Feature Engineering", "Explore the Data", 
     "Regression Analysis", "Extensive Data Analysis","Time Series Analysis","Causality Analysis", "Save"]
    )

    if primary_task == "Data Upload":
        upload_data()

    elif primary_task == "Feature Engineering":
        feature_engineering()

    elif primary_task == "Explore the Data":
        explore_data()

    elif primary_task == "Regression Analysis":
        evaluate_model_page()

    elif primary_task == "Extensive Data Analysis":
        advanced_data_analysis()
    elif primary_task == "Time Series Analysis":
        time_series_analysis()

    elif primary_task == "Save":
        save_data()
    elif primary_task == "Causality Analysis":
        causality_page()



if __name__ == "__main__":
    main_updated()
