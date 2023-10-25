from dowhy.causal_estimators import CausalEstimator
from dowhy.causal_refuters.data_subset_refuter import DataSubsetRefuter
from upload_data_page import upload_data
from feature_engineering import feature_engineering, display_data_preview, display_handle_missing_values, display_process_currency_percentage
from feature_engineering import display_drop_columns, display_data_transformation, display_encode_categorical, display_time_series_features, display_convert_to_datetime
from explore_data import explore_data, display_boxplot, display_binary_distribution, feature_comparison_graph_page
from regression import evaluate_model_page, display_model_performance_comparison, prepare_data, create_models, fit_models, evaluate_models, plot_model_performance
from regression import display_select_target_features_and_train, display_feature_importance, display_prediction_vs_actual, display_residuals_plot
from regression import display_correlation_heatmap, evaluate_model_page
from advance_data_analysis import advanced_data_analysis, perform_classification, perform_clustering, perform_dimensionality_reduction
from time_series_analysis import time_series_analysis, visualize_time_series_data,display_acf_pacf, fit_arima_model
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
import os
from dowhy import CausalModel
import re
import graphviz



def generate_dot_download_link(dot_representation, download_name="causal_graph.dot"):
    """Generate a download link for the DOT representation."""
    b64 = base64.b64encode(dot_representation.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{download_name}">Download DOT File</a>'
    return href


def parse_dot_content(dot_content):
    """Parse the DOT content to extract relationships."""
    lines = dot_content.split('\n')
    relationships = []
    for line in lines:
        line = line.strip()
        if '->' in line:
            cause, effect = line.split('->')
            cause = cause.strip(' "')
            effect = effect.strip(' ";')
            relationships.append((cause, effect))
    return relationships



def display_relationships_definition():
    """Sub-task for defining relationships and visualizing them as a causal graph."""
    
    st.subheader("Upload a DOT File (Optional)")
    uploaded_file = st.file_uploader("Choose a DOT file", type=["dot"])
    
    if uploaded_file:
        dot_content = uploaded_file.read().decode()
        st.session_state.dot_representation = dot_content
        st.success("DOT file uploaded successfully!")
        uploaded_graph = graphviz.Source(dot_content)
        st.graphviz_chart(uploaded_graph.source)
        
        # Parse the uploaded DOT content to get relationships
        st.session_state.relationships = parse_dot_content(dot_content)

    columns = list(st.session_state.data.columns)

    # Selectors for cause and effect
    cause_column = st.selectbox("Select Cause Column", columns, key="cause_column")
    effect_column = st.selectbox("Select Effect Column", columns, key="effect_column")

    # Button to add the relationship
    if st.button("Add Relationship"):
        if "relationships" not in st.session_state:
            st.session_state.relationships = []

        # Add the relationship
        st.session_state.relationships.append((cause_column, effect_column))
        st.success(f"Added relationship: {cause_column} -> {effect_column}")

    # Display defined relationships
    if "relationships" in st.session_state and st.session_state.relationships:
        st.subheader("Defined Relationships")
        
        # Multi-select widget for relationships
        to_remove = st.multiselect(
            "Select relationships to remove:",
            ["{} -> {}".format(relation[0], relation[1]) for relation in st.session_state.relationships]
        )
        
        # Button to remove selected relationships
        if st.button("Remove Selected Relationships"):
            for relation_str in to_remove:
                cause, effect = relation_str.split(" -> ")
                st.session_state.relationships.remove((cause, effect))
            st.success(f"Removed {len(to_remove)} relationships.")

    # Generate and display the causal graph
    if st.button("Generate Causal Graph"):
        dot_representation = "digraph {\n"
        for relation in st.session_state.relationships:
            dot_representation += f'    "{relation[0]}" -> "{relation[1]}";\n'
        dot_representation += "}"

        # Explicitly set it in the session state
        st.session_state.dot_representation = dot_representation

        graph = graphviz.Source(dot_representation)
        st.graphviz_chart(graph.source)
        st.session_state.generated_graph = True  # Mark that the graph has been generated

        # Provide the download link for the DOT file
        st.markdown(generate_dot_download_link(dot_representation), unsafe_allow_html=True)





def display_causal_model_creation():
    """Sub-task for creating the causal model based on the defined graph."""
    
    columns = list(st.session_state.data.columns)
    
    # Check if the causal graph has been generated
    if not st.session_state.get("generated_graph", False):
        st.warning("Please generate the causal graph first.")
        return

    # Ensure that the dot_representation is not empty
    dot_representation = st.session_state.get("dot_representation", "")
    if not dot_representation:
        st.warning("Please generate or upload a causal graph first.")
        return

    # Causal model creation
    treatment = st.selectbox("Select Treatment (cause) Variable", columns)
    outcome = st.selectbox("Select Outcome (effect) Variable", columns)
    st.write("""
    The treatment variable is what you believe to be the cause in your causal relationship, 
    and the outcome variable is the effect you are studying.
    """)

    if st.button("Create and Estimate Causal Model"):
        # Define Causal Model
        model = CausalModel(
            data=st.session_state.data,
            treatment=treatment,
            outcome=outcome,
            graph=st.session_state.get("dot_representation", "")
        )
        
        
        # Identification
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        st.session_state.identified_estimand = identified_estimand
        st.write("Identified estimand:", identified_estimand)

        # Estimation
        estimate = model.estimate_effect(identified_estimand,
                                         method_name="backdoor.linear_regression",
                                         control_value=0,
                                         treatment_value=1,
                                         confidence_intervals=True,
                                         test_significance=True)
        st.write("Causal Estimate:", estimate.value)
        
        st.session_state.causal_model = model
        st.session_state.estimate = estimate
        st.success("Causal model created and estimated successfully!")




def generate_download_link(filename, download_text):
    """Generate a download link for a given file and link text."""
    with open(filename, "rb") as f:
        file_data = f.read()
    b64 = base64.b64encode(file_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{download_text}</a>'
    return href




    



def display_refutation_tests():
    """Sub-task for running refutation tests."""
    st.subheader("Refutation Tests")

    # Ensure that the causal model is created
    if "causal_model" not in st.session_state:
        st.warning("Please create a causal model first.")
        return

    # Refutation methods
    methods = ["Data Subset Refuter"]
    chosen_method = st.selectbox("Choose a Refutation Method", methods)

    # Customize parameters based on the chosen method
    if chosen_method == "Data Subset Refuter":
        subset_fraction = st.slider("Choose a fraction of data to keep", 0.1, 1.0, 0.5)

    # Run refutation
    if st.button("Run Refutation"):
        if "identified_estimand" not in st.session_state or "estimate" not in st.session_state:
            st.warning("Please create and estimate the causal model first.")
            return

        refuter = DataSubsetRefuter(
            data=st.session_state.data,
            causal_model=st.session_state.causal_model,
            identified_estimand=st.session_state.identified_estimand,
            estimate=st.session_state.estimate,
            subset_fraction=subset_fraction
        )
        results = refuter.refute_estimate()
        st.write("Refutation Results:", results)
        
        # Extract p_value from the results
        # Assuming it's in the results string, you might need to adjust the extraction method
        p_value_str = re.search(r'p value:(\d+\.\d+)', str(results))
        if p_value_str:
            p_value = float(p_value_str.group(1))
        else:
            p_value = None

        # Interpretation based on p-value and difference in effects
        original_effect = st.session_state.estimate.value
        new_effect = results.new_effect

        if p_value and p_value > 0.05 and abs(original_effect - new_effect) < 0.05 * abs(original_effect):  # Assuming a 5% relative difference threshold for "close"
            st.write("Interpretation: The original causal estimate is consistent and robust, even when using a subset of the data.")
        elif p_value and p_value <= 0.05:
            st.write("Interpretation: The original causal estimate may not be reliable, as it changes significantly with a subset of the data.")






def causality_page():
    st.header("Causality Analysis")
    
    # Ensure data is uploaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload data first.")
        return
    
    # Moved task selection to sidebar
    task = st.sidebar.radio("Choose a Causality Sub-task", ["Define Relationships", "Create Causal Model", "Run Refutation Tests"])

    if task == "Define Relationships":
        display_relationships_definition()
    elif task == "Create Causal Model":
        display_causal_model_creation()
    elif task == "Run Refutation Tests":
        display_refutation_tests()


