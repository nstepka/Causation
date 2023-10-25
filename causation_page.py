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
from graphviz import Digraph
from io import BytesIO

def extract_relationships_from_dot(dot_representation):
    """Extract relationships from DOT format."""
    relationships = []
    
    # Split the string into lines and filter out non-edge lines
    lines = dot_representation.split("\n")
    edge_lines = [line.strip() for line in lines if "->" in line]
    
    for edge_line in edge_lines:
        # Extract cause and effect from the edge line
        cause, effect = edge_line.split("->")
        relationships.append((cause.strip(), effect.strip()))
    
    return relationships

def display_relationships_definition():
    st.subheader("Define Causal Relationships")
    
    # Ensure data is uploaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload data first.")
        return

    columns = list(st.session_state.data.columns)

    # Dropdowns to select cause and effect columns
    cause_column = st.selectbox("Select Cause Column", columns)
    effect_column = st.selectbox("Select Effect Column", columns)

    # Add relationship button
    if st.button("Add Relationship"):
        relationship = (cause_column, effect_column)
        if "relationships" not in st.session_state:
            st.session_state.relationships = []
        if relationship not in st.session_state.relationships:
            st.session_state.relationships.append(relationship)
            st.success(f"Added relationship: {cause_column} -> {effect_column}")

    # Display existing relationships and allow removal
    if "relationships" in st.session_state and st.session_state.relationships:
        st.write("Defined Relationships:")
        
        st.write("Select Relationship to Remove:")
        # Create a dropdown with all relationships formatted as "cause -> effect"
        relationship_options = [f"{cause} -> {effect}" for cause, effect in st.session_state.relationships]
        selected_relationship_str = st.selectbox("Select Relationship:", relationship_options)
        
        # Extract cause and effect from the selected string
        selected_cause, selected_effect = selected_relationship_str.split(" -> ")

        if st.button(f"Remove {selected_cause} -> {selected_effect}"):
            st.session_state.relationships.remove((selected_cause, selected_effect))
            st.success(f"Removed relationship: {selected_cause} -> {selected_effect}")

    # Upload a .dot file and load the graph
    uploaded_file = st.file_uploader("Upload a .dot file to load a causal graph", type="dot")
    if uploaded_file:
        uploaded_graph = uploaded_file.read().decode()
        
        # Extract relationships from the uploaded graph
        uploaded_relationships = extract_relationships_from_dot(uploaded_graph)
        
        # Add the extracted relationships to st.session_state.relationships
        if "relationships" not in st.session_state:
            st.session_state.relationships = []
        
        for relationship in uploaded_relationships:
            if relationship not in st.session_state.relationships:
                st.session_state.relationships.append(relationship)

        st.success(f"Loaded {len(uploaded_relationships)} relationships from the provided .dot file.")

    # Generate causal graph button
    if st.button("Generate Causal Graph"):
        if "relationships" not in st.session_state or not st.session_state.relationships:
            st.warning("Please define at least one relationship before generating the graph.")
        else:
            # Generate graph using Graphviz
            dot = Digraph()
            for cause, effect in st.session_state.relationships:
                dot.edge(cause, effect)
            
            # Convert dot to string and save in session state
            st.session_state.dot_representation = dot.source
            st.session_state.generated_graph = True
            
            # Display the graph
            st.graphviz_chart(dot)
            
            # Provide a download link for the graph
            # Save the graph representation as a temporary .dot file
            temp_filename = "causal_graph.dot"
            with open(temp_filename, "w") as f:
                f.write(dot.source)
            
            st.markdown(generate_download_link(temp_filename, "Download causal graph (.dot)"), unsafe_allow_html=True)


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
