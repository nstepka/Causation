import time
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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import base64
from sklearn.inspection import permutation_importance
import time
import plotly.express as px


def explore_data():
    if 'data' in st.session_state and st.session_state.data is not None:
        sub_page = st.sidebar.radio(
            "Choose an Explore the Data task:",
            ["Boxplot Visualization", "Binary Distribution",
             "Feature Comparison Graph", "Evaluate Feature Importance"]
        )
        if sub_page == "Boxplot Visualization":
            display_boxplot()
        elif sub_page == "Binary Distribution":
            display_binary_distribution()
        elif sub_page == "Feature Comparison Graph":
            feature_comparison_graph_page()
        elif sub_page == "Evaluate Feature Importance":
            evaluate_feature_importance()
    else:
        st.warning("Please upload data first.")



def display_boxplot():
    # Allow user to select a feature/column for the boxplot
    feature_to_plot = st.selectbox("Select a feature for the boxplot:", st.session_state.data.columns)

    if st.button("Submit"):
        # Generate initial boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        st.session_state.data.boxplot(column=feature_to_plot, ax=ax)
        ax.set_title(f"Boxplot for {feature_to_plot}")
        st.pyplot(fig)
        plt.close(fig)  # Close the figure to free up memory

        # Display some statistics for the selected feature
        avg_value = st.session_state.data[feature_to_plot].mean()
        min_value = st.session_state.data[feature_to_plot].min()
        max_value = st.session_state.data[feature_to_plot].max()
        st.write(f"Average value for {feature_to_plot}: {avg_value:.2f}")
        st.write(f"Minimum value for {feature_to_plot}: {min_value}")
        st.write(f"Maximum value for {feature_to_plot}: {max_value}")

        # Ask user if they want to set a min and max threshold
        min_val = st.number_input(f"Set a minimum value for {feature_to_plot}:", value=min_value)
        max_val = st.number_input(f"Set a maximum value for {feature_to_plot}:", value=max_value)

        if st.button("Apply Threshold"):
            # Filter only the rows where the selected feature's values are within the specified range
            filtered_data = st.session_state.data[(st.session_state.data[feature_to_plot] >= min_val) &
                                                  (st.session_state.data[feature_to_plot] <= max_val)]

            # Update the session state data
            st.session_state.data = filtered_data

            # Generate updated boxplot after applying threshold
            fig, ax = plt.subplots(figsize=(10, 6))
            st.session_state.data.boxplot(column=feature_to_plot, ax=ax)
            ax.set_title(f"Boxplot for {feature_to_plot} (After Applying Threshold)")
            st.pyplot(fig)
            plt.close(fig)  # Close the figure to free up memory

            # Display feedback to confirm data has been filtered
            new_avg = st.session_state.data[feature_to_plot].mean()
            new_min = st.session_state.data[feature_to_plot].min()
            new_max = st.session_state.data[feature_to_plot].max()
            st.write(f"New average value for {feature_to_plot}: {new_avg:.2f}")
            st.write(f"New minimum value for {feature_to_plot}: {new_min}")
            st.write(f"New maximum value for {feature_to_plot}: {new_max}")

def display_binary_distribution():
    # Identify binary columns (only containing 0s and 1s)
    binary_cols = [col for col in st.session_state.data.columns if st.session_state.data[col].nunique() == 2 and 
                   st.session_state.data[col].isin([0, 1]).all()]

    if not binary_cols:
        st.warning("No binary columns found in the dataset!")
        return

    # Allow user to select a binary column to visualize
    binary_col_to_plot = st.selectbox("Select a binary column to visualize:", binary_cols)

    if st.button("Plot"):
        fig, ax = plt.subplots()
        
        st.session_state.data[binary_col_to_plot].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Distribution of {binary_col_to_plot}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        st.pyplot(fig)
            

def feature_comparison_graph_page():
    data = st.session_state.data

    # Dropdown for selecting target feature
    target_feature = st.selectbox(
        "Select the target feature for y-axis",
        [col for col in data.columns]
    )

    # Grid of checkboxes for other features
    st.write("Select the features for visualization on x-axis:")
    features = [col for col in data.columns if col != target_feature]
    selected_features = []
    for i in range(0, len(features), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(features):
                is_checked = cols[j].checkbox(features[i + j])
                if is_checked:
                    selected_features.append(features[i + j])

    # Button to generate the visualizations
    if st.button("Generate Visualizations"):
        for feature in selected_features:
            plot_avg_for_feature(data, target_feature, feature)


def plot_avg_for_feature(data, target_feature, feature):
    avg_values = data.groupby(feature)[target_feature].mean()

    # Generate colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(avg_values)))

    # Plot
    plt.figure(figsize=(10, 6))
    avg_values.plot(kind='bar', color=colors)
    plt.ylabel(target_feature)
    plt.xlabel(feature)
    plt.title(f'Average {target_feature} by {feature}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()






def evaluate_feature_importance():
    st.write("""
    In this section, we will evaluate the importance of each feature using permutation importance. 
    The goal is to understand the contribution of each feature to the model's performance. 
    This can help in feature selection by identifying and keeping only the most influential features.
             
    THIS WILL TAKE A MOMENT TO RUN!  The page will not refresh after you submit the features
    you want to drop
    """)

    # Allow the user to select the target feature
    selected_target_feature = st.selectbox("Select the target feature:", st.session_state.data.columns)

    if st.button("Evaluate Features"):
        st.session_state.feature_evaluation_done = False
        st.session_state.selected_features = []

        # Using the already loaded data from st.session_state.data
        X_sample = st.session_state.data.drop(columns=[selected_target_feature])

        y_sample = st.session_state.data[selected_target_feature]

        # Convert all column names to strings
        X_sample.columns = X_sample.columns.astype(str)

        X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

        # Train the model with all features
        model = HistGradientBoostingRegressor(max_iter=50, max_depth=4)
        model.fit(X_train_sample, y_train_sample)

        # Estimate feature importances using permutation importance
        perm_importance = permutation_importance(model, X_train_sample, y_train_sample, n_repeats=10, random_state=42)
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        st.session_state.sorted_features = X_sample.columns[sorted_idx]  # Storing in session_state

        # Plotting feature importances using Plotly
        fig = px.bar(x=st.session_state.sorted_features, y=perm_importance.importances_mean[sorted_idx], labels={'x': 'Features', 'y': 'Importance'},
                     title='Feature Importances')
        st.plotly_chart(fig)

        st.session_state.feature_evaluation_done = True

    if "feature_evaluation_done" in st.session_state and st.session_state.feature_evaluation_done:
        # This code block runs after the feature evaluation is done.
        
        st.write("Top Features and Their Correlation Strengths:")
        st.write("Please UNCHECK the features you want to DROP from the dataset.")
        num_top_features = len(st.session_state.sorted_features)
        for feature in st.session_state.sorted_features:
            is_checked = st.checkbox(feature, value=True)
            if not is_checked:
                st.session_state.selected_features.append(feature)

        if st.button("Submit Features"):
            # Drop only the features that are in the dataframe and selected for removal
            features_to_drop = [feature for feature in st.session_state.selected_features if feature in st.session_state.data.columns]
            
            st.session_state.data.drop(columns=features_to_drop, inplace=True)
            
            # Refresh the page to show updated features
            st.experimental_rerun()

            st.write(f"Dropped the following features: {', '.join(features_to_drop)}")



