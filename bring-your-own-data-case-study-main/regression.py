from feature_engineering import display_data_preview

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB



def evaluate_model_page():
    if 'data' not in st.session_state:
        st.warning("Please upload data first.")
        return   

    # Previous task type (if exists)
    previous_task_type = st.session_state.get('task_type', None)

    # Ask the user for the type of machine learning task
    task_type = st.sidebar.radio(
        "Choose a machine learning task:",
        ["Regression", "Classification"]
    ).lower()

    # If task type is switched, clear the previous results and models
    if previous_task_type and previous_task_type != task_type:
        for key in ['results_df', 'y_preds', 'trained_models', 'X_train', 'y_test']:
            if key in st.session_state:
                del st.session_state[key]

    # Store the task type in the session state for use in other functions
    st.session_state.task_type = task_type

    # Sidebar for navigation within model evaluation
    sub_page = st.sidebar.radio(
        "Choose a task:",
        ["Data Preview", "Select Target & Features & Train", 
         "Model Performance Comparison", "Feature Importance", "Correlation Heatmap", 
         "Prediction vs. Actual Plot", "Residuals Plot"]
    )

    if sub_page == "Data Preview":
        display_data_preview()
    elif sub_page == "Select Target & Features & Train":
        display_select_target_features_and_train()
    elif sub_page == "Model Performance Comparison":
        display_model_performance_comparison()
    elif sub_page == "Feature Importance":
        display_feature_importance()
    elif sub_page == "Correlation Heatmap":
        display_correlation_heatmap()
    elif sub_page == "Prediction vs. Actual Plot":
        display_prediction_vs_actual()
    elif sub_page == "Residuals Plot":
        display_residuals_plot()



def display_correlation_heatmap():
    try:
        correlation_matrix = st.session_state.data[st.session_state.selected_features_grid].corr()
        # rest of the function
        st.write("Correlation Heatmap:")
    
        # Use seaborn to create the heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    
        # Display the heatmap in Streamlit
        st.pyplot(plt)
    except KeyError:
        st.warning("One or more selected features are not available. Please reselect target and features and retrain the model.")

    

def display_select_target_features_and_train():
    st.write("Select Target and Features")
    
    # Select target column
    st.session_state.target_col = st.selectbox("Select the target column", st.session_state.data.columns)

    # Feature selection grid
    st.write("Select features to use in the model:")

    # Break down the columns (excluding the target column) into chunks for the 3x3 grid
    available_features = [col for col in st.session_state.data.columns if col != st.session_state.target_col]
    col_chunks = [available_features[i:i + 3] for i in range(0, len(available_features), 3)]
    
    # Use session state to store selected features
    if 'selected_features_grid' not in st.session_state:
        st.session_state.selected_features_grid = available_features

    # Display the 3x3 grid with checkboxes
    for chunk in col_chunks:
        cols = st.columns(3)
        for i, col_name in enumerate(chunk):
            if len(chunk) > i:  # Check if the column exists (for the last row which might not be full)
                if cols[i].checkbox(col_name, key=col_name, value=(col_name in st.session_state.selected_features_grid)):
                    if col_name not in st.session_state.selected_features_grid:
                        st.session_state.selected_features_grid.append(col_name)
                else:
                    if col_name in st.session_state.selected_features_grid:
                        st.session_state.selected_features_grid.remove(col_name)

    # Allow users to select which models to use for training
    task_type = st.session_state.task_type
    if task_type == "regression":
        available_models = ["GradientBoosting", "RandomForest", "Linear", "HistGradientBoosting", "DecisionTree", "XGBoost", "KNN"]
    else:  # "classification"
        available_models = ["KNN", "Logistic Regression", "SVM", "Random Forest Classifier", "Gradient Boosting Classifier", "Neural Network", "Naive Bayes"]
    
    # Let the user decide on the max_iter value for Logistic Regression (default to 1000 if not set)
    logistic_max_iter = st.sidebar.number_input("Logistic Regression max_iter", min_value=10, max_value=10000, value=1000)

    
    selected_models = st.multiselect("Select models to train:", available_models, default=available_models)
    # In the display_select_target_features_and_train() function
    test_size = st.sidebar.slider("Choose test size (test/train split ratio):", min_value=0.1, max_value=0.9, value=0.2, step=0.05)
    st.session_state.test_size = test_size  # Store the test size in session state

    if st.button("Train and Evaluate Models"):
        X_train, X_test, y_train, y_test = prepare_data(st.session_state.data, st.session_state.target_col, st.session_state.selected_features_grid, task_type, test_size)

        # Displaying the training and test sample sizes:
        st.write(f"Training Sample Size: {len(X_train)}")
        st.write(f"Test Sample Size: {len(X_test)}")
        models = create_models(selected_models, task_type)
        models = fit_models(X_train, y_train, models)
        results_df, y_preds = evaluate_models(X_test, y_test, models, task_type)
        
        # Store results_df, y_preds, and trained_models in the session state
        st.session_state.results_df = results_df
        st.session_state.y_preds = y_preds
        st.session_state.trained_models = models
        st.session_state.X_train = X_train
        st.session_state.y_test = y_test




def display_model_performance_comparison():
    if 'selected_features_grid' not in st.session_state:
        st.warning("Please select target and features first.")
        return

    # Check if 'results_df' exists in st.session_state and if it's not empty
    if 'results_df' in st.session_state and not st.session_state.results_df.empty:
        st.write("Model Evaluation Results:")  # Display the results here
        st.write(st.session_state.results_df)
        
        # Display the test/train split percentage
        if 'test_size' in st.session_state:
            st.write(f"Test/Train Split: {st.session_state.test_size*100:.0f}% Test - {100*(1-st.session_state.test_size):.0f}% Train")

        # Plot model performance without passing the task_type argument
        st.write("Model Performance Comparison:")
        st.pyplot(plot_model_performance(st.session_state.results_df))
        
        if st.session_state.task_type == "classification":
            st.write("""
            **Metrics Explanation:**
            - **Accuracy**: The ratio of correctly predicted instances to the total instances. Gives a general idea of the model's performance.
            - **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. Measures a classifier's exactness.
            - **Recall**: The ratio of correctly predicted positive observations to all the observations in the actual class. Measures a classifier's completeness.
            - **F1 Score**: The weighted average of Precision and Recall. Tries to find the balance between precision and recall.
            """)
    else:
        st.warning("Please select target, features and train the models first.")



def prepare_data(df, target_col, selected_features, task_type="regression", test_size=0.2):
    """
    Prepare the data for model training and evaluation by splitting it into training and test sets.
    
    Parameters:
    - df: The input dataframe.
    - target_col: The target column name.
    - selected_features: The features selected by the user.
    - task_type: The machine learning task type. Default is "regression".
    - test_size: The proportion of the dataset to include in the test split. Default is 0.2.
    
    Returns:
    - X_train, X_test, y_train, y_test: Training and test sets.
    """
    # Ensure the target_col is not part of the selected features
    selected_features = [col for col in selected_features if col != target_col]
    
    X = df[selected_features]
    y = df[target_col]

    # If classification, ensure target is encoded properly
    if task_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return train_test_split(X, y, test_size=test_size, random_state=42)




def create_models(selected_models, task_type="regression"):
    """
    Create a dictionary of models based on the task type and selected models.
    
    Parameters:
    - selected_models: A list of model names selected by the user.
    - task_type: A string indicating the type of machine learning task. 
                 Accepts "regression" or "classification". Default is "regression".

    Returns:
    - A dictionary of models.
    """
    available_models = {}
    
    if task_type == "regression":
        available_models = {
            'GradientBoosting': GradientBoostingRegressor(),
            'RandomForest': RandomForestRegressor(),
            'Linear': LinearRegression(),
            'HistGradientBoosting': HistGradientBoostingRegressor(),
            'DecisionTree': DecisionTreeRegressor(),
            'XGBoost': xgb.XGBRegressor(),
            'KNN': KNeighborsRegressor()
        }
    elif task_type == "classification":
        available_models = {
            'KNN': KNeighborsClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(),
            'Random Forest Classifier': RandomForestClassifier(),
            'Gradient Boosting Classifier': GradientBoostingClassifier(),
            'Neural Network': MLPClassifier(max_iter=1000),
            'Naive Bayes': GaussianNB()
        }
    
    # Filter out the models not selected by the user
    return {name: model for name, model in available_models.items() if name in selected_models}





def fit_models(X_train, y_train, models):
    for name, model in models.items():
        model.fit(X_train.astype(float), y_train.astype(float))
        print(f"Finished training {name}")
    return models


def evaluate_models(X_test, y_test, models, task_type="regression"):
    results = []
    y_preds = {}  # Dictionary to store predictions for each model
    
    for name, model in models.items():
        y_pred = model.predict(X_test.astype(float))
        y_preds[name] = y_pred  # Store predictions in the dictionary

        if task_type == "regression":
            r2 = r2_score(y_test.astype(float), y_pred)
            mse = mean_squared_error(y_test.astype(float), y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test.astype(float), y_pred)
            results.append({
                'Model': name,
                'R2': r2,
                'RMSE': rmse,
                'MSE': mse,
                'MAE': mae
            })
        elif task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })

        print(f"Finished evaluating {name}")

    return pd.DataFrame(results), y_preds



def plot_model_performance(results_df):
    task_type = st.session_state.task_type
    if task_type == "regression":
        metrics = ['R2', 'RMSE', 'MSE', 'MAE']
    else:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Adjust the figure size to have a larger height
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 24))
    fig.suptitle('Model Performance Comparison', fontsize=20, y=1.1, color='darkblue')

    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df['Model'])))
    
    for idx, metric in enumerate(metrics):
        results_df.plot(x='Model', y=metric, kind='bar', ax=axes[idx], legend=None, color=colors)
        axes[idx].set_ylabel(metric, fontsize=16, color='darkgreen')
        axes[idx].set_title(f'{metric} Comparison Among Models', fontsize=18, color='darkred')
        axes[idx].set_ylim(bottom=0)
        axes[idx].set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=14)
        axes[idx].grid(axis='y')

        # Remove left and right spines for aesthetics
        axes[idx].spines['right'].set_visible(False)
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['left'].set_color('gray')
        axes[idx].spines['bottom'].set_color('gray')

    # Adjust the spacing between the subplots
    plt.subplots_adjust(hspace=0.7)  # Increase the vertical spacing
    
    return fig





def display_feature_importance():
    if 'selected_features_grid' not in st.session_state:
        st.warning("Please select target and features first.")
        return
    
    # Check if the trained models are available in the session state
    if 'trained_models' not in st.session_state:
        st.warning("Please train the models first.")
        return

    model_name = st.selectbox("Select a model to view feature importance", list(st.session_state.trained_models.keys()))
    model = st.session_state.trained_models[model_name]
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        # Ensure feature_names matches the features used in the trained model
        if hasattr(st.session_state.X_train, 'columns'):  # if X_train is a DataFrame
            feature_names = st.session_state.X_train.columns
        else:  # if X_train is a numpy array, fallback to the session state (might be unsafe)
            feature_names = st.session_state.selected_features_grid

        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
        # Convert the sorted Series to a DataFrame for display in Streamlit
        feature_importances_df = feature_importances.reset_index()
        feature_importances_df.columns = ['Feature', 'Importance']

        st.bar_chart(feature_importances_df.set_index('Feature'))
    else:
        st.warning(f"The selected model ({model_name}) does not provide feature importance.")


def display_prediction_vs_actual():
    # Ensure that all required session state variables exist
    if 'y_preds' not in st.session_state or 'trained_models' not in st.session_state or 'y_test' not in st.session_state:
        st.warning("Please train and evaluate models first.")
        return

    # Get models, y_preds, and y_test from session state
    models = st.session_state.trained_models
    y_preds = st.session_state.y_preds
    y_test = st.session_state.y_test

    model_name = st.selectbox("Select a model to view its predictions vs actual values", list(models.keys()))
    y_pred = y_preds[model_name]
    
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    st.pyplot(fig)

def display_residuals_plot():
    if 'y_preds' not in st.session_state or 'trained_models' not in st.session_state or 'y_test' not in st.session_state:
        st.warning("Please train and evaluate models first.")
        return

    # Get models, y_preds, and y_test from session state
    models = st.session_state.trained_models
    y_preds = st.session_state.y_preds
    y_test = st.session_state.y_test

    model_name = st.selectbox("Select a model to view its residuals", list(models.keys()))
    y_pred = y_preds[model_name]
    residuals = y_test - y_pred

    st.write("Residuals Plot:")
    st.line_chart(residuals)
