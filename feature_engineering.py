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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import base64
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler




def feature_engineering():
    if 'data' in st.session_state and st.session_state.data is not None:
        sub_page = st.sidebar.radio(
            "Choose a Feature Engineering task:",
            ["Data Preview", "Handle Missing Values", "Process Currency and Percentage", "Drop Columns", 
             "Data Transformation", "Encoding Categorical Variables", "Time Series Features", "Convert to Datetime", 
             "Binning/Bucketing","Custom Feature Engineering"]  # Added "Binning/Bucketing" to the list
        )
        if sub_page == "Data Preview":
            display_data_preview()
        elif sub_page == "Handle Missing Values":
            display_handle_missing_values()
        elif sub_page == "Process Currency and Percentage":
            display_process_currency_percentage()
        elif sub_page == "Drop Columns":
            display_drop_columns()
        elif sub_page == "Data Transformation":
            display_data_transformation()
        elif sub_page == "Encoding Categorical Variables":
            display_encode_categorical()
        elif sub_page == "Time Series Features":
            display_time_series_features()
        elif sub_page == "Convert to Datetime":
            display_convert_to_datetime()
        elif sub_page == "Binning/Bucketing":  # Added this section for "Binning/Bucketing"
            display_binning_bucketing()
        elif sub_page == "Custom Feature Engineering":
            display_custom_feature_engineering()
    else:
        st.warning("Please upload data first.")


def display_data_preview():
    st.write("Data Preview:")
    st.write(st.session_state.data.head(10))
    ##describe the data and write to screen
    st.write("Data Description:")
    st.write(st.session_state.data.describe())
    



def display_handle_missing_values():
    # Step 3: Check for missing values
    missing_data = st.session_state.data.isnull().sum()
    missing_columns = missing_data[missing_data > 0]

    if not missing_columns.empty:
        missing_values_placeholder = st.empty()
        missing_values_placeholder.write("Columns with missing values:")
        missing_values_placeholder.write(missing_columns)

        # Step 4: Handle missing values
        column_to_handle = st.selectbox("Select a column to handle:", missing_columns.index)
        action = st.selectbox("Choose an action:", ["Fill missing values", "Drop column", "Drop rows with missing values"])

        fill_method = None
        if action == "Fill missing values":
            fill_method = st.selectbox("Select a method to fill the missing values:", 
                                       ["mean", "median", "mode", "constant", "KNN imputation"])

        constant_value = ""
        if fill_method == "constant":
            constant_value = st.text_input("Enter the constant value:")

        knn_neighbors = 5
        if fill_method == "KNN imputation":
            knn_neighbors = st.slider("Select number of neighbors for KNN imputation:", 1, 50, 5)

        if st.button("Submit"):
            if action == "Fill missing values":
                if fill_method in ["mean", "median"] and not pd.api.types.is_numeric_dtype(st.session_state.data[column_to_handle]):
                    st.warning("Selected column is not numeric. Please choose another method or column.")
                else:
                    if fill_method == "mean":
                        st.session_state.data[column_to_handle] = st.session_state.data[column_to_handle].fillna(st.session_state.data[column_to_handle].mean())
                    elif fill_method == "median":
                        st.session_state.data[column_to_handle] = st.session_state.data[column_to_handle].fillna(st.session_state.data[column_to_handle].median())
                    elif fill_method == "mode":
                        st.session_state.data[column_to_handle] = st.session_state.data[column_to_handle].fillna(st.session_state.data[column_to_handle].mode()[0])
                    elif fill_method == "constant":
                        st.session_state.data[column_to_handle] = st.session_state.data[column_to_handle].fillna(constant_value)
                    elif fill_method == "KNN imputation":
                        imputer = KNNImputer(n_neighbors=knn_neighbors)
                        st.session_state.data[column_to_handle] = imputer.fit_transform(st.session_state.data[[column_to_handle]]).ravel()
                        st.success(f"Missing values in {column_to_handle} filled with KNN imputation!")
                    
            elif action == "Drop column":
                st.session_state.data.drop(columns=[column_to_handle], inplace=True)
                st.success(f"Column {column_to_handle} dropped!")

            elif action == "Drop rows with missing values":
                st.session_state.data.dropna(subset=[column_to_handle], inplace=True)
                st.success(f"Rows with missing values in {column_to_handle} dropped!")

            # Use experimental_rerun to refresh the app state
            st.experimental_rerun()

    else:
        st.success("There are no missing values in the dataset!")



def display_process_currency_percentage():
    # Detect columns with currency and percentage
    currency_cols = [col for col in st.session_state.data.columns if st.session_state.data[col].astype(str).str.contains('\$').any()]
    percent_cols = [col for col in st.session_state.data.columns if st.session_state.data[col].astype(str).str.contains('%').any()]

    # Initialize ignored columns in session state if not present
    if 'ignored_currency_cols' not in st.session_state:
        st.session_state.ignored_currency_cols = []
    if 'ignored_percent_cols' not in st.session_state:
        st.session_state.ignored_percent_cols = []

    # Filter out ignored columns from the main list
    currency_cols = [col for col in currency_cols if col not in st.session_state.ignored_currency_cols]
    percent_cols = [col for col in percent_cols if col not in st.session_state.ignored_percent_cols]

    col1, col2 = st.columns(2)

    if currency_cols:
        with col1:
            col1.write("Columns with currency values:")
            col1.write(currency_cols)

            # Allow user to select which currency column to process
            currency_column_to_process = col1.selectbox("Select a currency column to process:", currency_cols)
            currency_action = col1.radio("Choose an action:", ["Remove $ and convert to float", "Ignore"])

            if col1.button(f"Submit {currency_column_to_process}"):
                if currency_action == "Remove $ and convert to float":
                    st.session_state.data[currency_column_to_process] = st.session_state.data[currency_column_to_process].replace('[\$,]', '', regex=True).astype(float)
                    col1.success(f"Processed {currency_column_to_process} by removing $!")
                elif currency_action == "Ignore":
                    st.session_state.ignored_currency_cols.append(currency_column_to_process)

                # Use experimental_rerun to refresh the app state
                st.experimental_rerun()

    if percent_cols:
        with col1:
            col1.write("Columns with percentage values:")
            col1.write(percent_cols)

            # Allow user to select which percentage column to process
            percent_column_to_process = col1.selectbox("Select a percentage column to process:", percent_cols)
            percent_action = col1.radio("Choose an action for percentage column:", ["Convert % to fraction", "Ignore"])

            if col1.button(f"Submit {percent_column_to_process}", key=f"SubmitButton_{percent_column_to_process}"):
                if percent_action == "Convert % to fraction":
                    st.session_state.data[percent_column_to_process] = st.session_state.data[percent_column_to_process].str.rstrip('%').astype('float') / 100.0
                    col1.success(f"Processed {percent_column_to_process} by converting % to fraction!")
                elif percent_action == "Ignore":
                    st.session_state.ignored_percent_cols.append(percent_column_to_process)

                # Use experimental_rerun to refresh the app state
                st.experimental_rerun()

    with col2:
        col2.write("Ignored columns with currency values:")
        col2.write(st.session_state.ignored_currency_cols)
        col2.write("Ignored columns with percentage values:")
        col2.write(st.session_state.ignored_percent_cols)


def display_drop_columns():
    """Displays a grid of columns with checkboxes to allow users to drop selected columns."""
    st.write("Select the columns you wish to drop:")

    # Initialize selected_features_grid if it's not already in session state
    if 'selected_features_grid' not in st.session_state:
        st.session_state.selected_features_grid = []

    # Arrange columns in a grid
    num_columns = 4  # Define the number of columns for the grid
    col_chunks = [st.session_state.data.columns[i:i + num_columns] for i in range(0, len(st.session_state.data.columns), num_columns)]

    columns_to_drop = []
    for chunk in col_chunks:
        cols = st.columns(len(chunk))
        for i, col_name in enumerate(chunk):
            if cols[i].checkbox(col_name):
                columns_to_drop.append(col_name)

    # Button to submit the selected columns to drop
    if st.button("Drop Selected Columns"):
        st.session_state.data.drop(columns=columns_to_drop, inplace=True)
        
        # Remove the dropped columns from the selected features grid if present
        for col in columns_to_drop:
            if col in st.session_state.selected_features_grid:
                st.session_state.selected_features_grid.remove(col)
                
        st.success(f"Dropped columns: {', '.join(columns_to_drop)}!")
        st.experimental_rerun()


def display_data_transformation():
    st.write("Choose a data transformation method:")

    transformation_choice = st.selectbox(
        "Select a transformation method:",
        ["Normalization", "Standardization", "Log Transformation", "Robust Scaling"]  # Added "Robust Scaling" to the list
    )

    # Initialize feedback_message in session state if not present
    if 'feedback_message' not in st.session_state:
        st.session_state.feedback_message = ""

    if st.button("Submit"):
        # Reset feedback message
        st.session_state.feedback_message = ""

        # Select only numeric columns
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()

        # Filter out columns with all NaN values or infinite values
        valid_cols = [col for col in numeric_cols if not st.session_state.data[col].isna().all()]
        valid_cols = [col for col in valid_cols if not np.isinf(st.session_state.data[col]).any()]

        try:
            if transformation_choice == "Normalization":
                scaler = MinMaxScaler()
                st.session_state.data[valid_cols] = scaler.fit_transform(st.session_state.data[valid_cols])
                st.session_state.feedback_message = "Normalization successful!"

            elif transformation_choice == "Standardization":
                scaler = StandardScaler()
                st.session_state.data[valid_cols] = scaler.fit_transform(st.session_state.data[valid_cols])
                st.session_state.feedback_message = "Standardization successful!"

            elif transformation_choice == "Log Transformation":
                # Adding a small constant to avoid log(0)
                st.session_state.data[valid_cols] = np.log(st.session_state.data[valid_cols] + 1)
                st.session_state.feedback_message = "Log Transformation successful!"
            
            elif transformation_choice == "Robust Scaling":  # Added this section for "Robust Scaling"
                scaler = RobustScaler()
                st.session_state.data[valid_cols] = scaler.fit_transform(st.session_state.data[valid_cols])
                st.session_state.feedback_message = "Robust Scaling successful!"

        except Exception as e:
            st.session_state.feedback_message = f"{transformation_choice} failed! Error: {e}"

    # Display feedback message below the submit button
    st.write(st.session_state.feedback_message)



def display_encode_categorical():
    # Display a preview of the dataset
    st.write("Data preview:")
    st.write(st.session_state.data.head())

    # Allow user to select a column to encode
    categorical_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
    if not categorical_cols:
        st.warning("No categorical columns found in the dataset!")
        return
    
    column_to_encode = st.selectbox("Select a column to encode:", categorical_cols)
    
    # Display a preview of the selected column
    st.write(f"Preview of {column_to_encode}:")
    st.write(st.session_state.data[column_to_encode].head())

    # Allow user to choose encoding method
    encoding_method = st.selectbox("Choose an encoding method:", ["One-Hot Encoding", "Label Encoding"])
    
    if st.button("Encode"):
        if encoding_method == "One-Hot Encoding":
            # Split the comma-separated values to get a list of amenities
            split_data = st.session_state.data[column_to_encode].str.split(',').apply(lambda x: [i.strip() for i in x if i])

            # Use pd.get_dummies on these lists to one-hot encode the data
            dummies = pd.get_dummies(split_data.apply(pd.Series).stack()).sum(level=0)
            
            # Join the one-hot encoded columns to the original dataframe and drop the original column
            st.session_state.data = pd.concat([st.session_state.data, dummies], axis=1)
            st.session_state.data.drop(columns=[column_to_encode], inplace=True)
            st.success(f"{column_to_encode} encoded using One-Hot Encoding!")
            
        elif encoding_method == "Label Encoding":
            # Label encode the selected column
            le = LabelEncoder()
            st.session_state.data[column_to_encode] = le.fit_transform(st.session_state.data[column_to_encode])
            st.success(f"{column_to_encode} encoded using Label Encoding!")
        
        # Refresh the app state
        st.experimental_rerun()    

def display_time_series_features():
    st.write("Time Series Feature Engineering")

    # Heuristically detect potential datetime columns
    potential_datetime_cols = [col for col in st.session_state.data.columns if st.session_state.data[col].astype(str).str.contains('/|-').any()]

    if not potential_datetime_cols:
        st.warning("No potential datetime columns detected based on the heuristic!")
        return

    # Let the user choose which column to convert to datetime
    col_to_convert = st.selectbox("Select a column to potentially convert to datetime format:", potential_datetime_cols)

    # Display the first five rows of the selected column
    st.write(f"First five rows of {col_to_convert}:")
    st.write(st.session_state.data[col_to_convert].head())

    if st.button("Convert Selected Column"):
        try:
            st.session_state.data[col_to_convert] = pd.to_datetime(st.session_state.data[col_to_convert])
            st.success(f"Converted {col_to_convert} to datetime format!")
            st.experimental_rerun()
        except Exception:
            st.warning(f"Failed to convert {col_to_convert}. It may not have valid datetime values.")

    # Once columns are converted, display datetime columns for feature engineering
    datetime_cols = st.session_state.data.select_dtypes(include=[np.datetime64]).columns.tolist()

    if datetime_cols:
        column_to_process = st.selectbox("Select a datetime column to process:", datetime_cols)

        # Allow user to choose the feature extraction method
        feature_method = st.selectbox(
            "Choose a time series feature extraction method:",
            ["Date Extraction", "Lag Features", "Rolling Window"]
        )

        if feature_method == "Date Extraction":
            # Extract day, month, year, day of the week
            if st.button("Extract Date Features"):
                st.session_state.data[f"{column_to_process}_day"] = st.session_state.data[column_to_process].dt.day
                st.session_state.data[f"{column_to_process}_month"] = st.session_state.data[column_to_process].dt.month
                st.session_state.data[f"{column_to_process}_year"] = st.session_state.data[column_to_process].dt.year
                st.session_state.data[f"{column_to_process}_weekday"] = st.session_state.data[column_to_process].dt.weekday
                st.success(f"Date features extracted from {column_to_process}!")

        elif feature_method == "Lag Features":
            lag_period = st.number_input("Enter lag period:", value=1, min_value=1)
            if st.button("Generate Lag Features"):
                st.session_state.data[f"{column_to_process}_lag{lag_period}"] = st.session_state.data[column_to_process].shift(lag_period)
                st.success(f"Lag features with period {lag_period} generated for {column_to_process}!")

        elif feature_method == "Rolling Window":
            window_size = st.number_input("Enter window size for rolling average:", value=3, min_value=1)
            if st.button("Generate Rolling Window Feature"):
                st.session_state.data[f"{column_to_process}_rolling_avg{window_size}"] = st.session_state.data[column_to_process].rolling(window=window_size).mean()
                st.success(f"Rolling window feature with window size {window_size} generated for {column_to_process}!")

def display_convert_to_datetime():
    st.write("Select Columns to Convert to Datetime Format")

    # Break down the columns into chunks for the 3x3 grid
    col_chunks = [st.session_state.data.columns[i:i + 3] for i in range(0, len(st.session_state.data.columns), 3)]
    
    # Use session state to store selected columns to convert
    if 'columns_to_convert_grid' not in st.session_state:
        st.session_state.columns_to_convert_grid = []

    # Display the 3x3 grid with checkboxes
    for chunk in col_chunks:
        cols = st.columns(3)
        for i, col_name in enumerate(chunk):
            if len(chunk) > i:  # Check if the column exists (for the last row which might not be full)
                if cols[i].checkbox(col_name, key=col_name, value=(col_name in st.session_state.columns_to_convert_grid)):
                    if col_name not in st.session_state.columns_to_convert_grid:
                        st.session_state.columns_to_convert_grid.append(col_name)
                else:
                    if col_name in st.session_state.columns_to_convert_grid:
                        st.session_state.columns_to_convert_grid.remove(col_name)

    # Convert button
    if st.button("Convert Selected to Datetime"):
        for col in st.session_state.columns_to_convert_grid:
            try:
                st.session_state.data[col] = pd.to_datetime(st.session_state.data[col])
                st.success(f"{col} successfully converted to datetime format!")
            except Exception as e:
                st.error(f"Error converting {col} to datetime: {e}")
        # Clear the session state for selected columns to avoid carrying over selections
        st.session_state.columns_to_convert_grid = []
        # Refresh the app state
        st.experimental_rerun()

def display_binning_bucketing():
    st.write("Binning/Bucketing Continuous Variables")

    # Select continuous columns
    numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
    column_to_bin = st.selectbox("Select a column to bin/bucket:", numeric_cols)

    binning_choice = st.radio("Binning method:", ["Number of bins", "Custom bin edges"])
    
    if binning_choice == "Number of bins":
        n_bins = st.slider("Select the number of bins:", 2, 100, 5)
        bin_edges = None
    else:
        bin_edges_input = st.text_input("Enter custom bin edges separated by commas (e.g., 0,10,20,30):")
        try:
            bin_edges = [float(edge) for edge in bin_edges_input.split(",")]
        except ValueError:
            st.warning("Invalid bin edges input. Please enter numbers separated by commas.")
            return

    # Labeling choice
    label_choice = st.radio("Labeling method:", ["Numeric", "Custom labels"])
    labels = None
    if label_choice == "Custom labels":
        labels_input = st.text_input("Enter custom labels separated by commas:")
        labels = labels_input.split(",") if labels_input else []
        if bin_edges and len(labels) != (len(bin_edges) - 1):
            st.warning("Number of labels should be one less than the number of bin edges.")
            return

    if st.button("Bin/Bucket Column"):
        if binning_choice == "Number of bins":
            bins_result = pd.cut(st.session_state.data[column_to_bin], bins=n_bins, labels=labels)
        else:
            bins_result = pd.cut(st.session_state.data[column_to_bin], bins=bin_edges, labels=labels)
        
        st.session_state.data[f"{column_to_bin}_binned"] = bins_result
        st.success(f"{column_to_bin} has been binned/bucketed and stored in {column_to_bin}_binned column!")

        # Refresh the app state
        st.experimental_rerun()



def display_custom_feature_engineering():
    st.write("Custom Feature Engineering")

    # 1. Aggregate Columns
    st.subheader("Aggregate Columns")
    columns_to_aggregate = st.multiselect("Select columns to aggregate:", st.session_state.data.columns.tolist())
    aggregation_operation = st.selectbox("Select aggregation operation:", ["Sum", "Mean", "Min", "Max"])
    new_feature_name = st.text_input("Name for the new aggregated feature:")
    if st.button("Create Aggregated Feature"):
        if aggregation_operation == "Sum":
            st.session_state.data[new_feature_name] = st.session_state.data[columns_to_aggregate].sum(axis=1)
        elif aggregation_operation == "Mean":
            st.session_state.data[new_feature_name] = st.session_state.data[columns_to_aggregate].mean(axis=1)
        elif aggregation_operation == "Min":
            st.session_state.data[new_feature_name] = st.session_state.data[columns_to_aggregate].min(axis=1)
        elif aggregation_operation == "Max":
            st.session_state.data[new_feature_name] = st.session_state.data[columns_to_aggregate].max(axis=1)
        st.success(f"Aggregated feature '{new_feature_name}' created!")

    # 2. Create Binary Flags
    st.subheader("Create Binary Flags")
    col1 = st.selectbox("Select first column:", st.session_state.data.columns.tolist())
    col2 = st.selectbox("Select second column:", st.session_state.data.columns.tolist(), index=1)  # start from the second column by default
    binary_feature_name = st.text_input("Name for the new binary feature:")
    if st.button("Create Binary Feature"):
        st.session_state.data[binary_feature_name] = np.where(st.session_state.data[col1] == st.session_state.data[col2], 1, 0)
        st.success(f"Binary feature '{binary_feature_name}' created based on {col1} and {col2}!")

    