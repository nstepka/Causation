import streamlit as st
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for data manipulation

def decision_tree_page():
    st.title("Decision Tree Analysis")

    # Check if data is loaded
    if 'data' in st.session_state and st.session_state['data'] is not None:
        data = st.session_state['data']
        st.write(data.head())

        # Select features and target
        features = st.multiselect("Select features", data.columns.tolist(), default=data.columns[:-1].tolist())
        target = st.selectbox("Select target", data.columns.tolist(), index=len(data.columns)-1)

    
        # Check for numerical data
        if not all(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns if col != target):
            st.warning("⚠️ The dataset contains non-numerical features. Please visit the Feature Engineering section for proper encoding.")
            return  # Stop execution if non-numerical features are present

        st.write(data.head())
        st.write(data.head())

  

        # Split data into training and test sets
        if len(features) > 0 and target:
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Model type selection
            model_type = st.radio("Choose model type", ("Classifier", "Regressor"))

            # Hyperparameters input
            set_hyperparameters = st.checkbox("Set hyperparameters", value=False)
            max_depth = None
            min_samples_split = 2
            min_samples_leaf = 1
            max_features = None

            if set_hyperparameters:
                max_depth = st.number_input('Max Depth', min_value=1, value=5, step=1)
                min_samples_split = st.number_input('Min Samples Split', min_value=2, value=2, step=1)
                min_samples_leaf = st.number_input('Min Samples Leaf', min_value=1, value=1, step=1)
                max_features = st.selectbox('Max Features', ['auto', 'sqrt', 'log2'])

            # Initialize the appropriate model
            if model_type == "Classifier":
                model = DecisionTreeClassifier(max_depth=max_depth,
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf,
                                               max_features=max_features)
            else:  # Regressor
                model = DecisionTreeRegressor(max_depth=max_depth,
                                              min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              max_features=max_features)

            # Training the Decision Tree
            if st.button("Train Model"):
                model.fit(X_train, y_train)

                # Model evaluation
                predictions = model.predict(X_test)
                if model_type == "Classifier":
                    accuracy = model.score(X_test, y_test)
                    class_report = classification_report(y_test, predictions)
                    st.write("Model Accuracy:", accuracy)
                    st.text("Classification Report:")
                    st.text(class_report)
                else:  # Regressor
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    st.write("Mean Squared Error:", mse)
                    st.write("R^2 Score:", r2)

                # Visualize the tree
                fig, ax = plt.subplots(figsize=(12, 12))
                if model_type == "Classifier":
                    plot_tree(model, filled=True, feature_names=features, class_names=str(y.unique()), ax=ax)
                else:  # Regressor
                    plot_tree(model, filled=True, feature_names=features, ax=ax)
                st.pyplot(fig)
        else:
            st.warning("Please select at least one feature and a target.")
    else:
        st.warning("No data available. Please upload data first.")
