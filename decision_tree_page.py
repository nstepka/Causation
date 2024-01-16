import streamlit as st
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def decision_tree_page():
    st.title("Decision Tree Analysis")

    # Check if data is loaded
    if 'data' in st.session_state and st.session_state['data'] is not None:
        data = st.session_state['data']
        st.write(data.head())

        # Select features and target
        features = st.multiselect("Select features", data.columns.tolist(), default=data.columns[0])
        target = st.selectbox("Select target", data.columns.tolist(), index=len(data.columns)-1)

        # Split data into training and test sets
        if len(features) > 0:
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Model type selection
            model_type = st.radio("Choose model type", ("Classifier", "Regressor"))

            # Training the Decision Tree
            if st.button("Train Model"):
                if model_type == "Classifier":
                    model = DecisionTreeClassifier()
                else:
                    model = DecisionTreeRegressor()

                model.fit(X_train, y_train)

                # Model evaluation
                predictions = model.predict(X_test)
                if model_type == "Classifier":
                    st.write("Model Accuracy:", model.score(X_test, y_test))
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, predictions))
                else:
                    st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
                    st.write("R^2 Score:", r2_score(y_test, predictions))

                # Visualize the tree
                fig, ax = plt.subplots(figsize=(12, 12))
                plot_tree(model, filled=True, feature_names=features, class_names=str(y.unique()), ax=ax)
                st.pyplot(fig)
        else:
            st.warning("Please select at least one feature.")
    else:
        st.warning("No data available. Please upload data first.")
