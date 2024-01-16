def decision_tree_page():
    st.title("Decision Tree Analysis")

    # Check if data is loaded
    if 'data' in st.session_state and st.session_state['data'] is not None:
        data = st.session_state['data']
        st.write(data.head())

        # Select features and target
        features = st.multiselect("Select features", data.columns.tolist(), default=data.columns[:-1])
        target = st.selectbox("Select target", data.columns.tolist(), index=len(data.columns)-1)

        # Split data into training and test sets
        if len(features) > 0 and target:
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Hyperparameters
            max_depth = st.number_input('Max Depth', min_value=1, value=5, step=1)
            min_samples_split = st.number_input('Min Samples Split', min_value=2, value=2, step=1)
            min_samples_leaf = st.number_input('Min Samples Leaf', min_value=1, value=1, step=1)
            max_features = st.selectbox('Max Features', ['auto', 'sqrt', 'log2'])

            # Training the Decision Tree
            if st.button("Train Model"):
                model = DecisionTreeRegressor(max_depth=max_depth,
                                              min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              max_features=max_features)
                model.fit(X_train, y_train)

                # Model evaluation
                predictions = model.predict(X_test)
                st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
                st.write("R^2 Score:", r2_score(y_test, predictions))

                # Visualize the tree
                fig, ax = plt.subplots(figsize=(12, 12))
                plot_tree(model, filled=True, feature_names=features, ax=ax)
                st.pyplot(fig)
        else:
            st.warning("Please select at least one feature and a target.")
    else:
        st.warning("No data available. Please upload data first.")
