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
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet


def time_series_analysis():
    st.write("Time Series Analysis with ARIMA")

    # Check if data is uploaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload time series data first.")
        return

    # Let the user choose the datetime column
    date_column = st.sidebar.selectbox(
        "Select the datetime column for time series analysis:", 
        st.session_state.data.columns
    )

    # Let the user choose the y-axis column (numerical values)
    y_column = st.sidebar.selectbox(
        "Select the column for y-axis values:", 
        st.session_state.data.select_dtypes(include=[np.number]).columns
    )

    # Sidebar tasks under Time Series Analysis
    ts_task = st.sidebar.radio(
        "Choose a Time Series Analysis task:",
        ["Visualize Data", "ACF & PACF Plots", "Fit ARIMA Model", "Model Diagnostics", 
         "Forecast", "Model Evaluation", "Decomposition",
         "Fit Prophet Model", "Feature Engineering"]
    )
    
    if ts_task == "Visualize Data":
        visualize_time_series_data(date_column, y_column)

    elif ts_task == "ACF & PACF Plots":
        display_acf_pacf(y_column, date_column)
    elif ts_task == "Fit ARIMA Model":
        fit_arima_model(y_column, date_column)
    elif ts_task == "Model Diagnostics":
        time_series_model_diagnostics(y_column, date_column)
    elif ts_task == "Forecast":
        forecast_arima(y_column, date_column)
    elif ts_task == "Model Evaluation":
        model_evaluation(y_column, date_column)
    elif ts_task == "Decomposition":
        decompose_time_series(y_column, date_column)
    elif ts_task == "Fit Prophet Model":
        forecast_with_prophet(date_column, y_column)
    elif ts_task == "Feature Engineering":
        st.session_state.data = create_time_series_features(st.session_state.data, y_column)




    # ... and so on for other tasks


def get_arima_model(data):
    """Perform stepwise search and return the best ARIMA model."""
    if 'arima_model' not in st.session_state:
        st.session_state.arima_model = auto_arima(data, seasonal=True, trace=True, suppress_warnings=True)
    return st.session_state.arima_model



def time_series_model_diagnostics(y_column, date_column):
    st.write("Model Diagnostics")

    # Ensure the index is a DatetimeIndex
    data = st.session_state.data.set_index(date_column)[y_column]
    data.index = pd.to_datetime(data.index)
    data.index.freq = st.session_state.selected_freq

    # Get the ARIMA model
    auto_model = get_arima_model(data)
    p, d, q = auto_model.order

    # Fit the model
    model = ARIMA(data, order=(p,d,q))
    results = model.fit()

    # Extract residuals
    residuals = results.resid

    # Histogram of residuals
    st.subheader("Histogram of Residuals")
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.set_title("Histogram of Residuals")
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Q-Q plot
    st.subheader("Q-Q Plot")
    fig = sm.qqplot(residuals, fit=True, line='45')
    st.pyplot(fig)

    # ACF plot of residuals
    st.subheader("ACF Plot of Residuals")
    plot_acf(residuals, alpha=0.05)
    st.pyplot()

    # Ljung-Box test
    st.subheader("Ljung-Box Test")
    lb_stat, lb_pvalue = sm.stats.acorr_ljungbox(residuals, lags=10, return_df=False)
    
    # Display the results in a table format
    df_ljungbox = pd.DataFrame({
        "Lag": range(1, len(lb_stat) + 1),
        "Test Statistic": lb_stat,
        "P-value": lb_pvalue
    })

    st.table(df_ljungbox)

    # Durbin-Watson test
    st.subheader("Durbin-Watson Test")
    dw = sm.stats.durbin_watson(residuals)
    st.write("Durbin-Watson statistic:", dw)

    # RMSE, Log-Likelihood, AIC, and BIC
    rmse = np.sqrt(mean_squared_error(data, results.fittedvalues))
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"Log-Likelihood: {results.llf}")
    st.write(f"AIC: {results.aic}")
    st.write(f"BIC: {results.bic}")
    



def visualize_time_series_data(date_column, y_column):
    st.write("Time Series Data Visualization")

    # Use the selected datetime column for x-axis and y_column for y-axis
    st.line_chart(st.session_state.data.set_index(date_column)[y_column])

def check_stationarity(data):
    result = adfuller(data)
    return result[1] <= 0.05  # p-value

def apply_transformations(data, difference, log_transform, seasonal_difference):
    transformed_data = data.copy()
    
    if log_transform:
        transformed_data = np.log(transformed_data)
    
    if difference:
        transformed_data = transformed_data.diff().dropna()
    
    if seasonal_difference > 0:
        transformed_data = transformed_data.diff(seasonal_difference).dropna()

    # Handle inf and nan
    transformed_data = transformed_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    return transformed_data



def display_acf_pacf(y_column, date_column):
    st.write("Autocorrelation (ACF) and Partial Autocorrelation (PACF) Plots")
    
    # Check for stationarity
    if not check_stationarity(st.session_state.data[y_column]):
        st.warning("The time series appears to be non-stationary.")

        difference = st.checkbox("Apply Differencing")
        log_transform = st.checkbox("Apply Log Transformation")
        seasonal_difference = st.slider("Apply Seasonal Differencing (set to 0 for no seasonal differencing)", 0, 24, 0)

        if st.button("Submit"):
            transformed_data = apply_transformations(st.session_state.data[y_column], difference, log_transform, seasonal_difference)

            if check_stationarity(transformed_data):
                st.success("The transformations have made the time series stationary!")
            else:
                st.warning("The time series is still non-stationary. Consider other transformations or differencing options.")

            st.line_chart(transformed_data)

    # Infer the frequency and set it
    data = st.session_state.data.set_index(date_column)[y_column]
    data.index = pd.to_datetime(data.index)
    inferred_freq = infer_frequency(data.index)
    if inferred_freq:
        freq_options = ['D', 'M', 'Y']
        freq_desc = {
            'D': 'Daily',
            'M': 'Monthly',
            'Y': 'Yearly'
        }
        st.write(f"We've inferred that your data might be in {freq_desc[inferred_freq]} frequency.")
        selected_freq = st.selectbox("Please confirm or adjust the frequency:", options=freq_options, index=freq_options.index(inferred_freq))
        data.index.freq = selected_freq
        st.write(f"Using {freq_desc[selected_freq]} frequency for analysis.")
    else:
        st.warning("Unable to automatically infer the data frequency. Please select a frequency:")
        selected_freq = st.selectbox("Choose the frequency:", options=['D', 'M', 'Y'])
        data.index.freq = selected_freq

    # Store the selected frequency in session state for subsequent tasks
    st.session_state.selected_freq = selected_freq

    # Let the user select the number of lags
    max_lags = len(data) - 1
    lags = st.slider("Select number of lags:", 1, max_lags, 40)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    plot_acf(data, ax=ax[0], lags=lags, alpha=0.05)
    plot_pacf(data, ax=ax[1], lags=lags, alpha=0.05)

    # Add significance level lines (assuming a significance level of 0.05)
    ax[0].axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
    ax[0].axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
    ax[1].axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
    ax[1].axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')

    st.pyplot(fig)


def model_to_csv(model, test_data, forecast):
    # Create a DataFrame to store model details and forecasts
    df = pd.DataFrame()

    # Model order
    df["ARIMA_order"] = [model.order]

    # Coefficients
    ar_coeffs = model.arparams() if model.order[0] > 0 else []
    ma_coeffs = model.maparams() if model.order[2] > 0 else []

    for i, coeff in enumerate(ar_coeffs, 1):
        df[f"AR_coeff_{i}"] = [coeff]

    for i, coeff in enumerate(ma_coeffs, 1):
        df[f"MA_coeff_{i}"] = [coeff]

    # Forecasted vs actual values
    df_forecast = pd.DataFrame({"Actual": test_data, "Forecasted": forecast})

    # Combine the model details and forecast data
    df = pd.concat([df, df_forecast], axis=1)

    # Convert DataFrame to CSV format
    csv = df.to_csv(index=False)
    return csv

def infer_frequency(date_index):
    diffs = date_index[1:] - date_index[:-1]
    median_diff = pd.Series(diffs).median()

    if pd.Timedelta('0 days') < median_diff <= pd.Timedelta('1 days'):
        return 'D'
    elif pd.Timedelta('1 days') < median_diff <= pd.Timedelta('32 days'):
        return 'M'
    elif pd.Timedelta('32 days') < median_diff:
        return 'Y'
    else:
        return None


def fit_arima_model(y_column, date_column):
    st.write("Fitting ARIMA Model")

    # Split data into training and test sets
    split_ratio = st.sidebar.slider("Specify the percentage of data for testing:", 10, 50, 20) / 100
    split_index = int(len(st.session_state.data[y_column]) * (1 - split_ratio))

    # Ensure the index is a DatetimeIndex
    train_data = st.session_state.data.set_index(date_column)[:split_index][y_column]
    train_data.index = pd.to_datetime(train_data.index)

    # Set the frequency from session state directly
    train_data.index.freq = st.session_state.selected_freq

    # Get the ARIMA model
    auto_model = get_arima_model(train_data)
    st.write(f"Best ARIMA model: {auto_model.order}")

    # Forecast on test set
    forecast, conf_int = auto_model.predict(n_periods=len(st.session_state.data[y_column][split_index:]), return_conf_int=True)

    # Visualize actual vs. predicted values
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_data, label='Training Data', color='blue')
    
    forecast_dates = [train_data.index[-1] + pd.DateOffset(days=int(x)) for x in np.arange(1, len(forecast)+1)]
    ax.plot(forecast_dates, st.session_state.data[y_column][split_index:], label='Actual Test Data', color='green')
    ax.plot(forecast_dates, forecast, label='Forecast', color='red')
    ax.fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)

    ax.set_title("ARIMA Forecast vs Actuals")
    ax.legend()
    st.pyplot(fig)

    # Compute evaluation metrics
    rmse = np.sqrt(mean_squared_error(st.session_state.data[y_column][split_index:], forecast))
    mae = mean_absolute_error(st.session_state.data[y_column][split_index:], forecast)
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Save model as CSV (or any other format)
    if st.button("Save Model"):
        # Convert the model's parameters and other info to a dataframe
        model_info = pd.DataFrame({
            'Parameter': auto_model.params().index,
            'Value': auto_model.params().values
        })
        csv = model_info.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="arima_model.csv">Download ARIMA Model as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)



def forecast_arima(y_column, date_column):
    st.write("Forecasting with ARIMA Model")

    # Ensure the index is a DatetimeIndex
    data = st.session_state.data.set_index(date_column)[y_column]
    data.index = pd.to_datetime(data.index)
    data.index.freq = st.session_state.selected_freq

    #

    # Get the ARIMA model
    auto_model = get_arima_model(data)

    # How many periods to forecast
    periods_to_forecast = st.slider("Select number of periods to forecast into the future:", 1, 365, 30)

    # Forecast 
    forecast, conf_int = auto_model.predict(n_periods=periods_to_forecast, return_conf_int=True)

    # Visualize the forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data, label='Historical Data', color='blue')
    
    forecast_dates = [data.index[-1] + pd.DateOffset(days=int(x)) for x in np.arange(1, periods_to_forecast+1)]
    ax.plot(forecast_dates, forecast, label='Forecast', color='red')
    ax.fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    
    ax.set_title("ARIMA Forecast")
    ax.legend()
    st.pyplot(fig)

    # Option to download the forecast
    if st.button("Download Forecast"):
        df_forecast = pd.DataFrame({"Date": forecast_dates, "Forecasted": forecast})
        csv = df_forecast.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="arima_forecast.csv">Download Forecast as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)


def model_evaluation(y_column, date_column):
    st.write("Model Evaluation")
    
    # Split data into training and test sets
    split_ratio = st.sidebar.slider("Specify the percentage of data for testing (for evaluation):", 10, 50, 20) / 100
    split_index = int(len(st.session_state.data[y_column]) * (1 - split_ratio))
    
    # Ensure the index is a DatetimeIndex
    train_data = st.session_state.data.set_index(date_column)[:split_index][y_column]
    train_data.index = pd.to_datetime(train_data.index)
    test_data = st.session_state.data.set_index(date_column)[split_index:][y_column]
    test_data.index = pd.to_datetime(test_data.index)
    train_data.index.freq = st.session_state.selected_freq
    test_data.index.freq = st.session_state.selected_freq
    
    # Get the ARIMA model
    auto_model = get_arima_model(train_data)
    
    # Forecast on test set
    forecast, conf_int = auto_model.predict(n_periods=len(test_data), return_conf_int=True)
    
    # Compute evaluation metrics
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    mae = mean_absolute_error(test_data, forecast)
    r2 = r2_score(test_data, forecast)
    
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"R-squared (R2): {r2:.2f}")
    
    # Visualize actual vs. predicted values
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_data, label='Actual Test Data', color='green')
    ax.plot(test_data.index, forecast, label='Forecast', color='red')
    ax.fill_between(test_data.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    
    ax.set_title("Actual vs Forecasted Values on Test Set")
    ax.legend()
    st.pyplot(fig)


from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_time_series(y_column, date_column):
    st.write("Time Series Decomposition")

    # Ensure the index is a DatetimeIndex
    data = st.session_state.data.set_index(date_column)[y_column]
    data.index = pd.to_datetime(data.index)
    data.index.freq = st.session_state.selected_freq

    # Let the user choose the type of decomposition
    decomposition_type = st.selectbox(
        "Choose the type of decomposition:", ["additive", "multiplicative"]
    )

    # Decompose the time series
    decomposition = seasonal_decompose(data, model=decomposition_type)

    # Plot the components
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    
    # Original time series
    ax1.plot(data)
    ax1.set_title("Original Time Series")
    
    # Trend component
    ax2.plot(decomposition.trend)
    ax2.set_title("Trend Component")

    # Seasonal component
    ax3.plot(decomposition.seasonal)
    ax3.set_title("Seasonal Component")
    
    # Residuals
    ax4.plot(decomposition.resid)
    ax4.set_title("Residuals")
    
    plt.tight_layout()
    st.pyplot(fig)


def forecast_with_prophet(date_column, y_column):
    st.write("Forecasting with Prophet")

    # Ensure the dataframe is in the format Prophet expects
    df = st.session_state.data.rename(columns={date_column: 'ds', y_column: 'y'})

    # Create a Prophet model and fit
    model = Prophet()
    model.fit(df)

    # Create a dataframe for future dates
    periods = st.slider("Select number of periods to forecast into the future:", 1, 365, 30)
    future = model.make_future_dataframe(periods=periods)

    # Forecasting
    forecast = model.predict(future)
    
    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
    ax.set_title("Forecast with Prophet")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    
    st.pyplot(fig)  # Display the figure using Streamlit
    
    # Allow downloading the forecast
    if st.button("Download Prophet Forecast"):
        csv = forecast.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prophet_forecast.csv">Download Forecast as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)




def create_time_series_features(data, y_column):
    st.write("Time Series Feature Engineering")
    st.write("""
    Feature engineering in time series involves creating new variables or 
    transformations of the original data to improve the model's performance. 
    Common methods include lagging, rolling statistics, and encoding cyclical 
    patterns. By adding these features, we can capture temporal structures in the data.
    """)

    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        # Convert the index to DatetimeIndex
        data.index = pd.to_datetime(data.index)

    # Lagged features
    num_lags = st.slider("Number of lagged features:", 1, len(data)-1, 1, key="lag_slider")
    for i in range(1, num_lags + 1):
        data[f"lag_{i}"] = data[y_column].shift(i)
    if num_lags > 0:
        st.write(f"Created {num_lags} lagged features.")
        st.line_chart(data[[y_column] + [f"lag_{i}" for i in range(1, num_lags+1)]], use_container_width=True)

    # Rolling window statistics
    window_size = st.slider("Rolling window size:", 1, len(data)-1, 7, key="window_slider")
    data[f"rolling_mean_{window_size}"] = data[y_column].rolling(window=window_size).mean()
    data[f"rolling_std_{window_size}"] = data[y_column].rolling(window=window_size).std()
    st.write(f"Created rolling mean and standard deviation with window size of {window_size}.")
    st.line_chart(data[[y_column, f"rolling_mean_{window_size}", f"rolling_std_{window_size}"]], use_container_width=True)

    # Encode day of week and month as cyclical features
    encode_cyclical = st.checkbox("Encode day of week and month as cyclical features")
    if encode_cyclical:
        data['day'] = data.index.day
        data['month'] = data.index.month
        data['day_sin'] = np.sin(data['day']*(2.*np.pi/30))
        data['day_cos'] = np.cos(data['day']*(2.*np.pi/30))
        data['month_sin'] = np.sin(data['month']*(2.*np.pi/12))
        data['month_cos'] = np.cos(data['month']*(2.*np.pi/12))
        st.write("Encoded day of week and month as cyclical features.")

    # Download engineered data
    if st.button("Download Engineered Data"):
        csv = data.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="engineered_data.csv">Download Engineered Data as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    return data


