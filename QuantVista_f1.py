import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROJECT_ID = " "  # @param {type:"string"}
LOCATION = " "  # @param {type:"string"}
# Initialize Vertex AI
import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
from vertexai.preview.generative_models import (
    GenerativeModel,
)

seed = 42

import PIL.Image
import google.generativeai as genai
genai.configure(api_key=' ')


class QLearningAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.Q_values = np.zeros(n_actions)
        self.cumulative_reward = 0

    def select_action(self):
        return np.argmax(self.Q_values)

    def update_q_values(self, reward):
        # Update Q-values using a simple update rule
        learning_rate = 0.1
        discount_factor = 0.9
        self.Q_values = learning_rate * reward + discount_factor * np.max(self.Q_values)
        self.cumulative_reward += reward

def main():
    st.title("QuantVista")

    # Create sidebar with tabs
    app_mode = st.sidebar.selectbox("Select App Mode", ["Country GDP", "Company Financial Data"])

    # Render selected tab
    if app_mode == "Country GDP":
        gdp_tab()
    elif app_mode == "Company Financial Data":
        sales_tab()

def gdp_tab():
    st.header("Country GDP Analysis")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        st.success("File successfully uploaded!")

        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Display the DataFrame
        st.write("Preview of the uploaded data:")
        st.write(df)

        # Load list of countries from the uploaded CSV file
        all_countries = df.iloc[:, 0].tolist()

        # Dropdown for selecting a country
        selected_country = st.selectbox("Select a Country", [""] + all_countries)  # Add an empty option at the beginning

        # Display the selected country
        if selected_country:
            st.write(f"Selected Country: {selected_country}")

            # Perform GDP analysis using the selected country and uploaded data
            gdp_analysis(selected_country, df)

def sales_tab():
    st.header("Company Financial Data Analysis")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        st.success("File successfully uploaded!")

        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Display the DataFrame
        st.write("Preview of the uploaded data:")
        st.write(df)

        # Assuming the Date column is in the format 'dd-mm-yyyy'
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')

        # Set Date column as index
        df.set_index('Date', inplace=True)


def gdp_analysis(selected_country, df):
    # Filter data for the specified country
    country_data = df[df['Country Name'] == selected_country]

    # Extract the values from 1960 onwards for the specified country
    values_array = np.array(country_data.iloc[0, 4:]).astype(float)

    # Split data until 2018 and predict from 2019 to 2022
    train_data = values_array[:59]  # Data until 2018
    test_data = values_array[59:63]  # Data from 2019 to 2022

    # Plot the line chart for actual values
    fig_actual = plot_actual_values(train_data)
    st.pyplot(fig_actual)

    # Perform time series analysis and forecasting
    forecasted_values, fig_forecast = plot_time_series_analysis(train_data, selected_country, test_data)
    img_path = "forecast_image.png"
    fig_forecast.savefig(img_path)
    st.pyplot(fig_forecast)

    # Evaluate RL model for GDP and show results in a graph
    mse, rmse = evaluate_rl_model(train_data, test_data, forecasted_values)

    st.write("")
    st.title("Interact with the graph: Write in the Chat box below")
    prompt = st.chat_input("Type your Message")
    
    if prompt:
        st.info("AI Model: "+vision(img_path, text=prompt))

def vision(image,text):
    model = genai.GenerativeModel(model_name="gemini-pro-vision")

    img = PIL.Image.open(image)

    response = model.generate_content([text, img])
    return response.text

def sarima_forecast(data):
    # AutoARIMA to automatically select the best model parameters
    model = auto_arima(data, seasonal=True, m=12, stepwise=True, trace=True, suppress_warnings=True)

    # Fit the SARIMA model
    sarima_model = sm.tsa.statespace.SARIMAX(data, order=model.order, seasonal_order=model.seasonal_order)
    sarima_results = sarima_model.fit()

    # Forecast for the next 5 periods
    forecast_steps = 5
    forecast = sarima_results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean

    return forecast_mean  # Return the forecasted values

def rl_adjustment(data):
    # Initialize Q-learning agent
    n_actions = 10  # Assuming 5 actions for simplicity
    rl_agent = QLearningAgent(n_actions)

    # SARIMA forecast as the reward
    sarima_reward = np.mean(sarima_forecast(data)[5:])
    rl_agent.update_q_values(sarima_reward)

    # Make decisions using the Q-learning agent
    rl_actions = rl_agent.select_action()

    # Adjust the SARIMA forecast based on RL decisions
    adjusted_forecast_sarima_rl = sarima_forecast(data) * (1 + 0.067) ** (rl_actions + 1)

    # Display the adjusted forecast
    st.write(f"Adjusted forecast for the next 4 periods with RL:")
    st.write(adjusted_forecast_sarima_rl)

def plot_error_metrics(mse, rmse):
    # Plot both MSE and RMSE
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mse, label='MSE', color='blue')
    ax.plot(rmse, label='RMSE', color='red')
    ax.set_xlabel('Year')
    ax.set_ylabel('Error')
    ax.set_title('Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def evaluate_rl_model(train_data, test_data, forecasted_values):
    # SARIMA forecast as the reward
    sarima_reward = np.mean(sarima_forecast(train_data)[-5:])

    # Update Q-values using SARIMA reward
    rl_agent = QLearningAgent(n_actions=5)  # Assuming 5 actions for simplicity
    rl_agent.update_q_values(sarima_reward)

    # Track cumulative reward and convergence rate
    cumulative_reward = []
    convergence_rate = []

    # Perform Q-learning iterations
    for i in range(1, 1001):  # Number of iterations can be adjusted
        # Update Q-values
        rl_agent.update_q_values(sarima_reward)

        # Track cumulative reward
        cumulative_reward.append(rl_agent.cumulative_reward)

        # Track convergence rate
        if i > 1:
            convergence_rate.append(cumulative_reward[-1] - cumulative_reward[-2])

        # Check for convergence
        if i > 1 and cumulative_reward[-1] == cumulative_reward[-2]:
            break

    # Compute the mean squared error between the predicted values and actual test data
    mse = mean_squared_error(test_data, forecasted_values)
    rmse = np.sqrt(mse)

    # Plot convergence rate
    plot_convergence_rate(range(1, len(convergence_rate) + 1), convergence_rate)

    # Plot cumulative reward
    plot_cumulative_reward(range(1, len(cumulative_reward) + 1), cumulative_reward)

    # Print MSE and RMSE
    st.write("Mean Squared Error (MSE) between predicted and actual values from 2019 to 2022:", mse)
    st.write("Root Mean Squared Error (RMSE) between predicted and actual values from 2019 to 2022:", rmse)

    return mse, rmse





def plot_convergence_rate(iterations, convergence_rate):
    # Plot the convergence rate
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(iterations, convergence_rate, label='Convergence Rate', color='green')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Mean Q-value')
    ax.set_title('Convergence Rate of Q-learning for GDP')
    ax.legend()
    st.pyplot(fig)


def plot_cumulative_reward(iterations, cumulative_reward):
    # Plot the cumulative reward
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(iterations, cumulative_reward, label='Cumulative Reward', color='blue')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward over Q-learning Iterations')
    ax.legend()
    st.pyplot(fig)


def plot_actual_values(values_array):
    # Extract the years from the index
    years = range(1960, 1960 + len(values_array))

    # Create the line chart for actual values
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(years, values_array, marker='x', label='Actual')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title('Actual Values')
    ax.legend()
    ax.grid(True)
    return fig


def plot_time_series_analysis(train_data, selected_country, test_data):
    # Create a DataFrame from your data
    train_years = range(1960, 2019)
    test_years = range(2019, 2023)
    train_gdp_series = pd.Series(train_data, index=pd.to_datetime(train_years, format='%Y'))
    test_gdp_series = pd.Series(test_data, index=pd.to_datetime(test_years, format='%Y'))

    # Decompose the time series to identify trends and seasonality
    train_decomposition = sm.tsa.seasonal_decompose(train_gdp_series, model='additive')
    trend = train_decomposition.trend
    seasonal = train_decomposition.seasonal
    residual = train_decomposition.resid

    # Plot the decomposed components
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(train_gdp_series, label='Original')
    ax.legend(loc='upper left')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(trend, label='Trend')
    ax.legend(loc='upper left')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(seasonal, label='Seasonal')
    ax.legend(loc='upper left')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(residual, label='Residual')
    ax.legend(loc='upper left')
    plt.tight_layout()

    # Train a time series forecasting model (e.g., SARIMA)
    model = auto_arima(train_gdp_series, seasonal=True, m=12, stepwise=True, trace=True, suppress_warnings=True)

    # Fit the SARIMA model
    order = model.order
    seasonal_order = model.seasonal_order
    sarima_model = SARIMAX(train_gdp_series, order=order, seasonal_order=seasonal_order)
    sarima_results = sarima_model.fit()

    # Forecast GDP per capita for the next four years (2019-2022)
    forecast_steps = 4
    forecast = sarima_results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Plot the forecast with SARIMA
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_gdp_series, label='Observed', color='blue')
    ax.plot(forecast_mean.index, forecast_mean.values, color='red', label='SARIMA Forecast')
    ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', label='SARIMA Confidence Interval')
    ax.plot(test_gdp_series.index, test_gdp_series.values, color='green', label='Actual Test Data', linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel('GDP per Capita')
    ax.set_title('GDP per Capita Forecast with SARIMA')
    ax.legend()
    ax.grid(True)

    # Reinforcement Learning
    rl_agent = QLearningAgent(n_actions=forecast_steps)
    sarima_reward = np.mean(forecast_mean.values)
    rl_agent.update_q_values(sarima_reward)

    # Make decisions using the Q-learning agent
    rl_actions = rl_agent.select_action()

    # Adjust the SARIMA forecast based on RL decisions
    adjusted_forecast_sarima_rl = forecast_mean * (1 + 0.067) ** (rl_actions + 1)

    # Plot the forecast with SARIMA and RL adjustment
    fig_rl, ax_rl = plt.subplots(figsize=(12, 6))
    ax_rl.plot(train_gdp_series, label='Observed', color='blue')
    ax_rl.plot(forecast_mean.index, forecast_mean.values, color='red', label='SARIMA Forecast')
    ax_rl.plot(forecast_mean.index[rl_actions:], adjusted_forecast_sarima_rl.values[rl_actions:], color='green', label='Adjusted SARIMA Forecast (RL)')
    ax_rl.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', label='SARIMA Confidence Interval')
    ax_rl.plot(test_gdp_series.index, test_gdp_series.values, color='green', label='Actual Test Data', linestyle='--')
    ax_rl.set_xlabel('Year')
    ax_rl.set_ylabel('GDP per Capita')
    ax_rl.set_title('GDP per Capita Forecast with SARIMA and RL Adjustment for "'+selected_country+'" ')
    ax_rl.legend()
    ax_rl.grid(True)

    # Print the forecasted values for the next 4 years with RL adjustment
    forecasted_values_sarima_rl = adjusted_forecast_sarima_rl[-forecast_steps:]
    st.write("Adjusted Forecasted GDP per Capita for the Next 4 Years with RL:")
    st.write(pd.DataFrame({'Predicted': forecasted_values_sarima_rl, 'Actual': test_gdp_series}))

    return forecasted_values_sarima_rl, fig_rl



if __name__ == "__main__":
    main()
