# Time Series Analysis and Forecasting of Air Quality Index (AQI)
# This script performs data preprocessing, visualization, and forecasting using SARIMAX.

# Importing required libraries
import pandas as pd           # For data manipulation
import numpy as np            # For numerical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime    # To handle datetime operations
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX  # For SARIMAX modeling
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # ACF/PACF plots
from time import time  # To measure model fitting time

# Suppress warnings (optional)
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load and preprocess the dataset
# ------------------------------------------
# Replace the file path with the correct location of your CSV file
humidity = pd.read_csv('C:/Users/BEHINLAPTOP/Desktop/11.csv', index_col='date', parse_dates=['date'])

# Display the first few rows of the data
print(humidity.head())

# Remove the first row and fill missing values using forward fill
humidity = humidity.iloc[1:]
humidity = humidity.fillna(method='ffill')

# Step 2: Data visualization
# ---------------------------
# Plot the AQI column to visualize trends over time
humidity["AQI"].plot()
plt.title('Air Quality Index (AQI) Over Time')
plt.ylabel('AQI')
plt.xlabel('Date')
plt.show()

# Lag comparison: Original AQI vs Lagged AQI (shifted by 10 steps)
humidity["AQI"].plot(legend=True)
shifted = humidity["AQI"].shift(10).plot(legend=True)
plt.title('AQI vs AQI Lagged by 10 Steps')
plt.legend(['AQI', 'AQI_Lag'])
plt.show()

# ACF plot
plot_acf(humidity["AQI"], lags=25, title="Autocorrelation of AQI")
plt.show()

# PACF plot
plot_pacf(humidity["AQI"], lags=25, title="Partial Autocorrelation of AQI")
plt.show()

# Step 3: Differencing to achieve stationarity
# ---------------------------------------------
# Plot the differenced AQI data
humidity.diff().plot(figsize=(20, 6))
plt.title('First-Order Differenced AQI')
plt.ylabel('Difference in AQI')
plt.xlabel('Date')
plt.show()

# Step 4: SARIMAX modeling
# -------------------------
# Define the order and seasonal order for SARIMAX
my_order = (0, 1, 0)  # (p, d, q)
my_seasonal_order = (1, 0, 1, 12)  # (P, D, Q, s), s = 12 for monthly seasonality

# Prepare the training data (scaling AQI for demonstration purposes)
train_data = humidity["AQI"] * 2

# Define the SARIMAX model
model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)

# Fit the model and measure execution time
start = time()
model_fit = model.fit()
end = time()

print('Model Fitting Time:', end - start)
print(model_fit.summary())

# End of script
