import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pandas_datareader import data as pdr
import yfinance as yf

# Download the historical stock price data for Apple
yf.pdr_override()
df = pdr.get_data_yahoo("AAPL", start="2010-01-01", end="2022-04-01")

# Create a new dataframe with only the date and close columns
df = df[['Close']]

# Create a MinMaxScaler object to scale the data
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the data
scaled_data = scaler.fit_transform(df.values)

# Define the number of days to use as input for the LSTM model
num_days = 60

# Create empty lists for the input and output data
X_data = []
y_data = []

# Loop through the data to create the input and output data
for i in range(num_days, len(scaled_data)):
    X_data.append(scaled_data[i-num_days:i, 0])
    y_data.append(scaled_data[i, 0])

# Convert the data to numpy arrays
X_data = np.array(X_data)
y_data = np.array(y_data)

# Reshape the input data to be 3-dimensional
X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_data.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_data, y_data, epochs=50, batch_size=32)

# Make predictions on the test data
test_data = df[-num_days:].values
test_data = scaler.transform(test_data)
test_data = np.reshape(test_data, (1, num_days, 1))
predicted_price = model.predict(test_data)
predicted_price = scaler.inverse_transform(predicted_price)

print(predicted_price)
