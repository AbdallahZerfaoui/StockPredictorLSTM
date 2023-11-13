import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from Stock_Fetch_V2 import StockDataCollector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_sequences(data, sequences_length=10):
    sequences = []
    for i in range(len(data)-sequences_length):
        seq = data[i:i+sequences_length, :]
        sequences.append(seq)

    return np.array(sequences)


# Create the data collector object
symbol = "WMT"
collector = StockDataCollector()
historical_data = collector.get_historical_data(symbol=symbol) # Fetch the historical data
historical_data = collector.post_traitement_data(historical_data) # Post-process the data
historical_data = historical_data.dropna() # Drop the NaN values

# Extract the closing prices
columns_needed = ['Close', 'SMA_50'] # ['Close', 'SMA_10', 'SMA_20', 'SMA_50']
closing_prices = \
    historical_data[columns_needed].values.reshape(-1, len(columns_needed)) # Convert to numpy array

# Scale the closing prices
scaler = MinMaxScaler() # Create the scaler object
scaled_closing_prices = scaler.fit_transform(closing_prices) # Fit the data and transform

# Create the training and test sets
train_size = int(len(scaled_closing_prices) * 0.8)
test_size = len(scaled_closing_prices) - train_size
train_set = scaled_closing_prices[0:train_size,:]   # First 80% as training set
test_set = scaled_closing_prices[train_size:len(scaled_closing_prices),:] # Last 20% as test set


# Create sequences for the training set
sequences_length = 10
X_train = create_sequences(train_set, sequences_length)
X_test = create_sequences(test_set, sequences_length)

# training and fitting model (LSTM) using relu activation function and dropout layers
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, activation='relu', return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, train_set[sequences_length:], batch_size=20, epochs=50, verbose=1, validation_split=0.20)

# Access the training history
history = model.history

# Plot the training loss and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Predictions
predictions = model.predict(X_test)
predictions = predictions.reshape(-1, 1)
zeros_matrix = np.zeros((predictions.shape[0], len(columns_needed)-1)) # Create a matrix of zeros
predictions = \
    scaler.inverse_transform(np.hstack((predictions, zeros_matrix)))[:, 0]    # Inverse transform the predictions

# Plot the predictions
train = closing_prices[:train_size + sequences_length]
test = closing_prices[train_size + sequences_length:]
# results = pd.DataFrame({'Close': test.flatten(), 'Predictions': predictions.flatten()})
# Extract the 'Close' column from the original 'test' DataFrame
actual_close_values = np.array([elem[0] for elem in test]).flatten()
actual_sma_50_values = np.array([elem[1] for elem in test]).flatten()

# Create the 'results' DataFrame with the 'Close' column and the predicted values
results = pd.DataFrame({
    'Close': actual_close_values,
    'SMA_50': actual_sma_50_values,
    'Predictions': predictions.flatten()
})

# Save results variable to a csv file
results.to_csv(f"{symbol}_predictions_vs_test.csv", index=False)


#plt.plot(train, label="Training data")
#plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
plt.plot(results[['Close', 'SMA_50', 'Predictions']], label=["Close", "SMA_50", "Predictions"])
plt.legend()
plt.grid()
plt.show()


