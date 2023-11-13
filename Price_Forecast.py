import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Class for predicting stock prices using LSTM-based forecasting.

Attributes:
    historical_data (pd.DataFrame): DataFrame containing historical stock information.
    symbol (str): Stock symbol (default is "WMT" for Walmart).
    columns_needed (list): List of columns needed for analysis (default includes 'Close' and 'SMA_50').
    sequences_length (int): Length of sequences used for training and prediction (default is 10).
    scaler (MinMaxScaler): Scaler object for normalizing data.
    train_portion (float): Percentage of data used for training (default is 0.8).
    train_size (int): Size of the training set.
    test_size (int): Size of the test set.

Methods:
    __init__(self, historical_data): Initializes the PriceForecastEngine with historical data.
    get_financial_data(self): Extracts financial data from historical_data.
    scale_data(self, data): Scales the input data using MinMaxScaler.
    create_train_test_sets(self, scaled_financial_data): Splits the data into training and test sets.
    create_sequences(self, data): Generates sequences for LSTM model training.
    build_lstm_model(self, X_train): Builds and compiles an LSTM model.
    train_lstm_model(self, model, X_train, train_set, batch_size=20, epochs=50, validation_split=0.2): Trains the LSTM model.
    plot_train_validation_loss_history(self, model): Plots training and validation loss history.
    predict_prices(self, model, X_test): Predicts stock prices using the trained model.
    plot_results(self, financial_data, predictions, columns_to_plot): Plots actual and predicted prices.
    save_results(self, results): Saves prediction results to a CSV file.
    run_forcast_pipeline(self): Executes the entire forecasting pipeline.

Usage:
    engine = PriceForecastEngine(historical_data)
    engine.run_forcast_pipeline()
"""

class PriceForecastEngine:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.symbol = "WMT"
        self.columns_needed = ['Close', 'SMA_50']
        self.sequences_length = 10
        self.scaler = MinMaxScaler()
        self.train_portion = 0.8
        self.train_size=0
        self.test_size=0


    def get_financial_data(self):
        financial_data = \
            self.historical_data[self.columns_needed].values.reshape(-1, len(self.columns_needed))

        return financial_data


    def scale_data(self, data):
        scaled_data = self.scaler.fit_transform(data)

        return scaled_data


    def create_train_test_sets(self, scaled_financial_data):
        self.train_size = int(len(scaled_financial_data) * 0.8)
        self.test_size = len(scaled_financial_data) - self.train_size
        train_set = scaled_financial_data[0:self.train_size, :]  # First 80% as training set
        test_set = scaled_financial_data[self.train_size:len(scaled_financial_data), :]  # Last 20% as test set

        return train_set, test_set


    def create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.sequences_length):
            seq = data[i:i + self.sequences_length, :]
            sequences.append(seq)

        return np.array(sequences)


    def build_lstm_model(self, X_train):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu',
                                 return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, activation='relu', return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        return model


    def train_lstm_model(self, model, X_train, train_set, batch_size=20, epochs=50, validation_split=0.2):
        model.fit(X_train,
                  train_set[self.sequences_length:],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_split=validation_split)

        return model


    def plot_train_validation_loss_history(self, model):
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


    def predict_prices(self, model, X_test):
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1, 1)
        zeros_matrix = np.zeros((predictions.shape[0], len(self.columns_needed) - 1))  # Create a matrix of zeros
        predictions = \
            self.scaler.inverse_transform(np.hstack((predictions, zeros_matrix)))[:, 0]  # Inverse transform the predictions

        return predictions

    def plot_results(self, financial_data, predictions,
                     columns_to_plot=['Close', 'SMA_50', 'Predictions']):

        train = financial_data[:self.train_size + self.sequences_length]
        test = financial_data[self.train_size + self.sequences_length:]

        # Extract the 'Close' column from the original 'test' DataFrame
        actual_close_values = np.array([elem[0] for elem in test]).flatten()
        # Extract the 'SMA_50' column from the original 'test' DataFrame
        actual_sma_50_values = np.array([elem[1] for elem in test]).flatten()

        # Create the 'results' DataFrame with the 'Close' column and the predicted values
        results = pd.DataFrame({
            columns_to_plot[0]: actual_close_values,
            columns_to_plot[1]: actual_sma_50_values,
            columns_to_plot[2]: predictions.flatten()
        })

        plt.plot(results[columns_to_plot], label=columns_to_plot)
        plt.legend()
        plt.grid()
        plt.show()


    def save_results(self, results):
        results.to_csv(f"{self.symbol}_predictions_vs_test.csv", index=False)


    def run_forcast_pipeline(self):
        financial_data = self.get_financial_data()
        scaled_financial_data = self.scale_data(financial_data)
        train_set, test_set = self.create_train_test_sets(scaled_financial_data)
        X_train = self.create_sequences(train_set)
        X_test = self.create_sequences(test_set)
        model = self.build_lstm_model(X_train)
        model = self.train_lstm_model(model, X_train, train_set)
        self.plot_train_validation_loss_history(model)
        predictions = self.predict_prices(model, X_test)
        self.plot_results(financial_data, predictions)
