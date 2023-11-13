# StockPredictorLSTM

StockPredictorLSTM is a Python project that utilizes Long Short-Term Memory (LSTM) networks for predicting stock prices. The project fetches historical stock data, preprocesses it, trains an LSTM model, and makes predictions.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Fetches historical stock data using StockDataCollector.
- Applies post-processing and handles NaN values.
- Scales data using MinMaxScaler.
- Creates training and test sets, generates sequences for LSTM training.
- Builds, trains, and evaluates LSTM models.
- Plots training and validation loss history.
- Predicts stock prices and visualizes results.

## Prerequisites

- Python 3.x
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [StockDataCollector](#link-to-stock-data-collector) (replace with actual link)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/StockPredictorLSTM.git
   cd StockPredictorLSTM

# Usage
1. Fetch historical data and run the forecasting pipeline:
   
   ```bash
   python main.py

2. View the results and trained models in the project directory.

# Project Structure
The project is structured as follows:

- main.py: Main script to execute the forecasting pipeline.
- Stock_Fetch_V2.py: Module for fetching and processing historical stock data.
- Price_Forecast.py: Module containing the PriceForecastEngine class for LSTM forecasting.
- requirements.txt: List of Python dependencies.
