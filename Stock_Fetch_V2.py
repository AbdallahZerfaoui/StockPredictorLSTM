import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

"""
StockDataCollector is a class designed to collect and display historical stock data using the Yahoo Finance API.

Attributes:
    None

Methods:
    __init__: Initializes the StockDataCollector instance.

    get_historical_data: Fetches historical stock data for a given symbol within a specified date range.

        Parameters:
            symbol (str): The stock symbol for which historical data is to be collected.
            start_date (str): The start date for the historical data (default: "2019-11-01").
            end_date (str): The end date for the historical data (default: "2023-11-10").

        Returns:
            pd.DataFrame: A pandas DataFrame containing the historical stock data.

    display_historical_data: Displays a graphical representation of the historical stock data.

        Parameters:
            historical_data (pd.DataFrame): The DataFrame containing historical stock data.

        Returns:
            None
"""

class StockDataCollector:
    def __init__(self):
        pass


    def get_historical_data(self, symbol, start_date="2016-11-01", end_date="2023-11-10"):

        stock = yf.Ticker(symbol)
        historical_data = stock.history(start=start_date, end=end_date)

        historical_data = pd.DataFrame(historical_data)

        return historical_data

    def post_traitement_data(self, historical_data):
        # Add the Daily_pct_change column
        historical_data['Daily_pct_change'] = historical_data['Close'].pct_change()

        # Add profit_loss column 1 if positive, 0 if negative
        historical_data['profit_loss'] = historical_data['Daily_pct_change'].apply(lambda x: 1 if x > 0 else 0)

        # Create 3 new columns for SMA 10, 20, 50
        historical_data['SMA_10'] = historical_data['Close'].rolling(window=10).mean()
        historical_data['SMA_20'] = historical_data['Close'].rolling(window=20).mean()
        historical_data['SMA_50'] = historical_data['Close'].rolling(window=50).mean()

        return historical_data


    def display_historical_data(self, symbol, historical_data):
        # Display a graphic of the historical data
        historical_data = historical_data.reset_index()

        plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
        # Plot closing prices
        plt.plot(historical_data["Date"], historical_data["Close"], label="Close", color="blue")

        # Plot SMA curves
        plt.plot(historical_data["Date"], historical_data["SMA_10"], label="SMA 10", linestyle="--", color="orange")
        plt.plot(historical_data["Date"], historical_data["SMA_20"], label="SMA 20", linestyle="--", color="green")
        plt.plot(historical_data["Date"], historical_data["SMA_50"], label="SMA 50", linestyle="--", color="red")

        plt.title(f"{symbol} Historical Data with SMA Curves")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()  # Add legend to differentiate between curves
        plt.grid()
        plt.tight_layout()
        plt.show()



def check_StockerDataCollector():
    symbol = "AAPL"
    collector = StockDataCollector()
    historical_data = collector.get_historical_data(symbol)
    historical_data = collector.post_traitement_data(historical_data)     # Add the Daily_pct_change column
                                                                        # Add profit_loss column 1 if positive, 0 if negative
                                                                        # Create 3 new columns for SMA 10, 20, 50

    print(historical_data.columns)
    collector.display_historical_data(symbol, historical_data)



