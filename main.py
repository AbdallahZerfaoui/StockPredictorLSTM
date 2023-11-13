from Stock_Fetch_V2 import StockDataCollector
from Price_Forecast import PriceForecastEngine


symbol = "WMT"

# Create ths objects collector
collector = StockDataCollector()

# Fetch the historical data
historical_data = collector.get_historical_data(symbol=symbol) # Fetch the historical data
historical_data = collector.post_traitement_data(historical_data) # Post-process the data
historical_data = historical_data.dropna() # Drop the NaN values

# Create the PriceForecastEngine object
forcast_engine = PriceForecastEngine(historical_data)

# Run the pipeline
forcast_engine.run_forcast_pipeline()