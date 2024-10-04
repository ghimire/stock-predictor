import pandas as pd
import yfinance as yf
from pandas import DataFrame


class YahooFinanceAPI:
    """Class to interact with Yahoo Finance API for retrieving historical stock data."""

    def get_historical_data(self, ticker: str) -> DataFrame:
        """Fetches historical stock data for a given ticker symbol.

        Args:
            ticker (str): The stock ticker symbol.

        Returns:
            DataFrame: A pandas DataFrame containing the historical stock data.
        """
        # Fetch historical data using yfinance
        try:
            stock_data = yf.Ticker(ticker)
            historical_data = stock_data.history(period="max")
            if historical_data.empty:
                raise ValueError(f"No data found for ticker: {ticker}")
            return historical_data
        except Exception as e:
            print(f"An error occurred while fetching data for {ticker}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error
