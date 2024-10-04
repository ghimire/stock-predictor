import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from yahoo_finance_api import YahooFinanceAPI

logging.basicConfig(
    filename="stock_predictor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Predictor:
    """Class to predict stock prices using historical data and XGBoost."""

    def __init__(self):
        """Initializes the Predictor with an XGBoost model."""
        self.model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def predict_price(
        self,
        tickers: List[str],
        date: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        plot: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Predicts the stock price for given tickers and date or date range."""
        predictions = {}
        historical_data = {}
        logging.info(f"Starting predictions for tickers: {tickers}")
        for ticker in tickers:
            try:
                logging.info(f"Predicting for ticker: {ticker}")
                ticker_predictions = self._predict_single_ticker(
                    ticker, date, date_range
                )
                logging.info(f"Predictions for {ticker}: {ticker_predictions}")
                predictions[ticker] = ticker_predictions
                if plot:
                    historical_data[ticker] = YahooFinanceAPI().get_historical_data(
                        ticker
                    )
            except Exception as e:
                logging.error(f"Error predicting for {ticker}: {str(e)}")
                predictions[ticker] = {}

        if plot and historical_data:
            self._plot_predictions(tickers, historical_data, predictions)

        logging.info(f"Final predictions: {predictions}")
        return predictions

    def _predict_single_ticker(
        self,
        ticker: str,
        date: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, float]:
        """Predicts the stock price for a single ticker."""
        yahoo_api = YahooFinanceAPI()
        historical_data = yahoo_api.get_historical_data(ticker)

        logging.info(f"Historical data for {ticker}: {len(historical_data)} rows")
        logging.info(
            f"Historical data range: {historical_data.index[0]} to {historical_data.index[-1]}"
        )

        if historical_data.empty:
            logging.warning(f"No historical data found for {ticker}")
            return {}

        self.train_model(historical_data)

        historical_data.index = historical_data.index.tz_localize(None)

        if date:
            logging.info(f"Predicting single date for {ticker}: {date}")
            predictions = self._predict_single_date(historical_data, date)
        elif date_range:
            logging.info(
                f"Predicting date range for {ticker}: {date_range[0]} to {date_range[1]}"
            )
            predictions = self._predict_date_range(
                historical_data, date_range[0], date_range[1]
            )
        else:
            raise ValueError("Either date or date_range must be provided.")

        logging.info(f"Predictions for {ticker}: {predictions}")
        return predictions

    def _predict_single_date(
        self, historical_data: pd.DataFrame, date: str
    ) -> Dict[str, float]:
        """Predicts the stock price for a single date."""
        try:
            prediction_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            logging.error(f"Invalid date format: {date}")
            return {}

        if prediction_date <= historical_data.index[-1].date():
            return self._get_historical_price(historical_data, prediction_date)
        else:
            return self._predict_future_price(historical_data, prediction_date)

    def _get_historical_price(
        self, historical_data: pd.DataFrame, prediction_date: datetime.date
    ) -> Dict[str, float]:
        """Returns the actual historical price for a given date."""
        if prediction_date in historical_data.index.date:
            actual_price = historical_data.loc[
                historical_data.index.date == prediction_date, "Close"
            ].iloc[0]
            return {prediction_date.strftime("%Y-%m-%d"): actual_price}
        else:
            nearest_date = historical_data.index[
                historical_data.index.date <= prediction_date
            ][-1]
            actual_price = historical_data.loc[nearest_date, "Close"]
            return {nearest_date.strftime("%Y-%m-%d"): actual_price}

    def _predict_future_price(
        self, historical_data: pd.DataFrame, prediction_date: datetime.date
    ) -> Dict[str, float]:
        """Predicts the future price for a given date."""
        features = self._prepare_features(historical_data)
        X = features.drop(["Target", "Open", "High", "Low", "Close", "Volume"], axis=1)
        X_scaled = self.scaler.transform(X)

        predicted_price = round(float(self.model.predict(X_scaled)[-1]), 2)
        logging.info(
            f"Predicting for {prediction_date}: Predicted price = {predicted_price}"
        )
        return {prediction_date.strftime("%Y-%m-%d"): predicted_price}

    def _predict_date_range(
        self, historical_data: pd.DataFrame, start_date: str, end_date: str
    ) -> Dict[str, float]:
        """Predicts stock prices for a date range."""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            logging.error(f"Invalid date range format: {start_date}, {end_date}")
            return {}

        date_range = pd.date_range(start, end)
        predictions = {}
        current_data = historical_data.copy()

        logging.info(f"Predicting for date range: {start_date} to {end_date}")
        logging.info(f"Historical data last date: {historical_data.index[-1].date()}")

        for date in date_range:
            if date.date() <= historical_data.index[-1].date():
                prediction = self._get_historical_price(historical_data, date.date())
                logging.info(f"Using historical price for {date.date()}: {prediction}")
            else:
                prediction = self._predict_future_price(current_data, date.date())
                logging.info(f"Predicted future price for {date.date()}: {prediction}")
                # Update current_data with the new prediction
                new_row = current_data.iloc[-1].copy()
                new_row.name = date
                new_row["Close"] = round(list(prediction.values())[0], 2)
                current_data = pd.concat([current_data, pd.DataFrame([new_row])])

            predictions.update(prediction)

        logging.info(f"Predictions made: {len(predictions)}")
        return predictions

    def _plot_predictions(
        self,
        tickers: List[str],
        historical_data: Dict[str, pd.DataFrame],
        predictions: Dict[str, Dict[str, float]],
    ):
        """Plots the historical data and predictions for all tickers."""
        # Set the style to dark mode
        mpl.style.use("dark_background")

        num_tickers = len(tickers)
        fig, axs = plt.subplots(
            num_tickers, 1, figsize=(15, 7 * num_tickers), sharex=True
        )

        if num_tickers == 1:
            axs = [axs]  # Make axs a list when there's only one ticker

        for i, ticker in enumerate(tickers):
            data = historical_data[ticker]

            # Get the last 12 months of data
            last_12_months = data.loc[
                data.index >= (data.index[-1] - pd.DateOffset(months=12))
            ]

            # Price and predictions
            axs[i].plot(
                last_12_months.index,
                last_12_months["Close"],
                label="Historical",
                color="cyan",
            )
            pred_dates = [
                datetime.strptime(date, "%Y-%m-%d").date()
                for date in predictions[ticker].keys()
            ]
            pred_prices = list(predictions[ticker].values())
            axs[i].plot(
                pred_dates,
                pred_prices,
                label="Predicted",
                linestyle="--",
                color="magenta",
            )
            axs[i].set_title(f"{ticker} Stock Price", color="white")
            axs[i].set_ylabel("Price", color="white")
            axs[i].legend()
            axs[i].grid(True, color="gray")

        plt.tight_layout()
        plt.show()

    def _prepare_features(self, data: DataFrame) -> DataFrame:
        """Prepares the features for prediction."""
        data = data.copy()

        # Technical indicators
        data["SMA_5"] = data["Close"].rolling(window=5).mean()
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
        data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

        # MACD
        data["MACD"] = data["EMA_12"] - data["EMA_26"]
        data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        data["BB_Middle"] = data["Close"].rolling(window=20).mean()
        data["BB_Upper"] = (
            data["BB_Middle"] + 2 * data["Close"].rolling(window=20).std()
        )
        data["BB_Lower"] = (
            data["BB_Middle"] - 2 * data["Close"].rolling(window=20).std()
        )

        # Percentage changes
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            data[f"{col}_Pct_Change"] = data[col].pct_change()

        # Volatility
        data["Volatility"] = data["Close"].rolling(window=20).std()

        # Target variable: next day's closing price
        data["Target"] = data["Close"].shift(-1)

        # Drop rows with NaN values
        data.dropna(inplace=True)

        return data

    def train_model(self, data: DataFrame) -> None:
        """Trains the XGBoost model using historical stock data."""
        features = self._prepare_features(data)

        # Separate features and target
        X = features.drop(["Target", "Open", "High", "Low", "Close", "Volume"], axis=1)
        y = features["Target"]

        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        mse_scores = []
        r2_scores = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Scale the features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train the model
            self.model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = self.model.predict(X_val_scaled)

            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            mse_scores.append(mse)
            r2_scores.append(r2)

        logging.info(
            f"Mean MSE: {np.mean(mse_scores):.4f} (+/- {np.std(mse_scores):.4f})"
        )
        logging.info(f"Mean R2: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")

        # Retrain on the entire dataset
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def save_model(self, filename: str) -> None:
        """Saves the trained model to a file."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        joblib.dump(self.model, filename)
        joblib.dump(self.scaler, f"{filename}_scaler")
        logging.info(f"Model saved to {filename}")

    def load_model(self, filename: str) -> None:
        """Loads a trained model from a file."""
        self.model = joblib.load(filename)
        self.scaler = joblib.load(f"{filename}_scaler")
        logging.info(f"Model loaded from {filename}")
