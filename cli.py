import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Union


class CLI:
    """Class to handle command-line interface for stock price prediction."""

    def parse_arguments(self) -> Dict[str, Union[str, Tuple[str, str], List[str]]]:
        """Parses command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Predict stock prices using historical data."
        )

        parser.add_argument(
            "--tickers",
            type=self._validate_tickers,
            required=True,
            help="Comma-separated list of stock ticker symbols to predict.",
        )

        date_group = parser.add_mutually_exclusive_group(required=True)
        date_group.add_argument(
            "--date",
            type=self._validate_date,
            help="The date for which to predict the stock price (format: YYYY-MM-DD).",
        )
        date_group.add_argument(
            "--range",
            type=self._validate_date_range,
            help="Date range for prediction (format: YYYY-MM-DD,YYYY-MM-DD).",
        )

        parser.add_argument("--plot", action="store_true", help="Plot the predictions")
        parser.add_argument(
            "--save_model", type=str, help="Save the trained model to a file"
        )
        parser.add_argument(
            "--load_model", type=str, help="Load a trained model from a file"
        )

        args = parser.parse_args()
        return vars(args)

    @staticmethod
    def _validate_tickers(tickers_string: str) -> List[str]:
        """Validate the ticker symbols."""
        tickers = [ticker.strip().upper() for ticker in tickers_string.split(",")]
        if not all(ticker.isalpha() and len(ticker) <= 5 for ticker in tickers):
            raise argparse.ArgumentTypeError(
                "Invalid ticker format. Tickers should be alphabetic and up to 5 characters long."
            )
        return tickers

    @staticmethod
    def _validate_date(date_string: str) -> str:
        """Validate the date format."""
        try:
            datetime.strptime(date_string, "%Y-%m-%d")
            return date_string
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid date format. Please use YYYY-MM-DD."
            )

    @staticmethod
    def _validate_date_range(date_range: str) -> Tuple[str, str]:
        """Validate the date range format and ensure it's less than 365 days."""
        try:
            start_date, end_date = date_range.split(",")
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            if start > end:
                raise ValueError("Start date should be before end date.")
            if (end - start).days >= 365:
                raise ValueError("Date range should be less than 365 days.")
            return start_date, end_date
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Invalid date range: {str(e)}. Please use YYYY-MM-DD,YYYY-MM-DD and ensure the range is less than 365 days."
            )
