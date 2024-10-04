import argparse
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cli import CLI
from predictor import Predictor
from yahoo_finance_api import YahooFinanceAPI


@pytest.fixture
def mock_yahoo_finance_api():
    with patch("yahoo_finance_api.YahooFinanceAPI") as mock:
        yield mock


@pytest.fixture
def sample_historical_data():
    dates = pd.date_range(start="2020-01-01", end="2022-01-01", freq="D")
    data = pd.DataFrame(
        {
            "Open": np.random.rand(len(dates)) * 100 + 50,
            "High": np.random.rand(len(dates)) * 100 + 60,
            "Low": np.random.rand(len(dates)) * 100 + 40,
            "Close": np.random.rand(len(dates)) * 100 + 55,
            "Volume": np.random.randint(100, 1000, len(dates)),
            "Dividends": np.zeros(len(dates)),
            "Stock Splits": np.zeros(len(dates)),
        },
        index=dates,
    )
    return data


@patch("yahoo_finance_api.YahooFinanceAPI.get_historical_data")
def test_yahoo_finance_api(mock_get_historical_data, sample_historical_data):
    mock_get_historical_data.return_value = sample_historical_data
    api = YahooFinanceAPI()
    data = api.get_historical_data("AAPL")
    assert not data.empty
    assert set(data.columns) == set(
        ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
    )
    mock_get_historical_data.assert_called_once_with("AAPL")


def test_predictor_initialization():
    predictor = Predictor()
    assert predictor.model is not None
    assert predictor.scaler is not None


def test_predictor_prepare_features(sample_historical_data):
    predictor = Predictor()
    features = predictor._prepare_features(sample_historical_data)
    assert "SMA_5" in features.columns
    assert "MACD" in features.columns
    assert "RSI" in features.columns
    assert "Target" in features.columns


def test_predictor_train_model(sample_historical_data):
    predictor = Predictor()
    predictor.train_model(sample_historical_data)
    assert predictor.model is not None


@pytest.mark.parametrize(
    "date,expected_type", [("2023-06-01", dict), ("2025-01-01", dict)]
)
def test_predictor_predict_single_date(
    mock_yahoo_finance_api, sample_historical_data, date, expected_type
):
    mock_yahoo_finance_api.return_value.get_historical_data.return_value = (
        sample_historical_data
    )
    predictor = Predictor()
    prediction = predictor.predict_price(["AAPL"], date=date)
    assert isinstance(prediction, expected_type)
    assert "AAPL" in prediction
    assert isinstance(prediction["AAPL"], dict)


@patch("predictor.Predictor._predict_single_ticker")
def test_predictor_predict_date_range(
    mock_predict_single_ticker, mock_yahoo_finance_api, sample_historical_data
):
    mock_yahoo_finance_api.return_value.get_historical_data.return_value = (
        sample_historical_data
    )

    # Create a mock prediction for the date range
    start_date = "2023-06-01"
    end_date = "2023-06-30"
    mock_prediction = {}
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        mock_prediction[date_str] = 100.0  # Mock price
        current_date += timedelta(days=1)

    mock_predict_single_ticker.return_value = mock_prediction

    predictor = Predictor()
    prediction = predictor.predict_price(["AAPL"], date_range=(start_date, end_date))

    assert isinstance(prediction, dict)
    assert "AAPL" in prediction
    assert isinstance(prediction["AAPL"], dict)

    # Check if all dates in the range are present in the prediction
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        assert (
            date_str in prediction["AAPL"]
        ), f"Date {date_str} not found in prediction"
        assert isinstance(
            prediction["AAPL"][date_str], float
        ), f"Prediction for {date_str} is not a float"
        current_date += timedelta(days=1)

    # Additional assertions to check the structure of the prediction
    assert len(prediction) == 1, "Prediction should contain only one ticker"
    assert (
        len(prediction["AAPL"])
        == (end - datetime.strptime(start_date, "%Y-%m-%d")).days + 1
    ), "Prediction should contain all dates in the range"


def test_predictor_save_load_model(tmp_path):
    predictor = Predictor()
    model_path = tmp_path / "test_model.joblib"
    predictor.save_model(str(model_path))
    assert model_path.exists()

    new_predictor = Predictor()
    new_predictor.load_model(str(model_path))
    assert new_predictor.model is not None


def test_cli_parse_arguments():
    cli = CLI()
    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(
            tickers="AAPL,GOOGL", date="2023-06-01", range=None, plot=False
        ),
    ):
        args = cli.parse_arguments()
        assert args["tickers"] == "AAPL,GOOGL"
        assert args["date"] == "2023-06-01"
        assert args["range"] is None
        assert args["plot"] is False


def test_cli_validate_tickers():
    cli = CLI()
    assert cli._validate_tickers("AAPL,GOOGL") == ["AAPL", "GOOGL"]
    with pytest.raises(argparse.ArgumentTypeError):
        cli._validate_tickers("AAPL,INVALID!")


def test_cli_validate_date():
    cli = CLI()
    assert cli._validate_date("2023-06-01") == "2023-06-01"
    with pytest.raises(argparse.ArgumentTypeError):
        cli._validate_date("2023/06/01")


def test_cli_validate_date_range():
    cli = CLI()
    assert cli._validate_date_range("2023-06-01,2023-06-30") == (
        "2023-06-01",
        "2023-06-30",
    )
    with pytest.raises(argparse.ArgumentTypeError):
        cli._validate_date_range("2023-06-01,2024-06-01")  # More than 365 days


@patch("predictor.Predictor.predict_price")
@patch("predictor.Predictor.save_model")
@patch("predictor.Predictor.load_model")
def test_main(mock_load_model, mock_save_model, mock_predict_price, capsys):
    from stock_predictor import main

    mock_predict_price.return_value = {"AAPL": {"2023-06-01": 150.0}}
    with patch("sys.argv", ["main.py", "--tickers", "AAPL", "--date", "2023-06-01"]):
        main()
    captured = capsys.readouterr()
    assert "AAPL" in captured.out
    assert "150.0" in captured.out


if __name__ == "__main__":
    pytest.main()
