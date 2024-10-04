import json
import logging
from typing import List

from cli import CLI
from predictor import Predictor

logging.basicConfig(
    filename="stock_predictor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    """Main function to execute the stock price prediction process."""
    try:
        # Initialize the CLI and parse arguments
        cli = CLI()
        args = cli.parse_arguments()

        # Extract arguments
        tickers: List[str] = args["tickers"]
        date: str = args.get("date")
        date_range: tuple = args.get("range")
        plot: bool = args.get("plot", False)
        save_model: str = args.get("save_model")
        load_model: str = args.get("load_model")

        # Initialize the Predictor
        predictor = Predictor()

        if load_model:
            predictor.load_model(load_model)

        # Predict the stock price(s)
        if date:
            predictions = predictor.predict_price(tickers, date=date, plot=plot)
        elif date_range:
            predictions = predictor.predict_price(
                tickers, date_range=date_range, plot=plot
            )
        else:
            logging.error("Error: Either --date or --range must be provided.")
            return

        if save_model:
            predictor.save_model(save_model)

        # Output the result
        if not predictions:
            logging.warning(
                "Prediction could not be made due to insufficient data or invalid date(s)."
            )
        else:
            print(json.dumps(predictions, indent=2))

        if plot:
            input("Press Enter to close the plot and exit...")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
