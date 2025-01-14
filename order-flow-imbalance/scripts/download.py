import os
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)
sys.path.append(PROJECT_ROOT)

import datetime
import os

import databento as db
import pandas as pd



# --- Configuration ---

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)

CONFIG_ROOT = os.path.join(PROJECT_ROOT, "config")
with open(os.path.join(CONFIG_ROOT, "config.json"), "r") as f:
    DATA_CONFIG = json.load(f)

client = db.Historical(DATA_CONFIG["db-SHTimf6iQqxGSENt5NJp7te63MkjP"])

symbols = DATA_CONFIG["symbols"]
start_date = datetime.datetime.strptime(DATA_CONFIG["start_date"], "%Y-%m-%d")
end_date = datetime.datetime.strptime(DATA_CONFIG["end_date"], "%Y-%m-%d")

# --- Data Downloader Class ---

class DataDownloader:
    def __init__(self, client: db.Historical, dataset: str, schema: str):
        self.client = client
        self.dataset = dataset
        self.schema = schema

    def download_and_save_data(self, 
                               symbol: str,
                               start_date: datetime.datetime,
                               end_date: datetime.datetime):
        current_date = start_date
        while current_date < end_date:
            next_date = current_date + datetime.timedelta(days=1)
            print(f"Downloading {symbol} data for {current_date.strftime('%Y-%m-%d')}")

            data = self.client.timeseries.get_range(
                dataset=self.dataset,
                symbols=symbol,
                schema=self.schema,
                start=current_date.strftime("%Y-%m-%dT%H:%M:%S"),
                end=next_date.strftime("%Y-%m-%dT%H:%M:%S"),
            )

            tmp_df = data.to_df()
            current_date = next_date
            if len(tmp_df) == 0:
                continue
            tmp_df.to_csv(
                f"../data/raw/{symbol}_mbp-10_{tmp_df.index.max().strftime('%Y-%m-%d')}.csv"
            )


# --- Main Execution ---

if __name__ == "__main__":
    downloader = DataDownloader(client, "XNAS.ITCH", "mbp-10")  # Replace client with your actual client object
    for symbol in symbols:
        downloader.download_and_save_data(symbol, start_date, end_date)