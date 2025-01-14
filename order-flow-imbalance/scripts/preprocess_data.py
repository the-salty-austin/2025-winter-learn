import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
PROJECT_ROOT = os.path.dirname(current_dir)  # Get the project root directory
sys.path.append(PROJECT_ROOT)  # Add the project root to the Python path

import json
import itertools

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas_market_calendars

CONFIG_ROOT = os.path.join(PROJECT_ROOT, 'config')

with open( os.path.join(CONFIG_ROOT, 'config.json'), 'r' ) as f:
    CONFIG = json.load(f)

nyse = pandas_market_calendars.get_calendar('NYSE')
in_sample_dt_strs = [x.strftime('%Y-%m-%d') for x in nyse.valid_days(start_date=CONFIG['insample_start'], end_date=CONFIG['insample_end'])]
out_of_sample_dt_strs = [x.strftime('%Y-%m-%d') for x in nyse.valid_days(start_date=CONFIG['outofsample_start'], end_date=CONFIG['outofsample_end'])]
symbols = CONFIG["symbols"]
interval = CONFIG["interval"]
start_time = CONFIG["start_time"]
end_time = CONFIG["end_time"]


class OrderBookProcessor:
    def __init__(self, symbol, interval='10s'):
        self.symbol = symbol
        self.interval = interval
        self.levels = [str(x).rjust(2, '0') for x in range(5)]

    def load_and_preprocess(self, dt_str):
        mpb10df = pd.read_csv(f"../data/raw/{self.symbol}_mbp-10_{dt_str}.csv", index_col=0)
        mpb10df.index = pd.to_datetime(mpb10df.index)
        mpb10df['mid_price'] = mpb10df[['bid_px_00','ask_px_00']].sum(1) / 2
        return mpb10df

    def calculate_order_flow(self, mpb10df):
        for level, side in itertools.product(self.levels, ['bid', 'ask']):
            price_col = f'{side}_px_{level}'
            qty_col = f'{side}_ct_{level}'
            mpb10df[qty_col] = mpb10df[qty_col].astype(int)
            mpb10df[f'prev_{price_col}'] = mpb10df[price_col].shift(1)
            mpb10df[f'prev_{qty_col}'] = mpb10df[qty_col].shift(1)
            mpb10df[f'{side}_of_{level}'] = 0.0
            if side == 'bid':
                mpb10df.loc[(mpb10df[price_col] > mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = mpb10df.loc[(mpb10df[price_col] > mpb10df[f'prev_{price_col}']), qty_col]
                mpb10df.loc[(mpb10df[price_col] < mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = -mpb10df.loc[(mpb10df[price_col] < mpb10df[f'prev_{price_col}']), qty_col]
                mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), qty_col] - mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), f'prev_{qty_col}']
            else:
                mpb10df.loc[(mpb10df[price_col] > mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = -mpb10df.loc[(mpb10df[price_col] > mpb10df[f'prev_{price_col}']), qty_col]
                mpb10df.loc[(mpb10df[price_col] < mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = mpb10df.loc[(mpb10df[price_col] < mpb10df[f'prev_{price_col}']), qty_col]
                mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), qty_col] - mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), f'prev_{qty_col}']
        return mpb10df

    def calculate_ofi(self, mpb10df):
        ofi_df = pd.DataFrame()
        ofi_df[f'Q_{self.interval}'] = 0.0
        for level in self.levels:
            ofi_df[f'OFI_{level}_{self.interval}'] = mpb10df[f'bid_of_{level}'].resample(self.interval).sum() - mpb10df[f'ask_of_{level}'].resample(self.interval).sum()
            ofi_df[f'Q_{self.interval}_{level}'] = (1/(2 * mpb10df['price'].resample(self.interval).count())) * (mpb10df[f'bid_ct_{level}'].resample(self.interval).sum() + mpb10df[f'ask_ct_{level}'].resample(self.interval).sum())
        ofi_df[f'Q_{self.interval}'] = ofi_df[[f'Q_{self.interval}_{level}' for level in self.levels]].sum(axis=1) / len(self.levels)
        for level in self.levels:
            ofi_df[f'ofi_{level}_{self.interval}'] = ofi_df[f'OFI_{level}_{self.interval}']/ofi_df[f'Q_{self.interval}']
        return ofi_df

    def calculate_integrated_ofi(self, ofi_df: pd.DataFrame, w1: np.ndarray=None, scaler: StandardScaler=None):
        pca_cols = [f"ofi_{str(x).rjust(2, '0')}_{self.interval}" for x in range(5)]
        data = ofi_df[pca_cols].to_numpy()
        data[np.isnan(data)] = 0

        if scaler is None:
            scaler = StandardScaler()
            data_standardized = scaler.fit_transform(data)
        else:
            data_standardized = scaler.transform(data)

        if w1 is None:
            pca = PCA(n_components=1)
            pca.fit(data_standardized)
            w1 = pca.components_[0]

        ofi_df[f'ofi_I_{self.interval}'] = ((w1 / np.abs(w1).sum()).T * data).sum(1)
        return ofi_df, w1, scaler

    def calculate_other_features(self, mpb10df: pd.DataFrame, ofi_df: pd.DataFrame):
        ofi_df[f'px_{self.interval}'] = mpb10df['mid_price'].resample(self.interval).last()
        ofi_df[f'r_{self.interval}'] = np.log(1 + ofi_df[f'px_{self.interval}'].pct_change())
        ofi_df['volume'] = mpb10df[mpb10df['action'] == 'T']['size'].resample(self.interval).sum()
        ofi_df['volume'] = ofi_df['volume'].fillna(0.0)
        ofi_df['symbol'] = self.symbol
        return ofi_df

    def process_and_save(self, dt_str: str, 
                         w1: np.ndarray=None, scaler: StandardScaler=None,
                         start_time: str='14:45', end_time: str='20:45'):
        mpb10df = self.load_and_preprocess(dt_str)
        mpb10df = self.calculate_order_flow(mpb10df)
        ofi_df = self.calculate_ofi(mpb10df)
        ofi_df, w1, scaler = self.calculate_integrated_ofi(ofi_df, w1, scaler)
        ofi_df = self.calculate_other_features(mpb10df, ofi_df)

        # Select relevant columns
        keep_cols = ['symbol', f"px_{self.interval}", f"r_{self.interval}", 'volume']
        keep_cols.extend([f'ofi_{l}_{self.interval}' for l in self.levels])
        keep_cols.append(f'ofi_I_{self.interval}')

        ofi_df = ofi_df.loc[f'{dt_str} {start_time}':f'{dt_str} {end_time}']
        ofi_df[keep_cols].to_csv(f"../data/processed/{self.symbol}_mbp-10_{dt_str}.csv")
        return ofi_df, w1, scaler


if __name__ == "__main__":
    # --- Train PCA and Scaler on ALL in-sample data ---
    all_in_sample_data = []
    for symbol in symbols:
        processor = OrderBookProcessor(symbol, interval=interval)
        for dt_str in in_sample_dt_strs:
            print(symbol, dt_str)
            tmp, _, _ = processor.process_and_save(dt_str, start_time=start_time, end_time=end_time)
            all_in_sample_data.append(tmp)

    in_sample_data = pd.concat(all_in_sample_data, axis=0)
    processor = OrderBookProcessor(symbols[0], interval=interval)  # Use the first symbol's processor
    in_sample_data, w1, scaler = processor.calculate_integrated_ofi(in_sample_data)

    for symbol in symbols:
        processor = OrderBookProcessor(symbol, interval=interval)
        for dt_str in in_sample_dt_strs:
            print(symbol, dt_str)
            processor.process_and_save(dt_str, w1=w1, scaler=scaler)

    # --- Process out-of-sample data ---
    for symbol in symbols:
        processor = OrderBookProcessor(symbol, interval=interval)
        for dt_str in out_of_sample_dt_strs:
            print(symbol, dt_str)
            processor.process_and_save(dt_str, w1=w1, scaler=scaler)


# def process_data(symbol, interval='10s', dt_str='2024-12-05'):
#     # Load raw data
#     mpb10df = pd.read_csv(f"../data/raw/{symbol}_mbp-10_{dt_str}.csv", index_col=0)
#     mpb10df.index = pd.to_datetime(mpb10df.index)
#     mpb10df['mid_price'] = mpb10df[['bid_px_00','ask_px_00']].sum(1) / 2

#     # Calculate order flow for each level and side
#     levels = [str(x).rjust(2, '0') for x in range(5)]
#     sides = ['bid', 'ask']
#     for level, side in itertools.product(levels, sides):
#         price_col = f'{side}_px_{level}'
#         qty_col   = f'{side}_ct_{level}'
#         mpb10df[qty_col]   = mpb10df[qty_col].astype(int)
#         mpb10df[f'prev_{price_col}'] = mpb10df[price_col].shift(1)
#         mpb10df[f'prev_{qty_col}']   = mpb10df[qty_col].shift(1)
#         mpb10df[f'{side}_of_{level}'] = 0.0 # initialize
#         # Section 2.1 for order flow (OF) definition
#         if side == 'bid':
#             mpb10df.loc[(mpb10df[price_col] > mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = mpb10df.loc[(mpb10df[price_col] > mpb10df[f'prev_{price_col}']), qty_col]
#             mpb10df.loc[(mpb10df[price_col] < mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = -mpb10df.loc[(mpb10df[price_col] < mpb10df[f'prev_{price_col}']), qty_col]
#             mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), qty_col] - mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), f'prev_{qty_col}']
#         else:
#             mpb10df.loc[(mpb10df[price_col] > mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = -mpb10df.loc[(mpb10df[price_col] > mpb10df[f'prev_{price_col}']), qty_col]
#             mpb10df.loc[(mpb10df[price_col] < mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = mpb10df.loc[(mpb10df[price_col] < mpb10df[f'prev_{price_col}']), qty_col]
#             mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), f'{side}_of_{level}'] = mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), qty_col] - mpb10df.loc[(mpb10df[price_col] == mpb10df[f'prev_{price_col}']), f'prev_{qty_col}']

#     # Calculate OFI and normalized OFI
#     ofi_df = pd.DataFrame()
#     ofi_df[f'Q_{interval}'] = 0.0
#     for level in levels:
#         # Calculate OFI
#         ofi_df[f'OFI_{level}_{interval}'] = mpb10df[f'bid_of_{level}'].resample(interval).sum() - mpb10df[f'ask_of_{level}'].resample(interval).sum()
#         # Calculate normalized quantity
#         ofi_df[f'Q_{interval}_{level}'] = (1/(2 * mpb10df['price'].resample(interval).count())) * (mpb10df[f'bid_ct_{level}'].resample(interval).sum() + mpb10df[f'ask_ct_{level}'].resample(interval).sum())
#     # continue calculating normalized quantity
#     ofi_df[f'Q_{interval}'] = ofi_df[[f'Q_{interval}_{level}' for level in levels]].sum(axis=1) / len(levels)
#     # Calculate ofi
#     for level in levels:
#         ofi_df[f'ofi_{level}_{interval}'] = ofi_df[f'OFI_{level}_{interval}']/ofi_df[f'Q_{interval}']

#     ofi_df[f'px_{interval}'] = mpb10df['mid_price'].resample(interval).last()
#     ofi_df[f'r_{interval}'] = np.log(1 + ofi_df[f'px_{interval}'].pct_change())
#     ofi_df['volume'] = mpb10df[mpb10df['action'] == 'T']['size'].resample(interval).sum()
#     ofi_df['volume'] = ofi_df['volume'].fillna(0.0)
#     ofi_df['symbol'] = symbol

#     # Select relevant columns
#     keep_cols = ['symbol', f"px_{interval}", f"r_{interval}", 'volume']
#     keep_cols.extend([f'ofi_{l}_{interval}' for l in levels]) # ofi's

#     return ofi_df[keep_cols]


# in_sample_dt_strs = [x.strftime('%Y-%m-%d') for x in nyse.valid_days(start_date=DATA_CONFIG['insample_start'], end_date=DATA_CONFIG['insample_end'])]
# out_of_sample_dt_strs = [x.strftime('%Y-%m-%d') for x in nyse.valid_days(start_date=DATA_CONFIG['outofsample_start'], end_date=DATA_CONFIG['outofsample_end'])]

# symbols = DATA_CONFIG["symbols"]
# interval = DATA_CONFIG["interval"]

# # Collect ALL in-sample data for PCA training
# for symbol in symbols:
#     all_in_sample_data = []
#     for dt_str in in_sample_dt_strs:
#         print(symbol, dt_str)
#         tmp = process_data(symbol, interval=interval, dt_str=dt_str)
#         tmp = tmp.loc[f'{dt_str} 14:45':f'{dt_str} 20:45']
#         all_in_sample_data.append(tmp)

#     # Concatenate all in-sample data into a single DataFrame
#     in_sample_data = pd.concat(all_in_sample_data, axis=0)

#     # Train PCA on the combined in-sample data
#     pca_cols = [f"ofi_{str(x).rjust(2, '0')}_{interval}" for x in range(5)]
#     data = in_sample_data[pca_cols].to_numpy()
#     data[np.isnan(data)] = 0
#     scaler = StandardScaler()
#     data_standardized = scaler.fit_transform(data)
#     pca = PCA(n_components=1)
#     pca.fit(data_standardized)
#     w1 = pca.components_[0]

#     in_sample_data[f'ofi_I_{interval}'] = ((w1 / np.abs(w1).sum()).T * in_sample_data[pca_cols]).sum(1)

#     for dt_str in in_sample_dt_strs:
#         segment = in_sample_data.loc[f'{dt_str} 14:45':f'{dt_str} 20:45']
#         segment.to_csv(f"../data/processed/{symbol}_mbp-10_{dt_str}.csv")

#     # --- Process out-of-sample data ---
#     all_oos_data = []
#     for dt_str in out_of_sample_dt_strs:
#         print(symbol, dt_str)
#         tmp = process_data(symbol, interval=interval, dt_str=dt_str, w1=w1)  # Pass w1 to process_data()
#         tmp = tmp.loc[f'{dt_str} 14:45':f'{dt_str} 20:45']
#         all_oos_data.append(tmp)
    
#     oos_data = pd.concat(all_oos_data, axis=0)

#     oos_data[f'ofi_I_{interval}'] = ((w1 / np.abs(w1).sum()).T * oos_data[pca_cols]).sum(1)

#     for dt_str in out_of_sample_dt_strs:
#         segment = oos_data.loc[f'{dt_str} 14:45':f'{dt_str} 20:45']
#         segment.to_csv(f"../data/processed/{symbol}_mbp-10_{dt_str}.csv")