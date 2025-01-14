import os
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)
sys.path.append(PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
import statsmodels.formula.api as sm
import pandas_market_calendars


# --- Configuration ---

CONFIG_ROOT = os.path.join(PROJECT_ROOT, "config")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
with open(os.path.join(CONFIG_ROOT, "config.json"), "r") as f:
    DATA_CONFIG = json.load(f)

symbols = DATA_CONFIG["symbols"]
interval = DATA_CONFIG["interval"]
lags = DATA_CONFIG["lags"]
fwds = DATA_CONFIG["fwds"]

nyse = pandas_market_calendars.get_calendar("NYSE")
in_sample_dt_strs = [
    x.strftime("%Y-%m-%d")
    for x in nyse.valid_days(
        start_date=DATA_CONFIG["insample_start"], end_date=DATA_CONFIG["insample_end"]
    )
]
out_of_sample_dt_strs = [
    x.strftime("%Y-%m-%d")
    for x in nyse.valid_days(
        start_date=DATA_CONFIG["outofsample_start"],
        end_date=DATA_CONFIG["outofsample_end"],
    )
]


# --- Regression Functions ---

def regress(df: pd.DataFrame, dependent_var: str, independent_vars: list[str]):
    formula = f'{dependent_var} ~ {" + ".join(independent_vars)}'
    model = sm.ols(formula, data=df).fit()
    return model


def lasso_lagged_regress(
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    dependent_var: str,
    independent_vars: list[str],
    lags: list[int] = [1, 2, 3, 5, 10, 20, 30],
    fwd: int = 1,
):
    # Construct lagged variable names
    lagged_vars = [f"{var}_lag{lag}" for var in independent_vars for lag in lags]

    X_is = is_df[lagged_vars]
    y_is = is_df[f"{dependent_var}_fwd{fwd}"]

    # Perform Lasso regression with cross-validation
    model_cv = LassoCV(cv=5, random_state=0).fit(X_is, y_is)
    model = Lasso(alpha=model_cv.alpha_, random_state=0).fit(
        X_is, y_is
    )  # one-SE rule

    # Predict on IS data and calculate R-squared
    predictions_is = model.predict(X_is)
    ss_total_is = np.sum((y_is - np.mean(y_is)) ** 2)
    ss_residual_is = np.sum((y_is - predictions_is) ** 2)
    r_squared_is = 1 - (ss_residual_is / ss_total_is)
    print(f"IS R-squared for {dependent_var}: {r_squared_is:.4f}")

    # Prepare data for glmnet (OOS)
    X_oos = oos_df[lagged_vars]
    y_oos = oos_df[f"{dependent_var}_fwd{fwd}"]

    # Predict on OOS data and calculate R-squared
    predictions_oos = model.predict(X_oos)
    ss_total_oos = np.sum((y_oos - np.mean(y_oos)) ** 2)
    ss_residual_oos = np.sum((y_oos - predictions_oos) ** 2)
    r_squared_oos = 1 - (ss_residual_oos / ss_total_oos)
    print(f"OOS R-squared for {dependent_var}: {r_squared_oos:.4f}")

    return model


def ols_lagged_regress(
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    dependent_var: str,
    independent_vars: list,
    lags: list[int] = [1, 2, 3, 5, 10, 20, 30],
    fwd: int = 1,
):
    lagged_vars = [f"{var}_lag{lag}" for var in independent_vars for lag in lags]

    X_is = is_df[lagged_vars]
    y_is = is_df[f"{dependent_var}_fwd{fwd}"]

    # Perform OLS regression
    formula = (
        f'{y_is.name} ~ {" + ".join(X_is.columns)}'  # Use all lagged_vars
    )
    model = sm.ols(formula, data=pd.concat([X_is, y_is], axis=1)).fit()

    # Predict on IS data and calculate R-squared
    predictions_is = model.predict(X_is)
    ss_total_is = np.sum((y_is - np.mean(y_is)) ** 2)
    ss_residual_is = np.sum((y_is - predictions_is) ** 2)
    r_squared_is = 1 - (ss_residual_is / ss_total_is)
    print(f"IS R-squared for {dependent_var}: {r_squared_is:.4f}")

    # Prepare data for lm (OOS)
    X_oos = oos_df[lagged_vars]
    y_oos = oos_df[f"{dependent_var}_fwd{fwd}"]

    # Predict on OOS data and calculate R-squared
    predictions_oos = model.predict(X_oos)
    ss_total_oos = np.sum((y_oos - np.mean(y_oos)) ** 2)
    ss_residual_oos = np.sum((y_oos - predictions_oos) ** 2)
    r_squared_oos = 1 - (ss_residual_oos / ss_total_oos)
    print(f"OOS R-squared for {dependent_var}: {r_squared_oos:.4f}")

    return model


# --- Data Loading and Preprocessing Class ---

class DataPreprocessor:
    def __init__(self, symbols, interval, lags, fwds, in_sample_dt_strs, out_of_sample_dt_strs):
        self.symbols = symbols
        self.interval = interval
        self.lags = lags
        self.fwds = fwds
        self.in_sample_dt_strs = in_sample_dt_strs
        self.out_of_sample_dt_strs = out_of_sample_dt_strs
        self.lagcols = [f"ofi_I_{interval}"] + [f"ofi_{str(x).rjust(2, '0')}_{interval}" for x in range(5)]

    def load_and_process_data(self, dt_strs):
        all_data = []
        for symbol in self.symbols:
            inner = []
            for dt_str in dt_strs:
                tmp = pd.read_csv(
                    os.path.join(DATA_ROOT, "processed", f"{symbol}_mbp-10_{dt_str}.csv"),
                    index_col=0,
                )
                tmp.index = pd.to_datetime(tmp.index)
                tmp = tmp.loc[f"{dt_str} 14:45":f"{dt_str} 20:45"]
                for fwd in self.fwds:
                    tmp[f"r_{self.interval}_fwd{fwd}"] = tmp[f"r_{self.interval}"].shift(-fwd)
                for col in self.lagcols:
                    for lag in self.lags:
                        tmp[f"{col}_lag{lag}"] = tmp[col].shift(lag)
                tmp.columns = [f"{symbol}_{c}" for c in tmp.columns]
                inner.append(tmp)
            inner_df = pd.concat(inner, axis=0)
            all_data.append(inner_df)
        lagged_df = pd.concat(all_data, axis=1)
        return lagged_df.dropna()

    def preprocess_data(self):
        lagged_is_df = self.load_and_process_data(self.in_sample_dt_strs)
        lagged_oos_df = self.load_and_process_data(self.out_of_sample_dt_strs)
        return lagged_is_df, lagged_oos_df


# --- Main Execution ---

if __name__ == "__main__":
    preprocessor = DataPreprocessor(symbols, interval, lags, fwds, in_sample_dt_strs, out_of_sample_dt_strs)
    lagged_is_df, lagged_oos_df = preprocessor.preprocess_data()

    # --- Self Impact ---
    print("\n--- Self Impact ---\n")
    for symbol in symbols:
        independent_vars = [f"{symbol}_ofi_00_{interval}"]
        model = regress(lagged_is_df, f"{symbol}_r_{interval}", independent_vars)
        print(model.summary())
        # ... (Calculate and print OOS R-squared as before) ...

    # --- Cross Impact ---
    print("\n--- Cross Impact ---\n")
    for this_symbol in symbols:
        independent_vars = [f"{symbol}_ofi_00_{interval}" for symbol in symbols]
        model = regress(lagged_is_df, f"{this_symbol}_r_{interval}", independent_vars)
        print(model.summary())
        # ... (Calculate and print OOS R-squared as before) ...

    # --- Forward-Looking Best-Level & Integrated Impact (OLS) ---
    print("\n--- Forward-Looking Best-Level & Integrated Impact (OLS) ---\n")
    for this_symbol in symbols:
        independent_vars = [f"{this_symbol}_ofi_00_{interval}"]
        model = ols_lagged_regress(
            lagged_is_df, lagged_oos_df, f"{this_symbol}_r_{interval}", independent_vars
        )
        print(model.summary())

    # --- Forward-Looking Best-Level & Integrated Impact (LASSO) ---
    print("\n--- Forward-Looking Best-Level & Integrated Impact (LASSO) ---\n")
    for this_symbol in symbols:
        independent_vars = [f"{this_symbol}_ofi_00_{interval}"]
        model = lasso_lagged_regress(
            lagged_is_df, lagged_oos_df, f"{this_symbol}_r_{interval}", independent_vars
        )
        for lag, coef in zip(lags, model.coef_):
            print(f"{this_symbol} Best-level OFI lag {lag}: {coef}")
        print("------------")


# import os, sys
# import json

# current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
# PROJECT_ROOT = os.path.dirname(current_dir)  # Get the project root directory
# sys.path.append(PROJECT_ROOT)  # Add the project root to the Python path

# import pandas as pd
# import numpy as np
# from sklearn.linear_model import Lasso, LassoCV
# import statsmodels.formula.api as sm
# import pandas_market_calendars

# CONFIG_ROOT = os.path.join(PROJECT_ROOT, 'config')

# with open( os.path.join(CONFIG_ROOT, 'config.json'), 'r' ) as f:
#     CONFIG = json.load(f)

# nyse = pandas_market_calendars.get_calendar('NYSE')

# # --- Configuration ---

# current_dir = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(current_dir)
# sys.path.append(PROJECT_ROOT)

# CONFIG_ROOT = os.path.join(PROJECT_ROOT, "config")
# DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
# with open(os.path.join(CONFIG_ROOT, "config.json"), "r") as f:
#     DATA_CONFIG = json.load(f)

# symbols = DATA_CONFIG["symbols"]
# interval = DATA_CONFIG["interval"]
# in_sample_dt_strs = [x.strftime('%Y-%m-%d') for x in nyse.valid_days(start_date=CONFIG['insample_start'], end_date=CONFIG['insample_end'])]
# out_of_sample_dt_strs = [x.strftime('%Y-%m-%d') for x in nyse.valid_days(start_date=CONFIG['outofsample_start'], end_date=CONFIG['outofsample_end'])]

# # --- Regression Functions ---

# def regress(df: pd.DataFrame, dependent_var: str, independent_vars: list[str]):
#     formula = f'{dependent_var} ~ {" + ".join(independent_vars)}'
#     model = sm.ols(formula, data=df).fit()
#     return model


# def lasso_lagged_regress(
#     is_df: pd.DataFrame,
#     oos_df: pd.DataFrame,
#     dependent_var: str,
#     independent_vars: list[str],
#     lags: list[int]=[1, 2, 3, 5, 10, 20, 30],
#     fwd: int=1,
# ):
#     # Construct lagged variable names
#     lagged_vars = [f"{var}_lag{lag}" for var in independent_vars for lag in lags]

#     X_is = is_df[lagged_vars]
#     y_is = is_df[f"{dependent_var}_fwd{fwd}"]

#     # Perform Lasso regression with cross-validation
#     model_cv = LassoCV(cv=5, random_state=0).fit(X_is, y_is)
#     model = Lasso(alpha=model_cv.alpha_, random_state=0).fit(X_is, y_is) # one-SE rule

#     # Predict on IS data and calculate R-squared
#     predictions_is = model.predict(X_is)
#     ss_total_is = np.sum((y_is - np.mean(y_is)) ** 2)
#     ss_residual_is = np.sum((y_is - predictions_is) ** 2)
#     r_squared_is = 1 - (ss_residual_is / ss_total_is)
#     print(f"IS R-squared for {dependent_var}: {r_squared_is:.4f}")

#     # Prepare data for glmnet (OOS)
#     X_oos = oos_df[lagged_vars]
#     y_oos = oos_df[f"{dependent_var}_fwd{fwd}"]

#     # Predict on OOS data and calculate R-squared
#     predictions_oos = model.predict(X_oos)
#     ss_total_oos = np.sum((y_oos - np.mean(y_oos)) ** 2)
#     ss_residual_oos = np.sum((y_oos - predictions_oos) ** 2)
#     r_squared_oos = 1 - (ss_residual_oos / ss_total_oos)
#     print(f"OOS R-squared for {dependent_var}: {r_squared_oos:.4f}")

#     return model


# def ols_lagged_regress(
#     is_df: pd.DataFrame,
#     oos_df: pd.DataFrame,
#     dependent_var: str,
#     independent_vars: list,
#     lags: list[int]=[1, 2, 3, 5, 10, 20, 30],
#     fwd: int=1,
# ):
#     lagged_vars = [f"{var}_lag{lag}" for var in independent_vars for lag in lags]

#     X_is = is_df[lagged_vars]
#     y_is = is_df[f"{dependent_var}_fwd{fwd}"]

#     # Perform OLS regression
#     formula = f'{y_is.name} ~ {" + ".join(X_is.columns)}'  # Use all lagged_vars
#     model = sm.ols(formula, data=pd.concat([X_is, y_is], axis=1)).fit()

#     # Predict on IS data and calculate R-squared
#     predictions_is = model.predict(X_is)
#     ss_total_is = np.sum((y_is - np.mean(y_is)) ** 2)
#     ss_residual_is = np.sum((y_is - predictions_is) ** 2)
#     r_squared_is = 1 - (ss_residual_is / ss_total_is)
#     print(f"IS R-squared for {dependent_var}: {r_squared_is:.4f}")

#     # Prepare data for lm (OOS)
#     X_oos = oos_df[lagged_vars]
#     y_oos = oos_df[f"{dependent_var}_fwd{fwd}"]

#     # Predict on OOS data and calculate R-squared
#     predictions_oos = model.predict(X_oos)
#     ss_total_oos = np.sum((y_oos - np.mean(y_oos)) ** 2)
#     ss_residual_oos = np.sum((y_oos - predictions_oos) ** 2)
#     r_squared_oos = 1 - (ss_residual_oos / ss_total_oos)
#     print(f"OOS R-squared for {dependent_var}: {r_squared_oos:.4f}")

#     return model


# # --- Main Execution ---

# if __name__ == "__main__":
#     # --- Load IS data ---
#     lags=[1, 2, 3, 5, 10, 20, 30]
#     fwds=[1, 2, 3, 5, 10, 20, 30]
#     lagcols = [f'ofi_I_{interval}']
#     lagcols.extend([f"ofi_{str(x).rjust(2, '0')}_{interval}" for x in range(5)])

#     all_data = []
#     for symbol in symbols:
#         cols = [f"{x}" for x in lagcols]

#         inner = []
#         for dt_str in in_sample_dt_strs:
#             tmp = pd.read_csv(os.path.join(DATA_ROOT, "processed", f"{symbol}_mbp-10_{dt_str}.csv"), index_col=0)
#             tmp.index = pd.to_datetime(tmp.index)
#             tmp = tmp.loc[f'{dt_str} 14:45':f'{dt_str} 20:45']
#             for fwd in fwds:
#                 tmp[f'r_{interval}_fwd{fwd}'] = tmp[f'r_{interval}'].shift(-fwd)
#             for col in cols:
#                 for lag in lags:
#                     tmp[f'{col}_lag{lag}'] = tmp[col].shift(lag)
#             tmp.columns = [f'{symbol}_{c}' for c in tmp.columns]
#             inner.append(tmp)
#         inner_df = pd.concat(inner, axis=0)
#         all_data.append(inner_df)
#     lagged_is_df = pd.concat(all_data, axis=1)
#     lagged_is_df = lagged_is_df.dropna()
#     # print(lagged_is_df)

#     # print(lagged_is_df.columns)

#     all_data = []
#     for symbol in symbols:
#         inner = []
#         for dt_str in out_of_sample_dt_strs:
#             tmp = pd.read_csv(os.path.join(DATA_ROOT, "processed", f"{symbol}_mbp-10_{dt_str}.csv"), index_col=0)
#             tmp.index = pd.to_datetime(tmp.index)
#             tmp = tmp.loc[f'{dt_str} 14:45':f'{dt_str} 20:45']

#             for fwd in fwds:
#                 tmp[f'r_{interval}_fwd{fwd}'] = tmp[f'r_{interval}'].shift(-fwd)
#             for col in cols:
#                 for lag in lags:
#                     tmp[f'{col}_lag{lag}'] = tmp[col].shift(lag)
#             tmp.columns = [f"{symbol}_{c}" for c in tmp.columns]
#             inner.append(tmp)
#         inner_df = pd.concat(inner, axis=0)
#         all_data.append(inner_df)
#     lagged_oos_df = pd.concat(all_data, axis=1)
#     lagged_oos_df = lagged_oos_df.dropna()


#     # --- Self Impact ---
#     print("\n--- Self Impact ---\n")
#     for symbol in symbols:
#         independent_vars = [f"{symbol}_ofi_00_{interval}"]
#         model = regress(
#             lagged_is_df, f"{symbol}_r_{interval}", independent_vars
#         )
#         print(model.summary())

#         # Calculate and print OOS R-squared
#         predictions_oos = model.predict(lagged_oos_df)
#         ss_total_oos = np.sum(
#             (lagged_oos_df[f"{symbol}_r_{interval}"] - np.mean(lagged_oos_df[f"{symbol}_r_{interval}"])) ** 2
#         )
#         ss_residual_oos = np.sum(
#             (lagged_oos_df[f"{symbol}_r_{interval}"] - predictions_oos) ** 2
#         )
#         r_squared_oos = 1 - (ss_residual_oos / ss_total_oos)
#         print(f"OOS R-squared for {symbol}: {r_squared_oos:.4f}")

#     # --- Cross Impact ---
#     print("\n--- Cross Impact ---\n")
#     for this_symbol in symbols:
#         independent_vars = [f"{symbol}_ofi_00_{interval}" for symbol in symbols]
#         model = regress(
#             lagged_is_df, f"{this_symbol}_r_{interval}", independent_vars
#         )
#         print(model.summary())

#         # Calculate and print OOS R-squared
#         predictions_oos = model.predict(lagged_oos_df)
#         ss_total_oos = np.sum(
#             (lagged_oos_df[f"{this_symbol}_r_{interval}"] - np.mean(lagged_oos_df[f"{this_symbol}_r_{interval}"])) ** 2
#         )
#         ss_residual_oos = np.sum(
#             (lagged_oos_df[f"{this_symbol}_r_{interval}"] - predictions_oos) ** 2
#         )
#         r_squared_oos = 1 - (ss_residual_oos / ss_total_oos)
#         print(f"OOS R-squared for {this_symbol}: {r_squared_oos:.4f}")

#     # --- Forward-Looking Best-Level & Integrated Impact (OLS) ---
#     print("\n--- Forward-Looking Best-Level & Integrated Impact (OLS) ---\n")
#     for this_symbol in symbols:
#         independent_vars = [f"{this_symbol}_ofi_00_{interval}"]
#         model = ols_lagged_regress(
#             lagged_is_df,
#             lagged_oos_df,
#             f"{this_symbol}_r_{interval}",
#             independent_vars,
#         )
#         print(model.summary())

#     # --- Forward-Looking Best-Level & Integrated Impact (LASSO) ---
#     print("\n--- Forward-Looking Best-Level & Integrated Impact (LASSO) ---\n")

#     for this_symbol in symbols:
#         independent_vars = [f"{this_symbol}_ofi_00_{interval}"]
#         model = lasso_lagged_regress(
#             lagged_is_df,
#             lagged_oos_df,
#             f"{this_symbol}_r_{interval}",
#             independent_vars,
#         )
#         for lag, coef in zip(lags, model.coef_):
#             print(f"{this_symbol} Best-level OFI lag {lag}: {coef}")
#         print("------------")
#         # print(model.coef_)