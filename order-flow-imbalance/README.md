# Order Flow Imbalance and Price Impact

This repository contains code and analysis for replicating and extending the study by Cont et al. (2023) on the cross-impact of order flow imbalance in equity markets.

## Description

This project investigates the relationship between order flow imbalance (OFI) and price movements in a multi-asset setting, using Nasdaq TotalView-ITCH data for five stocks: AAPL, AMGN, TSLA, JPM, and XOM. The analysis focuses on:

* **Data preprocessing:** Cleaning, aggregating, and calculating OFI metrics from raw order book data.
* **Self-impact analysis:** Examining the impact of a stock's own OFI on its price.
* **Cross-impact analysis:** Investigating the impact of OFIs from other stocks on a given stock's price.
* **Predictive power:** Assessing the ability of lagged OFIs to predict future returns.

## Files

* `download.py`: Downloads raw order book data from Nasdaq TotalView-ITCH.
* `preprocess_data.py`: Processes the raw data, calculates OFI metrics, and applies PCA to obtain integrated OFIs.
* `regress.py`: Performs regression analysis (OLS and LASSO) to assess self-impact, cross-impact, and predictive power of OFIs.
* `main.py`:  Main script to orchestrate the data downloading, preprocessing, and regression analysis.
* `config.json`: Configuration file containing parameters for data processing and analysis.
* `requirements.txt`:  Lists the required Python packages.

## How to Run

1. **Clone the repository:** `git clone https://github.com/your-username/order-flow-imbalance.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Configure parameters:**  Modify `config.json` to set the desired dates, symbols, and other parameters.
4. **Download data:** `python download.py`
5. **Preprocess data:** `python preprocess_data.py`
6. **Run regression analysis:** `python regress.py`

## Results

The results of the analysis are presented in the `results` directory, including:

* **R-squared values:** Tables and figures summarizing the in-sample and out-of-sample R-squared values for different regression models.
* **Correlation matrices:** Heatmaps visualizing the correlations between OFIs at different levels of the order book.
* **Regression coefficients:**  Tables and figures showing the estimated coefficients for the OLS and LASSO regression models.

## Key Findings

* The analysis confirms the positive correlation between OFIs across multiple levels of the limit order book.
* For highly liquid stocks with significant retail investor participation, the best-level OFI exhibits the strongest correlation with other levels.
* Both best-level and integrated OFIs show significant self-impact on contemporaneous returns.
* Cross-impact analysis reveals limited explanatory power for contemporaneous returns.
* Lagged OFIs have negligible predictive power on future returns, suggesting limited short-term predictability.

## References

Cont, R., Cucuringu, M., & Zhang, C. (2023). Cross-Impact of Order Flow Imbalance in Equity Markets. arXiv preprint arXiv:2112.13213v4.