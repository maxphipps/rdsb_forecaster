# RDSB Forecaster

### Overview
This project contains a simple ridge regression model to predict the Shell share price based on its fundamentals.
The default config fits using a test set of the prior 10 years' data.

Specifically, the target variable is the daily Shell spot close price in USD. 
No adjustment for dividends is applied.

The model considers a number of features
(asterick indicates features in use for default config):
* Financials:
  * Income and balance sheet data(*)
  * Margins
  * Production volumes(*)
* Spot prices
  * Natural gas commercial(*)
  * WTI crude(*)
  * US Gasoline all retail(*)

### Data links:
* data/eia/prices/energy_prices.csv: [[click here]](https://www.eia.gov/outlooks/steo/data/browser/#/?v=8&f=M&s=0&start=199701&end=202212&ctype=linechart&maptype=0&linechart=WTIPUUS)
* data/cmg/macrotrendsdotnet/wti-crude-oil-prices-10-year-daily-chart.csv: [[click here]](https://www.macrotrends.net/2516/wti-crude-oil-prices-10-year-daily-chart)

Currently unused data:
* data/cmg/brent-crude-oil-prices-10-year-daily-chart: [[click here]](https://www.macrotrends.net/2480/brent-crude-oil-prices-10-year-daily-chart)
* data/fxspot/GBPUSD.csv: [[click here]](https://uk.investing.com/currencies/gbp-usd-historical-data)
* data/RDSB/RDSB.L.csv: [[click here]](https://uk.finance.yahoo.com/quote/SHEL.L/history?p=SHEL.L)