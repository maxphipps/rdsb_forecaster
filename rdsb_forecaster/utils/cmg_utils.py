import pathlib
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

path = pathlib.PurePath('rdsb_forecaster/data/cmg')


def get_cmg(corr_plot=False) -> pd.DataFrame:
    def get_daily_cmg_data(cmg_fn, tag) -> pd.DataFrame:
        """
        Loads daily commodities data.
        This data is available for the past 10 years only.
        :param cmg_fn: Commodities filename
        :param tag: Tag for this commodity
        :return:
        """
        df = pd.read_csv(path.joinpath(cmg_fn), parse_dates=['date'], skiprows=15, sep=';')
        df = df.rename(columns={'date': 'Date', ' value': tag})
        df = df.set_index('Date')

        # Slice to daterange of interest
        df = df[df.index > '2008-01-01']

        # Force pandas to resample beginning at a specific data
        # Note: this is end of fiscal year - important
        df.loc[pd.Timestamp('2007-12-29')] = None
        df = df.sort_index()  # sorting by index

        # Fill & interpolate missing dates:
        df = df.resample('D')
        df = df.interpolate(method='polynomial', order=2)
        # df = df.ffill()

        return df

    # Natural gas in dollars per thousand cubic feet
    # WTI and brent in dollars per barrel
    # gasoline and diesel fuel in cents per gallon
    cols_of_interest = {# Spot prices
                        'WTIPUUS': 'wticrude',
                        'BREPUUS': 'brent',
                        'NGHHMCF': 'naturalgas',  # high correlation
                        # Refiner Prices for Resale
                        'MGWHUUS': 'Gasoline',
                        # 'DSWHUUS': 'Diesel Fuel',  # high correlation
                        'D2WHUUS': 'Fuel Oil',
                        # Retail Prices Including Taxes
                        # 'MGRARUS': 'Gasoline Regular Grade', # high correlation
                        'MGEIAUS': 'Gasoline All Grades Retail Price',
                        # 'DSRTUUS': 'On-highway Diesel Fuel Retail Price', # high correlation
                        # U.S. Retail Prices
                        'NGICUUS': 'Natural Gas Price Industrial Sector',
                        'NGCCUUS': 'Natural Gas Price Commercial Sector',
                        # 'NGRCUUS': 'Natural Gas Price Residential Sector',
                        }

    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    dateparse = lambda x: datetime.strptime(x, '%b %Y')
    df = pd.read_csv(path.joinpath('eia/prices/energy_prices.csv'), parse_dates=['source key'], date_parser=dateparse, skiprows=5, sep=';')
    df = df.rename(columns={'source key': 'Date', **cols_of_interest})
    df = df.set_index('Date')
    df = df.sort_index()  # sorting by index

    # Values are the mean of the daily spot prices for the entire month, including the 1st date in the next month
    # E.g. "Jul 2021" refers to mean of 01/07/2021 to 02/08/2021 spot prices
    # Simply ffill to the next month
    df = df.reindex(df.index.union([df.index[-1] + timedelta(days=30)]))
    df = df.resample('D').ffill()

    # Override with exact daily values where available
    df_brent = get_daily_cmg_data('macrotrendsdotnet/brent-crude-oil-prices-10-year-daily-chart.csv', 'brent')
    df_wticrude = get_daily_cmg_data('macrotrendsdotnet/wti-crude-oil-prices-10-year-daily-chart.csv', 'wticrude')

    df['brent'] = df_brent.dropna().combine_first(df[['brent']])
    df['wticrude'] = df_wticrude.dropna().combine_first(df[['wticrude']])

    # Slice to columns of interest
    df = df[cols_of_interest.values()]

    if corr_plot:
        corr = df.corr()
        print(corr)
        sns.heatmap(corr,
                    xticklabels=corr.columns,
                    yticklabels=corr.columns)
        plt.tight_layout()
        plt.show()

    return df


def get_oil_supply_demand() -> pd.DataFrame:
    dateparse = lambda x: datetime.strptime(x, '%b %Y')
    df = pd.read_csv(path.joinpath('eia/supply_demand/production_consumption_and_inventories.csv'),
                     sep=';',
                     parse_dates=['source key'], date_parser=dateparse, index_col=1,
                     # parse_dates=['source key'], skiprows=4,
                     na_values='--')
    df = df.rename(columns={'source key': 'Date'})
    df = df.set_index('Date')

    # drop empty columns
    df = df.dropna(how='all', axis=1)

    # Resample to daily
    df = df.reindex(df.index.union([df.index[-1] + timedelta(days=30)]))
    df = df.resample('D').ffill()

    return df
