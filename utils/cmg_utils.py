from datetime import datetime, timedelta
import pandas as pd
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

path = pathlib.PurePath('data/cmg')


def __get_daily_cmg_data(cmg_fn, tag) -> pd.DataFrame:
    """
    Loads daily commodities data.
    This data is available for the past 10 years only.
    :param cmg_fn: Commodities filename
    :param tag: Tag for this commodity
    :return:
    """
    df = pd.read_csv(path.joinpath(cmg_fn), parse_dates=['date'], skiprows=15)
    df = df.rename(columns={'date': 'Date', ' value': tag})
    df = df.set_index('Date')

    # Cull to daterange of interest
    df = df[df.index > '2008-01-01']

    #### START OF RESAMPLING
    # Force pandas to resample beginning at a specific data (note how this is end of fiscal year - important)
    df.loc[pd.Timestamp('2007-12-29')] = None
    df = df.sort_index()  # sorting by index

    # Fill & interpolate missing dates (resampling):
    df = df.resample('D')
    df = df.interpolate(method='polynomial', order=2)
    # df = df.ffill()

    # df = df.reset_index()

    return df

def get_cmg() -> pd.DataFrame:
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
    df = pd.read_csv(path.joinpath('eia/prices/energy_prices.csv'), parse_dates=['source key'], date_parser=dateparse, skiprows=5)
    df = df.rename(columns={'source key': 'Date', **cols_of_interest})
    df = df.set_index('Date')
    df = df.sort_index()  # sorting by index

    # Values are the mean of the daily spot prices for the entire month, including the 1st date in the next month
    # E.g. "Jul 2021" refers to mean of 01/07/2021 to 02/08/2021 spot prices
    # # Since spot prices are readily available, simply shift to mid-month, and interpolate.
    # #   Cannot do this, since inconsistent with average
    # df.index = df.index + timedelta(days=14)
    # df = df.resample('D')
    # df = df.interpolate(method='polynomial', order=2)
    # Simply ffill to the next month
    df = df.reindex(df.index.union([df.index[-1] + timedelta(days=30)]))
    df = df.resample('D').ffill()

    # Override with exact daily values where available
    df_brent = __get_daily_cmg_data('brent-crude-oil-prices-10-year-daily-chart.csv', 'brent')
    df_wticrude = __get_daily_cmg_data('wti-crude-oil-prices-10-year-daily-chart.csv', 'wticrude')

    # # Visual data verification
    # ax = plt.gca()
    # df['wticrude'].plot(ax=ax)
    # df_wticrude.plot(ax=ax)
    # plt.show()
    #
    # ax = plt.gca()
    # df['brent'].plot(ax=ax)
    # df_brent.plot(ax=ax)
    # plt.show()
    # assert False

    df['brent'] = df_brent.dropna().combine_first(df[['brent']])
    df['wticrude'] = df_wticrude.dropna().combine_first(df[['wticrude']])

    # Cull to daterange of interest
    # df = df[df.index > '2008-01-01']

    # Slice to columns of interest
    df = df[cols_of_interest.values()]

    # # calculate the correlation matrix
    # corr = df.corr()
    # print(corr)
    # # plot the heatmap
    # sns.heatmap(corr,
    #             xticklabels=corr.columns,
    #             yticklabels=corr.columns)
    # plt.tight_layout()
    # plt.show()

    return df


def get_oil_demand() -> pd.DataFrame:
    # cols_of_interest = {#'PAPR_WORLD': 'world production',
    #                     #'PATC_WORLD': 'world consumption',
    #                     #'COPR_OPEC': 'OPEC 13 Crude Oil Production',
    #                     # 'PATC_OECD_EUROPE': 'Europe Petroleum Consumption',
    #                     # 'PATC_US': 'U.S. (50 States) Petroleum Consumption',
    #                     # '': '',
    #                     # '': '',
    #                     'T3_STCHANGE_WORLD': 'world inventory net withdrawals',  # million barrels per day
    #                     'PASC_US': 'U.S. Commercial Inventory Crude Oil and Other Liquids',  # million barrels
    #                     }

    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    dateparse = lambda x: datetime.strptime(x, '%b %Y')
    df = pd.read_csv(path.joinpath('eia/supply_demand/production_consumption_and_inventories.csv'),
                     sep=';', parse_dates=['source key'], date_parser=dateparse, index_col=1, # parse_dates=['source key'], skiprows=4,
                     na_values='--')
    # df = df.rename(columns={'source key': 'Date', **cols_of_interest})
    df = df.rename(columns={'source key': 'Date'})
    df = df.set_index('Date')

    # Change in inventories
    # df['']

    # drop empty columns
    df = df.dropna(how='all', axis=1)

    # Cull to daterange of interest
    # df = df[df.index > '2008-01-01']

    # Resample to daily
    df = df.reindex(df.index.union([df.index[-1] + timedelta(days=30)]))
    df = df.resample('D').ffill()

    # Slice to columns of interest
    # df = df[cols_of_interest.values()]

    # Scale to reasonable size
    # df *= 1E3

    return df
