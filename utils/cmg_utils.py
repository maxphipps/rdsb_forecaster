import pandas as pd
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

path = pathlib.PurePath('data/cmg')

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

    from datetime import datetime
    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    dateparse = lambda x: datetime.strptime(x, '%b %Y')
    df = pd.read_csv(path.joinpath('eia/prices/energy_prices.csv'), parse_dates=['source key'], date_parser=dateparse, skiprows=5)
    df = df.rename(columns={'source key': 'Date', **cols_of_interest})
    df = df.set_index('Date')

    # Convert index date to datetime type
    # df.index = pd.to_datetime(df)

    # Cull to daterange of interest
    # df = df[df.index > '2008-01-01']

    #### START OF RESAMPLING
    # Force pandas to resample beginning at a specific data (note how this is end of fiscal year - important)
    # df.loc[pd.Timestamp('2007-12-29')] = None
    df = df.sort_index()  # sorting by index

    # Fill & interpolate missing dates (resampling):
    df = df.resample('D')
    # df = df.interpolate(method='polynomial', order=2)
    df = df.ffill()
    df = df.bfill()

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
    cols_of_interest = {'PAPR_WORLD': 'world production',
                        'PATC_WORLD': 'world consumption',
                        # 'PATC_OECD_EUROPE': 'Europe Petroleum Consumption',
                        # 'PATC_US': 'U.S. (50 States) Petroleum Consumption',
                        # '': '',
                        # '': '',
                        'T3_STCHANGE_WORLD': 'world inventory net withdrawals',  # million barrels per day
                        'PASC_US': 'U.S. Commercial Inventory Crude Oil and Other Liquids',  # million barrels
                        }

    from datetime import datetime
    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    dateparse = lambda x: datetime.strptime(x, '%b %Y')
    df = pd.read_csv(path.joinpath('eia/supply_demand/production_consumption_and_inventories.csv'), parse_dates=['source key'], date_parser=dateparse, skiprows=4)
    df = df.rename(columns={'source key': 'Date', **cols_of_interest})
    df = df.set_index('Date')

    # Convert index date to datetime type
    # df.index = pd.to_datetime(df)

    # Cull to daterange of interest
    # df = df[df.index > '2008-01-01']

    #### START OF RESAMPLING
    # Force pandas to resample beginning at a specific data (note how this is end of fiscal year - important)
    # df.loc[pd.Timestamp('2007-12-29')] = None
    df = df.sort_index()  # sorting by index

    # Fill & interpolate missing dates (resampling):
    df = df.resample('D')
    # df = df.interpolate(method='polynomial', order=2)
    df = df.ffill()
    df = df.bfill()

    # Slice to columns of interest
    df = df[cols_of_interest.values()]

    # Scale to reasonable size
    df *= 1E3

    return df
