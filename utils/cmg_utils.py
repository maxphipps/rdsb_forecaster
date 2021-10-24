import pandas as pd
import pathlib

path = pathlib.PurePath('data/cmg')

def get_cmg_data(cmg_fn, tag) -> pd.DataFrame:
    """
    Loads commodities data
    :param cmg_fn: Commodities filename
    :param tag: Tag for this commodity
    :return:
    """
    df = pd.read_csv(path.joinpath(cmg_fn), parse_dates=['date'], skiprows=15)
    df = df.rename(columns={'date': 'Date', ' value': tag})
    df = df.set_index('Date')

    # Convert index date to datetime type
    # df.index = pd.to_datetime(df)

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

def get_oil_demand() -> pd.DataFrame:
    cols_of_interest = {'PAPR_WORLD': 'world production',
                        'PATC_WORLD': 'world consumption',
                        'T3_STCHANGE_WORLD': 'world inventory'}

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
    df = df.interpolate(method='polynomial', order=2)
    # df = df.ffill()

    # df = df.reset_index()

    # Slice to columns of interest
    df = df[cols_of_interest.values()]

    # Scale to reasonable size
    df *= 1E3
    print(df)

    return df
