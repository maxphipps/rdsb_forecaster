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
    from datetime import datetime
    dateparse = lambda x: datetime.strptime(x, '%d/%m/%y')
    df = pd.read_csv(path.joinpath('demand/data.csv'), parse_dates=['date'], skiprows=1)
    df = df.rename(columns={'date': 'Date'})
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
