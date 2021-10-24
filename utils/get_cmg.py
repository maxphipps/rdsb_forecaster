import pandas as pd


def brent_crude():
    df = {}
    fn = 'data/cmg/brent-crude-oil-prices-10-year-daily-chart.csv'
    df['brent'] = pd.read_csv(fn, parse_dates=['date'], skiprows=15)
    df['brent'] = df['brent'].rename(columns={'date': 'Date', ' value': 'Brent USD'})
    df['brent'] = df['brent'].set_index('Date')

    # Convert index date to datetime type
    # df['brent'].index = pd.to_datetime(df['brent'])

    # Cull to daterange of interest
    df['brent'] = df['brent'][df['brent'].index > '2008-01-01']

    #### START OF RESAMPLING
    # Force pandas to resample beginning at a specific data (note how this is end of fiscal year - important)
    df['brent'].loc[pd.Timestamp('2007-12-29')] = None
    df['brent'] = df['brent'].sort_index()  # sorting by index

    # Fill & interpolate missing dates (resampling):
    df['brent'] = df['brent'].resample('D')
    df['brent'] = df['brent'].interpolate(method='polynomial', order=2)
    # df['brent'] = df['brent'].ffill()

    # Rolling mean
    df['brent']['Brent rolling12M'] = df['brent']['Brent USD'].rolling(12 * 30).mean()
    df['brent']['Brent rolling6M'] = df['brent']['Brent USD'].rolling(6 * 30).mean()
    df['brent']['Brent rolling3M'] = df['brent']['Brent USD'].rolling(3 * 30).mean()
    df['brent'] = df['brent'].reset_index()

    # print(df['brent']); assert False

    return df


def wti_crude():
    df = {}
    fn = 'data/cmg/wti-crude-oil-prices-10-year-daily-chart.csv'
    df['wticrude'] = pd.read_csv(fn, parse_dates=['date'], skiprows=15)
    df['wticrude'] = df['wticrude'].rename(columns={'date': 'Date', ' value': 'wticrude USD'})
    df['wticrude'] = df['wticrude'].set_index('Date')

    # Convert index date to datetime type
    # df['wticrude'].index = pd.to_datetime(df['wticrude'])

    # Cull to daterange of interest
    df['wticrude'] = df['wticrude'][df['wticrude'].index > '2008-01-01']

    #### START OF RESAMPLING
    # Force pandas to resample beginning at a specific data (note how this is end of fiscal year - important)
    df['wticrude'].loc[pd.Timestamp('2007-12-29')] = None
    df['wticrude'] = df['wticrude'].sort_index()  # sorting by index

    # Fill & interpolate missing dates (resampling):
    df['wticrude'] = df['wticrude'].resample('D')
    df['wticrude'] = df['wticrude'].interpolate(method='polynomial', order=2)
    # df['wticrude'] = df['wticrude'].ffill()

    # Rolling mean
    df['wticrude']['wticrude rolling12M'] = df['wticrude']['wticrude USD'].rolling(12 * 30).mean()
    df['wticrude']['wticrude rolling6M'] = df['wticrude']['wticrude USD'].rolling(6 * 30).mean()
    df['wticrude']['wticrude rolling3M'] = df['wticrude']['wticrude USD'].rolling(3 * 30).mean()

    df['wticrude'] = df['wticrude'].reset_index()

    return df


def natgas():
    fn = 'data/cmg/natural-gas-prices-historical-chart_mjsp.csv'

    df = {}
    df['naturalgas'] = pd.read_csv(fn, parse_dates=['date'], skiprows=15)
    df['naturalgas'] = df['naturalgas'].rename(columns={'date': 'Date', ' value': 'Natural Gas USD'})
    df['naturalgas'] = df['naturalgas'].set_index('Date')

    # Convert index date to datetime type
    df['naturalgas'].index = pd.to_datetime(df['naturalgas'].index)

    # Cull to daterange of interest
    df['naturalgas'] = df['naturalgas'][df['naturalgas'].index > '2008-01-01']

    #### START OF RESAMPLING
    # Force pandas to resample beginning at a specific data (note how this is end of fiscal year - important)
    df['naturalgas'].loc[pd.Timestamp('2007-12-29')] = None
    df['naturalgas'] = df['naturalgas'].sort_index()  # sorting by index

    # Fill & interpolate missing dates (resampling):
    df['naturalgas'] = df['naturalgas'].resample('D')
    df['naturalgas'] = df['naturalgas'].interpolate(method='polynomial', order=2)
    # df['naturalgas'] = df['naturalgas'].ffill()

    # Rolling mean
    df['naturalgas']['Natural Gas rolling12M'] = df['naturalgas']['Natural Gas USD'].rolling(12 * 30).mean()
    df['naturalgas']['Natural Gas rolling6M'] = df['naturalgas']['Natural Gas USD'].rolling(6 * 30).mean()
    df['naturalgas']['Natural Gas rolling3M'] = df['naturalgas']['Natural Gas USD'].rolling(3 * 30).mean()

    df['naturalgas'] = df['naturalgas'].reset_index()

    # print(df['naturalgas']); assert False

    return df
