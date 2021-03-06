import datetime
import pathlib

import pandas as pd

path = pathlib.PurePath('rdsb_forecaster/data')


def read_usdgbp():
    dateparse = lambda x: datetime.datetime.strptime(x, '%b %d, %Y')
    df_usdgbp = pd.read_csv(path.joinpath('fxspot/GBPUSD.csv'), sep=';',
                            parse_dates=[0], date_parser=dateparse,
                            usecols=[0, 1], index_col=[0])
    df_usdgbp.rename(columns={'Price': 'GBPUSD'}, inplace=True)

    # upsample to daily
    df_usdgbp = df_usdgbp.ffill()
    df_usdgbp = df_usdgbp.resample('D')
    df_usdgbp = df_usdgbp.ffill()
    return df_usdgbp


def read_rdsb():
    df_sp = pd.read_csv(path.joinpath('rdsb/RDSB.L.csv'), parse_dates=['Date'], sep=';')
    df_sp = df_sp.set_index('Date')
    # interpolate to fill missing values
    # df['sp']['GBPUSD'] = df['sp']['GBPUSD'].interpolate(method='polynomial', order=2)
    # upsample to daily
    # this additional ffill required for some reason
    df_sp = df_sp.ffill()
    df_sp = df_sp.resample('D')
    df_sp = df_sp.ffill()
    return df_sp[['Close']]
