
import pandas as pd
import datetime

def read_usdgbp():
    dateparse = lambda x: datetime.datetime.strptime(x, '%b %d, %Y')
    df_usdgbp = pd.read_csv('data/fxspot/GBPUSD.csv', sep='\t',
                            parse_dates=[0], date_parser=dateparse,
                            usecols=[0, 1], index_col=[0])
    df_usdgbp.rename(columns={'Price': 'GBPUSD'}, inplace=True)

    # upsample to daily
    df_usdgbp = df_usdgbp.ffill()
    df_usdgbp = df_usdgbp.resample('D')
    df_usdgbp = df_usdgbp.ffill()
    return df_usdgbp

def read_rdsb():
    fn = 'data/rdsb/RDSB.L.csv'
    df_sp = pd.read_csv(fn, parse_dates=['Date'])
    df_sp = df_sp.set_index('Date')
    # interpolate to fill missing values
    # df['sp']['GBPUSD'] = df['sp']['GBPUSD'].interpolate(method='polynomial', order=2)
    # upsample to daily
    # this additional ffill required for some reason
    df_sp = df_sp.ffill()
    df_sp = df_sp.resample('D')
    df_sp = df_sp.ffill()
    return df_sp[['Close']]

