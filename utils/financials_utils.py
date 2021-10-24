import pandas as pd
import pathlib

'''
Script to convert xls financials to standard format
'''

path = pathlib.PurePath('data/rdsb')


def income_sheet():
    dftmp = pd.read_excel(path.joinpath('statement_of_income_2009_2019.xls'),
                          sheet_name='Statement of Income',
                          index_col=1,
                          skiprows=4)  # Transpose sheet to make index = dates
    dftmp = dftmp.transpose()
    # Drop empty rows
    dftmp = dftmp.dropna(how='all')
    # Drop empty columns
    dftmp = dftmp.dropna(how='all', axis='columns')
    # Remove leading & trailing whitespace in the columns
    dftmp.columns = dftmp.columns.str.strip()
    # Convert index to string
    dftmp.index = dftmp.index.astype(str)

    df = {}
    # Annual reported values:
    # Mask to drop where reported by quarter, leaving behind just years, e.g. 2019
    df_income_sheet_ann = dftmp[~dftmp.index.to_series().str.startswith('Q')]
    # Convert index to datetime
    df_income_sheet_ann.index = pd.to_datetime(df_income_sheet_ann.index + '1231', format='%Y%m%d')

    # Quarterly reported values (Q1,Q2,Q3,Q4; with fiscal year ending 31st Dec):
    df_income_sheet_quar = dftmp[dftmp.index.to_series().str.startswith('Q')]
    # Convert fiscal-year datetime format to format compatible with PeriodIndex
    df_income_sheet_quar['FiscalDate'] = df_income_sheet_quar.index.to_series().apply(lambda x: x[-4:] + x[:2])
    # use this to convert index to datetime format string
    df_income_sheet_quar.index = pd.PeriodIndex(df_income_sheet_quar['FiscalDate'].to_list(),
                                                   freq='Q-DEC').strftime('%d-%m-%Y')
    df_income_sheet_quar = df_income_sheet_quar.drop(columns='FiscalDate')
    # Convert index from string to datetime
    df_income_sheet_quar.index = pd.to_datetime(df_income_sheet_quar.index)

    df_income_sheet_quar = df_income_sheet_quar.sort_index()
    df_income_sheet_ann = df_income_sheet_ann.sort_index()

    # df_income_sheet_ann.head()
    # df_income_sheet_quar.head()

    # return df_income_sheet_ann
    return df_income_sheet_quar


def balance_sheet():
    dftmp = pd.read_excel(path.joinpath('statement_of_income_2009_2019.xls'),
                          sheet_name='Balance Sheet',
                          index_col=1,
                          skiprows=3)  # Transpose sheet to make index = dates
    dftmp = dftmp.transpose()
    # Drop empty rows
    dftmp = dftmp.dropna(how='all')
    # Drop empty columns
    dftmp = dftmp.dropna(how='all', axis='columns')
    # Remove leading & trailing whitespace in the columns
    dftmp.columns = dftmp.columns.str.strip()
    # Convert index to string
    dftmp.index = dftmp.index.astype(str)
    # De-dupe columns
    dftmp.columns = pd.io.parsers.base_parser.ParserBase({'names': dftmp.columns, 'usecols': None})._maybe_dedup_names(
        dftmp.columns)

    df = {}
    # # Annual reported values:
    # # Mask to drop where reported by quarter, leaving behind just years, e.g. 2019
    # # df_balance_sheet_ann = dftmp[~dftmp.index.to_series().str.startswith('Q')]
    # df_balance_sheet_ann = dftmp[dftmp.index.to_series().str.startswith('Q4')]
    # # Convert index to datetime
    # # df_balance_sheet_ann.index = pd.to_datetime(df_balance_sheet_ann.index+'1231', format='%Y%m%d')
    # df_balance_sheet_ann.index = pd.to_datetime(df_balance_sheet_ann.index+'1231', format='Q4 %Y%m%d')

    # Quarterly reported values (Q1,Q2,Q3,Q4; with fiscal year ending 31st Dec):
    df_balance_sheet_quar = dftmp[dftmp.index.to_series().str.startswith('Q')]
    # Convert fiscal-year datetime format to format compatible with PeriodIndex
    df_balance_sheet_quar['FiscalDate'] = df_balance_sheet_quar.index.to_series().apply(lambda x: x[-4:] + x[:2])
    # use this to convert index to datetime format string
    df_balance_sheet_quar.index = pd.PeriodIndex(df_balance_sheet_quar['FiscalDate'].to_list(),
                                                    freq='Q-DEC').strftime('%d-%m-%Y')
    df_balance_sheet_quar = df_balance_sheet_quar.drop(columns='FiscalDate')
    # Convert index from string to datetime
    df_balance_sheet_quar.index = pd.to_datetime(df_balance_sheet_quar.index)

    df_balance_sheet_quar = df_balance_sheet_quar.sort_index()
    # df_balance_sheet_ann = df_balance_sheet_ann.sort_index()

    # df_balance_sheet_ann.head()
    # df_balance_sheet_quar.head()

    return df_balance_sheet_quar


def shares():
    dftmp = pd.read_excel(path.joinpath('statement_of_income_2009_2019.xls'),
                          sheet_name='EPS and EPS ADS',
                          index_col=1,
                          skiprows=3)  # Transpose sheet to make index = dates
    dftmp = dftmp.transpose()
    # Drop empty rows
    dftmp = dftmp.dropna(how='all')
    # Drop empty columns
    dftmp = dftmp.dropna(how='all', axis='columns')
    # Remove leading & trailing whitespace in the columns
    dftmp.columns = dftmp.columns.str.strip()
    # Convert index to string
    dftmp.index = dftmp.index.astype(str)

    df = {}
    # Annual reported values:
    # Mask to drop where reported by quarter, leaving behind just years, e.g. 2019
    df_shares_ann = dftmp[~dftmp.index.to_series().str.startswith('Q')]
    # Convert index to datetime
    df_shares_ann.index = pd.to_datetime(df_shares_ann.index + '1231', format='%Y%m%d')

    # Quarterly reported values (Q1,Q2,Q3,Q4; with fiscal year ending 31st Dec):
    df_shares_quar = dftmp[dftmp.index.to_series().str.startswith('Q')]
    # Convert fiscal-year datetime format to format compatible with PeriodIndex
    df_shares_quar['FiscalDate'] = df_shares_quar.index.to_series().apply(lambda x: x[-4:] + x[:2])
    # use this to convert index to datetime format string
    df_shares_quar.index = pd.PeriodIndex(df_shares_quar['FiscalDate'].to_list(), freq='Q-DEC').strftime(
        '%d-%m-%Y')
    df_shares_quar = df_shares_quar.drop(columns='FiscalDate')
    # Convert index from string to datetime
    df_shares_quar.index = pd.to_datetime(df_shares_quar.index)

    df_shares_quar = df_shares_quar.sort_index()
    df_shares_ann = df_shares_ann.sort_index()

    # df_shares_ann.head()
    # df_shares_quar.head()

    return df_shares_quar


def margins():
    fn = 'data/rdsb/statement_of_income_2009_2019.xls'

    dftmp = pd.read_excel(path.joinpath('statement_of_income_2009_2019.xls'),
                          sheet_name='Price & Margin Information',
                          index_col=1,
                          skiprows=3)  # Transpose sheet to make index = dates
    dftmp = dftmp.transpose()
    # Drop empty rows
    dftmp = dftmp.dropna(how='all')
    # Drop empty columns
    dftmp = dftmp.dropna(how='all', axis='columns')
    # Remove leading & trailing whitespace in the columns
    dftmp.columns = dftmp.columns.str.strip()
    # Convert index to string
    dftmp.index = dftmp.index.astype(str)

    df = {}
    # Annual reported values:
    # Mask to drop where reported by quarter, leaving behind just years, e.g. 2019
    df_PriceAndMarginInfo_ann = dftmp[~dftmp.index.to_series().str.startswith('Q')]
    # Convert index to datetime
    df_PriceAndMarginInfo_ann.index = pd.to_datetime(df_PriceAndMarginInfo_ann.index + '1231', format='%Y%m%d')

    # Quarterly reported values (Q1,Q2,Q3,Q4; with fiscal year ending 31st Dec):
    df_PriceAndMarginInfo_quar = dftmp[dftmp.index.to_series().str.startswith('Q')]
    # Convert fiscal-year datetime format to format compatible with PeriodIndex
    df_PriceAndMarginInfo_quar['FiscalDate'] = df_PriceAndMarginInfo_quar.index.to_series().apply(
        lambda x: x[-4:] + x[:2])
    # use this to convert index to datetime format string
    df_PriceAndMarginInfo_quar.index = pd.PeriodIndex(df_PriceAndMarginInfo_quar['FiscalDate'].to_list(),
                                                         freq='Q-DEC').strftime('%d-%m-%Y')
    df_PriceAndMarginInfo_quar = df_PriceAndMarginInfo_quar.drop(columns='FiscalDate')
    # Convert index from string to datetime
    df_PriceAndMarginInfo_quar.index = pd.to_datetime(df_PriceAndMarginInfo_quar.index)

    df_PriceAndMarginInfo_quar = df_PriceAndMarginInfo_quar.sort_index()
    df_PriceAndMarginInfo_ann = df_PriceAndMarginInfo_ann.sort_index()

    # How many USD Shell managed to sell it's oil for
    df_PriceAndMarginInfo_quar['Realised oil price global'] = df_PriceAndMarginInfo_quar['Global'].iloc[:, 0]
    # How many USD Shell managed to sell it's gas for
    df_PriceAndMarginInfo_quar['Realised gas price global'] = df_PriceAndMarginInfo_quar['Global'].iloc[:, 1]

    # df_PriceAndMarginInfo_ann.head()
    # df_PriceAndMarginInfo_quar.head()
    return df_PriceAndMarginInfo_quar


def volumes():
    dftmp = pd.read_excel(path.joinpath('statement_of_income_2009_2019.xls'),
                          sheet_name='Oil & Gas Volumes',
                          index_col=1,
                          skiprows=3)  # Transpose sheet to make index = dates
    dftmp = dftmp.transpose()
    # Drop empty rows
    dftmp = dftmp.dropna(how='all')
    # Drop empty columns
    dftmp = dftmp.dropna(how='all', axis='columns')
    # Remove leading & trailing whitespace in the columns
    dftmp.columns = dftmp.columns.str.strip()
    # Convert index to string
    dftmp.index = dftmp.index.astype(str)

    df = {}
    # Annual reported values:
    # Mask to drop where reported by quarter, leaving behind just years, e.g. 2019
    df_volumes_ann = dftmp[~dftmp.index.to_series().str.startswith('Q')]
    # Convert index to datetime
    df_volumes_ann.index = pd.to_datetime(df_volumes_ann.index + '1231', format='%Y%m%d')

    # Quarterly reported values (Q1,Q2,Q3,Q4; with fiscal year ending 31st Dec):
    df_volumes_quar = dftmp[dftmp.index.to_series().str.startswith('Q')]
    # Convert fiscal-year datetime format to format compatible with PeriodIndex
    df_volumes_quar['FiscalDate'] = df_volumes_quar.index.to_series().apply(
        lambda x: x[-4:] + x[:2])
    # use this to convert index to datetime format string
    df_volumes_quar.index = pd.PeriodIndex(df_volumes_quar['FiscalDate'].to_list(),
                                              freq='Q-DEC').strftime('%d-%m-%Y')
    df_volumes_quar = df_volumes_quar.drop(columns='FiscalDate')
    # Convert index from string to datetime
    df_volumes_quar.index = pd.to_datetime(df_volumes_quar.index)

    df_volumes_quar = df_volumes_quar.sort_index()
    df_volumes_ann = df_volumes_ann.sort_index()

    # df_volumes_ann.head()
    # print(df_volumes_quar.head())
    # print(df_volumes_quar.columns)

    # Rename a few columns
    # Volumes columns:
    # 'Europe', 'Asia', 'Oceania', 'SPDC1 - Nigeria', 'Other Africa',
    #        'North America', 'South America', 'Total liquids production',
    #        'Integrated Gas', 'Upstream', 'Europe', 'Asia', 'Oceania',
    #        'SPDC1 - Nigeria', 'Other Africa', 'North America', 'South America',
    #        'Total natural gas production', 'Integrated Gas', 'Upstream', 'Europe',
    #        'Asia', 'Oceania', 'SPDC1 - Nigeria', 'Other Africa', 'North America',
    #        'South America', 'Total production', 'Integrated Gas', 'Upstream',
    #        'LNG liquefaction volumes (million tonnes)',
    #        'LNG sales volumes (million tonnes)']

    # df_volumes_quar = df_volumes_quar.rename(
    #     columns={'Total production': 'Total barrels of oil equivalent production'})

    # # How much volume Shell managed to sell
    # df_volumes_quar['Brent sale volume'] = df_volumes_quar['Europe'].iloc[:, 0] + \
    #                                        df_volumes_quar['Asia'].iloc[:, 0] + \
    #                                        df_volumes_quar['Oceana'].iloc[:, 0] + \
    #                                        df_volumes_quar['SPDC1 - Nigeria'].iloc[:, 0] + \
    #                                        df_volumes_quar['Other Africa'].iloc[:, 0]
    # df_volumes_quar['WTI sale volume']


    # Fill & interpolate missing dates (resampling):
    # df_volumes_quar = df_volumes_quar.resample('D')
    # df_volumes_quar = df_volumes_quar.interpolate(method='polynomial', order=2)
    # df_volumes_quar = df_volumes_quar.ffill()

    # print(df_volumes_quar); assert False

    return df_volumes_quar
