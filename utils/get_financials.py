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
    df['income_sheet_ann'] = dftmp[~dftmp.index.to_series().str.startswith('Q')]
    # Convert index to datetime
    df['income_sheet_ann'].index = pd.to_datetime(df['income_sheet_ann'].index + '1231', format='%Y%m%d')

    # Quarterly reported values (Q1,Q2,Q3,Q4; with fiscal year ending 31st Dec):
    df['income_sheet_quar'] = dftmp[dftmp.index.to_series().str.startswith('Q')]
    # Convert fiscal-year datetime format to format compatible with PeriodIndex
    df['income_sheet_quar']['FiscalDate'] = df['income_sheet_quar'].index.to_series().apply(lambda x: x[-4:] + x[:2])
    # use this to convert index to datetime format string
    df['income_sheet_quar'].index = pd.PeriodIndex(df['income_sheet_quar']['FiscalDate'].to_list(),
                                                   freq='Q-DEC').strftime('%d-%m-%Y')
    df['income_sheet_quar'] = df['income_sheet_quar'].drop(columns='FiscalDate')
    # Convert index from string to datetime
    df['income_sheet_quar'].index = pd.to_datetime(df['income_sheet_quar'].index)

    df['income_sheet_quar'] = df['income_sheet_quar'].sort_index()
    df['income_sheet_ann'] = df['income_sheet_ann'].sort_index()

    # df['income_sheet_ann'].head()
    # df['income_sheet_quar'].head()

    # return df['income_sheet_ann']
    return df['income_sheet_quar']


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
    # # df['balance_sheet_ann'] = dftmp[~dftmp.index.to_series().str.startswith('Q')]
    # df['balance_sheet_ann'] = dftmp[dftmp.index.to_series().str.startswith('Q4')]
    # # Convert index to datetime
    # # df['balance_sheet_ann'].index = pd.to_datetime(df['balance_sheet_ann'].index+'1231', format='%Y%m%d')
    # df['balance_sheet_ann'].index = pd.to_datetime(df['balance_sheet_ann'].index+'1231', format='Q4 %Y%m%d')

    # Quarterly reported values (Q1,Q2,Q3,Q4; with fiscal year ending 31st Dec):
    df['balance_sheet_quar'] = dftmp[dftmp.index.to_series().str.startswith('Q')]
    # Convert fiscal-year datetime format to format compatible with PeriodIndex
    df['balance_sheet_quar']['FiscalDate'] = df['balance_sheet_quar'].index.to_series().apply(lambda x: x[-4:] + x[:2])
    # use this to convert index to datetime format string
    df['balance_sheet_quar'].index = pd.PeriodIndex(df['balance_sheet_quar']['FiscalDate'].to_list(),
                                                    freq='Q-DEC').strftime('%d-%m-%Y')
    df['balance_sheet_quar'] = df['balance_sheet_quar'].drop(columns='FiscalDate')
    # Convert index from string to datetime
    df['balance_sheet_quar'].index = pd.to_datetime(df['balance_sheet_quar'].index)

    df['balance_sheet_quar'] = df['balance_sheet_quar'].sort_index()
    # df['balance_sheet_ann'] = df['balance_sheet_ann'].sort_index()

    # df['balance_sheet_ann'].head()
    # df['balance_sheet_quar'].head()

    return df['balance_sheet_quar']


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
    df['shares_ann'] = dftmp[~dftmp.index.to_series().str.startswith('Q')]
    # Convert index to datetime
    df['shares_ann'].index = pd.to_datetime(df['shares_ann'].index + '1231', format='%Y%m%d')

    # Quarterly reported values (Q1,Q2,Q3,Q4; with fiscal year ending 31st Dec):
    df['shares_quar'] = dftmp[dftmp.index.to_series().str.startswith('Q')]
    # Convert fiscal-year datetime format to format compatible with PeriodIndex
    df['shares_quar']['FiscalDate'] = df['shares_quar'].index.to_series().apply(lambda x: x[-4:] + x[:2])
    # use this to convert index to datetime format string
    df['shares_quar'].index = pd.PeriodIndex(df['shares_quar']['FiscalDate'].to_list(), freq='Q-DEC').strftime(
        '%d-%m-%Y')
    df['shares_quar'] = df['shares_quar'].drop(columns='FiscalDate')
    # Convert index from string to datetime
    df['shares_quar'].index = pd.to_datetime(df['shares_quar'].index)

    df['shares_quar'] = df['shares_quar'].sort_index()
    df['shares_ann'] = df['shares_ann'].sort_index()

    # df['shares_ann'].head()
    # df['shares_quar'].head()

    return df['shares_quar']


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
    df['PriceAndMarginInfo_ann'] = dftmp[~dftmp.index.to_series().str.startswith('Q')]
    # Convert index to datetime
    df['PriceAndMarginInfo_ann'].index = pd.to_datetime(df['PriceAndMarginInfo_ann'].index + '1231', format='%Y%m%d')

    # Quarterly reported values (Q1,Q2,Q3,Q4; with fiscal year ending 31st Dec):
    df['PriceAndMarginInfo_quar'] = dftmp[dftmp.index.to_series().str.startswith('Q')]
    # Convert fiscal-year datetime format to format compatible with PeriodIndex
    df['PriceAndMarginInfo_quar']['FiscalDate'] = df['PriceAndMarginInfo_quar'].index.to_series().apply(
        lambda x: x[-4:] + x[:2])
    # use this to convert index to datetime format string
    df['PriceAndMarginInfo_quar'].index = pd.PeriodIndex(df['PriceAndMarginInfo_quar']['FiscalDate'].to_list(),
                                                         freq='Q-DEC').strftime('%d-%m-%Y')
    df['PriceAndMarginInfo_quar'] = df['PriceAndMarginInfo_quar'].drop(columns='FiscalDate')
    # Convert index from string to datetime
    df['PriceAndMarginInfo_quar'].index = pd.to_datetime(df['PriceAndMarginInfo_quar'].index)

    df['PriceAndMarginInfo_quar'] = df['PriceAndMarginInfo_quar'].sort_index()
    df['PriceAndMarginInfo_ann'] = df['PriceAndMarginInfo_ann'].sort_index()

    # How many USD Shell managed to sell it's oil for
    df['PriceAndMarginInfo_quar']['Realised oil price global'] = df['PriceAndMarginInfo_quar']['Global'].iloc[:, 0]
    # How many USD Shell managed to sell it's gas for
    df['PriceAndMarginInfo_quar']['Realised gas price global'] = df['PriceAndMarginInfo_quar']['Global'].iloc[:, 1]

    # df['PriceAndMarginInfo_ann'].head()
    # df['PriceAndMarginInfo_quar'].head()
    return df['PriceAndMarginInfo_quar']


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
    df['volumes_ann'] = dftmp[~dftmp.index.to_series().str.startswith('Q')]
    # Convert index to datetime
    df['volumes_ann'].index = pd.to_datetime(df['volumes_ann'].index + '1231', format='%Y%m%d')

    # Quarterly reported values (Q1,Q2,Q3,Q4; with fiscal year ending 31st Dec):
    df['volumes_quar'] = dftmp[dftmp.index.to_series().str.startswith('Q')]
    # Convert fiscal-year datetime format to format compatible with PeriodIndex
    df['volumes_quar']['FiscalDate'] = df['volumes_quar'].index.to_series().apply(
        lambda x: x[-4:] + x[:2])
    # use this to convert index to datetime format string
    df['volumes_quar'].index = pd.PeriodIndex(df['volumes_quar']['FiscalDate'].to_list(),
                                              freq='Q-DEC').strftime('%d-%m-%Y')
    df['volumes_quar'] = df['volumes_quar'].drop(columns='FiscalDate')
    # Convert index from string to datetime
    df['volumes_quar'].index = pd.to_datetime(df['volumes_quar'].index)

    df['volumes_quar'] = df['volumes_quar'].sort_index()
    df['volumes_ann'] = df['volumes_ann'].sort_index()

    # df['volumes_ann'].head()
    # print(df['volumes_quar'].head())
    # print(df['volumes_quar'].columns)

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

    df['volumes_quar'] = df['volumes_quar'].rename(
        columns={'Total production': 'Total barrels of oil equivalent production'})
    df['volumes_quar'] = df['volumes_quar'][
        ['Total natural gas production', 'Total barrels of oil equivalent production']]

    # Fill & interpolate missing dates (resampling):
    # df['volumes_quar'] = df['volumes_quar'].resample('D')
    # df['volumes_quar'] = df['volumes_quar'].interpolate(method='polynomial', order=2)
    # df['volumes_quar'] = df['volumes_quar'].ffill()

    # print(df['volumes_quar']); assert False

    return df['volumes_quar']
