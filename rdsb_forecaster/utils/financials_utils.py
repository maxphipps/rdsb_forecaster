import datetime
import pathlib

import pandas as pd

'''
Utility functions to convert xls financials to pandas dataframes
'''

path = pathlib.PurePath('rdsb_forecaster/data/rdsb')


def get_financials_frame():
    fin_list = (FinancialsExtractor.income_sheet(),
                FinancialsExtractor.balance_sheet(),
                FinancialsExtractor.shares(),
                FinancialsExtractor.margins(),
                FinancialsExtractor.volumes())
    df_financials = pd.concat(fin_list, join='outer', axis=1)

    # Unaudited accounts are published 30 days after financial date
    # Shift the financial data forwards to avoid look forward
    # E.g. Q3 = 1/7 to 30/9, but market cannot see until 31/10
    df_financials.index = df_financials.index + datetime.timedelta(days=30)
    # Backfill to populate all days in the quarter
    df_financials = df_financials.resample('D').bfill()
    return df_financials


class FinancialsExtractor:
    def extract_quarterly_frame(read_excel_kwargs, dedupe_cols=True):
        df_raw = pd.read_excel(path.joinpath('statement_of_income_2009_2019.xls'),
                              index_col=1,
                              **read_excel_kwargs)
        # Transpose sheet to make index = dates
        df_raw = df_raw.transpose()
        # Drop empty rows
        df_raw = df_raw.dropna(how='all')
        # Drop empty columns
        df_raw = df_raw.dropna(how='all', axis='columns')
        # Remove leading & trailing whitespace in the columns
        df_raw.columns = df_raw.columns.str.strip()
        # Convert index to string
        df_raw.index = df_raw.index.astype(str)
        # De-dupe columns
        if dedupe_cols:
            df_raw.columns = pd.io.parsers.base_parser.ParserBase(
                {'names': df_raw.columns, 'usecols': None})._maybe_dedup_names(df_raw.columns)

        # Quarterly reported values (Q1,Q2,Q3,Q4; with fiscal year ending 31st Dec):
        df_quarterly = df_raw[df_raw.index.to_series().str.startswith('Q')]
        # Convert fiscal-year datetime format to format compatible with PeriodIndex
        df_quarterly['FiscalDate'] = df_quarterly.index.to_series().apply(lambda x: x[-4:] + x[:2])
        # use this to convert index to datetime format string
        df_quarterly.index = pd.PeriodIndex(df_quarterly['FiscalDate'].to_list(),
                                                    freq='Q-DEC').strftime('%d-%m-%Y')
        df_quarterly = df_quarterly.drop(columns='FiscalDate')
        # Convert index from string to datetime
        df_quarterly.index = pd.to_datetime(df_quarterly.index)
        df_quarterly = df_quarterly.sort_index()
        return df_quarterly

    @staticmethod
    def income_sheet():
        read_excel_kwargs = {'sheet_name': 'Statement of Income',
                             'skiprows': 4}
        return FinancialsExtractor.extract_quarterly_frame(read_excel_kwargs)

    @staticmethod
    def balance_sheet():
        read_excel_kwargs = {'sheet_name': 'Balance Sheet',
                             'skiprows': 3}
        return FinancialsExtractor.extract_quarterly_frame(read_excel_kwargs)

    @staticmethod
    def shares():
        read_excel_kwargs = {'sheet_name': 'EPS and EPS ADS',
                             'skiprows': 3}
        return FinancialsExtractor.extract_quarterly_frame(read_excel_kwargs)

    @staticmethod
    def margins():
        read_excel_kwargs = {'sheet_name': 'Price & Margin Information',
                             'skiprows': 3}
        df_PriceAndMarginInfo_quar = FinancialsExtractor.extract_quarterly_frame(dedupe_cols=False,
                                                                                 read_excel_kwargs=read_excel_kwargs)

        # How many USD Shell managed to sell it's oil for
        df_PriceAndMarginInfo_quar['Realised oil price global'] = df_PriceAndMarginInfo_quar['Global'].iloc[:, 0]
        # How many USD Shell managed to sell it's gas for
        df_PriceAndMarginInfo_quar['Realised gas price global'] = df_PriceAndMarginInfo_quar['Global'].iloc[:, 1]

        return df_PriceAndMarginInfo_quar

    @staticmethod
    def volumes():
        read_excel_kwargs = {'sheet_name': 'Oil & Gas Volumes',
                             'skiprows': 3}
        return FinancialsExtractor.extract_quarterly_frame(read_excel_kwargs)
