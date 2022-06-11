import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from utils import fxspot_utils, cmg_utils
from utils.financials_utils import get_financials_frame
from utils.ml_utils import feature_selection_and_fit_model

sns.set()


class RDSBModel:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def forecast_production_volume(df_feat_target, target_col, plot_forecast=False):
        """
        Forecasts production volumes
        :param df_feat_target: Unprocessed (raw) features and target variable
        :param target_col: target variable column
        :param plot_forecast: whether to plot the final prediction
        :return:
        """
        # Function to get current feature columns
        get_feat_cols = lambda _df: [col for col in _df if col != target_col]
        # Enrich input data with lagging features
        for col in get_feat_cols(df_feat_target):
            # looking n days behind
            for lag in range(10, 160, 10): # [30, 60, 90, 140, 150, 160]:
                df_feat_target[col + f'_lag{lag}'] = df_feat_target[col].shift(lag)
        # Normalise feature columns
        liq_idx0 = df_feat_target[target_col].dropna().iloc[0]
        for col in get_feat_cols(df_feat_target):
            df_feat_target[col] *= liq_idx0 / df_feat_target[col].dropna().iloc[0]
        df_feat_target = df_feat_target.sort_index()

        # Feature selection and fit model
        df_results = feature_selection_and_fit_model(target_col, df_feat_target)
        # unpack
        feature_cols = df_results['features']
        regr = df_results['mdl']

        # Make predictions across all dates
        X_test = df_feat_target[feature_cols].dropna()
        Y_test_pred = regr.predict(X_test)

        # Plot
        if plot_forecast:
            plt.plot(X_test.index, Y_test_pred, alpha=0.5)
            plt.plot(df_feat_target.index, df_feat_target[target_col], label='train')
            plt.legend()
            plt.show()

        # Return dataframe of predicted production volumes
        df_pred = pd.DataFrame(index=X_test.index,
                               data={f'{target_col} forecast': Y_test_pred})
        return df_pred

    def read_data(self):
        """
        Reads data
        :return:
        """
        # Read share price in gbp
        df_usdgbp = fxspot_utils.read_usdgbp()
        df_rdsb_price = fxspot_utils.read_rdsb()
        # Convert to usd
        df_rdsb_price = df_rdsb_price.join(df_usdgbp,how='left',on='Date')
        df_rdsb_price['Close US cents'] = df_rdsb_price['Close'].multiply(df_rdsb_price['GBPUSD'], axis=0)

        df_financials = get_financials_frame()

        # Construct main dataframe
        df_commodities_prices = cmg_utils.get_cmg()
        df_oil_supply_demand = cmg_utils.get_oil_supply_demand()
        self.df_main = pd.concat([df_financials,
                                  df_commodities_prices,
                                  df_oil_supply_demand,
                                  df_rdsb_price], join='outer', axis=1)

        # Forecast the production volumes
        for volume_col in ['Total natural gas production', 'Total liquids production']:
            # Construct features
            df_liq = pd.concat([self.df_main[volume_col], df_oil_supply_demand], join='outer', axis=1)
            df_pred_volume = self.forecast_production_volume(df_liq, target_col=volume_col)
            # Override exact onto forecast values
            df_pred_volume = df_pred_volume.rename(columns={f'{volume_col} forecast':
                                                            volume_col})
            self.df_main[volume_col] = self.df_main[[volume_col]].combine_first(df_pred_volume)

            # # Visual data verification: successful join
            # ax = plt.gca()
            # self.df_main[volume_col].plot(ax=ax,alpha=0.5,color='r')
            # df_pred_volume.plot(ax=ax,alpha=0.5,color='g')
            # plt.show()


        # # Approximation: Backfill production volumes data from 2014
        # self.df_main['Total natural gas production'] = self.df_main['Total natural gas production'].bfill()
        # self.df_main['Total liquids production'] = self.df_main['Total liquids production'].bfill()
        # # self.df_main['Total barrels of oil equivalent production'] = self.df_main['Total barrels of oil equivalent production'].bfill()
        # self.df_main['PAPR_WORLD'] = self.df_main['PAPR_WORLD'].bfill()
        # self.df_main['PATC_WORLD'] = self.df_main['PATC_WORLD'].bfill()

        # Fill in gaps in data
        self.df_main['Total assets'] = self.df_main['Total assets'].interpolate(method='polynomial', order=1, limit_direction='backward')
        self.df_main['Total liabilities'] = self.df_main['Total liabilities'].interpolate(method='polynomial', order=1, limit_direction='backward')
        # TODO: backcasting
        # TODO: harvest from public data
        self.df_main['Shares outstanding at the end of the period'] = self.df_main['Shares outstanding at the end of the period'].bfill()
        backcast_features = False
        if backcast_features:
            self.df_main = self.df_main.bfill()

        forecast_features = True
        if forecast_features:
            # Income from 5Y rolling mean
            for col in ['Income from continuing operations',
                        'Income attributable to Royal Dutch Shell plc shareholders']:
                self.df_main[col] = self.df_main[col].fillna(self.df_main[col].rolling(5*365, min_periods=1).mean())
                # self.df_main[col] = self.df_main[col].fillna(self.df_main[col].rolling(5*365, min_periods=1).max())
            # All others via ffill
            self.df_main = self.df_main.ffill()

    def extract_features(self):
        # Calculate possible target variables to choose from
        self.df_main['Revenue per share $'] = self.df_main['Revenue'] / self.df_main['Shares outstanding at the end of the period']
        self.df_main['Shares outstanding at the end of the period'] = \
            self.df_main['Shares outstanding at the end of the period'].ffill()
        self.df_main['Market cap'] = self.df_main['Close US cents'] * self.df_main['Shares outstanding at the end of the period']

        # Calculate -ve of purchase expenditure, since model forces +ve coefficients
        self.df_main['Purchases_neg'] = -1.0*self.df_main['Purchases']
        self.df_main['Depreciation, depletion and amortisation_neg'] = -1.0*self.df_main['Depreciation, depletion and amortisation']
        self.df_main['Total liabilities_neg'] = -1.0*self.df_main['Total liabilities']
        self.df_main['NAV'] = self.df_main['Total assets'] - self.df_main['Total liabilities']

        self.df_main['Revenue-Depreciation'] = self.df_main['Revenue'] - self.df_main['Depreciation, depletion and amortisation']

        self.df_main['world consumption-production'] = self.df_main['PATC_WORLD'] - self.df_main['PAPR_WORLD']
        self.df_main['world consumption/production'] = self.df_main['PATC_WORLD'] / self.df_main['PAPR_WORLD']

        # V1: Capital employed = volume produced
        # Calculate cmg price x volume produced
        self.df_main['naturalgas x vol'] = self.df_main['naturalgas'] * self.df_main['Total natural gas production']
        # TODO: Brent vs wti split is approximate
        brent_wti_prod_ratio = 0.5
        self.df_main['brent x vol'] = self.df_main['brent'] * self.df_main['Total liquids production'] * brent_wti_prod_ratio
        self.df_main['wticrude x vol'] = self.df_main['wticrude'] * self.df_main['Total liquids production'] * (1-brent_wti_prod_ratio)
        # self.df_main['cmg x vol'] = self.df_main[['naturalgas x vol', 'brent x vol', 'wticrude x vol']].sum()

        # Using Shell's own average sale prices
        for col in ['Realised oil price global',
                    'wticrude',
                    'brent',
                    'Gasoline',
                    'Fuel Oil',
                    'Gasoline All Grades Retail Price']:
            self.df_main[col+' x vol'] = self.df_main[col] * self.df_main['Total liquids production']
        for col in ['Realised gas price global',
                    'naturalgas',
                    'Natural Gas Price Industrial Sector',
                    'Natural Gas Price Commercial Sector',
                    # 'Natural Gas Price Residential Sector',
                    ]:
            self.df_main[col+' x vol'] = self.df_main[col] * self.df_main['Total natural gas production']

        # print(self.df_main[['naturalgas x vol', 'brent x vol', 'wticrude x vol']])
        # assert False

        # Calculate rolling means
        for col in ['Natural Gas Price Commercial Sector x vol',
                    'Gasoline All Grades Retail Price x vol',
                    'brent x vol', 'wticrude x vol', 'naturalgas x vol',
                    'Gasoline All Grades Retail Price x vol', 'Natural Gas Price Industrial Sector x vol',
                    'Depreciation, depletion and amortisation_neg', 'Purchases_neg', 'Revenue',
                    'Revenue-Depreciation',
                    'Total assets', 'Total liabilities_neg',
                    'Income from continuing operations', 'Income attributable to Royal Dutch Shell plc shareholders',
                    'NAV',
                    'Debt',
                    'PAPR_WORLD', 'PATC_WORLD', 'world consumption-production',
                    'Close US cents', 'Market cap']:
            # self.df_main[col+'36M'] = self.df_main[col].rolling(36 * 30).mean()
            self.df_main[col+'24M'] = self.df_main[col].rolling(24 * 30).mean()
            self.df_main[col+'12M'] = self.df_main[col].rolling(12 * 30).mean()
            self.df_main[col+'6M'] = self.df_main[col].rolling(6 * 30).mean()
            self.df_main[col+'3M'] = self.df_main[col].rolling(3 * 30).mean()
            self.df_main[col+'1M'] = self.df_main[col].rolling(1 * 30).mean()
            self.df_main[col+'26D'] = self.df_main[col].rolling(1 * 26).mean()
            self.df_main[col+'12D'] = self.df_main[col].rolling(1 * 12).mean()

    def train_test_split(self):

        # Visualise the features
        self.df_main[self.feature_cols].plot(style='x-', markersize=2)
        plt.show()

        # Data to regress
        self.df_regdata = self.df_main[self.feature_cols+[self.regress_col]]

        # Avoid look-forward: shift the columns forward
        # since these value won't be available until the latest report is released
        for col in feature_cols:
            if col in self.df_regdata.columns:
                self.df_regdata[col] = self.df_regdata[col].shift(1)

        # Drop rows with NaNs
        self.df_regdata = self.df_regdata.dropna()

        # Cull to daterange of interest
        # self.df_regdata = self.df_regdata[self.df_regdata.index > '2008-01-01']
        self.df_regdata = self.df_regdata[self.df_regdata.index > '2001-02-01']
        # self.df_regdata = self.df_regdata[self.df_regdata.index < '2021-10-01']

        self.X_all_dates = self.df_regdata.index.values
        self.X_all = self.df_regdata[feature_cols].values

        # target variable revenue
        self.y_all = self.df_regdata[regress_col].values

        # n_test = 14 # 1800
        # n_test = 15#10 # 12 # 1800
        n_test = int(np.ceil(self.years_test * 365))

        # Split the data into training/testing sets
        self.X_train = self.X_all[:-n_test]
        self.X_dates_train = self.X_all_dates[:-n_test]
        self.X_test = self.X_all[-n_test:]

        # Split the targets into training/testing sets
        self.Y_train = self.y_all[:-n_test]
        self.Y_test = self.y_all[-n_test:]

    def train(self):
        """
        Optimising the alpha coefficient for ridge regression
        """
        coefs = []
        errors = []

        regr = self.mdl

        alphas = np.logspace(4, 16, 200)
        # alphas = np.logspace(9, 14, 200)
        # alphas = np.logspace(11, 13, 200)

        # Train the model with different regularisation strengths
        for a in alphas:
            regr.set_params(alpha=a)
            # Train the model using the training sets
            regr.fit(self.X_train, self.Y_train)
            # Make predictions using the testing set
            self.Y_test_pred = regr.predict(self.X_test)
            coefs.append(regr.coef_)
            errors.append(mean_squared_error(self.Y_test,self.Y_test_pred))

        # Find alpha that minimises error
        alpha_err_list = list(zip(alphas, errors))
        alpha_err_list.sort(key=lambda x: x[1])
        self.alpha_minerr = alpha_err_list[0][0]
        print(f'alpha value that minimises error = {self.alpha_minerr}')

        # Display results
        plt.figure(figsize=(20, 6))

        plt.subplot(121)
        ax = plt.gca()
        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        plt.axvline(self.alpha_minerr, color='k', linestyle='--')
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Ridge coefficients as a function of the regularization')
        plt.axis('tight')

        plt.subplot(122)
        ax = plt.gca()
        ax.plot(alphas, errors)
        ax.set_xscale('log')
        plt.axvline(self.alpha_minerr, color='k', linestyle='--')
        plt.xlabel('alpha')
        plt.ylabel('error')
        plt.title('Coefficient error as a function of the regularization')
        plt.axis('tight')
        plt.show()

    def predict(self):
        regr = self.mdl

        # regr = Ridge(alpha=alpha_minerr, positive=True)
        # regr = ElasticNet(alpha=alpha_minerr, l1_ratio=0.05, positive=True)
        params = regr.get_params(deep=True)
        params.update({'alpha': self.alpha_minerr})
        regr.set_params(**params)

        # Train the model using the training sets
        regr.fit(self.X_train, self.Y_train)

        # Make predictions using the testing set
        self.Y_test_pred = regr.predict(self.X_test)

        # The coefficients
        print('Coefficients: \n', regr.coef_)
        print('Intercept: \n', regr.intercept_)
        # The mean squared error
        print('Mean squared error: %.2f'
              % mean_squared_error(self.Y_test, self.Y_test_pred))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f'
              % r2_score(self.Y_test, self.Y_test_pred))

        # Print the coefficients scaled by avg value of its column
        # Effectively an importance plot
        unsorted_list = []
        final_train_date = self.X_dates_train.max()
        for col, coeff in zip(self.df_regdata.columns, regr.coef_):
            # coeff_norm_abs = abs(coeff/self.df_regdata[col].median())
            coeff_norm_abs = abs(coeff/self.df_regdata[col][:final_train_date].median())
            unsorted_list.append([col, coeff_norm_abs, coeff])
        unsorted_list.sort(key=lambda x: x[1])
        print(f'{"Feature name":60s} Coefficient  Negative Coeff')
        for tup in unsorted_list:
            if tup[1] != 0.0:
                print(f'{tup[0]:60s} {tup[1]:.5e}  {tup[2] < 0.0}')

        # View the entire train+test plot
        # Make predictions using the training set
        y_all_pred = regr.predict(self.X_all)
        y_err = abs(y_all_pred - self.y_all)
        plt.figure(figsize=(15, 4))
        # daily resolved versions of the target variable
        for target_col in ['Revenue per share $', 'Revenue', 'Total revenue and other income',
                           'Market cap', 'Close US cents']:
            if self.regress_col.startswith(target_col):
                df_daily = self.df_main[target_col][self.X_all_dates].dropna()
        plt.plot(self.X_all_dates, df_daily.values, color='black', linewidth=1, label='test-daily');
        # model train and predict lines
        plt.plot(self.X_all_dates, self.y_all, color='green', alpha=0.5, linewidth=1, label='test');
        plt.plot(self.X_all_dates, y_all_pred, color='blue', linewidth=1.5, label='pred');
        plt.plot(self.X_all_dates, y_err, color='black', linewidth=1, label='error');
        plt.axvline(self.X_dates_train[-1])
        plt.legend();
        plt.ylim([0, plt.ylim()[1]*1.1]);
        plt.show()


if __name__ == '__main__':
    feature_cols = [
        'Gasoline All Grades Retail Price x vol',
        'Natural Gas Price Commercial Sector x vol',
        # 'naturalgas x vol',
        # 'brent x vol',
        'wticrude x vol',
        'Income attributable to Royal Dutch Shell plc shareholders',
        'Total assets', 'Total liabilities_neg',
    ]

    regress_col = 'Close US cents'
    # regress_col = 'Close US cents1M'
    # regress_col = 'Close US cents3M'
    # regress_col = 'Market cap'
    # regress_col = 'Market cap1M'
    # regress_col = 'Market cap3M'
    # regress_col = 'Revenue per share $'
    # regress_col = 'Revenue'
    # regress_col = 'Total revenue and other income'

    # Create linear regression object
    # regr_mdl = TweedieRegressor(power=0)
    regr_mdl = Ridge(positive=False)
    # regr_mdl = ElasticNet(l1_ratio=0.05, positive=True)
    # regr_mdl = BayesianRidge()

    param = {'regress_col': regress_col, 'feature_cols': feature_cols,
             'years_test': 10, 'mdl': regr_mdl}
    rdsb_mdl = RDSBModel(**param)
    rdsb_mdl.read_data()
    rdsb_mdl.extract_features()
    rdsb_mdl.train_test_split()
    rdsb_mdl.train()
    rdsb_mdl.predict()
