import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns
sns.set()

import pandas as pd
# pd.options.display.max_rows = 10
# pd.options.display.max_columns = 6

import xlrd
# import natsort

from utils import utilities, get_financials, get_cmg

''''''

# Read data
df_usdgbp = utilities.read_usdgbp()
df_sp = utilities.read_rdsb()

# Join with usdgbp rate
df_sp = df_sp.join(df_usdgbp,how='left',on='Date')
df_sp['Close US cents'] = df_sp['Close'].multiply(df_sp['GBPUSD'], axis=0)

# # Plot
# plt.figure(figsize=(15,4))
# sns.lineplot(data=df_sp[['Close','Close US cents']])
# plt.show()

df = {}
df['income_sheet_quar'] = get_financials.income_sheet()
df['balance_sheet_quar'] = get_financials.balance_sheet()
df['shares_quar'] = get_financials.shares()
df['PriceAndMarginInfo_quar'] = get_financials.margins()
df['volumes_quar'] = get_financials.volumes()

# Get cmg prices
df.update(get_cmg.brent_crude())
df.update(get_cmg.wti_crude())
df.update(get_cmg.natgas())

# Commodities dataframe is join of all individual commodities frames
df['commodities'] = df['brent'].set_index('Date').join(df['naturalgas'].set_index('Date'),how='outer')
df['commodities'] = df['commodities'].join(df['wticrude'].set_index('Date'),how='outer')
# df['commodities'] = df['commodities'].ffill()

df_main = df['income_sheet_quar'].join(df['shares_quar'], rsuffix='_shares')
df_main = df_main.join(df['PriceAndMarginInfo_quar'], rsuffix='_margin', how='outer')
df_main = df_main.join(df['balance_sheet_quar'], rsuffix='_bs', how='outer')
df_main = df_main.join(df['volumes_quar'], rsuffix='_vol', how='outer')
df_main = df_main.join(df['commodities'], rsuffix='_cmg', how='outer')
df_main = df_main.join(df_sp['2008-01-01':][['Close', 'Close US cents']], how='outer')
# df_main = df_main.interpolate(method='polynomial', order=2)  # TODO CHEAT
df_main = df_main.ffill()

# Approximation: Backfill production volumes data from 2014
df_main['Total natural gas production'] = df_main['Total natural gas production'].bfill()
df_main['Total barrels of oil equivalent production'] = df_main['Total barrels of oil equivalent production'].bfill()

# Calculate pricing measures (i.e. possible target variables to choose from)
df_main['Revenue per share $'] = df_main['Revenue'] / df_main['Shares outstanding at the end of the period']
df_main['Shares outstanding at the end of the period'] = \
    df_main['Shares outstanding at the end of the period'].ffill()
df_main['Market cap'] = df_main['Close US cents'] * df_main['Shares outstanding at the end of the period']

# # Display comparison of pricing measures
# plt.figure(figsize=(20, 6))
# plt.subplot(121)
# ax = plt.gca()
# df_main['Revenue per share $'].plot(ax=ax)
# plt.axis('tight')
# plt.subplot(122)
# ax = plt.gca()
# df_main['Market cap'].plot(ax=ax)
# plt.axis('tight')
# plt.show()

# for col in ['Close US cents', 'Revenue per share $']:
#     (df_main[col]/df_main[col].dropna().iloc[-1]).plot()
# plt.show()

# print(df_main[['Close US cents','Market cap']])
# print(df_main['Shares outstanding at the end of the period'])
# # .drop_duplicates())
# # df_main['Close US cents'].plot()
# # plt.show()
# assert False

'''Features'''
# Calculate -ve of purchase expenditure, since model forces +ve coefficients
df_main['Purchases_neg'] = -df_main['Purchases']
df_main['Depreciation, depletion and amortisation_neg'] = - df_main['Depreciation, depletion and amortisation']
df_main['Total liabilities_neg'] = -df_main['Total liabilities']
df_main['NAV'] = df_main['Total assets'] - df_main['Total liabilities']

df_main['Revenue-Depreciation'] = df_main['Revenue'] - df_main['Depreciation, depletion and amortisation']

# Calculate cmg price x volume
df_main['natural gas price x vol'] = df_main['Natural Gas USD'] * df_main['Total natural gas production']
# TODO: Brent vs wti split is approximate:
#  based on 2014 production values and assumed constant
brent_wti_prod_ratio = 819 / 3245
df_main['brent price x vol'] = df_main['Brent USD'] * df_main['Total barrels of oil equivalent production'] * brent_wti_prod_ratio
df_main['wti price x vol'] = df_main['wticrude USD'] * df_main['Total barrels of oil equivalent production'] * (1-brent_wti_prod_ratio)

# Calculate rolling means
for col in ['brent price x vol', 'wti price x vol', 'natural gas price x vol',
            'Depreciation, depletion and amortisation_neg', 'Purchases_neg', 'Revenue',
            'Revenue-Depreciation',
            'Total assets', 'Total liabilities_neg', 'NAV']:
    # df_main[col+'36M'] = df_main[col].rolling(36 * 30).mean()
    df_main[col+'24M'] = df_main[col].rolling(24 * 30).mean()
    df_main[col+'12M'] = df_main[col].rolling(12 * 30).mean()
    df_main[col+'6M'] = df_main[col].rolling(6 * 30).mean()
    df_main[col+'3M'] = df_main[col].rolling(3 * 30).mean()

# print(df_main['Natural Gas USD'].drop_duplicates())
# assert False

# plt.figure(figsize=(15,4))
# sns.lineplot(data=df['wticrude'].set_index('Date')['wticrude USD'], color='k');
# sns.lineplot(data=df['wticrude'].set_index('Date')['wticrude rolling12M'], color='k');
# plt.show()
# plt.figure(figsize=(15,4))
# sns.lineplot(data=df['naturalgas'].set_index('Date')['Natural Gas USD'], color='darkblue');
# sns.lineplot(data=df['naturalgas'].set_index('Date')['Natural Gas rolling12M'], color='darkblue');
# plt.show()


# # Plot income
# cols = ['Revenue',\
#         'Share of profit of joint ventures and associates',\
#         'Interest and other income1']
# plt.figure(figsize=(12,3*3))
#
# ax = [None for _ in range(3)]
# for ii,col in enumerate(cols):
#     ax[ii]=plt.subplot(3, 1, ii+1)
#     sns.lineplot(data=df['income_sheet_quar'][col], label=col, ax=ax[ii])
#     # Set y limit
#     ax[ii].set_ylim(min(0.,ax[ii].get_ylim()[0]),ax[ii].get_ylim()[1]*1.1);
#     ax[ii].set_title(col)
#
# plt.tight_layout()
# plt.show()
#
# cols = ['Revenue',\
#         'Share of profit of joint ventures and associates']
# plt.figure(figsize=(15,4))
# ax = sns.lineplot(data=df['income_sheet_quar'][cols[0]], label=cols[0], color='y')
# ax2 = plt.twinx()
# sns.lineplot(data=df['income_sheet_quar'][cols[1]], label=cols[1], ax=ax2)
# # remove right hand grid lines
# ax2.grid(None)
# # High correlation between "revenue" and "profit of joint ventures and associates"
# print(df['income_sheet_quar'][cols].corr(method='pearson'))
# plt.show()


''' Model '''
# capital = 'Property, plant and equipment'

# # Join prices of commodities on open market (frame) with
# # prices RDSB has reported it has sold its commodities for (frame)
# df['regdata'] = df['commodities'].join(dfOilUSDperBbl.rename(columns={'Global':'Realised oil price global'})['Realised oil price global'])
# df['regdata'] = df['regdata'].join(dfGasUSDperThousandSCF.rename(columns={'Global':'Realised gas price global'})['Realised gas price global'])

regress_col = 'Close US cents'
# regress_col = 'Market cap'
# regress_col = 'Revenue per share $'
# regress_col = 'Revenue'
# regress_col = 'Total revenue and other income'

feature_cols = [
                    'Property, plant and equipment',
                  'Brent USD',
#                   'Brent rolling3M',
#                   'Brent rolling6M',
#                   'Brent rolling12M',
                  'wticrude USD',
#                   'wticrude rolling3M',
#                   'wticrude rolling6M',
#                   'wticrude rolling12M',
                  'Natural Gas USD',
#                   'Natural Gas rolling3M',
#                   'Natural Gas rolling6M',
#                   'Natural Gas rolling12M',
#                   'Realised oil price global', 'Realised gas price global',
#                     'natural gas price x vol',
#                     'brent price x vol',
                    'wti price x vol',
                    # 'natural gas price x vol3M',
                    # 'brent price x vol3M',
                    # 'wti price x vol3M',
                    # 'natural gas price x vol6M',
                    # 'brent price x vol6M',
                    # 'wti price x vol6M',
                    # 'natural gas price x vol12M',
                    # 'brent price x vol12M',
                    # 'wti price x vol12M',
                    'natural gas price x vol24M',
                    # 'brent price x vol24M',
                    # 'wti price x vol24M',
    ### Revenue
    # 'Revenue',
    # 'Revenue3M',
    # 'Revenue6M',
    # 'Revenue12M',
    # 'Revenue24M',
    ### Significant investments
    # 'Purchases_neg',
    # 'Purchases_neg3M',
    # 'Purchases_neg6M',
    # 'Purchases_neg12M',
    'Purchases_neg24M',
    # 'Depreciation, depletion and amortisation_neg',
    # 'Depreciation, depletion and amortisation_neg3M',
    # 'Depreciation, depletion and amortisation_neg6M',
    # 'Depreciation, depletion and amortisation_neg12M',
    # 'Depreciation, depletion and amortisation_neg24M',
    ### Critical expenditure that is sunk cost i.e. not an investment
    # 'Revenue-Depreciation',
    'Revenue-Depreciation3M',
    # 'Revenue-Depreciation6M',
    # 'Revenue-Depreciation12M',
    # 'Revenue-Depreciation24M',
# 'Total assets', 'Total liabilities_neg',
# 'Total assets6M', 'Total liabilities_neg6M',
# 'Total assets12M', 'Total liabilities_neg12M',
# 'Total assets24M', 'Total liabilities_neg24M',
    ### Assets-liabilities
    'NAV',
    # 'NAV3M',
    # 'NAV6M',
    'NAV12M',
    # 'NAV24M',
]


# Commodities frame joined with share price frame, and Revenue
# df['regdata'] = df['regdata'].join(
#     df_main[feature_cols+[regress_col]],
#     how='inner')
df['regdata'] = df_main[feature_cols+[regress_col]]

# TODO CHEAT: temporarily commented
# Avoid look-forward: shift the below columns forward
#    'Property, plant and equipment', 'Realised oil price global', 'Realised gas price global'
# since these value won't be available until the latest report is released
# for col in ['Property, plant and equipment', 'Realised oil price global', 'Realised gas price global']:
# for col in feature_cols:
#     if col in df['regdata'].columns:
#         df['regdata'][col] = df['regdata'][col].shift(1)

# temporary. remove:
# df['regdata']['Brent USD'] = df['regdata']['Brent USD'].shift(1)
# df['regdata']['Natural Gas USD'] = df['regdata']['Natural Gas USD'].shift(1)
# df['regdata']['Brent rolling3M'] = df['regdata']['Brent rolling3M'].shift(1)
# df['regdata']['Natural Gas rolling3M'] = df['regdata']['Natural Gas rolling3M'].shift(1)

# Drop rows with NaNs
df['regdata'] = df['regdata'].dropna()

# Calculate the regressors:
# multiply commodity prices by capital employed
commodity_cols = list(df['commodities'].columns)
# Do not include, since capital employed is the determinant of volume produced
# # include prices multiplied by volume produced
# for col in['natural gas price x vol',
#                     'brent price x vol',
#                     'wti price x vol']:
#     commodity_cols += [col]
#     commodity_cols += [col+'3M']
#     commodity_cols += [col+'6M']
#     commodity_cols += [col+'12M']
#     commodity_cols += [col+'24M']

SCALE_CAPITAL_EMPLOYED = True
if SCALE_CAPITAL_EMPLOYED:
    for comcol in commodity_cols:
        if comcol in df['regdata'].columns:
            df['regdata'][comcol] = \
                    df['regdata'][comcol].multiply(df['regdata']['Property, plant and equipment'],axis=0)

# Drop 'Property, plant and equipment' from features
df['regdata'] = df['regdata'].drop(columns=['Property, plant and equipment'])
feature_cols = feature_cols[1:]

# # initial trial values for regression
# days_in_a_quarter_year = (365/4.)
# liquids_thousand_barrels_per_day_2019 = 1876 # 6561 # 156
# liquids_bbl_per_quarter_millions = (liquids_thousand_barrels_per_day_2019*1000.*days_in_a_quarter_year)/1000000
#
# nat_gas_million_scf_per_day_2019 = 10377
# nat_gas_thousand_scf_per_quarter_millions = (nat_gas_million_scf_per_day_2019*1000000.*days_in_a_quarter_year/1000.)/1000000
# # print((liquids_bbl_per_quarter_millions, nat_gas_thousand_scf_per_quarter_millions))
#
#
# # back of the envelope revenue calculation
# liq_million_usd = liquids_bbl_per_quarter_millions * df_main['Realised oil price global'].dropna().iloc[-1]
# gas_million_usd = nat_gas_thousand_scf_per_quarter_millions * df_main['Realised gas price global'].dropna().iloc[-1]
# _ = gas_million_usd + liq_million_usd
# print(f'Back of the envelope revenue Q4 2019 = USD {_:.0f}m.\n\
# Liq = USD {liq_million_usd:.0f}m\n\
# Gas = USD {gas_million_usd:.0f}m')

# Normalisation. But note collinearity issues.
# for col in df['regdata']:
#     df['regdata'][col] /= abs(df['regdata'][col].iloc[0])
# print(df['regdata'])
# assert False

X_all_dates = df['regdata'].index.values
# X_all = df['all'][['Brent rolling12M','Natural Gas rolling12M','Shares outstanding at the end of the period']].values
# X_all = df['all'][['Brent USD','Natural Gas USD','Shares outstanding at the end of the period']].values
X_all = df['regdata'][feature_cols].values

# target variable revenue
y_all = df['regdata'][regress_col].values

# n_test = 14 # 1800
# n_test = 15#10 # 12 # 1800
n_test = 5 *12*30

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

# Split the data into training/testing sets
X_train = X_all[:-n_test]
X_dates_train = X_all_dates[:-n_test]
X_test = X_all[-n_test:]
X_dates_test = X_all_dates[-n_test:]

# Split the targets into training/testing sets
y_train = y_all[:-n_test]
y_test = y_all[-n_test:]



'''
Optimising the alpha coefficient for ridge regression
'''
coefs = []
errors = []

# clf = Ridge(positive=True)
clf = ElasticNet(l1_ratio=0.05, positive=True)

alphas = np.logspace(4, 16, 200)
# alphas = np.logspace(9, 14, 200)
# alphas = np.logspace(11, 13, 200)

# Train the model with different regularisation strengths
for a in alphas:
    clf.set_params(alpha=a)
    # Train the model using the training sets
    clf.fit(X_train, y_train)
    # Make predictions using the testing set
    y_test_pred = clf.predict(X_test)
    coefs.append(clf.coef_)
    errors.append(mean_squared_error(y_test,y_test_pred))

# Display results
plt.figure(figsize=(20, 6))

plt.subplot(121)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Coefficient error as a function of the regularization')
plt.axis('tight')
plt.show()


# Create linear regression object
# regr = LinearRegression(fit_intercept=False)
# regr = LinearRegression()

# regr = Ridge(alpha=10**10, positive=True)
regr = ElasticNet(alpha=10**5, l1_ratio=0.05, positive=True)

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_test_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_test_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_test_pred))

# Print the coefficients scaled by avg value of its column
# Effectively an importance plot
unsorted_list = []
for col, coeff in zip(df['regdata'].columns, regr.coef_):
    coeff_norm = coeff/abs(df['regdata'][col].median())
    unsorted_list.append([col, coeff_norm])
unsorted_list.sort(key=lambda x: x[1])
for tup in unsorted_list:
    print(tup)

# View the entire train+test plot
# Make predictions using the training set
y_all_pred = regr.predict(X_all)
y_err = abs(y_all_pred - y_all)
plt.figure(figsize=(15, 4))
plt.plot(X_all_dates, y_all, color='green', linewidth=1, label='test');
plt.plot(X_all_dates, y_all_pred, color='blue', linewidth=1, label='pred');
plt.plot(X_all_dates, y_err, color='black', linewidth=1, label='error');
plt.axvline(X_dates_train[-1])
plt.legend();
plt.ylim([0, plt.ylim()[1]*1.1]);
plt.show()



