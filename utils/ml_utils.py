import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def feature_importance(X_data, regr):
    """
    Simple feature importance metric.
    Prints the coefficients scaled by median value of the X_data passed in.
    :param X_data: features
    :param regr: model
    :return:
    """
    unsorted_list = []
    for col, coeff in zip(X_data.columns, regr.coef_):
        coeff_norm_abs = abs(coeff / X_data[col].median())
        unsorted_list.append([col, coeff_norm_abs, coeff])
    unsorted_list.sort(key=lambda x: x[1])
    print(f'{"Feature name":60s} Coefficient  Negative Coeff')
    for tup in unsorted_list:
        if tup[1] != 0.0:
            print(f'{tup[0]:60s} {tup[1]:.5e}  {tup[2] < 0.0}')


def feature_selection_and_fit_model(target_col, df_feat_target, do_feature_importance=True, test_train_split=0.33):
    """
    Feature selection:
    Returns the top n features most correlated with the target variable
    that minimises the test MSE.
    :param target_col: the target column to predict
    :param df_feat_target: dataframe of features and target column
    :param do_feature_importance: if True, perform feature importance
    :param test_train_split: Split ratio of test dataset to the training dataset
    :return:
    """
    # Perform test train split
    df_test_train = df_feat_target.dropna()
    n_test = int(np.ceil(len(df_test_train) * test_train_split))
    X_train_all_feat = df_test_train.drop(columns=[target_col]).iloc[:-n_test]
    Y_train = df_test_train[target_col].iloc[:-n_test]
    X_test_all_feat = df_test_train.drop(columns=[target_col]).iloc[-n_test:]
    Y_test = df_test_train[target_col][-n_test:]

    # calculate correlation matrix
    corr = abs(df_test_train.corr())
    corr = corr.sort_values(by=target_col, ascending=False)[[target_col]]

    # print(corr)
    # # plot the heatmap
    # sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.index)
    # plt.tight_layout()
    # plt.show()

    def get_trial_features(nfeat):
        """
        Top n most correlated features
        :return:
        """
        feature_cols = list(corr.index[1:nfeat + 1].values)
        assert target_col not in feature_cols
        return feature_cols

    # Feature extraction: top n correlated features
    df_results = pd.DataFrame(columns=['nfeatures', 'error', 'mdl'])
    for i, liq_feat_top_n in enumerate(range(1, 25)):
        # Isolate features
        feature_cols = get_trial_features(liq_feat_top_n)
        X_train = X_train_all_feat[feature_cols]
        X_test = X_test_all_feat[feature_cols]

        # Train the model
        regr = LinearRegression()
        regr.fit(X_train, Y_train)
        # Make predictions
        Y_test_pred = regr.predict(X_test)

        # For debugging and analytics
        if do_feature_importance:
            feature_importance(X_test, regr)

        # Calculate error
        _err = mean_squared_error(Y_test, Y_test_pred)
        df_results = df_results.append({'features': feature_cols,
                                        'nfeatures': liq_feat_top_n,
                                        'error': _err,
                                        'mdl': copy.deepcopy(regr)}, ignore_index=True)
    #     # Plot
    #     plt.plot(dates_test, Y_test_pred, label=f'pred(n={liq_feat_top_n}); MSE={_err:.1f}', alpha=0.5)
    # plt.plot(df_test_train[target_col], label='train')
    # plt.legend()
    # plt.show()

    df_results = df_results.set_index('nfeatures')
    # # Plot error curve
    # df_results['error'].plot()
    # plt.show()
    # Optimise number of features
    opt_n_features = int(df_results.index[df_results['error'].argmin()])
    # loc rather than iloc, since we want the index with value=opt_n_features
    return df_results.loc[opt_n_features]
