from sklearn.linear_model import Ridge

from rdsb_forecaster.model.rdsb_model import RDSBModel

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
