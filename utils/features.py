import talib as ta
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')


def generate_features(df, ticker, multiindex=True):
    _ = df.copy()
    if multiindex:
        _.columns = _.columns.droplevel()

    # Overlap Studies Functions
    _['BBANDS_U'], _['BBANDS_M'], _['BBANDS_D'] = ta.BBANDS(_['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    _['DEMA'] = ta.DEMA(_['Close'], timeperiod=30)
    _['EMA'] = ta.EMA(_['Close'], timeperiod=30)
    _['HT_TRENDLINE'] = ta.HT_TRENDLINE(_['Close'])
    _['KAMA'] = ta.KAMA(_['Close'], timeperiod=30)
    _['MA'] = ta.MA(_['Close'], timeperiod=30, matype=0)
    _['MIDPOINT'] = ta.MIDPOINT(_['Close'], timeperiod=14)
    _['MIDPRICE'] = ta.MIDPRICE(_['High'], _['Low'], timeperiod=14)
    _['SAR'] = ta.SAR(_['High'], _['Low'], acceleration=0, maximum=0)
    _['SAREXT'] = ta.SAREXT(_['High'], _['Low'], startvalue=0, offsetonreverse=0, accelerationinitlong=0,
                            accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0,
                            accelerationmaxshort=0)
    _['SMA'] = ta.SMA(_['Close'], timeperiod=30)
    _['T3'] = ta.T3(_['Close'], timeperiod=5, vfactor=0)
    _['TEMA'] = ta.TEMA(_['Close'], timeperiod=30)
    _['TRIMA'] = ta.TRIMA(_['Close'], timeperiod=30)
    _['WMA'] = ta.WMA(_['Close'], timeperiod=30)

    # Momentum Indicator Functions
    _['ADX'] = ta.ADX(_['High'], _['Low'], _['Close'], timeperiod=14)
    _['ADXR'] = ta.ADXR(_['High'], _['Low'], _['Close'], timeperiod=14)
    _['APO'] = ta.APO(_['Close'], fastperiod=12, slowperiod=26, matype=0)
    _['AROON_UP'], _['AROON_DOWN'] = ta.AROON(_['High'], _['Low'], timeperiod=14)
    _['AROONOSC'] = ta.AROONOSC(_['High'], _['Low'], timeperiod=14)
    _['BOP'] = ta.BOP(_['Open'], _['High'], _['Low'], _['Close'])
    _['CCI'] = ta.CCI(_['High'], _['Low'], _['Close'], timeperiod=14)
    _['CMO'] = ta.CMO(_['Close'], timeperiod=14)
    _['DX'] = ta.DX(_['High'], _['Low'], _['Close'], timeperiod=14)
    _['MACD'], _['MACD_SIGNAL'], _['MACD_HIST'] = ta.MACD(_['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    _['MFI'] = ta.MFI(_['High'], _['Low'], _['Close'], _['Volume'], timeperiod=14)
    _['MINUS_DI'] = ta.MINUS_DI(_['High'], _['Low'], _['Close'], timeperiod=14)
    _['MINUS_DM'] = ta.MINUS_DM(_['High'], _['Low'], timeperiod=14)
    _['MOM'] = ta.MOM(_['Close'], timeperiod=10)
    _['PLUS_DI'] = ta.PLUS_DI(_['High'], _['Low'], _['Close'], timeperiod=14)
    _['PLUS_DM'] = ta.PLUS_DM(_['High'], _['Low'], timeperiod=14)
    _['PPO'] = ta.PPO(_['Close'], fastperiod=12, slowperiod=26, matype=0)
    _['RSI'] = ta.RSI(_['Close'], timeperiod=14)
    _['STOCH_K'], _['STOCH_D'] = ta.STOCH(_['High'], _['Low'], _['Close'], fastk_period=5, slowk_period=3,
                                          slowk_matype=0, slowd_period=3, slowd_matype=0)
    _['STOCHF_K'], _['STOCHF_D'] = ta.STOCHF(_['High'], _['Low'], _['Close'], fastk_period=5, fastd_period=3,
                                             fastd_matype=0)
    _['STOCHRSI_K'], _['STOCHRSI_D'] = ta.STOCHRSI(_['Close'], timeperiod=14, fastk_period=5, fastd_period=3,
                                                   fastd_matype=0)
    _['TRIX'] = ta.TRIX(_['Close'], timeperiod=30)
    _['ULTOSC'] = ta.ULTOSC(_['High'], _['Low'], _['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    _['WILLR'] = ta.WILLR(_['High'], _['Low'], _['Close'], timeperiod=14)

    # Volume Indicators
    _['AD'] = ta.AD(_['High'], _['Low'], _['Close'], _['Volume'])
    _['ADOSC'] = ta.ADOSC(_['High'], _['Low'], _['Close'], _['Volume'], fastperiod=3, slowperiod=10)
    _['OBV'] = ta.OBV(_['Close'], _['Volume'])

    # Volatility Indicators
    _['ATR'] = ta.ATR(_['High'], _['Low'], _['Close'], timeperiod=14)
    _['NATR'] = ta.NATR(_['High'], _['Low'], _['Close'], timeperiod=14)
    _['TRANGE'] = ta.TRANGE(_['High'], _['Low'], _['Close'])

    # Cycle Indicators
    _['HT_DCPERIOD'] = ta.HT_DCPERIOD(_['Close'])
    _['HT_DCPHASE'] = ta.HT_DCPHASE(_['Close'])
    _['HT_PHASOR_INPHASE'], _['HT_PHASOR_QUADRATURE'] = ta.HT_PHASOR(_['Close'])
    _['HT_SINE'], _['HT_LEADSINE'] = ta.HT_SINE(_['Close'])
    # _['HT_TRENDMODE'] = ta.HT_TRENDMODE(_['Close'])

    # Statistic Functions
    _['BETA'] = ta.BETA(_['High'], _['Low'], timeperiod=5)
    _['CORREL'] = ta.CORREL(_['High'], _['Low'], timeperiod=30)
    _['LINEARREG'] = ta.LINEARREG(_['Close'], timeperiod=14)
    _['LINEARREG_ANGLE'] = ta.LINEARREG_ANGLE(_['Close'], timeperiod=14)
    _['LINEARREG_INTERCEPT'] = ta.LINEARREG_INTERCEPT(_['Close'], timeperiod=14)
    _['LINEARREG_SLOPE'] = ta.LINEARREG_SLOPE(_['Close'], timeperiod=14)
    _['STDDEV'] = ta.STDDEV(_['Close'], timeperiod=5, nbdev=1)
    _['TSF'] = ta.TSF(_['Close'], timeperiod=14)
    _['VAR'] = ta.VAR(_['Close'], timeperiod=5, nbdev=1)

    _.columns = pd.MultiIndex.from_product([[ticker.split('.')[0]], _.columns],
                                           names=['ticker', 'observation'])
        
    _.dropna(inplace=True)
    return _


def eval_features(df, num_feats):
    _ = df.copy()
    if 'Date' in _.columns:
        _.drop('Date', axis=1, inplace=True)

    x = _.loc[:, _.columns != 'target']
    y = _.iloc[:, -1]
    cor_list = []
    feature_name = x.columns.tolist()
    for i in x.columns.tolist():
        cor = np.corrcoef(x[i], y)[0, 1]
        cor_list.append(cor)
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = x.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]
    print(str(len(cor_feature)), 'selected features')

    x_norm = MinMaxScaler().fit_transform(x)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(x_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = x.loc[:, chi_support].columns.tolist()
    print(str(len(chi_feature)), 'selected features')

    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(x_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = x.loc[:, rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features')

    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    embeded_lr_selector.fit(x_norm, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = x.loc[:, embeded_lr_support].columns.tolist()
    print(str(len(embeded_lr_feature)), 'selected features')

    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(x, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = x.loc[:, embeded_rf_support].columns.tolist()
    print(str(len(embeded_rf_feature)), 'selected features')

    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                          reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
    embeded_lgb_selector.fit(x, y)
    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = x.loc[:, embeded_lgb_support].columns.tolist()
    print(str(len(embeded_lgb_feature)), 'selected features')

    feature_selection_df = pd.DataFrame(
        {'Feature': feature_name, 'Pearson': cor_support, 'Chi-2': chi_support, 'RFE': rfe_support,
         'Logistics': embeded_lr_support, 'Random Forest': embeded_rf_support, 'LightGBM': embeded_lgb_support})
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df) + 1)

    feature_selection_df = feature_selection_df.sort_values(by='Total', ascending=False)
    
    columns = feature_selection_df['Feature'][0:20].tolist()
    columns.append('target')
    _ = _[columns]
    return _, feature_selection_df
