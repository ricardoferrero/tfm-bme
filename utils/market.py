import os
import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from utils.features import generate_features, eval_features
from utils.utils import clean_data, get_dates_in_out, filter_df_dates, generate_labels, create_train_test_data, \
    get_features_col
from backtest.backtest import generate_trades


class Market(object):
    def __init__(self, config):

        self.config = config

        self.index = self.config['GENERAL SETTINGS']['index']
        self.start_date = self.config['GENERAL SETTINGS']['start date']
        self.end_date = self.config['GENERAL SETTINGS']['end date']
        self.data_path = f'raw_data/{self.index}/'

        self.tickers = []
        self.data = {}

        self.index_ticker = None
        self.index_dataframe = None

        self.trades = None
        self.dataset = None
        self.dataset_evaluated = None
        self.feature_importance = None

    def get_tickers_from_folder(self):
        data_path = f'raw_data/{self.index}/'
        self.tickers = [f.rstrip('.csv') for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        self.tickers.sort()
        self.index_ticker = self.tickers.pop(0)

    def run(self):
        self.get_tickers_from_folder()
        self.load_data()
        self.generate_trades()
        self.generate_dataset()
        self.create_index_dataframe()
        create_train_test_data('dataset_eval.csv')

    def load_data(self):
        print(f'Loading {self.index} data:')
        tickers_to_remove = []
        for ticker in tqdm(self.tickers):
            data = self.load_dataframe(ticker)
            if len(data) < 100:
                tickers_to_remove.append(ticker)
                data = None
            if data is None:
                continue
            ticker_name = ticker.split('.')[0]
            features = generate_features(data, ticker_name, multiindex=True)
            self.data[ticker_name] = {
                'data': data,
                'features': features
            }
        self.tickers = [x for x in self.tickers if x not in tickers_to_remove]
        self.index_dataframe = self.load_dataframe(self.index_ticker)

    def load_dataframe(self, ticker):
        data_path = f'{self.data_path}{ticker}.csv'
        df = pd.read_csv(data_path, parse_dates=True, index_col=['Date'])
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        ticker_name = ticker.split('.')[0]
        df.columns = pd.MultiIndex.from_product([[ticker_name], ['Open', 'High', 'Low', 'Close', 'Volume']],
                                                names=['ticker', 'observation'])
        df = clean_data(df)
        return df

    def index_join_leave_tickers(self):
        _ = pd.read_csv(f'raw_data/{self.index}_JL.txt', sep='\t', parse_dates=True, index_col=['Date'])
        _['RIC'] = [x.split(' ')[0] for x in _['RIC']]
        _.sort_index(inplace=True)
        index_tickers = list(set(_['RIC']))
        index_tickers.sort()
        na_tickers = self.check_if_downloaded(index_tickers)
        index_tickers = [x for x in index_tickers if x not in na_tickers and x in self.tickers]
        return _, index_tickers

    def check_if_downloaded(self, tickers):
        x = []
        for ticker in tickers:
            data_path = f'raw_data/{self.index}/{ticker}.csv'
            if not os.path.exists(data_path):
                x.append(ticker)
        return x

    def create_index_dataframe(self):
        print(f'Creating {self.index} dataframe:')
        jl_df, tickers = self.index_join_leave_tickers()
        colums = get_features_col()
        df = pd.DataFrame(columns=pd.MultiIndex.from_product([['Ticker'], colums],
                                                             names=['ticker', 'observation']))
        for ticker in tqdm(tickers):
            x = self.load_dataframe(ticker)
            if x is None:
                continue
            dates_in, dates_out = get_dates_in_out(jl_df, ticker, x.index.min(), x.index.max())
            tmp = filter_df_dates(x, dates_in, dates_out, ticker)
            if len(tmp) == 0:
                continue
            df = df.join(tmp, how='outer')
        df.dropna(axis=0, how='all', inplace=True)
        df.pop('Ticker')
        df = df.loc[(df.index >= self.start_date) & (df.index <= self.end_date)]
        df = df.reindex(columns=colums, level=1)
        df.to_pickle(f'{self.index}.pkl')
        self.index_dataframe = df

    def generate_trades(self):
        print('Generating trades:')
        result = pd.DataFrame()
        tickers = [x.split('.')[0] for x in self.tickers]
        strategy = self.config['BACKTEST SETTINGS']['strategy']
        trades_config = self.config['BACKTEST SETTINGS']
        for ticker in tqdm(tickers):
            df = self.data[ticker]['features'].copy()
            df.columns = df.columns.droplevel()

            if (df['Close'].values > 10000).any():
                continue

            bt = generate_trades(ticker, df, trades_config)
            result = result.append(bt)
        self.trades = result
        self.trades.to_csv(f'backtest/trades_{strategy}.csv', sep=',', index=False)

    def generate_dataset(self, evaluate_features=True):
        print('Generating dataset:')
        result = pd.DataFrame()
        df_trades_dates = self.trades.copy()
        df_trades_dates['Days'] = [x.days for x in df_trades_dates['Duration']]
        df_trades_dates = df_trades_dates[df_trades_dates['Days'] <= 150]
        tickers = df_trades_dates['Ticker'].unique()

        for ticker in tqdm(tickers):
            df_features = self.data[ticker]['features'].copy()
            df_features.columns = df_features.columns.droplevel()
            df_trades = df_trades_dates[df_trades_dates['Ticker'] == ticker].copy()
            df_trades.index = df_trades['EntryTime']
            df_trades.index = df_trades.index.rename('Date')
            df_trades = df_trades.loc[:, ['ReturnPct']]
            df_features = df_features[df_features.index.isin(df_trades.index)]
            df_features = df_features.iloc[:, 5:]
            df = df_trades.merge(df_features, left_index=True, right_index=True)
            result = result.append(df)

        result = generate_labels(result)
        result.pop('ReturnPct')

        # Drop last 5 years
        result.index = pd.to_datetime(result.index)
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date) - relativedelta(years=5)

        result = result.loc[str(start_date):str(end_date)]

        result.reset_index(inplace=True)
        result.sort_values(by='Date', inplace=True)
        result.to_csv('dataset.csv', index=False)
        self.dataset = result

        if evaluate_features:
            print('Evaluating features:')
            self.dataset_evaluated, self.feature_importance = eval_features(result, 30)
            self.dataset_evaluated.to_csv('dataset_eval.csv', index=False)
