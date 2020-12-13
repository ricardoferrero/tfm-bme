import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict
import quantstats as qs


class Portfolio(object):
    def __init__(self, config):
        self.config = config

        self.index = self.config['GENERAL SETTINGS']['index']
        self.index_ticker = self.config['GENERAL SETTINGS']['index ticker']
        self.start_date = self.config['PORTFOLIO SETTINGS']['start date']
        self.end_date = self.config['PORTFOLIO SETTINGS']['end date']
        self.commission = self.config['PORTFOLIO SETTINGS']['commission']

        self.initial_cash = self.config['PORTFOLIO SETTINGS']['cash']

        self.data = None
        self.benchmark = None

        self.scaler = None
        self.neat_model = None
        self.tf_model = None

        self.history = {
            'RSI': {
                'Buy_signals': defaultdict(list),
                'Open_positions': defaultdict(list),
                'Cash_history': defaultdict(list),
                'Equity_history': defaultdict(list),
                'Cash': self.initial_cash,
                'Trades': pd.DataFrame()
            },
            'TF': {
                'Buy_signals': defaultdict(list),
                'Open_positions': defaultdict(list),
                'Cash_history': defaultdict(list),
                'Equity_history': defaultdict(list),
                'Cash': self.initial_cash,
                'Trades': pd.DataFrame()
            },
            'NEAT': {
                'Buy_signals': defaultdict(list),
                'Open_positions': defaultdict(list),
                'Cash_history': defaultdict(list),
                'Equity_history': defaultdict(list),
                'Cash': self.initial_cash,
                'Trades': pd.DataFrame()
            }
        }

        self.init()

    def init(self):
        self.load_data()
        self.load_benchmark()
        self.load_scaler()
        self.load_tf_model()
        self.load_neat_model()
        self.generate_signal()
        self.run()
        self.create_report()

    def run(self):
        print('Running backtest')
        dates = self.get_trading_dates()

        for day in tqdm(dates):
            for k in self.history.keys():
                prev_day = self.get_previous_date(day)
                if prev_day is None:
                    continue
                if prev_day in self.history[k]['Open_positions'].keys():
                    if len(self.history[k]['Open_positions'][prev_day]) > 0:
                        sells = self.history[k]['Open_positions'][prev_day]
                        for s in sells:
                            price_out = self.get_price(s[0], day, 'Open')
                            cash_in = self.close_position(k, day, s, price_out)
                            self.history[k]['Cash'] += cash_in

            tickers = self.get_available_ticker(day)
            for ticker in tickers:
                if self.data.loc[day, (ticker, 'BuySignal')] == 1:
                    self.history['RSI']['Buy_signals'][day].append(ticker)
                    features = self.get_features(ticker, day)
                    tf_output, neat_output = self.evaluate_signal(features)
                    if tf_output == 1:
                        self.history['TF']['Buy_signals'][day].append(ticker)
                    if neat_output == 1:
                        self.history['NEAT']['Buy_signals'][day].append(ticker)

            for k in self.history.keys():
                if day in self.history[k]['Buy_signals'].keys():
                    if len(self.history[k]['Buy_signals'][day]) > 0:
                        buys = self.history[k]['Buy_signals'][day]
                        n_buys = len(buys)
                        cash = self.distribute_cash(self.history[k]['Cash'], n_buys)
                        total_cash = cash * n_buys

                        if self.history[k]['Cash'] <= total_cash:
                            continue

                        for b in buys:
                            price = self.get_price(b, day, 'Close')
                            size = self.calculate_size(cash, price)
                            if size is None:
                                continue
                            cash_out = self.open_poistion(k, day, b, price, size)
                            self.history[k]['Cash'] -= cash_out

                        self.history[k]['Cash_history'][day] = self.history[k]['Cash']

    def open_poistion(self, k, day, ticker, price, size):
        self.history[k]['Open_positions'][day].append([ticker, size, price, day])
        cash_out = price * size * (1 + self.commission)

        return cash_out

    def generate_trade(self, k, t, size, price_in, price_out, date_in, date_out):
        buy_commission = price_in * size * self.commission
        sell_commission = price_out * size * self.commission
        gross_return = price_out / price_in - 1
        net_return = (price_out * size * (1 - self.commission) / (price_in * size * (1 + self.commission))) - 1

        data = {
            'Ticker': [t],
            'Size': [size],
            'EntryPrice': [price_in],
            'ExitPrice': [price_out],
            'EntryTime': [date_in],
            'ExitTime': [date_out],
            'BuyCommission': [buy_commission],
            'SellCommission': [sell_commission],
            'GrossReturn': [gross_return],
            'NetReturn': [net_return],
        }
        self.history[k]['Trades'] = self.history[k]['Trades'].append(pd.DataFrame(data))

    def close_position(self, k, day, position, price_out):
        t = position[0]
        size = position[1]
        price_in = position[2]
        date_in = position[3]
        date_out = day
        cash_in = price_out * size * (1 - self.commission)

        self.generate_trade(k, t, size, price_in, price_out, date_in, date_out)

        return cash_in

    def calculate_size(self, cash, price):
        if price > cash:
            return None
        size = cash // price

        return size

    def get_previous_date(self, day):
        prev_day = self.data.index.get_loc(day)
        if prev_day == 0:
            return None

        return self.data.index[prev_day - 1]

    def get_price(self, ticker, date, col):
        price = self.data.loc[date, (ticker, col)]
        if np.isnan(price):
            price = self.get_last_price(ticker, date, col)

        return price

    def get_last_price(self, ticker, date, col):
        last_date = self.get_previous_date(date)
        price = self.data.loc[last_date, (ticker, col)]

        return price

    def distribute_cash(self, cash, n_tickers):
        discount = 0.5
        money = cash / n_tickers * discount

        return money

    def generate_signal(self):
        tickers = list(set(self.data.dropna().columns.droplevel(1)))
        tickers.sort()
        for ticker in tickers:
            self.data[ticker, 'BuySignal'] = self.create_signal(self.data[ticker, 'RSI'])

    def create_signal(self, data):
        signal = np.where(((data >= 30.0) & (data.shift() < 30.0)) | ((data <= 30.0) & (data.shift() > 30.0)), 1, 0)

        return signal

    def evaluate_signal(self, features):
        features = features.reshape(1, -1)
        scaled_features = self.standarize_data(features)
        scaled_features = np.array(scaled_features)
        tf_output = self.predict_tf(scaled_features)
        neat_output = self.predict_neat(scaled_features)

        return tf_output, neat_output

    def get_trading_dates(self):
        return self.data.index.tolist()

    def get_available_ticker(self, date):
        tickers = list(set(self.data.loc[str(date)].dropna(how='any').index.droplevel(1)))
        tickers.sort()

        return tickers

    def load_benchmark(self):
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        columns = pd.MultiIndex.from_product([[self.index_ticker], ['Open', 'High', 'Low', 'Close', 'Volume']],
                                             names=['ticker', 'observation'])
        self.benchmark = pd.read_csv(f'raw_data/{self.index}/{self.index_ticker}.csv', index_col='Date')
        self.benchmark.columns = columns
        self.benchmark = self.benchmark.loc[str(start_date):str(end_date)]

    def load_data(self):
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        self.data = pd.read_pickle(f'{self.index}.pkl')
        self.data = self.data.loc[str(start_date):str(end_date)]
        self.data.dropna(how='all', axis=1, inplace=True)

    def get_features(self, ticker, date):
        return self.data.loc[str(date), ticker][5:-1].values

    def create_report(self):
        for k in self.history.keys():
            self.history[k]['Trades'].to_csv(f'backtest/results/{k}_trades.csv', index=False)
        df_strategies = self.create_df_strategies()
        benchmark_df = self.prepare_benchmark_df()
        self.create_hmml_report(df_strategies, benchmark_df)
        self.generate_result()

    def create_df_strategies(self):
        df_strategies = pd.DataFrame()
        idx = pd.bdate_range(self.start_date, self.end_date)
        for k in self.history.keys():
            tmp = self.history[k]['Trades'].groupby('EntryTime')['GrossReturn'].mean()
            tmp.index = pd.to_datetime(tmp.index)
            tmp = tmp.reindex(idx, fill_value=0)
            df_strategies[k] = tmp

        df_strategies.dropna(inplace=True)
        return df_strategies

    def prepare_benchmark_df(self):
        df = pd.read_csv('raw_data/NASDAQ100/.NDX.csv', index_col='Date', parse_dates=True)
        df = df.loc[self.start_date:, 'CLOSE']
        df = df.pct_change()
        idx = pd.bdate_range(self.start_date, self.end_date)
        df = df.reindex(idx, fill_value=0)
        return df

    def create_hmml_report(self, df_strategies, benchmark):
        strategies = df_strategies.columns
        for strat in strategies:
            qs.reports.html(df_strategies[strat], benchmark, output=f'backtest/results/{strat}_report.html')

    def load_tf_model(self):
        self.tf_model = tf.keras.models.load_model('models/tf_model/checkpoints/model_1/')

    def load_neat_model(self):
        self.neat_model = pickle.load(open('models/neat_model/checkpoints/winner.pkl', 'rb'))

    def predict_neat(self, data):
        output = np.array(self.neat_model.activate(data))
        if output >= 0.5:
            output = 1.0
        else:
            output = 0.0

        return output

    def generate_result(self):
        for k in self.history.keys():
            df = self.history[k]['Trades'].copy()
            df = df.loc[:, ['Ticker', 'EntryTime', 'ExitTime']]
            df.rename(columns={'EntryTime': 'bar_in', 'ExitTime': 'bar_out'}, inplace=True)
            slot_pct = 0.05
            df['slot_pct'] = slot_pct
            df.to_csv(f'backtest/results/{k}_signaling.csv', index=False)

    def predict_tf(self, data):
        data = data.reshape((1, data.shape[0]))
        output = self.tf_model((data,), training=False)
        if output >= 0.5:
            output = 1.0
        else:
            output = 0.0

        return output

    def load_scaler(self):
        self.scaler = pickle.load(open('models/scaler_models/scaler.pkl', 'rb'))

    def standarize_data(self, data):
        std_data = self.scaler.transform(data)
        std_data_flatten = [item for elem in std_data for item in elem]

        return std_data_flatten
