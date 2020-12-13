from backtesting import Backtest
from backtest.strategies import RsiStrategy
import numpy as np

np.seterr(divide='ignore')


def generate_trades(ticker, df, config):
	params = config.copy()
	if params['strategy'] == 'RsiStrategy':
		params.update({'strategy': RsiStrategy})
	else:
		params.update({'strategy': RsiStrategy})

	params.update({
		'trade_on_close': bool(params['trade_on_close']),
		'hedging': bool(params['hedging']),
		'exclusive_orders': bool(params['exclusive_orders'])
	})

	bt = Backtest(df, **params)
	stats = bt.run()
	stats._trades['Ticker'] = ticker
	return stats._trades
