from backtesting import Strategy
from backtesting.lib import crossover
import talib as ta


class RsiStrategy(Strategy):
	def init(self):
		price = self.data.Close
		self.rsi = self.I(ta.RSI, price, 12, plot=True, name='RSI')

	def next(self):
		if not self.position:
			if crossover(self.rsi, 30) or crossover(30, self.rsi):
				self.buy()

		if self.position:
			self.position.close()
