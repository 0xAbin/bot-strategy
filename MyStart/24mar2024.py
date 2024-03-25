from functools import reduce
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IntParameter, DecimalParameter, BooleanParameter
from datetime import datetime
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ImageBasedStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Strategy parameters
    minimal_roi = {"0": 0.02}  # Adjusted for smaller but more frequent profits
    stoploss = -0.05  # Tighter stop loss for scalping
    timeframe = '1m'

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.04  #
    trailing_stop_positive_offset = 0.05  #
    trailing_only_offset_is_reached = False

    # Stochastic RSI parameters
    stoch_rsi_period = IntParameter(3, 15, default=14, space="buy_sell")
    stoch_rsi_overbought = IntParameter(70, 90, default=80, space="buy_sell")
    stoch_rsi_oversold = IntParameter(10, 30, default=20, space="buy_sell")

    # MACD parameters
    macd_fast = IntParameter(8, 16, default=12, space="buy_sell")
    macd_slow = IntParameter(18, 32, default=26, space="buy_sell")
    macd_signal = IntParameter(3, 15, default=9, space="buy_sell")

    # Take profit parameters
    take_profit = DecimalParameter(0.01, 0.05, default=0.02, space="sell")  # Adjusted for scalping

    # Maximum holding time (in minutes)
    max_holding_time = IntParameter(2, 10, default=5, space="sell")  # Adjusted for scalping

    def heikin_ashi(self, dataframe):
        heikin_ashi = dataframe.copy()
        heikin_ashi['HA_Close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        heikin_ashi['HA_Open'] = (heikin_ashi['open'].shift(1) + heikin_ashi['close'].shift(1)) / 2
        heikin_ashi.loc[0, 'HA_Open'] = (heikin_ashi['open'].iloc[0] + heikin_ashi['close'].iloc[0]) / 2
        heikin_ashi['HA_High'] = heikin_ashi[['high', 'HA_Open', 'HA_Close']].max(axis=1)
        heikin_ashi['HA_Low'] = heikin_ashi[['low', 'HA_Open', 'HA_Close']].min(axis=1)
        return heikin_ashi

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Heikin Ashi Calculation
        heikin_ashi = self.heikin_ashi(dataframe)
        dataframe['HA_close'] = heikin_ashi['HA_Close']
        dataframe['HA_open'] = heikin_ashi['HA_Open']
        dataframe['HA_high'] = heikin_ashi['HA_High']
        dataframe['HA_low'] = heikin_ashi['HA_Low']

        # Define bullish and bearish Heikin Ashi candles
        dataframe['HA_bullish_candle'] = heikin_ashi['HA_Close'] > heikin_ashi['HA_Open']
        dataframe['HA_bearish_candle'] = heikin_ashi['HA_Close'] < heikin_ashi['HA_Open']

        # Stochastic RSI
        stoch_rsi = ta.STOCHRSI(dataframe, fastk_period=self.stoch_rsi_period.value, fastd_period=3, fastd_mamode=0)
        dataframe['stoch_rsi'] = stoch_rsi['fastd']

        # MACD
        macd = ta.MACD(dataframe, fastperiod=self.macd_fast.value, slowperiod=self.macd_slow.value, signalperiod=self.macd_signal.value)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Entry Signals
        dataframe.loc[
            (
                (dataframe['stoch_rsi'] < self.stoch_rsi_oversold.value) &
                (dataframe['macdhist'] < 0) &
                (dataframe['HA_close'] < dataframe['HA_open']) &  # Enter long at the bottom of a bearish (red) Heikin Ashi candle
                (dataframe['HA_close'].shift(1) > dataframe['HA_open'].shift(1))  # Previous candle was bullish
            ),
            'enter_long'] = 1

        # Short Entry Signals
        dataframe.loc[
            (
                (dataframe['stoch_rsi'] > self.stoch_rsi_overbought.value) &
                (dataframe['macdhist'] > 0) &
                (dataframe['HA_close'] > dataframe['HA_open']) &  # Enter short at the top of a bullish (green) Heikin Ashi candle
                (dataframe['HA_close'].shift(1) < dataframe['HA_open'].shift(1))  # Previous candle was bearish
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit Long Signals
        dataframe.loc[
            (
                (dataframe['macdhist'] > 15) &  # Exit long when MACD histogram is above 15
                (dataframe['HA_close'] > dataframe['HA_open']) &  # Exit on a bullish (green) Heikin Ashi candle
                (dataframe['HA_close'].shift(1) < dataframe['HA_open'].shift(1))  # Previous candle was bearish
            ),
            'exit_long'] = 1

        # Exit Short Signals
        dataframe.loc[
            (
                (dataframe['macdhist'] < -12) &  # Exit short when MACD histogram is below -12
                (dataframe['HA_close'] < dataframe['HA_open']) &  # Exit on a bearish (red) Heikin Ashi candle
                (dataframe['HA_close'].shift(1) > dataframe['HA_open'].shift(1))  # Previous candle was bullish
            ),
            'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs) -> float:
        return self.stoploss  # Tighter stop loss for scalping

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        # Adjust leverage based on market conditions or volatility
        if self.dataframe['stoch_rsi'].iloc[-1] > 80:
            return 20.0  # Increase leverage when Stochastic RSI is overbought
        elif self.dataframe['stoch_rsi'].iloc[-1] < 20:
            return 25.0  # Increase leverage when Stochastic RSI is oversold
        else:
            return 15.0  # Default leverage

    def check_exit_timeout(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs) -> bool:
        """
        Check if the maximum holding time has been reached for the current trade.
        """
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60
        if trade_duration > self.max_holding_time.value:
            return True
        return False