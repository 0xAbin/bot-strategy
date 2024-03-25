from functools import reduce
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IntParameter, DecimalParameter, BooleanParameter
from datetime import datetime
import freqtrade.vendor.qtpylib.indicators as qtpylib

class CClassicnew(IStrategy):
    INTERFACE_VERSION = 3

    # Strategy parameters
    minimal_roi = {"0": 0.02}  # Adjusted for smaller but more frequent profits
    stoploss = -0.05  # Tighter stop loss for scalping
    timeframe = '1m'

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.04  
    trailing_stop_positive_offset = 0.05  
    trailing_only_offset_is_reached = False

    # Williams %R parameters
    williams_period = IntParameter(6, 20, default=14, space="buy_sell")
    williams_upper = DecimalParameter(-7, -2, default=-4, space="buy_sell")
    williams_lower = DecimalParameter(-88, -94, default=-92, space="buy_sell")

    # MACD parameters
    macd_fast = IntParameter(8, 16, default=12, space="buy_sell")
    macd_slow = IntParameter(18, 32, default=26, space="buy_sell")
    macd_signal = IntParameter(3, 15, default=9, space="buy_sell")

    # RSI parameters
    rsi_period = IntParameter(5, 25, default=14, space="buy_sell")
    rsi_overbought = IntParameter(65, 90, default=80, space="buy_sell")
    rsi_oversold = IntParameter(10, 35, default=23, space="buy_sell")

    # Whether to use the Williams %R, MACD, and RSI as buy/sell signals
    use_williams = BooleanParameter(default=True, space="buy_sell")
    use_macd = BooleanParameter(default=True, space="buy_sell")
    use_rsi = BooleanParameter(default=True, space="buy_sell")

    # Take profit parameters
    take_profit = DecimalParameter(0.01, 0.05, default=0.02, space="sell")  

    # Maximum holding time (in minutes)
    max_holding_time = IntParameter(2, 10, default=5, space="sell")  

    # Stochastic Oscillator parameters
    stoch_k_period = IntParameter(3, 15, default=14, space="buy_sell")
    stoch_d_period = IntParameter(3, 15, default=3, space="buy_sell")
    stoch_oversold = IntParameter(10, 30, default=20, space="buy_sell")
    stoch_overbought = IntParameter(70, 90, default=80, space="buy_sell")

    # Moving Average parameters
    ma_short = IntParameter(5, 20, default=5, space="buy_sell")
    ma_long = IntParameter(20, 50, default=20, space="buy_sell")
    ma_trend = IntParameter(100, 200, default=200, space="buy_sell")

    # Parabolic SAR parameters
    psar_af = DecimalParameter(0.01, 0.04, default=0.02, space="buy_sell")
    psar_max_af = DecimalParameter(0.1, 0.5, default=0.2, space="buy_sell")

    # Resistance calculation window parameter
    resistance_period = IntParameter(15, 50, default=30, space="buy_sell")

    def heikin_ashi(self, dataframe):
        heikin_ashi = dataframe.copy()
        heikin_ashi['HA_Close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        heikin_ashi['HA_Open'] = (heikin_ashi['open'].shift(1) + heikin_ashi['close'].shift(1)) / 2
        heikin_ashi.loc[0, 'HA_Open'] = (heikin_ashi['open'].iloc[0] + heikin_ashi['close'].iloc[0]) / 2
        heikin_ashi['HA_High'] = heikin_ashi[['high', 'HA_Open', 'HA_Close']].max(axis=1)
        heikin_ashi['HA_Low'] = heikin_ashi[['low', 'HA_Open', 'HA_Close']].min(axis=1)
        return heikin_ashi

    def find_resistance(self, dataframe: DataFrame) -> DataFrame:
        # Determine resistance using a simple rolling maximum
        resistance_period = self.resistance_period.value
        dataframe['resistance'] = dataframe['high'].rolling(window=resistance_period).max()
        return dataframe

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

        # Include resistance calculations
        dataframe = self.find_resistance(dataframe)

        # Williams %R indicator
        if self.use_williams.value:
            dataframe['williams_r'] = ta.WILLR(dataframe, timeperiod=self.williams_period.value)

        # MACD indicator
        if self.use_macd.value:
            macd = ta.MACD(dataframe, fastperiod=self.macd_fast.value, slowperiod=self.macd_slow.value, signalperiod=self.macd_signal.value)
            dataframe['macd'] = macd['macd']
            dataframe['macdsignal'] = macd['macdsignal']
            dataframe['macdhist'] = macd['macdhist']

        # RSI indicator
        if self.use_rsi.value:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # Stochastic Oscillator
        stoch = ta.STOCH(dataframe, fastk_period=self.stoch_k_period.value, slowk_period=3, slowd_period=self.stoch_d_period.value)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # Moving Averages
        dataframe['ma_short'] = ta.SMA(dataframe,timeperiod=self.ma_short.value)
        dataframe['ma_long'] = ta.SMA(dataframe, timeperiod=self.ma_long.value)
        dataframe['ma_trend'] = ta.SMA(dataframe, timeperiod=self.ma_trend.value)

        # Parabolic SAR
        dataframe['psar'] = ta.SAR(dataframe, acceleration=self.psar_af.value, maximum=self.psar_max_af.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Heikin Ashi Calculation
        heikin_ashi = self.heikin_ashi(dataframe)

        dataframe.loc[
            (
                # Add your entry conditions here, incorporating resistance level check
                (self.use_williams.value & (dataframe['williams_r'] <= self.williams_lower.value)) &
                (self.use_macd.value & (dataframe['macdhist'] < 0)) &
                (self.use_rsi.value & (dataframe['rsi'] <= self.rsi_oversold.value)) &
                (dataframe['HA_close'] < dataframe['HA_open']) &  
                (dataframe['HA_close'].shift(1) > dataframe['HA_open'].shift(1)) &  
                (dataframe['close'] > dataframe['resistance'].shift(1))  
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # Add your entry conditions for short positions incorporating resistance level check
                (self.use_williams.value & (dataframe['williams_r'] >= self.williams_upper.value)) &
                (self.use_macd.value & (dataframe['macdhist'] > 0)) &
                (self.use_rsi.value & (dataframe['rsi'] >= self.rsi_overbought.value)) &
                (dataframe['HA_close'] > dataframe['HA_open']) &  
                (dataframe['HA_close'].shift(1) < dataframe['HA_open'].shift(1)) &  
                (dataframe['close'] < dataframe['resistance'].shift(1))  
            ),
            'enter_short'] = 1

        # Additional entry conditions based on the provided scalping strategies
        # Stochastic Oscillator
        dataframe.loc[
            (
                # Add your entry conditions incorporating resistance level check
                (dataframe['stoch_k'] < self.stoch_oversold.value) &
                (dataframe['stoch_d'] < self.stoch_oversold.value) &
                (dataframe['stoch_k'].shift(1) <= dataframe['stoch_d'].shift(1)) &  
                (dataframe['HA_close'] < dataframe['HA_open']) &  
                (dataframe['close'] > dataframe['resistance'].shift(1))  
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # Add your entry conditions for short positions incorporating resistance level check
                (dataframe['stoch_k'] > self.stoch_overbought.value) &
                (dataframe['stoch_d'] > self.stoch_overbought.value) &
                (dataframe['stoch_k'].shift(1) >= dataframe['stoch_d'].shift(1)) &  
                (dataframe['HA_close'] > dataframe['HA_open']) &  
                (dataframe['close'] < dataframe['resistance'].shift(1))  
            ),
            'enter_short'] = 1

        # Moving Averages
        dataframe.loc[
            (
                # Add your entry conditions incorporating resistance level check
                (dataframe['ma_short'] > dataframe['ma_long']) &  
                (dataframe['close'] > dataframe['ma_trend']) &  
                (dataframe['HA_close'] < dataframe['HA_open']) &  
                (dataframe['close'] > dataframe['resistance'].shift(1))  
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # Add your entry conditions for short positions incorporating resistance level check
                (dataframe['ma_short'] < dataframe['ma_long']) &  
                (dataframe['close'] < dataframe['ma_trend']) &  
                (dataframe['HA_close'] > dataframe['HA_open']) &  
                (dataframe['close'] < dataframe['resistance'].shift(1))  
            ),
            'enter_short'] = 1

        # Parabolic SAR
        dataframe.loc[
            (
                # Add your entry conditions incorporating resistance level check
                (dataframe['close'] > dataframe['psar']) &  
                (dataframe['HA_close'] < dataframe['HA_open']) &  
                (dataframe['close'] > dataframe['resistance'].shift(1))  
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # Add your entry conditions for short positions incorporating resistance level check
                (dataframe['close'] < dataframe['psar']) &  
                (dataframe['HA_close'] > dataframe['HA_open']) &  
                (dataframe['close'] < dataframe['resistance'].shift(1))  
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Add your exit conditions incorporating resistance level check
                (dataframe['macdhist'] > 15) &  
                (dataframe['williams_r'] >= -31) &  
                (dataframe['HA_close'] > dataframe['HA_open']) &  
                (dataframe['HA_close'].shift(1) < dataframe['HA_open'].shift(1)) &  
                (dataframe['close'] > dataframe['resistance'].shift(1))  
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                # Add your exit conditions for short positions incorporating resistance level check
                (dataframe['macdhist'] < -12) &  
                (dataframe['williams_r'] <= -84) &  
                (dataframe['HA_close'] < dataframe['HA_open']) &  
                (dataframe['HA_close'].shift(1) > dataframe['HA_open'].shift(1)) &  
                (dataframe['close'] < dataframe['resistance'].shift(1))  
            ),
            'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs) -> float:
        return self.stoploss  

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        # Adjust leverage based on market conditions or volatility
        if self.use_rsi.value and self.dataframe['rsi'].iloc[-1] > 70:
            return 20.0  
        elif self.use_rsi.value and self.dataframe['rsi'].iloc[-1] < 30:
            return 25.0
        else:
            return 5.0  

    def check_exit_timeout(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs) -> bool:
        """
        Check if the maximum holding time has been reached for the current trade.
        """
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60
        if trade_duration > self.max_holding_time.value:
            return True
        return False
