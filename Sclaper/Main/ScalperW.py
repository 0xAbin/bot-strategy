from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Tuple
from datetime import datetime
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class AdvancedScalp(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.01}
    stoploss = -0.03  
    timeframe = '1m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['sma'] = ta.SMA(dataframe['close'], timeperiod=30)
        macd, signal, _ = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd
        dataframe['macdsignal'] = signal
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 25) & (dataframe['rsi'] < 29)
            ),
            'enter_long'] = 1
        dataframe.loc[
            (
                (dataframe['rsi'] < 80) & (dataframe['rsi'] > 75)
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) |
                (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])) |
                (dataframe['close'] > dataframe['sma'])
            ),
            'exit_long'] = 1
        dataframe.loc[
            (
                (dataframe['rsi'] < 32) |
                (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) |
                (dataframe['close'] < dataframe['sma'])
            ),
            'exit_short'] = 1
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Dynamic stoploss that adjusts to protect profit.
        """
        # If the current profit is greater than or equal to 1%, set the stop loss to protect 1% of the profit.
        if current_profit >= 0.01:
            # Adjust stoploss to protect 1% of the profits
            return 0.01
        # Otherwise, use the default stoploss value
        return self.stoploss

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 2.0