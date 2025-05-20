import pandas as pd
import numpy as np
import streamlit as st

class Backtester:
    def __init__(self, df):
        """Initialize the Backtester with a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with price and indicator data
        """
        self.df = df.copy()
        
    def backtest_indicators(self, indicators, lookback_period=5, threshold=0.7):
        """Backtest a list of indicators for predicting negative price streaks.
        
        Args:
            indicators (list): List of indicator column names to backtest
            lookback_period (int): Number of days to look back for signal detection
            threshold (float): Threshold for signal strength (0-1)
            
        Returns:
            dict: Dictionary of backtest results for each indicator
        """
        if 'BTC_Close' not in self.df.columns:
            st.error("BTC_Close column not found in the data.")
            return {}
        
        results = {}
        
        for indicator in indicators:
            if indicator not in self.df.columns:
                st.warning(f"Indicator {indicator} not found in the data.")
                continue
            
            # Generate trading signals based on the indicator
            signals = self._generate_signals(indicator, lookback_period, threshold)
            
            # Backtest the signals
            backtest_result = self._run_backtest(signals)
            
            # Store the results
            results[indicator] = backtest_result
        
        return results
    
    def _generate_signals(self, indicator, lookback_period, threshold):
        """Generate trading signals based on an indicator.
        
        Args:
            indicator (str): Name of the indicator column
            lookback_period (int): Number of days to look back for signal detection
            threshold (float): Threshold for signal strength (0-1)
            
        Returns:
            pd.Series: Series of trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        df = self.df.copy()
        
        # Skip if indicator has too many NaN values
        if df[indicator].isna().sum() > len(df) * 0.3:
            return pd.Series(0, index=df.index)
        
        # Fill NaN values with median
        df[indicator] = df[indicator].fillna(df[indicator].median())
        
        # Initialize signals
        signals = pd.Series(0, index=df.index)
        
        # Handle different indicator types
        
        # Binary indicators (like crossover events)
        if df[indicator].nunique() <= 2:
            # For binary indicators, a value of 1 is a signal
            signals.loc[df[indicator] == 1] = 1
        
        # RSI-like indicators (lower values indicate oversold conditions)
        elif 'RSI' in indicator:
            # For RSI, values below 30 are considered oversold (buy signal)
            signals.loc[df[indicator] < 30] = 1
            # Values above 70 are considered overbought (sell signal)
            signals.loc[df[indicator] > 70] = -1
        
        # MACD-like indicators
        elif 'MACD' in indicator:
            # MACD crossing above signal line is a buy signal
            signals.loc[(df[indicator] > 0) & (df[indicator].shift(1) <= 0)] = 1
            # MACD crossing below signal line is a sell signal
            signals.loc[(df[indicator] < 0) & (df[indicator].shift(1) >= 0)] = -1
        
        # Price to moving average ratio indicators
        elif 'Ratio' in indicator:
            # Price below moving average by threshold percentage is a buy signal
            buy_threshold = 1 - threshold
            signals.loc[df[indicator] < buy_threshold] = 1
            
            # Price above moving average by threshold percentage is a sell signal
            sell_threshold = 1 + threshold
            signals.loc[df[indicator] > sell_threshold] = -1
        
        # Volatility indicators
        elif 'Volatility' in indicator:
            # High volatility may indicate market reversals
            volatility_threshold = df[indicator].quantile(0.75)
            
            # High volatility with price below SMA can be a buy signal
            if 'SMA_20' in df.columns:
                signals.loc[(df[indicator] > volatility_threshold) & 
                           (df['BTC_Close'] < df['SMA_20'])] = 1
        
        # Default case for other indicators
        else:
            # Consider extreme values as signals
            lower_threshold = df[indicator].quantile(threshold)
            upper_threshold = df[indicator].quantile(1 - threshold)
            
            signals.loc[df[indicator] <= lower_threshold] = 1
            signals.loc[df[indicator] >= upper_threshold] = -1
        
        return signals
    
    def _run_backtest(self, signals):
        """Run a backtest with given signals.
        
        Args:
            signals (pd.Series): Series of trading signals
            
        Returns:
            dict: Dictionary with backtest results
        """
        df = self.df.copy()
        
        # Initialize trade tracking variables
        in_position = False
        entry_price = 0
        entry_date = None
        
        trade_entries = []
        trade_exits = []
        trade_results = []
        
        # Loop through the data
        for i in range(1, len(df)):
            current_date = df.index[i]
            previous_date = df.index[i-1]
            
            current_price = df.iloc[i]['BTC_Close']
            signal = signals.iloc[i]
            
            # Enter a position
            if not in_position and signal > 0:
                in_position = True
                entry_price = current_price
                entry_date = current_date
                trade_entries.append(current_date)
            
            # Exit a position
            elif in_position and (signal < 0 or i == len(df) - 1):
                in_position = False
                exit_price = current_price
                exit_date = current_date
                trade_exits.append(current_date)
                
                # Calculate trade result
                pct_return = (exit_price - entry_price) / entry_price * 100
                trade_results.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'return_pct': pct_return,
                    'hold_days': (exit_date - entry_date).days
                })
        
        # Calculate backtest metrics
        if trade_results:
            trades_df = pd.DataFrame(trade_results)
            
            # Calculate hit rate (% of profitable trades)
            profitable_trades = (trades_df['return_pct'] > 0).sum()
            hit_rate = profitable_trades / len(trades_df) * 100
            
            # Calculate average return per trade
            avg_return = trades_df['return_pct'].mean()
            
            # Calculate max drawdown
            trades_df['cumulative_return'] = (1 + trades_df['return_pct'] / 100).cumprod() * 100 - 100
            max_drawdown = trades_df['cumulative_return'].min() if trades_df['cumulative_return'].min() < 0 else 0
            
            # Calculate false positives (trades that resulted in loss)
            false_positives = (trades_df['return_pct'] <= 0).sum()
            
            # Return the results
            return {
                'hit_rate': hit_rate,
                'avg_return': avg_return,
                'max_drawdown': max_drawdown,
                'false_positives': false_positives,
                'total_trades': len(trades_df),
                'trade_entries': trade_entries,
                'trade_exits': trade_exits,
                'trades_df': trades_df
            }
        else:
            # Return empty results if no trades
            return {
                'hit_rate': 0,
                'avg_return': 0,
                'max_drawdown': 0,
                'false_positives': 0,
                'total_trades': 0,
                'trade_entries': [],
                'trade_exits': [],
                'trades_df': pd.DataFrame()
            }
