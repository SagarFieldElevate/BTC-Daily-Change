import pandas as pd
import numpy as np
import streamlit as st
import ta
from datetime import datetime, timedelta

class TechnicalAnalysis:
    def __init__(self):
        """Initialize the TechnicalAnalysis class."""
        pass
    
    def find_negative_streaks(self, df, streak_length=5):
        """Find instances of consecutive days with negative price changes.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            streak_length (int): Length of the negative streak to find
            
        Returns:
            list: A list of lists, where each inner list contains the dates of a negative streak
        """
        if 'BTC_Daily_Change' not in df.columns:
            st.error("BTC_Daily_Change column not found in the data.")
            return []
        
        # Create a mask for negative price changes
        negative_mask = df['BTC_Daily_Change'] < 0
        
        # Find streaks
        streaks = []
        current_streak = []
        
        for date, is_negative in zip(df.index, negative_mask):
            if is_negative:
                current_streak.append(date)
            else:
                # Check if we found a streak of the required length
                if len(current_streak) >= streak_length:
                    streaks.append(current_streak)
                # Reset the streak
                current_streak = []
        
        # Check the last streak if it's ongoing
        if len(current_streak) >= streak_length:
            streaks.append(current_streak)
        
        return streaks
    
    def calculate_indicators(self, df):
        """Calculate various technical indicators on the price data.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        if 'BTC_Close' not in df.columns:
            st.error("BTC_Close column not found in the data.")
            return df
        
        # Make a copy to avoid modifying the original
        df_indicators = df.copy()
        
        # Check for OHLC data
        has_ohlc = all(col in df_indicators.columns for col in ['BTC_Open', 'BTC_High', 'BTC_Low'])
        
        try:
            # Add trend indicators
            df_indicators['SMA_20'] = ta.trend.sma_indicator(df_indicators['BTC_Close'], window=20)
            df_indicators['SMA_50'] = ta.trend.sma_indicator(df_indicators['BTC_Close'], window=50)
            df_indicators['SMA_200'] = ta.trend.sma_indicator(df_indicators['BTC_Close'], window=200)
            
            df_indicators['EMA_12'] = ta.trend.ema_indicator(df_indicators['BTC_Close'], window=12)
            df_indicators['EMA_26'] = ta.trend.ema_indicator(df_indicators['BTC_Close'], window=26)
            
            # MACD
            macd = ta.trend.MACD(df_indicators['BTC_Close'])
            df_indicators['MACD'] = macd.macd()
            df_indicators['MACD_Signal'] = macd.macd_signal()
            df_indicators['MACD_Diff'] = macd.macd_diff()
            
            # Add momentum indicators
            df_indicators['RSI'] = ta.momentum.RSIIndicator(df_indicators['BTC_Close']).rsi()
            
            if has_ohlc:
                # Bollinger Bands
                bollinger = ta.volatility.BollingerBands(df_indicators['BTC_Close'])
                df_indicators['BB_High'] = bollinger.bollinger_hband()
                df_indicators['BB_Low'] = bollinger.bollinger_lband()
                df_indicators['BB_Mid'] = bollinger.bollinger_mavg()
                df_indicators['BB_Width'] = (df_indicators['BB_High'] - df_indicators['BB_Low']) / df_indicators['BB_Mid']
                
                # Add volume-based indicators if volume data is available
                if 'Volume' in df_indicators.columns:
                    df_indicators['OBV'] = ta.volume.OnBalanceVolumeIndicator(
                        df_indicators['BTC_Close'], 
                        df_indicators['Volume']
                    ).on_balance_volume()
            
            # Add custom indicators
            df_indicators['Price_SMA20_Ratio'] = df_indicators['BTC_Close'] / df_indicators['SMA_20']
            df_indicators['SMA_20_50_Ratio'] = df_indicators['SMA_20'] / df_indicators['SMA_50']
            
            # Identify crossovers
            df_indicators['Golden_Cross'] = (
                (df_indicators['SMA_50'] > df_indicators['SMA_200']) & 
                (df_indicators['SMA_50'].shift(1) <= df_indicators['SMA_200'].shift(1))
            ).astype(int)
            
            df_indicators['Death_Cross'] = (
                (df_indicators['SMA_50'] < df_indicators['SMA_200']) & 
                (df_indicators['SMA_50'].shift(1) >= df_indicators['SMA_200'].shift(1))
            ).astype(int)
            
            # Price change rate
            df_indicators['Price_Change_Rate'] = df_indicators['BTC_Daily_Change'] / df_indicators['BTC_Close'].shift(1)
            
            # Volatility indicator
            df_indicators['5d_Volatility'] = df_indicators['BTC_Close'].rolling(5).std() / df_indicators['BTC_Close'].rolling(5).mean()
            
            return df_indicators
            
        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def analyze_indicator_correlations(self, df_indicators, negative_streaks):
        """Analyze which indicators correlate with negative price streaks.
        
        Args:
            df_indicators (pd.DataFrame): DataFrame with price data and indicators
            negative_streaks (list): List of negative price streak date ranges
            
        Returns:
            dict: Dictionary mapping indicator names to correlation scores
        """
        if not negative_streaks:
            return {}
        
        # Get all indicator columns
        indicator_columns = [
            col for col in df_indicators.columns 
            if col not in ['BTC_Open', 'BTC_High', 'BTC_Low', 'BTC_Close', 'BTC_Volume', 'BTC_Daily_Change']
        ]
        
        if not indicator_columns:
            st.warning("No indicator columns found for correlation analysis.")
            return {}
        
        # Create a mask for dates before negative streaks
        pre_streak_mask = pd.Series(False, index=df_indicators.index)
        
        # Define lookback period (days before streak to check for signals)
        lookback_period = 5
        
        for streak in negative_streaks:
            streak_start = streak[0]
            lookback_start = df_indicators.index[
                max(0, df_indicators.index.get_loc(streak_start) - lookback_period)
            ]
            
            # Mark the period before the streak
            pre_streak_mask.loc[lookback_start:streak_start] = True
        
        # Create a mask for the negative streak periods
        streak_mask = pd.Series(False, index=df_indicators.index)
        
        for streak in negative_streaks:
            streak_start = streak[0]
            streak_end = streak[-1]
            streak_mask.loc[streak_start:streak_end] = True
        
        # Calculate correlations and indicator appearance rates
        correlations = {}
        
        for indicator in indicator_columns:
            # Skip columns with too many NaN values
            if df_indicators[indicator].isna().sum() > len(df_indicators) * 0.3:
                continue
            
            # Fill NaN values with the median for correlation calculation
            filled_indicator = df_indicators[indicator].fillna(df_indicators[indicator].median())
            
            # Binary values like crossovers need special handling
            if df_indicators[indicator].nunique() <= 2:
                # For binary indicators, check how often they appear before streaks
                pre_streak_appearance = df_indicators.loc[pre_streak_mask, indicator].mean()
                overall_appearance = df_indicators[indicator].mean()
                
                correlation_score = pre_streak_appearance / overall_appearance if overall_appearance > 0 else 0
                appearance_rate = pre_streak_appearance
            else:
                # Calculate correlation between indicator and future negative streaks
                # For continuous indicators, use rolling windows to detect patterns
                
                # Create shifted streak mask for leading correlation
                future_streak = streak_mask.shift(-lookback_period).fillna(False)
                
                # Calculate point-biserial correlation
                correlation_score = np.corrcoef(filled_indicator, future_streak)[0, 1]
                
                # Calculate how often significant indicator values precede streaks
                # Define significant as values in the top/bottom quartile
                significant_threshold = df_indicators[indicator].quantile(0.25)
                significant_mask = filled_indicator <= significant_threshold
                
                # Rate of significant indicator values before streaks
                significant_before_streaks = (significant_mask & pre_streak_mask).sum() / pre_streak_mask.sum()
                appearance_rate = significant_before_streaks
            
            # Store the correlation and appearance rate
            correlations[indicator] = (abs(correlation_score), appearance_rate)
        
        # Sort by correlation strength
        sorted_correlations = {
            k: v for k, v in sorted(correlations.items(), key=lambda item: item[1][0], reverse=True)
        }
        
        return sorted_correlations
