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
                
            # Check if the column is numeric
            if not pd.api.types.is_numeric_dtype(df_indicators[indicator]):
                st.warning(f"Skipping non-numeric column: {indicator}")
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
        
    def analyze_cross_dataset_correlations(self, primary_dataset, secondary_datasets, datasets_dict, negative_streaks, max_lag_periods=5):
        """Analyze correlations between datasets, especially related to negative BTC price streaks.
        
        Args:
            primary_dataset (str): Name of the primary dataset (usually Bitcoin)
            secondary_datasets (list): Names of secondary datasets to correlate with
            datasets_dict (dict): Dictionary of datasets by name
            negative_streaks (list): List of negative BTC price streak date ranges
            max_lag_periods (int): Maximum lag periods to consider for correlation
            
        Returns:
            dict: Dictionary of cross-dataset correlations with lag information
        """
        st.write("### Cross-Dataset Correlation Analysis Debug Information")
        
        if not negative_streaks:
            st.warning("No negative streaks provided for analysis.")
            return {}
            
        if not secondary_datasets:
            st.warning("No secondary datasets selected for analysis.")
            return {}
        
        # Get the primary dataset
        primary_df = datasets_dict.get(primary_dataset)
        if primary_df is None:
            st.error(f"Primary dataset {primary_dataset} not found.")
            return {}
            
        # Display primary dataset info for debugging
        st.write(f"Primary dataset: {primary_dataset}")
        st.write(f"Primary dataset shape: {primary_df.shape}")
        st.write(f"Primary dataset date range: {primary_df.index.min()} to {primary_df.index.max()}")
        
        # Find BTC price column in primary dataset
        btc_price_col = None
        for col in primary_df.columns:
            if any(term in col.lower() for term in ['btc', 'bitcoin']) and any(term in col.lower() for term in ['close', 'price']):
                btc_price_col = col
                break
                
        if btc_price_col is None:
            st.warning(f"Could not find Bitcoin price column in primary dataset {primary_dataset}.")
            if 'BTC_Close' in primary_df.columns:
                btc_price_col = 'BTC_Close'
            else:
                # Use the first numeric column as a fallback
                for col in primary_df.columns:
                    if pd.api.types.is_numeric_dtype(primary_df[col]):
                        btc_price_col = col
                        st.info(f"Using {col} as the primary price column.")
                        break
                        
        if btc_price_col is None:
            st.error("No suitable price column found in primary dataset.")
            return {}
            
        st.write(f"Using price column: {btc_price_col}")
        
        # Display negative streak information for debugging
        st.write("### Negative Streak Information")
        for i, streak in enumerate(negative_streaks):
            st.write(f"Streak #{i+1}: {streak[0]} to {streak[-1]} ({len(streak)} days)")
        
        # Create a simple boolean Series to track streak periods
        # Initialize with all False values and length of primary dataset
        streak_mask = pd.Series(False, index=primary_df.index)
        
        # Simpler approach: explicitly loop through each primary index date
        # and mark it as True if it falls within any streak
        for idx in primary_df.index:
            for streak in negative_streaks:
                streak_start = pd.Timestamp(streak[0]) if not isinstance(streak[0], pd.Timestamp) else streak[0]
                streak_end = pd.Timestamp(streak[-1]) if not isinstance(streak[-1], pd.Timestamp) else streak[-1]
                
                if idx >= streak_start and idx <= streak_end:
                    streak_mask.loc[idx] = True
                    break  # Break once we've found a match
            
        # Check for streak mask coverage
        if streak_mask.sum() == 0:
            st.warning("No negative streaks found in the time range of the primary dataset.")
            return {}
            
        cross_correlations = {}
        
        # Analyze each secondary dataset
        for sec_dataset_name in secondary_datasets:
            sec_df = datasets_dict.get(sec_dataset_name)
            
            if sec_df is None:
                st.warning(f"Secondary dataset {sec_dataset_name} not found.")
                continue
                
            # Find potential indicator columns in secondary dataset
            numeric_cols = [col for col in sec_df.columns 
                         if pd.api.types.is_numeric_dtype(sec_df[col]) and 
                         col not in ['record_name']]
                         
            if not numeric_cols:
                st.warning(f"No numeric columns found in secondary dataset {sec_dataset_name}.")
                continue
                
            # Test different lag periods for each column
            best_correlation = 0
            best_lag = 0
            best_column = None
            
            for col in numeric_cols:
                for lag in range(max_lag_periods + 1):
                    # Create lagged dataset
                    lagged_df = self.create_lagged_dataset(primary_df, sec_df, col, lag)
                    
                    if lagged_df is None or lagged_df.empty:
                        continue
                        
                    # Skip if not enough overlap
                    if len(lagged_df) < 10:  # Require at least 10 data points
                        continue
                        
                    # Find the lagged column name
                    lagged_col = f"{sec_dataset_name}_{col}_lag{lag}"
                    
                    if lagged_col not in lagged_df.columns or btc_price_col not in lagged_df.columns:
                        continue
                        
                    # Calculate correlation with price and streaks
                    try:
                        # Point correlation with price
                        price_corr = lagged_df[lagged_col].corr(lagged_df[btc_price_col])
                        
                        # Create streak mask for the lagged dataset
                        lagged_streak_mask = streak_mask.reindex(lagged_df.index).fillna(False)
                        
                        # Calculate mean values for streak and non-streak periods
                        streak_mean = lagged_df.loc[lagged_streak_mask, lagged_col].mean()
                        non_streak_mean = lagged_df.loc[~lagged_streak_mask, lagged_col].mean()
                        
                        # Calculate streak correlation (difference in means)
                        if not pd.isna(streak_mean) and not pd.isna(non_streak_mean):
                            streak_diff = abs(streak_mean - non_streak_mean)
                            overall_std = lagged_df[lagged_col].std()
                            
                            if overall_std > 0:
                                streak_corr = streak_diff / overall_std
                            else:
                                streak_corr = 0
                        else:
                            streak_corr = 0
                            
                        # Combine price correlation and streak correlation
                        combined_corr = (abs(price_corr) + streak_corr) / 2
                        
                        if combined_corr > best_correlation:
                            best_correlation = combined_corr
                            best_lag = lag
                            best_column = col
                    except Exception as e:
                        st.error(f"Error calculating correlation for {sec_dataset_name}, {col}, lag {lag}: {str(e)}")
                        continue
            
            if best_column is not None:
                cross_correlations[sec_dataset_name] = (best_lag, best_correlation, best_column)
                
        # Sort by correlation strength
        sorted_cross_correlations = {
            k: v for k, v in sorted(cross_correlations.items(), key=lambda item: item[1][1], reverse=True)
        }
        
        return sorted_cross_correlations
        
    def create_lagged_dataset(self, primary_df, secondary_df, column, lag_periods):
        """Create a dataset with a lagged column from the secondary dataset.
        
        Args:
            primary_df (pd.DataFrame): Primary dataset
            secondary_df (pd.DataFrame): Secondary dataset
            column (str): Column from secondary dataset to lag
            lag_periods (int): Number of periods to lag
            
        Returns:
            pd.DataFrame: Combined dataset with lagged column
        """
        try:
            # Make copies to avoid modifying originals
            primary_copy = primary_df.copy()
            secondary_copy = secondary_df.copy()
            
            # Ensure both dataframes have datetime indices
            if not isinstance(primary_copy.index, pd.DatetimeIndex):
                date_cols = [col for col in primary_copy.columns if 'date' in col.lower()]
                if date_cols:
                    primary_copy = primary_copy.set_index(date_cols[0])
                    primary_copy.index = pd.to_datetime(primary_copy.index)
                else:
                    st.warning("No date column found in primary dataset for creating lagged dataset.")
                    return None
                    
            if not isinstance(secondary_copy.index, pd.DatetimeIndex):
                date_cols = [col for col in secondary_copy.columns if 'date' in col.lower()]
                if date_cols:
                    secondary_copy = secondary_copy.set_index(date_cols[0])
                    secondary_copy.index = pd.to_datetime(secondary_copy.index)
                else:
                    st.warning("No date column found in secondary dataset for creating lagged dataset.")
                    return None
            
            # Check if the column exists
            if column not in secondary_copy.columns:
                st.warning(f"Column {column} not found in secondary dataset.")
                return None
                
            # Create a lagged version of the column
            secondary_name = secondary_copy['record_name'].iloc[0] if 'record_name' in secondary_copy.columns else "secondary"
            lagged_col_name = f"{secondary_name}_{column}_lag{lag_periods}"
            
            # Create a Series with the lagged values
            lagged_series = secondary_copy[column].shift(-lag_periods)
            
            # Create a new dataframe with just the index and lagged column
            lagged_df = pd.DataFrame(lagged_series).rename(columns={column: lagged_col_name})
            
            # Ensure both indexes are in datetime format
            lagged_df.index = pd.to_datetime(lagged_df.index)
            primary_copy.index = pd.to_datetime(primary_copy.index)
            
            # Merge with primary dataframe on index
            result_df = pd.merge(primary_copy, lagged_df, how='inner', left_index=True, right_index=True)
            
            # Check if there's enough data after merging
            if len(result_df) < 10:  # Require at least 10 data points
                return None
                
            return result_df
            
        except Exception as e:
            st.error(f"Error creating lagged dataset: {str(e)}")
            return None
