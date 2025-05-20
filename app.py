import streamlit as st
import pandas as pd
import os
import time
from airtable_service import AirtableService
from data_processor import DataProcessor
from technical_analysis import TechnicalAnalysis
from backtest import Backtester
from visualizations import Visualizer

# Page configuration
st.set_page_config(
    page_title="Bitcoin Price Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Bitcoin Price Analysis")
st.markdown("""
This application analyzes Bitcoin price data, identifies negative price streaks, 
and discovers correlated indicators for backtesting.
""")

# Initialize services
@st.cache_resource
def initialize_services():
    airtable_api_key = os.getenv("AIRTABLE_API_KEY", "")
    airtable_base_id = os.getenv("AIRTABLE_BASE_ID", "")
    
    if not airtable_api_key or not airtable_base_id:
        st.error("Airtable API key or Base ID not found in environment variables.")
        return None, None, None, None
    
    airtable_service = AirtableService(airtable_api_key, airtable_base_id)
    data_processor = DataProcessor()
    technical_analysis = TechnicalAnalysis()
    visualizer = Visualizer()
    
    return airtable_service, data_processor, technical_analysis, visualizer

airtable_service, data_processor, technical_analysis, visualizer = initialize_services()

# Check if services are initialized properly
if not all([airtable_service, data_processor, technical_analysis, visualizer]):
    st.stop()

# Sidebar for operations
st.sidebar.title("Operations")
operation = st.sidebar.radio(
    "Select Operation",
    ["Load Data", "Find Negative Streaks", "Analyze Indicators", "Backtest Indicators"]
)

# Main content based on operation
if operation == "Load Data":
    st.header("Load Data from Airtable")
    
    with st.spinner("Fetching data from Airtable..."):
        try:
            records = airtable_service.get_records_from_table("daily")
            if not records:
                st.warning("No records found in the 'daily' table.")
                st.stop()
            
            st.success(f"Successfully fetched {len(records)} records from Airtable!")
            
            # Process and display sample data
            with st.spinner("Processing data..."):
                df = data_processor.process_airtable_records(records, airtable_service)
                
                if df is not None and not df.empty:
                    st.session_state['df'] = df
                    st.subheader("Sample Data")
                    st.dataframe(df.head())
                    
                    # Calculate BTC_Daily_Change
                    if 'BTC_Close' in df.columns:
                        df['BTC_Daily_Change'] = df['BTC_Close'].diff()
                        st.session_state['df'] = df
                        
                        st.subheader("Bitcoin Daily Change")
                        st.line_chart(df[['BTC_Daily_Change']].dropna())
                    else:
                        st.error("BTC_Close column not found in the data.")
                else:
                    st.error("Failed to process data from Airtable records.")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

elif operation == "Find Negative Streaks":
    st.header("Negative Price Streaks Analysis")
    
    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.warning("Please load data first.")
        st.stop()
    
    df = st.session_state['df']
    
    # Find 5-day negative streaks
    with st.spinner("Finding 5-day negative price streaks..."):
        negative_streaks = technical_analysis.find_negative_streaks(df, streak_length=5)
        st.session_state['negative_streaks'] = negative_streaks
        
        if negative_streaks:
            st.success(f"Found {len(negative_streaks)} instances of 5-day negative streaks.")
            
            # Display the negative streaks
            for i, streak in enumerate(negative_streaks):
                st.subheader(f"Negative Streak #{i+1}")
                start_date = streak[0]
                end_date = streak[-1]
                
                # Get the streak data
                streak_df = df.loc[start_date:end_date]
                
                # Display streak information
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Start Date: {start_date}")
                    st.write(f"End Date: {end_date}")
                    st.write(f"Duration: {len(streak)} days")
                    
                    # Calculate total price change during streak
                    start_price = df.loc[start_date, 'BTC_Close']
                    end_price = df.loc[end_date, 'BTC_Close']
                    total_change = end_price - start_price
                    percent_change = (total_change / start_price) * 100
                    
                    st.write(f"Total Price Change: ${total_change:.2f} ({percent_change:.2f}%)")
                
                with col2:
                    # Visualize the streak
                    visualizer.plot_price_streak(streak_df, 'BTC_Close')
                    
                # Show the detailed data for the streak
                with st.expander("View Detailed Data"):
                    st.dataframe(streak_df[['BTC_Close', 'BTC_Daily_Change']])
        else:
            st.info("No 5-day negative streaks found in the data.")

elif operation == "Analyze Indicators":
    st.header("Indicator Correlation Analysis")
    
    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.warning("Please load data first.")
        st.stop()
        
    if 'negative_streaks' not in st.session_state or not st.session_state['negative_streaks']:
        st.warning("Please find negative streaks first.")
        st.stop()
    
    df = st.session_state['df']
    negative_streaks = st.session_state['negative_streaks']
    
    with st.spinner("Calculating technical indicators..."):
        # Calculate technical indicators
        df_with_indicators = technical_analysis.calculate_indicators(df)
        st.session_state['df_with_indicators'] = df_with_indicators
        
        # Analyze which indicators correlate with negative streaks
        indicator_correlations = technical_analysis.analyze_indicator_correlations(
            df_with_indicators, negative_streaks
        )
        st.session_state['indicator_correlations'] = indicator_correlations
        
        if indicator_correlations:
            st.success("Analysis completed!")
            
            # Display correlation results
            st.subheader("Indicator Correlations with Negative Streaks")
            
            # Convert to DataFrame for better display
            correlation_df = pd.DataFrame({
                'Indicator': list(indicator_correlations.keys()),
                'Correlation Score': [score for score, _ in indicator_correlations.values()],
                'Appearance Rate': [rate for _, rate in indicator_correlations.values()]
            }).sort_values('Correlation Score', ascending=False)
            
            st.dataframe(correlation_df)
            
            # Visualize top indicators
            st.subheader("Top Indicators Visualization")
            top_indicators = correlation_df.head(5)['Indicator'].tolist()
            
            for indicator in top_indicators:
                st.write(f"### {indicator}")
                visualizer.plot_indicator_vs_price(df_with_indicators, 'BTC_Close', indicator)
        else:
            st.info("No significant correlations found.")

elif operation == "Backtest Indicators":
    st.header("Backtest Promising Indicators")
    
    if 'df_with_indicators' not in st.session_state or st.session_state['df_with_indicators'] is None:
        st.warning("Please analyze indicators first.")
        st.stop()
        
    if 'indicator_correlations' not in st.session_state or not st.session_state['indicator_correlations']:
        st.warning("No indicator correlations available for backtesting.")
        st.stop()
    
    df_with_indicators = st.session_state['df_with_indicators']
    indicator_correlations = st.session_state['indicator_correlations']
    
    # Allow user to select indicators for backtesting
    available_indicators = list(indicator_correlations.keys())
    selected_indicators = st.multiselect(
        "Select indicators to backtest",
        available_indicators,
        default=available_indicators[:3] if len(available_indicators) >= 3 else available_indicators
    )
    
    if not selected_indicators:
        st.warning("Please select at least one indicator for backtesting.")
        st.stop()
    
    # Set backtest parameters
    st.subheader("Backtest Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        lookback_period = st.slider("Lookback Period (days)", 1, 30, 5)
    with col2:
        threshold = st.slider("Signal Threshold", 0.0, 1.0, 0.7)
    
    # Run backtest
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            backtester = Backtester(df_with_indicators)
            backtest_results = backtester.backtest_indicators(
                selected_indicators, 
                lookback_period=lookback_period,
                threshold=threshold
            )
            
            if backtest_results:
                st.success("Backtest completed!")
                
                # Display backtest results
                st.subheader("Backtest Results")
                
                for indicator, result in backtest_results.items():
                    st.write(f"### {indicator}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Hit Rate", f"{result['hit_rate']:.2f}%")
                        st.metric("Avg. Return", f"{result['avg_return']:.2f}%")
                    with col2:
                        st.metric("False Positives", f"{result['false_positives']}")
                        st.metric("Max Drawdown", f"{result['max_drawdown']:.2f}%")
                    
                    # Visualize backtest results
                    visualizer.plot_backtest_results(
                        df_with_indicators, 
                        indicator, 
                        result['trade_entries'],
                        result['trade_exits']
                    )
                    
                    # Show detailed trades
                    with st.expander(f"View {indicator} Detailed Trades"):
                        if result['trades_df'] is not None and not result['trades_df'].empty:
                            st.dataframe(result['trades_df'])
                        else:
                            st.info("No trades were executed during the backtest period.")
            else:
                st.warning("Backtest could not be completed. Please try different parameters.")
