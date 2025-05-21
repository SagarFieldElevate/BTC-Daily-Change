import streamlit as st
import pandas as pd
import os
import time
from airtable_service import AirtableService
from data_processor import DataProcessor
from technical_analysis import TechnicalAnalysis
from backtest import Backtester
from visualizations import Visualizer
from registry import StreamRegistry
from connectors.airtable_connector import AirtableConnector

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
        return None, None, None, None, None

    registry = StreamRegistry("config/data_streams.yaml")
    airtable_connector = AirtableConnector(airtable_api_key, airtable_base_id)
    data_processor = DataProcessor()
    technical_analysis = TechnicalAnalysis()
    visualizer = Visualizer()

    return registry, airtable_connector, data_processor, technical_analysis, visualizer

registry, airtable_connector, data_processor, technical_analysis, visualizer = initialize_services()

# Check if services are initialized properly
if not all([registry, airtable_connector, data_processor, technical_analysis, visualizer]):
    st.stop()

# Sidebar for operations
st.sidebar.title("Operations")
operation = st.sidebar.radio(
    "Select Operation",
    ["Load Data", "Find Negative Streaks", "Analyze Indicators", "Backtest Indicators"]
)

# Select data stream from registry
approved = registry.approved_streams()
stream_names = list(approved.keys())
selected_stream = st.sidebar.selectbox("Select Data Stream", stream_names)

# Main content based on operation
if operation == "Load Data":
    st.header("Load Data from Airtable")
    
    with st.spinner("Fetching data from Airtable..."):
        try:
            stream_cfg = registry.get(selected_stream)
            df = airtable_connector.fetch(stream_cfg)
            
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
    
    # Show analysis options
    st.subheader("Analysis Options")
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Technical Indicators", "Cross-Dataset Correlations", "Combined Analysis"]
    )
    
    if analysis_type == "Technical Indicators" or analysis_type == "Combined Analysis":
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
                st.success("Technical indicator analysis completed!")
                
                # Display correlation results
                st.subheader("Technical Indicator Correlations with Negative Streaks")
                
                # Convert to DataFrame for better display
                correlation_df = pd.DataFrame({
                    'Indicator': list(indicator_correlations.keys()),
                    'Correlation Score': [score for score, _ in indicator_correlations.values()],
                    'Appearance Rate': [rate for _, rate in indicator_correlations.values()]
                }).sort_values('Correlation Score', ascending=False)
                
                st.dataframe(correlation_df)
                
                # Visualize top indicators
                st.subheader("Top Technical Indicators Visualization")
                top_indicators = correlation_df.head(5)['Indicator'].tolist()
                
                for indicator in top_indicators:
                    st.write(f"### {indicator}")
                    visualizer.plot_indicator_vs_price(df_with_indicators, 'BTC_Close', indicator)
            else:
                st.info("No significant technical indicator correlations found.")
    
    if analysis_type == "Cross-Dataset Correlations" or analysis_type == "Combined Analysis":
        st.subheader("Cross-Dataset Correlation Analysis")
        
        if 'datasets_by_name' not in st.session_state:
            st.warning("No multiple datasets available for cross-correlation analysis.")
            st.stop()
            
        datasets = st.session_state['datasets_by_name']
        
        # List available datasets for correlation
        dataset_names = list(datasets.keys())
        
        if len(dataset_names) <= 1:
            st.warning("At least two datasets are required for cross-correlation analysis.")
            st.stop()
            
        # Primary dataset (likely Bitcoin)
        primary_datasets = [name for name in dataset_names if 'btc' in name.lower() or 'bitcoin' in name.lower()]
        if primary_datasets:
            primary_dataset = st.selectbox("Select Primary Dataset (Bitcoin)", primary_datasets)
        else:
            primary_dataset = st.selectbox("Select Primary Dataset", dataset_names)
            
        # Secondary datasets to correlate with
        other_datasets = [name for name in dataset_names if name != primary_dataset]
        selected_datasets = st.multiselect(
            "Select Secondary Datasets to Correlate With", 
            other_datasets,
            default=other_datasets[:2] if len(other_datasets) >= 2 else other_datasets
        )
        
        if not selected_datasets:
            st.warning("Please select at least one secondary dataset for correlation analysis.")
            st.stop()
            
        # Configure lag periods for cross-correlation
        lag_periods = st.slider("Maximum Lag Period (Days)", 0, 30, 5)
        
        if st.button("Run Cross-Dataset Correlation Analysis"):
            with st.spinner("Analyzing cross-dataset correlations..."):
                # Perform cross-correlation analysis
                cross_correlations = technical_analysis.analyze_cross_dataset_correlations(
                    primary_dataset, selected_datasets, datasets, negative_streaks, lag_periods
                )
                
                if cross_correlations:
                    st.success("Cross-dataset correlation analysis completed!")
                    
                    # Display correlation results
                    st.subheader("Cross-Dataset Correlations with Negative Streaks")
                    
                    # Convert to DataFrame for better display
                    cross_corr_df = pd.DataFrame({
                        'Dataset': list(cross_correlations.keys()),
                        'Best Lag': [lag for lag, _, _ in cross_correlations.values()],
                        'Correlation Score': [score for _, score, _ in cross_correlations.values()],
                        'Indicator Column': [col for _, _, col in cross_correlations.values()]
                    }).sort_values('Correlation Score', ascending=False)
                    
                    st.dataframe(cross_corr_df)
                    
                    # Visualize top cross-correlations
                    if not cross_corr_df.empty:
                        st.subheader("Top Cross-Dataset Correlations Visualization")
                        for i, row in cross_corr_df.head(3).iterrows():
                            dataset = row['Dataset']
                            lag = row['Best Lag']
                            indicator_col = row['Indicator Column']
                            
                            st.write(f"### {dataset} ({indicator_col}, Lag: {lag} days)")
                            
                            # Plot the lagged correlation
                            lagged_df = technical_analysis.create_lagged_dataset(
                                datasets[primary_dataset], 
                                datasets[dataset], 
                                indicator_col, 
                                lag
                            )
                            
                            if lagged_df is not None and not lagged_df.empty:
                                # Find the BTC price column
                                btc_cols = [col for col in lagged_df.columns if 'btc_close' in col.lower()]
                                if btc_cols:
                                    visualizer.plot_indicator_vs_price(
                                        lagged_df, 
                                        btc_cols[0], 
                                        f"{dataset}_{indicator_col}_lag{lag}"
                                    )
                                else:
                                    st.warning(f"Could not find BTC price column in lagged dataset.")
                            else:
                                st.warning(f"Could not create lagged dataset for visualization.")
                else:
                    st.info("No significant cross-dataset correlations found.")

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
