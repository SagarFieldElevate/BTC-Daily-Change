import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class Visualizer:
    def __init__(self):
        """Initialize the Visualizer class."""
        pass
        
    def plot_price_streak(self, df, price_col):
        """Plot price data for a given streak.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            price_col (str): Column name for price data
        """
        if price_col not in df.columns:
            st.error(f"Column {price_col} not found in the data.")
            return
        
        fig = px.line(
            df, 
            y=price_col,
            title=f"Bitcoin Price During Streak",
            labels={price_col: "Price ($)"},
            template="plotly_white"
        )
        
        fig.update_traces(line=dict(color="#FF9500", width=2))
        
        # Add daily change as annotations
        if 'BTC_Daily_Change' in df.columns:
            for i, row in df.iterrows():
                daily_change = row.get('BTC_Daily_Change', 0)
                if not pd.isna(daily_change):
                    color = "red" if daily_change < 0 else "green"
                    fig.add_annotation(
                        x=i,
                        y=row[price_col],
                        text=f"${daily_change:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=color,
                        arrowsize=1,
                        arrowwidth=1,
                        ax=0,
                        ay=-30,
                        font=dict(color=color, size=10)
                    )
        
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def plot_indicator_vs_price(self, df, price_col, indicator_col):
        """Plot an indicator against price data.
        
        Args:
            df (pd.DataFrame): DataFrame with price and indicator data
            price_col (str): Column name for price data
            indicator_col (str): Column name for indicator data
        """
        if price_col not in df.columns or indicator_col not in df.columns:
            st.error(f"Columns {price_col} or {indicator_col} not found in the data.")
            return
        
        # Create subplot with dual y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[price_col],
                name=price_col,
                line=dict(color="#FF9500", width=2)
            ),
            secondary_y=False
        )
        
        # Handle binary indicators (0/1) differently
        if df[indicator_col].nunique() <= 2:
            # For binary indicators, use markers instead of a line
            mask = df[indicator_col] == 1
            
            if mask.sum() > 0:
                fig.add_trace(
                    go.Scatter(
                        x=df.index[mask],
                        y=df.loc[mask, price_col],
                        mode='markers',
                        name=indicator_col,
                        marker=dict(
                            symbol='circle',
                            size=10,
                            color='red',
                        )
                    ),
                    secondary_y=False
                )
        else:
            # Add indicator line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[indicator_col],
                    name=indicator_col,
                    line=dict(color="#1F77B4", width=1.5)
                ),
                secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title=f"{indicator_col} vs. {price_col}",
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text=f"{price_col} ($)", secondary_y=False)
        fig.update_yaxes(title_text=indicator_col, secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_backtest_results(self, df, indicator_name, trade_entries, trade_exits):
        """Plot backtest results with entry and exit points.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            indicator_name (str): Name of the indicator used for the backtest
            trade_entries (list): List of entry dates
            trade_exits (list): List of exit dates
        """
        if 'BTC_Close' not in df.columns:
            st.error("BTC_Close column not found in the data.")
            return
        
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BTC_Close'],
                name='BTC Price',
                line=dict(color="#FF9500", width=2)
            )
        )
        
        # Add entry points
        if trade_entries:
            entry_prices = [df.loc[date, 'BTC_Close'] for date in trade_entries if date in df.index]
            entry_dates = [date for date in trade_entries if date in df.index]
            
            fig.add_trace(
                go.Scatter(
                    x=entry_dates,
                    y=entry_prices,
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=1, color='green')
                    )
                )
            )
        
        # Add exit points
        if trade_exits:
            exit_prices = [df.loc[date, 'BTC_Close'] for date in trade_exits if date in df.index]
            exit_dates = [date for date in trade_exits if date in df.index]
            
            fig.add_trace(
                go.Scatter(
                    x=exit_dates,
                    y=exit_prices,
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=1, color='red')
                    )
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"Backtest Results: {indicator_name}",
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="BTC Price ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
