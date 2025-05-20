import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

class DataProcessor:
    def __init__(self):
        """Initialize the data processor."""
        pass
        
    def process_airtable_records(self, records, airtable_service):
        """Process records from Airtable, extracting data from attachments.
        
        Args:
            records (list): List of records from Airtable
            airtable_service (AirtableService): AirtableService instance
            
        Returns:
            pd.DataFrame: DataFrame containing the processed data
        """
        all_data = []
        
        # Progress bar for processing records
        progress_bar = st.progress(0)
        
        for i, record in enumerate(records):
            try:
                # Update progress
                progress = (i + 1) / len(records)
                progress_bar.progress(progress)
                
                fields = record.get('fields', {})
                record_name = fields.get('Name', 'Unknown')
                
                # Check for attachments
                attachments = fields.get('Attachments', [])
                
                if not attachments:
                    st.warning(f"No attachments found for record: {record_name}")
                    continue
                
                # Process the first attachment (assuming it's an Excel file)
                attachment = attachments[0]
                
                # Extract data from the Excel file
                excel_data = airtable_service.get_excel_data_from_attachment(attachment)
                
                if excel_data is not None and not excel_data.empty:
                    # Add record name as a reference
                    excel_data['record_name'] = record_name
                    
                    # Add the data to our collection
                    all_data.append(excel_data)
                else:
                    st.warning(f"Could not extract data from attachment for record: {record_name}")
            
            except Exception as e:
                st.error(f"Error processing record {i}: {str(e)}")
        
        # Clear the progress bar
        progress_bar.empty()
        
        if not all_data:
            st.error("No data could be extracted from any attachments.")
            return None
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Process and clean the combined data
        processed_data = self.clean_and_process_data(combined_data)
        
        return processed_data
    
    def clean_and_process_data(self, df):
        """Clean and process the combined data.
        
        Args:
            df (pd.DataFrame): Raw DataFrame from Excel files
            
        Returns:
            pd.DataFrame: Cleaned and processed DataFrame
        """
        try:
            # Make a copy to avoid modifying the original
            processed_df = df.copy()
            
            # Check for date column - common names
            date_columns = [col for col in processed_df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_columns:
                date_col = date_columns[0]
                
                # Ensure date column is datetime
                processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')
                
                # Drop rows with invalid dates
                processed_df = processed_df.dropna(subset=[date_col])
                
                # Set date as index
                processed_df = processed_df.set_index(date_col)
                
                # Sort by date
                processed_df = processed_df.sort_index()
            
            # Look for Bitcoin price columns
            btc_price_cols = [
                col for col in processed_df.columns 
                if any(term in col.lower() for term in ['btc', 'bitcoin']) and 
                any(term in col.lower() for term in ['price', 'close', 'open', 'high', 'low'])
            ]
            
            # If we have price columns, ensure they're numeric
            for col in btc_price_cols:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
            # If we don't have a BTC_Close column but have other price columns, create it
            if 'BTC_Close' not in processed_df.columns and btc_price_cols:
                # Use the first found price column as BTC_Close
                processed_df['BTC_Close'] = processed_df[btc_price_cols[0]]
            
            # Calculate the daily change if not already present
            if 'BTC_Close' in processed_df.columns and 'BTC_Daily_Change' not in processed_df.columns:
                processed_df['BTC_Daily_Change'] = processed_df['BTC_Close'].diff()
            
            # Drop rows with missing crucial data
            if 'BTC_Close' in processed_df.columns:
                processed_df = processed_df.dropna(subset=['BTC_Close'])
            
            return processed_df
            
        except Exception as e:
            st.error(f"Error cleaning and processing data: {str(e)}")
            return df  # Return original if processing fails
