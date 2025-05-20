import os
import io
import tempfile
import pandas as pd
from pyairtable import Api, Base, Table
import streamlit as st

class AirtableService:
    def __init__(self, api_key, base_id):
        """Initialize the Airtable service.
        
        Args:
            api_key (str): Airtable API key
            base_id (str): Airtable base ID
        """
        self.api_key = api_key
        self.base_id = base_id
        self.api = Api(api_key)
        self.base = Base(api_key, base_id)
        
    def get_records_from_table(self, table_name):
        """Get records from an Airtable table.
        
        Args:
            table_name (str): The name of the table to fetch records from
            
        Returns:
            list: A list of records from the table
        """
        try:
            table = Table(self.api_key, self.base_id, table_name)
            records = table.all()
            return records
        except Exception as e:
            st.error(f"Error fetching records from table {table_name}: {str(e)}")
            return []
            
    def download_attachment(self, attachment):
        """Download an attachment from Airtable.
        
        Args:
            attachment (dict): Attachment information from Airtable
            
        Returns:
            io.BytesIO: A BytesIO object containing the attachment data
        """
        import requests
        
        try:
            url = attachment.get('url')
            if not url:
                return None
                
            response = requests.get(url)
            if response.status_code == 200:
                return io.BytesIO(response.content)
            else:
                st.error(f"Failed to download attachment: HTTP status {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error downloading attachment: {str(e)}")
            return None
    
    def get_excel_data_from_attachment(self, attachment):
        """Extract data from an Excel attachment.
        
        Args:
            attachment (dict): Attachment information from Airtable
            
        Returns:
            pd.DataFrame: A DataFrame containing the Excel data
        """
        try:
            file_obj = self.download_attachment(attachment)
            if file_obj is None:
                return None
                
            # Read Excel file
            df = pd.read_excel(file_obj)
            return df
        except Exception as e:
            st.error(f"Error processing Excel attachment: {str(e)}")
            return None
