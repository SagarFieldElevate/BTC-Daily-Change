import logging
from airtable_service import AirtableService
from data_processor import DataProcessor

logger = logging.getLogger(__name__)

class AirtableConnector:
    """Fetch and process data from Airtable tables."""

    def __init__(self, api_key: str, base_id: str):
        self.service = AirtableService(api_key, base_id)
        self.processor = DataProcessor()

    def fetch(self, stream_cfg: dict):
        table = stream_cfg.get("table")
        if not table:
            raise ValueError("Airtable stream requires a 'table' field")
        try:
            records = self.service.get_records_from_table(table)
            df = self.processor.process_airtable_records(records, self.service)
            if df is not None:
                df.attrs["frequency"] = stream_cfg.get("frequency")
            return df
        except Exception:
            logger.exception("Failed to fetch Airtable table %s", table)
            return None
