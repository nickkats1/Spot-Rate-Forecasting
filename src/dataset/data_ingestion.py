from tools.logger import logger

import pandas as pd
from constants import *



class DataIngestion:
    """Utility class to fetch data from FredAPI"""
    
    def __init__(self, series_ids: str):
        """Initialize DataIngestion class.

        Args:
            series_ids (str): The name of the data from Fred API._
        """
        self.series_ids = series_ids
        
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from fred and returns pd.Dataframe.
        
        Returns:
            data (pd.DataFrame): A pd.DataFrame consisting of clean fred data.
        """
        
        try:
            data = fred.get_series(series_id=self.series_ids)
            
            # set name for variable in series_ids
            data.name = self.series_ids
            data = pd.DataFrame(data).dropna().drop_duplicates()
            
            # reset index
            data.reset_index(inplace=True)
            
            # change index to data and drop index
            data['date'] = data['index']
            data.drop("index", inplace=True, axis=1)
            return data
        except ValueError as ve:
            logger.info(f"Invalid fred series id: {ve}")