# import data ingestion
from src.dataset.data_ingestion import DataIngestion

# import logger
from tools.logger import logger

# sklearn
from sklearn.preprocessing import MinMaxScaler

# pandas
import pandas as pd

# type-hinting
from typing import Tuple

import numpy as np

class DataTransformation:
    """Utility class for transforming data fetched from data ingestion.
    """
    def __init__(self, series_ids: str):
        self.series_ids = series_ids
        self.scaler = MinMaxScaler()
        
        
    def split_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into training and testing split.
        
        Returns:
            train_scaled, test_scaled (Tuple[np.ndarray, np.ndarray])
        """
        
        try:
            data = DataIngestion(series_ids=self.series_ids).fetch_data()
            
            # split data into training and testing features
            
            training = data.iloc[:, 0:1].values

            
            # training split
            train_size = int(len(training) * 0.80)
            logger.info(f"Train Size: {train_size}")
            
            train_data = training[:train_size]
            test_data = training[train_size:]
            
            # scale training and testing data
            self.scaler.fit(training)
            train_scaled = self.scaler.transform(train_data)
            test_scaled = self.scaler.transform(test_data)
            
   
   
            logger.info(f"Shape of scaled training data: {train_scaled.shape}")
            logger.info(f"Shape of scaled testing data: {test_scaled.shape}")

            
            return train_scaled, test_scaled
        except ValueError as ve:
            logger.info(f"The Series Id was not correct or does not exist: {ve}")
        except Exception as e:
            logger.info(f"could not split data: {e}")
        return [], []
    