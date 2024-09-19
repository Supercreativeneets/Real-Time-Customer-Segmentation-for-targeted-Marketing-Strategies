import sys
import os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

import pandas as pd



@dataclass
class DataIngestionConfig:
    data_path: str = os.path.join('artifacts', 'df.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self):
        logging.info("Starting to load the dataset")
        try:
            df = pd.read_csv('/Users/neeti/Documents/Customer-Segmentation-Kubernetes-CI-CD/notebook/data/Customer Data.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return self.ingestion_config.data_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    data_df = obj.start_data_ingestion()

