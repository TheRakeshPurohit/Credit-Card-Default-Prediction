import numpy as np
import pandas as pd
from application_logger import CustomApplicationLogger
from tqdm import tqdm
import requests


class DataIngestion:
    def __init__(self) -> None:
        self.file_obj = open(
            r"E:\QnAMedical\Credit Card Fraud\logs\DataIngestionLogs.txt", "a+"
        )
        self.logger = CustomApplicationLogger()

    def data_ingestion_from_local_system(self, train_data_path, test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            self.logger.logger(self.file_obj, "Data Ingestion is Successful")
            return train_data, test_data
        except Exception as e:
            self.logger.logger(self.file_obj, str(e))
            raise e
        finally:
            self.file_obj.close()

    def download_data(self): 
        try: 
            # download data from the link and stoe it in data/raw_data/ 
            url = "https://ayushml.blob.core.windows.net/data/train.csv" 
            r = requests.get(url, allow_redirects=True) 
            open(r'E:\QnAMedical\Credit Card Fraud\data\raw_data\train.csv', 'wb').write(r.content)  
            # url = "https://ayushml.blob.core.windows.net/data/test.csv" 
            # r = requests.get(url, allow_redirects=True) 
            # open('data/raw_data/test.csv', 'wb').write(r.content) 
            self.logger.logger(self.file_obj, "Data Download is Successful") 
        except Exception as e: 
            self.logger.logger(self.file_obj, str(e)) 
            raise e 
        finally: 
            self.file_obj.close()


if __name__ == "__main__": 
    data_ingestion = DataIngestion()
    data_ingestion.download_data()