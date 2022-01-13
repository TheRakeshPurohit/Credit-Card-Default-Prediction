import numpy as np
import pandas as pd
from application_logger import CustomApplicationLogger
import json


class DataValidation:
    def __init__(self, data) -> None:
        self.data = data
        self.file_obj = open(
            r"E:\QnAMedical\Credit Card Fraud\logs\TrainingDataValidationLogs.txt", "a+"
        )
        self.logger = CustomApplicationLogger()
        self.schema_path = r"E:\QnAMedical\Credit Card Fraud\src\schema_training.json"

    def train_data_validation(self):
        """
        Input: 
            None
        Output: 
            Logs the message to the file object 
        """
        try:
            # validate the data from schema_training.json 
            # check if the filename "train.csv" is same as in schema
            if json.load(open(self.schema_path))["FileName"] == "train.csv":
                self.logger.logger(self.file_obj, "File name is matching")
            else:
                self.logger.logger(
                    self.file_obj, "File name is not matching"
                )  # log the error message

            # check if the the number of columns is same as in schema
            if (
                len(self.data.columns)
                == json.load(open(self.schema_path))["NumberofColumns"]
            ):
                self.logger.logger(self.file_obj, "No. of Columns are matching")
            else:
                self.logger.logger(
                    self.file_obj, "No. of Columns not matching"
                )  # log the error message

            for column in self.data.columns:
                if (
                    self.data[column].dtype
                    != json.load(open(self.schema_path))["ColumnNames"][column]
                ):
                    self.logger.logger(self.file_obj, "Data types are not matching")
                    raise ValueError("Data types are not matching")
                else:
                    self.logger.logger(
                        self.file_obj, f"Data type of {column} are matching"
                    )

        except Exception as e:
            self.logger.logger(self.file_obj, str(e))
            raise e
        finally:
            self.file_obj.close()


if __name__ == "__main__":
    pass 