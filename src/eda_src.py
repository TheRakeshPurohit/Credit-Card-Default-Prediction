import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from application_logger import CustomApplicationLogger
from data_ingestion import DataIngestion
import plotly
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
import plotly.offline as offline
import plotly.graph_objs as go


class EDA:
    def __init__(self, data) -> None:
        self.data = data
        self.file_object = open(
            r"E:\QnAMedical\Credit Card Fraud\logs\EDA_logs.txt", "a+"
        )
        self.logger = CustomApplicationLogger()

    def check_class_distributions(self):
        try:
            data_utils = DataIngestion()
            train_data, test_data = data_utils.data_ingestion_from_local_system(
                train_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\train.csv",
                test_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\test.csv",
            ) 

            # plot a pie chart in plotly for every column except Review and ID
            columns = train_data[
                [
                    "gender",
                    "owns_car",
                    "owns_house",
                    "migrant_worker",
                    "no_of_children",
                    "prev_defaults",
                    "default_in_last_6months",
                    "credit_card_default",
                ]
            ]
            for column in columns:
                # plt the pie chart
                labels = train_data[column].unique()
                values = train_data[column].value_counts()
                fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                fig.update_layout(title_text=column)
                fig.show()
            self.logger.logger(
                self.file_object, "Checking class distributions is successful"
            )
        except Exception as e:
            self.logger.logger(self.file_object, str(e))
            raise e

    def cross_tabulation(self):
        try:
            data_utils = DataIngestion()
            train_data, test_data = data_utils.data_ingestion_from_local_system(
                train_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\train.csv",
                test_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\test.csv",
            )
            pd.crosstab(train_data["owns_car"], train_data["owns_house"]).plot(
                kind="bar"
            )
            pd.crosstab(train_data["gender"], train_data["credit_card_default"]).plot(
                kind="bar"
            )

            plt.show()
            self.logger.logger(self.file_object, "Cross Tabulation Done")
        except Exception as e:
            self.logger.logger(self.file_object, str(e))
            raise e
    def check_null_values(self): 
        try: 
            data_utils = DataIngestion()
            train_data, test_data = data_utils.data_ingestion_from_local_system(
                train_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\train.csv",
                test_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\test.csv",
            )
            missing_values = train_data.isnull().sum()
            missing_values_percentage = 100 * train_data.isnull().sum() / len(train_data)
            missing_values_table = pd.concat(
                [missing_values, missing_values_percentage], axis=1
            )
            missing_values_table_ren_columns = missing_values_table.rename(
                columns={0: "Missing Values", 1: "% of Total Values"}
            )
            # plot the missing values
            missing_values_table_ren_columns.plot(kind="bar", figsize=(20, 10))

            plt.show() 
            self.logger.logger(self.file_object, "Checking Null Values is successful") 
        except Exception as e: 
            self.logger.logger(self.file_object, str(e)) 
            raise e 
    
    def check_outliers(self): 
        try: 
            data_utils = DataIngestion()
            train_data, test_data = data_utils.data_ingestion_from_local_system(
                train_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\train.csv",
                test_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\test.csv",
            )
            # plot the outliers
            sns.boxplot(x="credit_card_default", y="age", data=train_data)
            plt.show()
            self.logger.logger(self.file_object, "Checking Outliers is successful")
        except Exception as e:
            self.logger.logger(self.file_object, str(e))
            raise e


if __name__ == "__main__": 
    eda = EDA(data=None) 
    eda.check_class_distributions() 
    eda.cross_tabulation() 
    eda.check_null_values() 
    eda.check_outliers()
