import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import logging
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from application_logger import CustomApplicationLogger
from data_ingestion import DataIngestion


class DataProcessingTest:
    def __init__(self, data):
        self.data = data
        self.file_object = open(
            r"E:\QnAMedical\Credit Card Fraud\logs\DataProcessingLogs.txt", "a+"
        )
        self.logger = CustomApplicationLogger()

    def is_null_present(self):

        try:
            self.null_present = False
            self.cols_with_missing_values = []
            self.cols = self.data.columns
            self.null_counts = self.data.isnull().sum()
            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True
                    self.cols_with_missing_values.append(self.cols[i])
            if (
                self.null_present
            ):  # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null["columns"] = self.data.columns
                self.dataframe_with_null["missing values count"] = np.asarray(
                    self.data.isna().sum()
                )
            self.logger.logger(self.file_object, "Null Values Method Success")
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger.logger(self.file_object, str(e))
            raise e

            # Error while  checking for missing values

    def impute_missing_values(self, data, cols_with_missing_values):
        try:

            self.data = data
            self.cols_with_missing_values = cols_with_missing_values

            self.data["owns_car"].fillna(self.data["owns_car"].mode()[0], inplace=True)
            self.data["no_of_children"].fillna(
                self.data["no_of_children"].mode()[0], inplace=True
            )
            self.data["no_of_days_employed"].fillna(
                self.data["no_of_days_employed"].mean(), inplace=True
            )
            self.data["total_family_members"].fillna(
                self.data["total_family_members"].mode()[0], inplace=True
            )
            self.data["migrant_worker"].fillna(
                self.data["migrant_worker"].mode()[0], inplace=True
            )
            self.data["yearly_debt_payments"].fillna(
                self.data["yearly_debt_payments"].mean(), inplace=True
            )
            self.data["credit_score"].fillna(
                self.data["credit_score"].mean(), inplace=True
            )

            null_present, cols_with_missing_values = self.is_null_present()
            self.logger.logger(self.file_object, "Imputing missing values is a success")
            return data
        except Exception as e:
            self.logger.logger(self.file_obj, str(e))
            raise e

    def drop_columns(self, data, cols_to_drop):
        try:
            self.data = data
            self.cols_to_drop = cols_to_drop

            self.data.drop(self.cols_to_drop, axis=1, inplace=True)

            self.logger.logger(self.file_object, "Drop columns is a success")
           
            return self.data
        except Exception as e:
            self.logger.logger(self.file_obj, str(e))
            raise e

    def encode_categorical_data(self, data, cols_to_exclude):
        logging.info("In the Encode categorical data method of the Preprocessor class")
        try:
            self.data = data
            self.drop_columns(self.data, cols_to_exclude)
            Encoder = OrdinalEncoder()
            categorical_columns = data.select_dtypes(include=["object"]).columns
            self.data[categorical_columns] = Encoder.fit_transform(
                self.data[categorical_columns]
            )
            logging.info(
                "Done Encoding categorical data. Exited the Encode categorical data method of the Preprocessor class"
            )
            return self.data
        except Exception as e:
            logging.error("Error while encoding categorical data")
            raise e

    def scale_numerical_features(self, data):
        self.data = data
        logging.info("In the scale numerical features method of the Preprocessor class")
        try:
            self.num_df = self.data[
                [
                    "age",
                    "net_yearly_income",
                    "no_of_days_employed",
                    "yearly_debt_payments",
                    "credit_limit",
                    "credit_limit_used(%)",
                    "credit_score",
                ]
            ]
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(
                data=self.scaled_data, columns=self.num_df.columns
            )

            logging.info(
                "scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class"
            )
            return self.scaled_num_df
        except Exception as e:
            logging.error("Error while scaling numerical features")
            raise e

    def prepare_df_for_new_scaled_features(self, data):
        self.data = data
        logging.info(
            "In the prepare_df_for_new_scaled_features method of the Preprocessor class"
        )
        try:
            scaled_numerical_features = self.scale_numerical_features(self.data)
            self.data.drop(
                [
                    "age",
                    "net_yearly_income",
                    "no_of_days_employed",
                    "yearly_debt_payments",
                    "credit_limit",
                    "credit_limit_used(%)",
                    "credit_score",
                ],
                axis=1,
                inplace=True,
            )
            self.data = pd.concat([self.data, scaled_numerical_features], axis=1)
            final_processed_data = self.encode_categorical_data(
                self.data, ["customer_id", "name"]
            )
            logging.info(
                "Exited the prepare_df_for_new_scaled_features method of the Preprocessor class"
            )
            return final_processed_data
        except Exception as e:
            logging.error("Error while preparing data for new scaled features")
            raise e

    def handle_imbalanced_data(self, X, y):
        logging.info("In the handle_imbalanced_data method of the Preprocessor class")
        try:
            sm = RandomOverSampler(random_state=42)
            X_res, y_res = sm.fit_sample(X, y)
            logging.info(
                "Exited the handle_imbalanced_data method of the Preprocessor class"
            )
            return X_res, y_res
        except Exception as e:
            logging.error("Error while handling imbalanced data")
            raise e


class DatasetDevelopment: 
    def __init__(self) -> None:
        self.file_object = open(r"E:\QnAMedical\Credit Card Fraud\logs\data_dividelogs.txt", "a+") 
        self.logger = CustomApplicationLogger() 

    def split_data(self): 
        try: 
            data_utils = DataIngestion()
            train_data, test_data = data_utils.data_ingestion_from_local_system(
                train_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\train.csv",
                test_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\test.csv",
            )  
            X = train_data.drop(["credit_card_default"], axis=1) 
            x_train, x_test, y_train, y_test = train_test_split( 
                X, train_data["credit_card_default"], test_size=0.2, random_state=42 
            ) 
            self.logger.logger(self.file_object, "Splitting data is a success") 
            return x_train, x_test, y_train, y_test 
        except Exception as e: 
            self.logger.logger(self.file_object, str(e)) 
            raise e

         

