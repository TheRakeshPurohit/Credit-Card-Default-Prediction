import numpy as np
import pandas as pd
from scipy.sparse import data
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler
import xgboost
import data_ingestion


class DataProcessingTest:
    def __init__(self, data):
        self.data = data

    def is_null_present(self):

        logging.info("Entered the is_null_present method of the Preprocessor class",)
        self.null_present = False
        self.cols_with_missing_values = []
        self.cols = self.data.columns

        try:
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
            logging.info(
                "Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class",
            )

            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            logging.error("Error while  checking for missing values")
            raise e

    def impute_missing_values(self, data, cols_with_missing_values):
        logging.info(
            "Entered the impute_missing_values method of the Preprocessor class",
        )
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
            logging.info(
                "Number of null values present in the dataframe after Imputing: ",
                null_present,
            )

            return data
        except Exception as e:
            logging.error("Error while imputing missing values")
            raise e

    def drop_columns(self, data, cols_to_drop):
        logging.info("Entered the drop_columns method of the Preprocessor class",)
        try:
            self.data = data
            self.cols_to_drop = cols_to_drop

            self.data.drop(self.cols_to_drop, axis=1, inplace=True)

            logging.info(
                "Dropping columns is a success.Data written to the dropped columns file. Exited the drop_columns method of the Preprocessor class",
            )

            return self.data
        except Exception as e:
            logging.error("Error while dropping columns")
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
            X_res, y_res = sm.fit_resample(X, y)
            logging.info(
                "Exited the handle_imbalanced_data method of the Preprocessor class"
            )
            return X_res, y_res
        except Exception as e:
            logging.error("Error while handling imbalanced data")
            raise e


class Submission:
    def __init__(self, model, test_data) -> None:
        self.model = model
        self.test_data = test_data

    def get_submission_csv(self, name):
        self.test_data_copy = self.test_data.copy()
        data_process = DataProcessingTest(self.test_data_copy)
        # null_present, cols_with_missing_values = data_process.is_null_present()
        # test_data_copy = data_process.impute_missing_values(
        #     self.test_data_copy, cols_with_missing_values
        # )
        # self.test_data_copy = data_process.encode_categorical_data(
        #     test_data_copy, ["customer_id", "name"]
        # )
        self.predictions = self.model.predict(self.test_data_copy)
        submission = pd.DataFrame(
            {
                "customer_id": self.test_data["customer_id"],
                "credit_card_default": self.predictions,
            }
        )
        submission.to_csv(
            r"E:\Hackathon\Credit Card Default Risk\sub.csv", index=False,
        )
        return submission

    def get_sub(self):
        test_copy = self.test_data.copy()
        data_process = DataProcessingTest(test_copy)
        null_present, cols_with_missing_values = data_process.is_null_present()
        test_copy = data_process.impute_missing_values(
            test_copy, cols_with_missing_values
        )
        test_copy = data_process.encode_categorical_data(
            test_copy, ["customer_id", "name"]
        )

        import joblib
        import xgboost as xgb

        xgboost = joblib.load("xgboost_model.pkl")
        data_test = xgb.DMatrix(test_copy)
        prediction = np.array(
            xgboost.predict(data_test, ntree_limit=xgboost.best_iteration)
        )
        ypred_bst = prediction > 0.2
        ypred_bst = ypred_bst.astype(int)
        submission = pd.DataFrame(
            {
                "customer_id": self.test_data["customer_id"],
                "credit_card_default": ypred_bst,
            }
        )
        submission.to_csv(r"E:\Hackathon\Credit Card Default Risk\sub1.csv",)
        return submission

    # except Exception as e:
    #     logging.error("Error in training get_submission_csv")
    #     logging.error(e)
    #     return None
