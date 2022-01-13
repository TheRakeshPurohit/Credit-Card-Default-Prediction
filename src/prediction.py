import logging
import numpy as np
import pandas as pd

from application_logger import CustomApplicationLogger
from data_processing import DataProcessingTest
from utils import File_Ops
import os


class Prediction:
    def __init__(self) -> None:
        self.file_object = open(
            r"E:\QnAMedical\Credit Card Fraud\logs\PredictionLogs.txt", "a+"
        )
        self.loggings = CustomApplicationLogger()

    def get_prediction_data(self):
        try:
            self.loggings.logger(
                self.file_object, "Prediction class called successfully"
            )
            test_pred_file = pd.read_csv(
                r"E:\QnAMedical\Credit Card Fraud\data\raw_data\test.csv"
            )
            self.loggings.logger(
                self.file_object, "Prediction class read the test file successfully"
            )
            return test_pred_file
        except Exception as e:
            self.loggings.logger(
                self.file_object, "Error occured in prediction class" + str(e)
            )
            raise Exception()

    def process_prediction_csv(self):
        try:
            self.loggings.logger(self.file_object, "Processing Prediction Csv")
            test_pred_file = self.get_prediction_data()
            data_processing = DataProcessingTest(data=test_pred_file)
            null_present, cols_with_missing_values = data_processing.is_null_present()
            test_pred_file = data_processing.impute_missing_values(
                test_pred_file, cols_with_missing_values
            )
            test_pred_file = data_processing.encode_categorical_data(
                test_pred_file, ["customer_id", "name"]
            )
            self.loggings.logger(
                self.file_object, "Processing Prediction Csv completed"
            )
            return test_pred_file
        except Exception as e:
            self.loggings.logger(
                self.file_object, "Error occured in processing prediction csv" + str(e)
            )
            raise Exception()

    def predict(self):
        try:

            file_loader = File_Ops(self.file_object, self.loggings)
            kmeans = file_loader.load_models("KMeans")

            X = self.process_prediction_csv()
            clusters = kmeans.predict(X)
            X["clusters"] = clusters
            clusters = X["clusters"].unique()
            predictions = []
            for i in clusters:
                cluster_data = X[X["clusters"] == i]
                cluster_data = cluster_data.drop(["clusters"], axis=1)
                model_name = file_loader.get_correct_models(i)
                model = file_loader.load_models(model_name)
                result = model.predict(cluster_data)

            final = pd.DataFrame(list(zip(result)), columns=["Predictions"])
            path = r"E:\QnAMedical\Credit Card Fraud\data\predictions\Predictions.csv"
            final.to_csv(
                r"E:\QnAMedical\Credit Card Fraud\data\predictions\Predictions.csv",
                header=True,
                mode="a+",
            )  # appends result to prediction file
            self.loggings.logger(self.file_object, "End of Prediction")
        except Exception as e:
            self.loggings.logger(
                self.file_object,
                "Error occured while running the prediction!! Error:: %s" % e,
            )
            raise e

    def get_prediction_for_single_example(
        self,
        age,
        gender,
        owns_car,
        owns_house,
        no_of_children,
        net_yearly_income,
        no_of_days_employed,
        occupation_type,
        total_family_members,
        migrant_worker,
        yearly_debt_payments,
        credit_limit,
        credit_limit_used,
        credit_score,
        prev_defaults,
        default_in_last_6months,
    ):
        try:
            self.loggings.logger(
                self.file_object, "Prediction class called successfully"
            )
            inputs = [
                [
                    age,
                    gender,
                    owns_car,
                    owns_house,
                    no_of_children,
                    net_yearly_income,
                    no_of_days_employed,
                    occupation_type,
                    total_family_members,
                    migrant_worker,
                    yearly_debt_payments,
                    credit_limit,
                    credit_limit_used,
                    credit_score,
                    prev_defaults,
                    default_in_last_6months,
                ]
            ]

            file_loader = File_Ops(self.file_object, self.loggings)

            kmeans = file_loader.load_models("KMeans")
            clusters = kmeans.predict(inputs)
            if clusters == 0:
                decision_tree = file_loader.load_models("dt_model_0")
                lightgbm = file_loader.load_models("lgbm_model_0")
                xgboost = file_loader.load_models("xgb_model_0")
                # give the result which got maximum number of vites
                gbm_result = lightgbm.predict(inputs)
                return gbm_result[0]

            if clusters == 1:
                decision_tree = file_loader.load_models("dt_model_1")
                lightgbm = file_loader.load_models("lgbm_model_1")
                xgboost = file_loader.load_models("xgb_model_1")
                # give the result which got maximum number of vites
                gbm_result = lightgbm.predict(inputs)
                return gbm_result[0]

            self.loggings.logger(
                self.file_object, "Prediction class read the test file successfully"
            )
            # return result
        except Exception as e:
            self.loggings.logger(
                self.file_object,
                "Error occured while running the prediction!! Error:: %s" % e,
            )
            raise e


if __name__ == "__main__":
    # prediction = Prediction()
    # prediction.predict()
    # inputs = [52, 0, 1, 0, 0, 23000, 998, 12, 0, 14000, 26000, 4, 779, 0, 0]
    # prediction.get_prediction_for_single_example(
    #     age=52,
    #     gender=0,
    #     owns_car=1,
    #     owns_house=0,
    #     no_of_children=0,
    #     net_yearly_income=230000,
    #     no_of_days_employed=998,
    #     occupation_type=12,
    #     total_family_members=0,
    #     migrant_worker=0,
    #     yearly_debt_payments=14000,
    #     credit_limit=26000,
    #     credit_limit_used=4,
    #     credit_score=779,
    #     prev_defaults=0,
    #     default_in_last_6months=0,
    # ) 
    pass 
