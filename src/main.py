import numpy as np
from application_logger import CustomApplicationLogger
from evaluation import *
from model_development import ModelTraining, BestModelFinder
from data_ingestion import DataIngestion
from eda_src import EDA
from data_processing import DataProcessingTest, DatasetDevelopment
from data_validation import DataValidation
from clustering import KMeansClustering
from utils import File_Ops
from sklearn.model_selection import train_test_split


def main():
    try:
        data_utils = DataIngestion()
        train_data, test_data = data_utils.data_ingestion_from_local_system(
            train_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\train.csv",
            test_data_path=r"E:\QnAMedical\Credit Card Fraud\data\raw_data\test.csv",
        )

        eda = EDA(data=None)
        eda.check_class_distributions()
        eda.cross_tabulation()
        eda.check_null_values()
        eda.check_outliers()

        data_validation = DataValidation(data=train_data)
        data_validation.train_data_validation()

        data_processing = DataProcessingTest(data=train_data)
        null_present, cols_with_missing_values = data_processing.is_null_present()
        data = data_processing.impute_missing_values(
            train_data, cols_with_missing_values
        )
        train_data = data_processing.encode_categorical_data(
            train_data, ["customer_id", "name"]
        )

        X_train = train_data.drop(["credit_card_default"], axis=1)
        file_object = open(
            r"E:\QnAMedical\Credit Card Fraud\logs\ModelTrainingLogs.txt", "a+"
        )
        loggings = CustomApplicationLogger()
        kmeans = KMeansClustering(file_object, loggings)
        number_of_clusters = kmeans.elbow_plot(X_train)

        # Divide the data into clusters
        X_train = kmeans.create_clusters(X_train, number_of_clusters)

        # create a new column in the dataset consisting of the corresponding cluster assignments.
        Y = train_data["credit_card_default"]
        X_train["Labels"] = Y

        list_of_clusters = X_train["Cluster"].unique()

        """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""
        X_train["Cluster"].value_counts()
        for i in list_of_clusters:
            cluster_data = X_train[X_train["Cluster"] == i]
            cluster_features = cluster_data.drop(["Labels", "Cluster"], axis=1)
            cluster_label = cluster_data["Labels"]

            # x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)
            dataset_dev = DatasetDevelopment()
            X = X_train.drop(["Labels", "Cluster"], axis=1)
            y = X_train[["Labels"]]

            # use ravel on y
            y = y.values.ravel()
            y.reshape(1, -1)
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=355
            )
            ModelTrainer = BestModelFinder()
            dt_model, rf_model, lgbm_model, xgb_model = ModelTrainer.train_all_models(
                x_train, y_train, x_test, y_test
            )
            # saving the best model to the directory.
            file_op = File_Ops(file_object, loggings)
            dt_save_model = file_op.save_models(dt_model, "dt_model_" + str(i))
            rf_save_model = file_op.save_models(rf_model, "rf_model_" + str(i))
            lgbm_save_model = file_op.save_models(lgbm_model, "lgbm_model_" + str(i))
            xgb_save_model = file_op.save_models(xgb_model, "xgb_model_" + str(i))

            best_model = ModelEvaluater(x_test, y_test)
            best_model = best_model.evaluate_trained_models(
                dt_model, rf_model, lgbm_model, xgb_model
            )
            # logging the successful Training
            loggings.logger(file_object, "Successful End of Training")

    except Exception as e:
        # logging the unsuccessful Training
        loggings.logger(file_object, "Unsuccessful End of Training")
        raise Exception


if __name__ == "__main__":
    main()
