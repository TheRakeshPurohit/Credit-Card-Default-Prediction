from itertools import combinations
import logging
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from data_ingestion import DataUtils
# from processing1 import DataProcessingTest


class DatavalidationTest:
    def __init__(self, data) -> None:
        self.data = data

    def dataset_development(self):
        logging.info("Dataset development")
        try:
            train_ratio = 0.75
            validation_ratio = 0.15
            test_ratio = 0.10
            X = self.data.drop(["credit_card_default"], axis=1)
            y = self.data["credit_card_default"]
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=1 - train_ratio, random_state=48
            )
            logging.info("Dataset development completed")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error("Dataset development failed")
            logging.error(e)
            return None

    def outlier_detection(self, x_train, x_val, x_test, y_train, y_val, y_test):
        logging.info("Outlier detection")
        try:
            iso = IsolationForest(contamination=0.1)
            iso.fit(x_train)
            y_hat = iso.predict(x_train)
            y_hat_val = iso.predict(x_val)
            y_hat_test = iso.predict(x_test)
            mask_train = y_hat != -1
            x_train1, y_train1 = x_train[mask_train, :], y_train[mask_train]
            mask_val = y_hat_val != -1
            x_val1, y_val1 = x_val[mask_val, :], y_val[mask_val]
            mask_test = y_hat_test != -1
            x_test1, y_test1 = x_test[mask_test, :], y_test[mask_test]
            logging.info("Outlier detection completed")
            print(x_train1)
        except Exception as e:
            logging.error("Outlier detection failed")
            logging.error(e)
            return None


# data_utils = DataUtils()
# train_data, test_data = data_utils.load_data()
# data_process = DataProcessingTest(train_data)
# null_present, cols_with_missing_values = data_process.is_null_present()
# train_data = data_process.impute_missing_values(train_data, cols_with_missing_values)
# train_data = data_process.encode_categorical_data(train_data, ["customer_id", "name"])

# data_validation = DatavalidationTest(train_data)
# x_train, x_val, x_test, y_train, y_val, y_test = data_validation.dataset_development()

# print(x_train.shape)
# print(x_val.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)

