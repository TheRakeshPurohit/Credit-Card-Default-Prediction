from pytorch_tabnet.tab_model import TabNetClassifier  
import pandas as pd 
import numpy as np
from data_ingestion import DataIngestion 
import data_ingestion
from feature_engineering import FeatureEngineering 
from utils import DataProcessingTest, Submission
from validate_data import DatavalidationTest

data_utils = data_ingestion.DataUtils()
train_data, test_data = data_utils.load_data()
train_data_copy = train_data.copy()
feature_engineering = FeatureEngineering(train_data_copy)
train_data_copy = feature_engineering.add_features()

    # train_data_csv = pd.read_csv(r'E:\Credit Card Default Risk\dataset\train_data.csv')
data_process = DataProcessingTest(train_data_copy)
null_present, cols_with_missing_values = data_process.is_null_present()
train_data = data_process.impute_missing_values(
    train_data_copy, cols_with_missing_values
)
    # train_data = data_process.encode_categorical_data(train_data_copy, ["customer_id", "name"])
    # train_data.drop("Unnamed: 0", axis=1, inplace=True)
data_dev = DatavalidationTest(train_data)
x_train, x_test, y_train, y_test = data_dev.dataset_development()

    # model_devtrain = model_dev.ModelTraining(x_train,y_train, x_test, y_test)
    # xgboost_model = model_devtrain.xgboost(fine_tuning=True)
    # prediction = xgboost_model.predict(x_test)
    # prediction_proba = xgboost_model.predict_proba(x_test)

    # # plot the feature importances
    # feature_importance = xgboost_model.get_booster().get_score(importance_type='weight')
    # feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    # feature_importance = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    # feature_importance
    # take the cols which are of object types
# feature_engineering1 = FeatureEngineering(test_data)
# test_data_copy = feature_engineering1.add_features()
# data_process = DataProcessingTest(test_data_copy)
# null_present, cols_with_missing_values = data_process.is_null_present()
# test_data_copy = data_process.impute_missing_values(
#         test_data_copy, cols_with_missing_values
# )
#     # test_data_copy.to_csv("test_data.csv", index=False)

train_t, test_t, encoder = feature_engineering.mean_encoding(
        x_train, y_train, x_test, test=False
)
train_t.drop(["customer_id", "name"], axis=1, inplace=True)
test_t.drop(["customer_id", "name"], axis=1, inplace=True)

train_t.fillna(0, inplace=True)
test_t.fillna(0, inplace=True)



# tets_dp = DataProcessingTest(test_data)
# null_present, cols_with_missing_values = tets_dp.is_null_present()
# test_datas = tets_dp.impute_missing_values(test_data, cols_with_missing_values)

# test_datas = encoder.transform(test_datas)
# test_datas = test_datas[cols_to_reserve]


# train_data_csv = pd.read_csv(r'E:\Credit Card Default Risk\dataset\train_data.csv')
# data_process = DataProcessingTest(train_data_csv)
# null_present, cols_with_missing_values = data_process.is_null_present()
# train_data = data_process.impute_missing_values(train_data_csv, cols_with_missing_values)
# train_data = data_process.encode_categorical_data(train_data_csv, ["customer_id", "name"])
# train_data.drop("Unnamed: 0", axis=1, inplace=True) 
# data_dev = DatavalidationTest(train_data)
# x_train, x_test, y_train, y_test = data_dev.dataset_development()  

# # fill the missing values with the mean of the column 
# x_train.fillna(x_train.mean(), inplace=True) 
# y_train.isnull().sum().sum() 

x_test.fillna(x_test.mean(), inplace=True)
clf = TabNetClassifier()  #TabNetRegressor()
clf.fit(
  train_t, y_train,
  eval_set=[(test_t, y_test)]
)
preds = clf.predict(x_test)
