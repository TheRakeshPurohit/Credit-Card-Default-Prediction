import catboost
from catboost.core import train
import xgboost
from validate_data import DatavalidationTest
from utils import DataProcessingTest, Submission
import data_ingestion
import model_dev
import evaluate_models
import utils
import process_data
import xgboost as xgb
import numpy as np
from utils import DataProcessingTest
import utils
from feature_engineering import FeatureEngineering
import pandas as pd


def main():
    data_utils = data_ingestion.DataUtils()
    train_data, test_data = data_utils.load_data()
    train_data_copy = train_data.copy()
    feature_engineering = FeatureEngineering(train_data_copy)
    train_data_copy = feature_engineering.add_features()

    train_data_copy
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
    cols_to_drop = [
        col for col in train_data.columns if train_data[col].dtype == "object"
    ]
    # catboost_model = model_devtrain.catboost(fine_tuning=True)

    feature_engineering1 = FeatureEngineering(test_data)
    test_data_copy = feature_engineering1.add_features()
    data_process = DataProcessingTest(test_data_copy)
    null_present, cols_with_missing_values = data_process.is_null_present()
    test_data_copy = data_process.impute_missing_values(
        test_data_copy, cols_with_missing_values
    )
    # test_data_copy.to_csv("test_data.csv", index=False)

    train_t, test_t, encoder = feature_engineering.mean_encoding(
        x_train, y_train, x_test, test=False
    )

    train_t.drop(["customer_id", "name"], axis=1, inplace=True)
    test_t.drop(["customer_id", "name"], axis=1, inplace=True)
    train_t
    model_devtrain = model_dev.ModelTraining(train_t, y_train, test_t, y_test)
    xgboost_model = model_devtrain.xgboost(fine_tuning=True)

    # =============================================================================
    # plot the feature importances

    from xgboost import plot_importance

    plot_importance(xgboost_model, max_num_features=10)
    train_t.columns
    # remoe all the columns which less important
    cols_to_reserve = [
        "yearly_debt_payments",
        "credit_limit",
        "coverage_ratio",
        "credit_score",
        "net_yearly_income",
        "credit_limit_used(%)",
        "age",
        "credit_score_per_year",
        "no_of_days_employed",
        "precentage_of_number_of_days_employed",
    ]
    train_t = train_t[cols_to_reserve]
    test_t = test_t[cols_to_reserve]
    y_pred = xgboost_model.predict(test_t)
    evals = evaluate_models.Evaluation(y_test, y_pred)
    evals.evaluate_model()

    tets_dp = DataProcessingTest(test_data)
    null_present, cols_with_missing_values = tets_dp.is_null_present()
    test_datas = tets_dp.impute_missing_values(test_data, cols_with_missing_values)

    test_datas = encoder.transform(test_datas)
    test_datas = test_datas[cols_to_reserve]

    prediction = xgboost_model.predict(test_datas)
    submission = pd.DataFrame(
        {"customer_id": test_data["customer_id"], "credit_card_default": prediction}
    )
    submission
    submission.to_csv("Xgboostmeancodingwithfeaturesselection.csv")

    return train_data_copy, test_data



train_data, test_data = main()
train_data.to_csv(r"E:\Credit Card Default Risk\dataset\train_data.csv")

