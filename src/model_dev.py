from numpy.lib.function_base import gradient
import pandas as pd
import numpy as np
import logging

from scipy.sparse import data
from scipy.sparse.construct import random
from validate_data import DatavalidationTest

# import GrideSearchCV
from sklearn.model_selection import GridSearchCV
from data_ingestion import DataUtils
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import optuna
from sklearn import linear_model
from evaluate_models import Evaluation
from utils import Submission
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import catboost
from mlxtend.classifier import StackingClassifier


class Hyperparameters_Optimization:
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize_logistic_regression(self, trial):
        C = trial.suggest_loguniform("C", 1e-7, 10.0)
        solver = trial.suggest_categorical("solver", ("lbfgs", "saga"))
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        clf = LogisticRegression(C=C, solver=solver)
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)

        return val_accuracy

    def optimize_naive_bayes(self, trial):
        var_smoothing = trial.suggest_loguniform("var_smoothing", 1e-7, 10.0)
        clf = GaussianNB(var_smoothing=var_smoothing)
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_decisiontrees(self, trial):
        criterion = trial.suggest_categorical("criterion", ("gini", "entropy"))
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_svc(self, trial):
        C = trial.suggest_loguniform("C", 1e-7, 10.0)
        kernel = trial.suggest_categorical("kernel", ("rbf", "linear", "poly"))
        gamma = trial.suggest_loguniform("gamma", 1e-7, 10.0)
        clf = SVC(C=C, kernel=kernel, gamma=gamma)
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_randomforest(self, trial):
        logging.info("optimize_randomforest")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_adaboost(self, trial):
        logging.info("optimize_adaboost")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_gradientboosting(self, trial):
        logging.info("optimize_gradientboosting")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_xgboost(self, trial):
        logging.info("optimize_xgboost")
        param = {
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-7, 10.0),
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_uniform("colsample_bylevel", 0.5, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-7, 10.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-7, 10.0),
            "scale_pos_weight": trial.suggest_loguniform(
                "scale_pos_weight", 1e-7, 10.0
            ),
        }
        clf = xgb.XGBClassifier(**param)
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_catboost(self, trial):
        logging.info("optimize_catboost")
        train_data = catboost.Pool(self.x_train, label=self.y_train)
        test_data = catboost.Pool(self.x_test, label=self.y_test)

        param = {
            "loss_function": trial.suggest_categorical(
                "loss_function", ("Logloss", "CrossEntropy")
            ),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10.0),
            "max_bin": trial.suggest_int("max_bin", 200, 400),
            "subsample": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "learning_rate": trial.suggest_uniform("learning_rate", 0.006, 0.018),
            "n_estimators": trial.suggest_int("n_estimators", 1, 2000),
            "max_depth": trial.suggest_categorical("max_depth", [7, 10, 14, 16]),
            "random_state": trial.suggest_categorical("random_state", [24, 48, 2020]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 300),
        }

        clf = catboost.CatBoostClassifier(**param)
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy


class ModelTraining:
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def logistic_regression(self, fine_tuning=True):
        logging.info("Entered for training Logistic regression model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_logistic_regression, n_trials=100)
                trial = study.best_trial
                C = trial.params["C"]
                solver = trial.params["solver"]
                max_iter = trial.params["max_iter"]
                clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
                clf.fit(self.x_train, self.y_train)

                print(
                    "C : {} and solver : {} and max_iter : {}".format(
                        C, solver, max_iter
                    )
                )

                logging.info("Logistic Regression model fine tuned ands trained")
                return clf

            else:
                logging.info("Logistic Regression model is being trained")
                model = LogisticRegression(
                    C=0.00015200974256076671, solver="lbfgs", max_iter=641
                )
                model.fit(self.x_train, self.y_train)
                logging.info("Logistic Regression model trained")
                return model

        except Exception as e:
            logging.error("Error in training Logistic regression model")
            logging.error(e)
            return None

    def naive_bayes(self, fine_tuning=True):
        logging.info("Entered for training Naive Bayes model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_naive_bayes, n_trials=100)
                trial = study.best_trial
                var_smoothing = trial.params["var_smoothing"]
                print("var_smoothing : {}".format(var_smoothing))
                clf = GaussianNB(var_smoothing=var_smoothing)
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                model = GaussianNB(var_smoothing=7.127035668355718)
                model.fit(self.x_train, self.y_train)
                return model

        except Exception as e:
            logging.error("Error in training Naive Bayes model")
            logging.error(e)
            return None

    def decision_trees(self, fine_tuning=True):
        logging.info("Entered for training Decision Trees model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_decisiontrees, n_trials=100)
                trial = study.best_trial
                criterion = trial.params["criterion"]
                max_depth = trial.params["max_depth"]
                min_samples_split = trial.params["min_samples_split"]
                print("Best parameters : ", trial.params)
                clf = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                )
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                model = DecisionTreeClassifier(
                    criterion="entropy", max_depth=1, min_samples_split=7
                )

                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Decision Trees model")
            logging.error(e)
            return None

    def support_vector_classifier(self, fine_tuning=True):
        """ 
        not able to work on this coz of computation power! 
        """
        logging.info("Entered for training Support Vector Classifier model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_svc, n_trials=100)
                trial = study.best_trial
                C = trial.params["C"]
                kernel = trial.params["kernel"]
                gamma = trial.params["gamma"]
                clf = SVC(C=C, kernel=kernel, gamma=gamma)
                clf.fit(self.x_train, self.y_train)
                return clf
        except Exception as e:
            logging.error("Error in training Support Vector Classifier model")
            logging.error(e)
            return None

    def random_forest(self, fine_tuning=True):
        logging.info("Entered for training Random Forest model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_randomforest, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                min_samples_split = trial.params["min_samples_split"]
                print("Best parameters : ", trial.params)
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                )
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                model = RandomForestClassifier(
                    n_estimators=92, max_depth=19, min_samples_split=4
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Random Forest model")
            logging.error(e)
            return None

    def adaboost(self, fine_tuning=True):
        logging.info("Entered for training AdaBoost model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_adaboost, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                print("Best parameters : ", trial.params)
                clf = AdaBoostClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate
                )
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                model = AdaBoostClassifier(
                    n_estimators=143, learning_rate=0.2374269674908056
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training AdaBoost model")
            logging.error(e)
            return None

    def gradient_boosting(self, fine_tuning=True):
        logging.info("Entered for training Gradient Boosting model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_gradientboosting, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                clf = GradientBoostingClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate
                )
                clf.fit(self.x_train, self.y_train)
                return clf
        except Exception as e:
            logging.error("Error in training Gradient Boosting model")
            logging.error(e)
            return None

    def voting_classifier(self, fine_tuning=True):
        logging.info("Entered for training Voting Classifier model")
        try:
            if fine_tuning:
                clf = VotingClassifier(
                    estimators=[
                        (
                            "logistic_regression",
                            self.logistic_regression(fine_tuning=False),
                        ),
                        ("decision_trees", self.decision_trees(fine_tuning=False)),
                        ("random_forest", self.random_forest(fine_tuning=False)),
                        ("adaboost", self.adaboost(fine_tuning=False)),
                        (
                            "gradient_boosting",
                            self.gradient_boosting(fine_tuning=False),
                        ),
                        ("xgboost", self.xgboost(fine_tuning=False)),
                        ("catboost", self.catboost(fine_tuning=False)),
                    ],
                    voting="hard",
                )
                clf.fit(self.x_train, self.y_train)
                return clf
        except Exception as e:
            logging.error("Error in training Voting Classifier model")
            logging.error(e)
            return None

    def xgboost(self, fine_tuning=True):
        logging.info("Entered for training XGBoost model")
        try:
            if fine_tuning:
                hy_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hy_opt.optimize_xgboost, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                max_depth = trial.params["max_depth"]
                min_child_weight = trial.params["min_child_weight"]
                subsample = trial.params["subsample"]
                colsample_bytree = trial.params["colsample_bytree"]
                reg_alpha = trial.params["reg_alpha"]
                reg_lambda = trial.params["reg_lambda"]
                clf = XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                )
                print("Best parameters : ", trial.params)
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                params = {
                    "objective": "binary:logistic",
                    "use_label_encoder": True,
                    "base_score": 0.5,
                    "booster": "gbtree",
                    "colsample_bylevel": 1,
                    "colsample_bynode": 1,
                    "colsample_bytree": 0.9865465799558366,
                    "enable_categorical": False,
                    "gamma": 0,
                    "gpu_id": -1,
                    "importance_type": None,
                    "interaction_constraints": "",
                    "learning_rate": 0.1733839701849005,
                    "max_delta_step": 0,
                    "max_depth": 6,
                    "min_child_weight": 1,
                    "n_estimators": 73,
                    "n_jobs": 8,
                    "num_parallel_tree": 1,
                    "predictor": "auto",
                    "random_state": 0,
                    "reg_alpha": 8.531151528439326e-06,
                    "reg_lambda": 0.006678010524298995,
                    "scale_pos_weight": 1,
                    "subsample": 0.7761340636250333,
                    "tree_method": "exact",
                    "validate_parameters": 1,
                    "verbosity": None,
                }
                clf = XGBClassifier(**params)
                clf.fit(self.x_train, self.y_train)
                return clf

        except Exception as e:
            logging.error("Error in training XGBoost model")
            logging.error(e)
            return None

    def catboost(self, fine_tuning=True, best_trial=None):
        logging.info("Entered for training CatBoost model")
        if fine_tuning:
            hy_opt = Hyperparameters_Optimization(
                self.x_train, self.y_train, self.x_test, self.y_test
            )
            study = optuna.create_study(direction="maximize")
            study.optimize(hy_opt.optimize_catboost, n_trials=10)
            trial = study.best_trial

            max_depth = trial.params["max_depth"]
            l2_leaf_reg = trial.params["l2_leaf_reg"]
            max_bin = trial.params["max_bin"]
            bagging_fraction = trial.params["bagging_fraction"]
            learning_rate = trial.params["learning_rate"]
            loss_function = trial.params["loss_function"]
            n_estimators = trial.params["n_estimators"]
            random_state = trial.params["random_state"]
            min_data_in_leaf = trial.params["min_data_in_leaf"]

            clf = CatBoostClassifier(
                max_depth=max_depth,
                l2_leaf_reg=l2_leaf_reg,
                max_bin=max_bin,
                bagging_fraction=bagging_fraction,
                learning_rate=learning_rate,
                loss_function=loss_function,
                n_estimators=n_estimators,
                random_state=random_state,
                min_data_in_leaf=min_data_in_leaf,
            )

            clf.fit(self.x_train, self.y_train)
            return clf
        else:
            clf = CatBoostClassifier(
                loss_function="Logloss",
                l2_leaf_reg=0.005603859124543057,
                max_bin=332,
                learning_rate=0.0075037108414941255,
                n_estimators=1297,
                max_depth=16,
                random_state=24,
                min_data_in_leaf=94,
            )
            clf.fit(self.x_train, self.y_train)
            return clf

    def stacking(self):
        logging.info("Entered for stacking model")
        try:
            decision_tree = DecisionTreeClassifier(
                criterion="entropy", max_depth=1, min_samples_split=7
            )
            rf_tree = RandomForestClassifier(
                n_estimators=92, max_depth=19, min_samples_split=4
            )
            adaboost = AdaBoostClassifier(
                n_estimators=143, learning_rate=0.2374269674908056
            )
            params = {
                "objective": "binary:logistic",
                "use_label_encoder": True,
                "base_score": 0.5,
                "booster": "gbtree",
                "colsample_bylevel": 1,
                "colsample_bynode": 1,
                "colsample_bytree": 0.9865465799558366,
                "enable_categorical": False,
                "gamma": 0,
                "gpu_id": -1,
                "importance_type": None,
                "interaction_constraints": "",
                "learning_rate": 0.1733839701849005,
                "max_delta_step": 0,
                "max_depth": 6,
                "min_child_weight": 1,
                "n_estimators": 73,
                "n_jobs": 8,
                "num_parallel_tree": 1,
                "predictor": "auto",
                "random_state": 0,
                "reg_alpha": 8.531151528439326e-06,
                "reg_lambda": 0.006678010524298995,
                "scale_pos_weight": 1,
                "subsample": 0.7761340636250333,
                "tree_method": "exact",
                "validate_parameters": 1,
                "verbosity": None,
            }
            xgb_clf = XGBClassifier(**params)
            cat_clf = CatBoostClassifier(
                loss_function="Logloss",
                l2_leaf_reg=0.005603859124543057,
                max_bin=332,
                learning_rate=0.0075037108414941255,
                n_estimators=1297,
                max_depth=16,
                random_state=24,
                min_data_in_leaf=94,
            )
            lr = LogisticRegression()
            clf = StackingClassifier(
                classifiers=[decision_tree, rf_tree, adaboost, xgb_clf, cat_clf, lr],
                meta_classifier=lr,
            )
            clf.fit(self.x_train, self.y_train)
            return clf
        except Exception as e:
            logging.error("Error in stacking model")
            logging.error(e)
            return None


# data_utils = DataUtils()
# train_data, test_data = data_utils.load_data()
# data_process = DataProcessing(train_data)
# null_present, cols_with_missing_values = data_process.is_null_present()
# train_data = data_process.impute_missing_values(train_data, cols_with_missing_values)
# train_data = data_process.encode_categorical_data(train_data, ["customer_id", "name"])

# data_dev = DatavalidationTest(train_data)
# x_train, x_test, y_train, y_test = data_dev.dataset_development()
# model_train = ModelTraining(x_train, y_train, x_test, y_test)

# model = model_train.logistic_regression(fine_tuning=False)
# predictions = model.predict(x_test)
# predictions_prob = model.predict_proba(x_test)
# evaluation = Evaluation(y_test, predictions)
# TP, FP, TN, FN = evaluation.get_tpfptnfn()
# print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))
# evaluation.get_confusion_metrics()
# evaluation.get_precision_recall_score()
# evaluation.f1_score()
# evaluation.get_score()
# Submission_obj = Submission(model, test_data)
# Submission_obj.get_submission_csv("logreg.csv")

# naive_bayes_model = model_train.naive_bayes(fine_tuning=True)
# naive_bayes_modelpredictions = naive_bayes_model.predict(x_test)
# evaluation_naive = Evaluation(
#     y_test, naive_bayes_modelpredictions
# )
# TP, FP, TN, FN = evaluation_naive.get_tpfptnfn()
# print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))

# evaluation_naive.get_confusion_metrics()
# evaluation_naive.get_precision_recall_score()
# evaluation_naive.f1_score()
# Submission_obj = Submission(naive_bayes_model, test_data)
# Submission_obj.get_submission_csv("naivebayes.csv")


# decision_trees_model = model_train.decision_trees(fine_tuning=True)
# decision_trees_modelpredictions = decision_trees_model.predict(x_test)
# evaluation_decision_trees = Evaluation(y_test, decision_trees_modelpredictions)
# TP, FP, TN, FN = evaluation_decision_trees.get_tpfptnfn()
# print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))
# evaluation_decision_trees.get_confusion_metrics()
# evaluation_decision_trees.get_precision_recall_score()
# evaluation_decision_trees.f1_score()
# evaluation_decision_trees.get_score()
# Submission_obj = Submission(decision_trees_model, test_data)
# Submission_obj.get_submission_csv("decision_trees_model.csv")

# support_vector_classifier_model = model_train.support_vector_classifier(fine_tuning=True)
# support_vector_classifier_model_predict_prob = support_vector_classifier_model.predict_proba(x_test)
# support_vector_classifier_modelpredictions = support_vector_classifier_model.predict(x_test)
# evaluation_support_vector_classifier = Evaluation(y_test, support_vector_classifier_modelpredictions, support_vector_classifier_model_predict_prob)
# TP, FP, TN, FN = evaluation_support_vector_classifier.get_tpfptnfn()
# print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))
# evaluation_support_vector_classifier.get_confusion_metrics()
# evaluation_support_vector_classifier.get_precision_recall_score()
# evaluation_support_vector_classifier.f1_score()

# random_forest_model = model_train.random_forest(fine_tuning=False)
# random_forest_modelpredictions = random_forest_model.predict(x_test)
# evaluation_random_forest = Evaluation(y_test, random_forest_modelpredictions)
# TP, FP, TN, FN = evaluation_random_forest.get_tpfptnfn()
# print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))
# evaluation_random_forest.get_confusion_metrics()
# evaluation_random_forest.get_precision_recall_score()
# evaluation_random_forest.f1_score()
# evaluation_random_forest.get_score()
# Submission_obj = Submission(random_forest_model, test_data)
# Submission_obj.get_submission_csv("random_forest_model.csv")

# adaboost = model_train.adaboost(fine_tuning=False)
# adaboost_predictions = adaboost.predict(x_test)
# evaluation_adaboost = Evaluation(y_test, adaboost_predictions)
# TP, FP, TN, FN = evaluation_adaboost.get_tpfptnfn()
# print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))
# evaluation_adaboost.get_confusion_metrics()
# evaluation_adaboost.get_precision_recall_score()
# evaluation_adaboost.f1_score()
# evaluation_adaboost.get_score()
# Submission_obj = Submission(adaboost, test_data)
# Submission_obj.get_submission_csv("adaboosts.csv")
# adaboost.get_params()
# gradient_boosting = model_train.gradient_boosting(fine_tuning=True)
# gradient_boosting_predict_prob = gradient_boosting.predict_proba(x_test)
# gradient_boosting_predictions = gradient_boosting.predict(x_test)
# evaluation_gradient_boosting = Evaluation(
#     y_test, gradient_boosting_predictions, gradient_boosting_predict_prob
# )
# TP, FP, TN, FN = evaluation_gradient_boosting.get_tpfptnfn()
# print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))
# evaluation_gradient_boosting.get_confusion_metrics()
# evaluation_gradient_boosting.get_precision_recall_score()
# evaluation_gradient_boosting.f1_score()

# hy_opt = Hyperparameters_Optimization(x_train, y_train, x_test, y_test)
# study = optuna.create_study(direction="minimize")
# study.optimize(hy_opt.optimize_xgboost, n_trials=50)
# print("Number of finished trials:", len(study.trials))
# print("Best trial:", study.best_trial.params)
# xgboost = model_train.xgboost(fine_tuning=True)
# xgboost_predictions = xgboost.predict(x_test)
# evaluation_xgboost = Evaluation(y_test, xgboost_predictions)
# TP, FP, TN, FN = evaluation_xgboost.get_tpfptnfn()
# print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))
# evaluation_xgboost.get_confusion_metrics()
# evaluation_xgboost.get_precision_recall_score()
# evaluation_xgboost.f1_score()
# evaluation_xgboost.get_score()
# Submission_obj = Submission(adaboost, test_data)
# Submission_obj.get_submission_csv("xgboost.csv")
# xgboost.get_params()
# import joblib

# catboost = model_train.catboost(fine_tuning=False)
# catboost_model = joblib.load("catboost_model_1.pkl")
# catboost_predictions = catboost_model.predict(x_test)
# evaluation_catboost = Evaluation(y_test, catboost_predictions)
# TP, FP, TN, FN = evaluation_catboost.get_tpfptnfn()
# print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))
# evaluation_catboost.get_confusion_metrics()
# evaluation_catboost.get_precision_recall_score()
# evaluation_catboost.f1_score()
# evaluation_catboost.get_score()
# Submission_obj = Submission(catboost_model, test_data)
# Submission_obj.get_submission_csv("catboost_model.csv")


# voting_classifier = model_train.voting_classifier(fine_tuning=True)
# voting_classifier_predictions = voting_classifier.predict(x_test)
# evaluation_voting_classifier = Evaluation(y_test, voting_classifier_predictions)
# TP, FP, TN, FN = evaluation_voting_classifier.get_tpfptnfn()
# print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))
# evaluation_voting_classifier.get_confusion_metrics()
# evaluation_voting_classifier.get_precision_recall_score()
# evaluation_voting_classifier.f1_score()


# stacking_classificier = model_train.stacking()
# stacking_classificier_predictions = stacking_classificier.predict(x_test)
# stacking_classificier.predict_proba(x_test)
# evaluation_stacking_classificier = Evaluation(y_test, stacking_classificier_predictions)
# TP, FP, TN, FN = evaluation_stacking_classificier.get_tpfptnfn()
# print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))
# evaluation_stacking_classificier.get_confusion_metrics()
# evaluation_stacking_classificier.get_precision_recall_score()
# evaluation_stacking_classificier.f1_score()
# evaluation_stacking_classificier.get_score()
# Submission_obj = Submission(stacking_classificier, test_data)
# Submission_obj.get_submission_csv("stacking_model.csv")
# save the stacking classifier
# import joblib
# joblib.dump(stacking_classificier, "stacking_model_1.pkl")

