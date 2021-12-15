import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    log_loss,
)
import logging
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluation:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def get_tpfptnfn(self):
        logging.info("Entered the method named get_tpfptnfn")
        try:
            TP = np.sum(np.logical_and(self.y_pred == 1, self.y_true == 1))

            # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
            TN = np.sum(np.logical_and(self.y_pred == 0, self.y_true == 0))

            # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
            FP = np.sum(np.logical_and(self.y_pred == 1, self.y_true == 0))

            # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
            FN = np.sum(np.logical_and(self.y_pred == 0, self.y_true == 1))
            logging.info("Exited the method named get_tpfptnfn")
            return TP, FP, TN, FN
        except Exception as e:
            logging.error("Exception occured in method named get_tpfptnfn: " + str(e))
            return None

    def get_confusion_metrics(self):
        logging.info("Entered the method named get_confusion_metrics")
        try:
            confusion = confusion_matrix(self.y_true, self.y_pred)
            plt.figure(figsize=(20, 4))
            labels = [0, 1]
            cmap = sns.light_palette("blue")
            plt.subplot(1, 3, 1)
            sns.heatmap(
                confusion,
                annot=True,
                cmap=cmap,
                fmt=".3f",
                xticklabels=labels,
                yticklabels=labels,
            )
            plt.xlabel("Predicted Class")
            plt.ylabel("Original Class")
            plt.title("Confusion matrix")

        except Exception as e:
            logging.error("Error occured: " + str(e))
            return None

    def get_precision_recall_score(self):
        logging.info("Entered the method get_precision_recall_score")
        try:
            precision_sc = precision_score(self.y_true, self.y_pred)
            recall_fc = recall_score(self.y_true, self.y_pred)
            logging.info("Exited get_precision_recall_score method")
            return precision_sc, recall_fc
        except Exception as e:
            logging.error(
                "Exception occured in method named get_precision_recall_score: "
                + str(e)
            )
            return None

    def f1_score(self):
        logging.info("Entered the method f1_score")
        try:
            f1_scores = f1_score(self.y_true, self.y_pred)
            logging.info("Exited get_precision_recall_score method")
            return f1_scores
        except Exception as e:
            logging.error(
                "Exception occured in method named get_precision_recall_score: "
                + str(e)
            )
            return None

    def get_score(self):
        logging.info("Entered the method get_score")
        try:
            score = 100 * (f1_score(self.y_true, self.y_pred, average="macro"))
            logging.info("Exited the method get_score")
            return score
        except Exception as e:
            logging.error("Exception occured in method named get_score: " + str(e))
            return None

    def evaluate_model(self):
        logging.info("Entered the method named evaluate_model")
        try:
            TP, FP, TN, FN = self.get_tpfptnfn()
            print("TP : {} and FP : {} and TN : {} and FN : {}".format(TP, FP, TN, FN))
            precision_sc, recall_fc = self.get_precision_recall_score()
            print(
                "Precision Score : {} and Recall Score : {}".format(
                    precision_sc, recall_fc
                )
            )
            f1_scores = self.f1_score()
            print("F1 Score : {}".format(f1_scores))
            score = self.get_score()
            print("Score : {}".format(score))
            self.get_confusion_metrics()
            logging.info("Exited the method named evaluate_model")
        except Exception as e:
            logging.error("Exception occured in method named evaluate_model: " + str(e))
            return None
