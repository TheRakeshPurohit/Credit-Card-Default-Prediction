import numpy as np
import pandas as pd
from application_logger import CustomApplicationLogger


class TestUtils:
    def __init__(self) -> None:
        pass

    def test_application_logger(self):
        """
        Input: 
            None
        Output: 
            Logs the message to the file object 
        """
        logger = CustomApplicationLogger()
        file_obj = open(r"E:\QnAMedical\Credit Card Fraud\logs\logTest.txt", "a+")
        logger.logger(file_obj, "This is a test message")
        file_obj.close()


if __name__ == "__main__":
    test_utils = TestUtils()
    test_utils.test_application_logger()
