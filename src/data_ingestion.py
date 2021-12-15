import logging
import pandas as pd


class DataIngestion:
    def __init__(self, training_file, testing_file, sample_submission_file) -> None:
        self.training_file = training_file
        self.testing_file = testing_file
        self.sample_submission_file = sample_submission_file

    def get_data(self):
        logging.info("Entered the phase to load the data")
        try:
            logging.info("Reading training data")
            training_data = pd.read_csv(self.training_file)
            logging.info("Successfully loaded training data")
            logging.info("Reading testing data")
            testing_data = pd.read_csv(self.testing_file)
            logging.info("Successfully loaded testing data")
            return training_data, testing_data
        except Exception as e:
            logging.error("Error while loading the data")
            raise e

    def get_sample_submision(self):
        logging.info("Reading sample submission data")
        try:
            sample_submission_data = pd.read_csv(self.sample_submission_file)
            logging.info("Successfully loaded sample submission data")
            return sample_submission_data

        except Exception as e:
            logging.error("Error while loading the sample submission data")
            raise e


class DataUtils: 
    def __init__(self) -> None:
         pass 

    @staticmethod
    def load_data():  

        training_path = r"E:\Hackathon\Credit Card Default Risk\dataset\train.csv" 
        testing_path  = r"E:\Hackathon\Credit Card Default Risk\dataset\test.csv" 
        sample_file_path = r"E:\Hackathon\Credit Card Default Risk\dataset\sample_submission.csv" 
        
        data_ingest = DataIngestion(training_path, testing_path, sample_file_path) 
        training_data, testing_data = data_ingest.get_data() 

        return training_data, testing_data 
