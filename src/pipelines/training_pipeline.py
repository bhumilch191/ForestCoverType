import os,sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingistion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TraininingPipeline:
    
    def start_model_train(self):
        try:
            obj = DataIngestion()
            train_data_path, test_data_path = obj.initiate_data_ingestion()
            print(train_data_path,test_data_path)
            
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(train_arr, test_arr)

        except Exception as e:
            logging.info("Error accured in training pipeline")
            raise CustomException(e, sys)