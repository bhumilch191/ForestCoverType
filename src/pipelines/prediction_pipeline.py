import sys
from flask import request
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from dataclasses import dataclass

## Create prediction pipeline config
@dataclass
class PredictionPipelineConfig:
    pred_output_dirname: str = "predictions"
    pred_file_name = "predicted_file.csv"
    prediction_file_path:str = os.path.join(pred_output_dirname, pred_file_name)


class PredictPipeline:
    def __init__(self, request: request):
        self.prediction_pipeline_config = PredictionPipelineConfig()
        self.request = request
        
    
    def save_input_files(self)-> str:

        """
            Method Name :   save_input_files
            Description :   This method saves the input file to the prediction artifacts directory. 
            
            Output      :   input dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            #creating the file
            pred_file_input_dir = "prediction_art"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files.get('file')
            if input_csv_file:
                pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
                logging.info(f'Prediction file path: {pred_file_path} and file name: {input_csv_file.filename}')
                
                input_csv_file.save(pred_file_path)
                logging.info("")
                return pred_file_path
            else:
                logging.error('No file provided in the request.')
                return None
            
        except Exception as e:
            logging.info("Error accured in given file for prediction.")
            raise CustomException(e,sys)
        

    def predict(self,features):
        try:
            logging.info("Start Model Prediction")
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return pred
            
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
    
    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):

        """
            Method Name :   get_predicted_dataframe
            Description :   this method returns the dataframw with a new column containing predictions

            
            Output      :   predicted dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
   
        try:
            logging.info(f'input df path: {input_dataframe_path}')
            prediction_column_name : str = "Cover_Type"
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            input_dataframe = input_dataframe.drop(labels=['Soil_Type7', 'Soil_Type15'], axis=1)
            
            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            logging.info("predictions completed.")
            output_col = input_dataframe['Cover_Type']
            logging.info(f'Input dataframe {output_col}')

            os.makedirs(self.prediction_pipeline_config.pred_output_dirname, exist_ok= True)

            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)
            logging.info(f'Successfuly save predicted data at {self.prediction_pipeline_config.prediction_file_path}')
            
        except Exception as e:
            logging.info(f"Error accured in get_predicted_dataframe")
            raise CustomException(e, sys) from e
        
    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            # input_csv_path = 'data/test_new.csv'
            if input_csv_path:
                self.get_predicted_dataframe(input_csv_path)
                return self.prediction_pipeline_config
            else:
                return None

        except Exception as e:
            logging.exception("Error occurred in run pipeline.")
            raise CustomException(e, sys)  
        