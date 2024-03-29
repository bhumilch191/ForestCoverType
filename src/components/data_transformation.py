import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.pipeline import Pipeline ## pipelines

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


## Create A Data Transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


## Create A Data Transformation class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformation_object(self):
         try:
            logging.info('Data Transformation initiated')
            logging.info('Pipeline Initiated')
            preprocessor=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info('Pipeline Completed')
            return preprocessor

         except Exception as e:  
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
         
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Cover_Type'
            drop_columns = [target_column_name, 'Soil_Type7', 'Soil_Type15']

            ## features into independent and dependent features
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]-1

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]-1

            ## apply the transformation
            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info('Processsor pickle in created and saved')
            return(
                train_arr,
                test_arr,
                # self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info("Exception accured in the initiate_datatransformation")
            raise CustomException(e,sys)