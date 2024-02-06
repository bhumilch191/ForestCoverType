import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import *

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

## Create Model Trainer Config class
@dataclass
class ModelTrainerConfig:
    train_model_path = os.path.join('artifacts', 'model.pkl')

## Create Model Trainer Class
class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            # list_clf = [('bagg', clf1), ('Ext', clf2), ('Hgrad', clf3), ('XGB', clf4), ('Grad', clf5), 
            #             ('KN', clf6), ('log_reg', clf7), ('svm', clf8), ('rand_forest', clf9)]
            
            models = {
                'bagg_extra_tree': BaggingClassifier(ExtraTreesClassifier(criterion='entropy', max_features='sqrt'), 
                                        n_estimators=10, verbose=1),
                'extra_tree': ExtraTreesClassifier(n_estimators=130, criterion='entropy', max_features='sqrt'),
                'hist_grad_boost': HistGradientBoostingClassifier(interaction_cst='pairwise', learning_rate=0.4),
                'xgb': XGBClassifier(learning_rate=0.2, max_depth=8, n_estimators=300),
                'grad': GradientBoostingClassifier(n_estimators=350, learning_rate=0.3, max_depth=5, 
                                                max_features='sqrt'),
                'kn': KNeighborsClassifier(algorithm='auto', leaf_size=10, n_neighbors=4, p=1, 
                                            weights='distance'),
                'log_reg': LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=10),
                'svc': SVC(),
                'random': RandomForestClassifier()
            }
                # 'voting_hard': VotingClassifier(estimators=list_clf, voting='hard'),
                # 'voting_soft': VotingClassifier(estimators=list_clf, voting='soft')

            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report: {model_report}')

            #To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, Score: {best_model_score}')
            
            save_object(file_path=self.model_trainer_config.train_model_path, obj=best_model)
        
        except Exception as e:
            logging.info('Error accured in Model Training')
            raise CustomException(e, sys)