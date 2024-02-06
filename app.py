from flask import Flask, render_template, request, send_file
from src.exception import CustomException
from src.logger import logging
import sys

from src.pipelines.training_pipeline import TraininingPipeline
from src.pipelines.prediction_pipeline import PredictPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to my application"

@app.route("/train")
def train_route():
    try:
        train_pipeline = TraininingPipeline()
        train_pipeline.start_model_train()
        return "Training Completed."
    
    except Exception as e:
        logging.info("Error accured in train_rout")
        raise CustomException(e,sys)

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':
            # it is a object of prediction pipeline
            prediction_pipeline = PredictPipeline(request)

            #now we are running this run pipeline method
            prediction_file_detail = prediction_pipeline.run_pipeline()

            logging.info("prediction completed. Downloading prediction file.")
            logging.info(f"prediction_file_path: {prediction_file_detail.prediction_file_path}")

            try:
                return send_file(prediction_file_detail.prediction_file_path,
                                 download_name= prediction_file_detail.pred_file_name,
                                 as_attachment= True)
            except Exception as e:
                logging.info('Error accured in send_file')
                raise CustomException(e, sys)
        else:
            return render_template('upload_file.html')
        
    except Exception as e:
        logging.info("Error in upload function")
        raise CustomException(e,sys)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)