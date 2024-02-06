import logging
import os
from datetime import datetime

logfile_name = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'
log_path = os.path.join(os.getcwd(),"logs",logfile_name)
os.makedirs(log_path,exist_ok=True)

logfile_path = os.path.join(log_path,logfile_name)

logging.basicConfig(
    filename=logfile_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)