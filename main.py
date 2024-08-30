import logging
import os
import sys
from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_training import model_training
from src.make_predictions import make_predictions

# Configure logging
log_info = '[%(asctime)s: %(levelname)s: %(module)s: %(message)s]'
os.makedirs('artifacts/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format=log_info, handlers=[
    logging.FileHandler('artifacts/logs/pipeline.log'),
    logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger()

def main_pipeline():
    logger.info("Pipeline execution started.")
    
    # Run the pipeline
    X, y = data_ingestion()
    X_train, X_test, y_train, y_test, scaler = data_preprocessing(X, y)
    model = model_training(X_train, y_train)
    predictions, accuracy = make_predictions(X_test, y_test)
    
    logger.info("Pipeline execution completed.")

# if __name__ == "__main__":
#     main_pipeline()

