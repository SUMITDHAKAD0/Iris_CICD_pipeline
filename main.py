import logging
import os
from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_training import model_training
from src.make_predictions import make_predictions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/pipeline.log')
logger.addHandler(file_handler)

if __name__ == "__main__":
    logger.info("Pipeline execution started.")
    
    # Run the pipeline
    X, y = data_ingestion()
    X_train, X_test, y_train, y_test, scaler = data_preprocessing(X, y)
    model = model_training(X_train, y_train)
    predictions, accuracy = make_predictions(X_test, y_test)
    
    logger.info("Pipeline execution completed.")
