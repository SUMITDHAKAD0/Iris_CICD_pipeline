from sklearn.metrics import accuracy_score
import joblib
import logging
import pandas as pd
import os
from src.plot_utils import plot_model_performance

logger = logging.getLogger(__name__)

def save_predictions(predictions, y_test):
    os.makedirs('predictions', exist_ok=True)
    prediction_df = pd.DataFrame({'True_Label': y_test, 'Prediction': predictions})
    prediction_df.to_csv('predictions/predictions.csv', index=False)
    logger.info("Predictions saved.")


def make_predictions(X_test, y_test):
    logger.info("Starting prediction...")
    
    # Load the model
    model_path = 'models/iris_model.pkl'
    model = joblib.load(model_path)
    
    # Make predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Save predictions
    save_predictions(predictions, y_test)
    
    # Plot and save model performance
    plot_model_performance(y_test, predictions)
    
    logger.info(f"Prediction completed with an accuracy of {accuracy:.4f}.")
    return predictions, accuracy
