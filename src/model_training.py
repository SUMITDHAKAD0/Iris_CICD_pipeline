from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging
from src.plot_utils import plot_feature_distributions

logger = logging.getLogger(__name__)

def model_training(X_train, y_train):
    logger.info("Starting model training...")
    
    # Plot and save feature distributions
    plot_feature_distributions(X_train)
    
    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = 'artifacts/models/iris_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model trained and saved at {model_path}.")
    
    return model
