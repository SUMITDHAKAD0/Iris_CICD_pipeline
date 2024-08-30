from sklearn.metrics import accuracy_score
import joblib
import logging
import pandas as pd
import os
from pydantic import BaseModel
from src.plot_utils import plot_model_performance

logger = logging.getLogger(__name__)

def save_predictions(predictions, y_test):
    os.makedirs('artifacts/predictions', exist_ok=True)
    prediction_df = pd.DataFrame({'True_Label': y_test, 'Prediction': predictions})
    prediction_df.to_csv('artifacts/predictions/predictions.csv', index=False)
    logger.info("Predictions saved.")


def make_predictions(X_test, y_test):
    logger.info("Starting prediction...")
    
    # # Load the model
    # model_path = 'artifacts/models/iris_model.pkl'
    # model = joblib.load(model_path)
    
    # Define the path to the model
    MODEL_PATH = "artifacts/models/iris_model.pkl"

    # Load the pre-trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # Make predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Save predictions
    save_predictions(predictions, y_test)
    
    # Plot and save model performance
    plot_model_performance(y_test, predictions)
    
    logger.info(f"Prediction completed with an accuracy of {accuracy:.4f}.")
    return predictions, accuracy


# Define a request model for input data
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def make_prediction(iris: IrisInput):
    """
    This function takes an IrisInput object, prepares the input data, 
    and returns the predicted class.
    """
    # Define the path to the model
    MODEL_PATH = "artifacts/models/iris_model.pkl"

    # Load the pre-trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # Prepare input data for prediction
    input_data = [[
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    ]]
    iris_species = {
        0: "Iris Setosa",
        1: "Iris Versicolor",
        2: "Iris Virginica"
    }
    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = int(prediction[0])
    output = iris_species[predicted_class]
    output = f"{output}({predicted_class})"
    return output

