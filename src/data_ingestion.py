import pandas as pd
from sklearn.datasets import load_iris
import logging

logger = logging.getLogger(__name__)

def data_ingestion():
    logger.info("Starting data ingestion...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    
    logger.info("Data successfully loaded.")
    return X, y
