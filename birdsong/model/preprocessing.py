import numpy as np
from birdsong.config import config

from sklearn.pipeline import make_pipeline, Pipeline

def preprocess_data(data: np.ndarray)-> np.ndarray:
    """
    Preprocess the data
    """
    data = np.log(data)
    return data
