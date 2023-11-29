"""
This file contains functions for preprocessing the data
"""
import numpy as np
from birdsong.config import config

from sklearn.pipeline import make_pipeline, Pipeline

"""TODO: Add docstrings
    move preprocessing to audiotransform module
"""

def preprocess_data(data: np.ndarray)-> np.ndarray:
    """
    Preprocess the data
    """
    data = np.log(data)
    return data
