"""
this file contains the functions / class
describing the Deep Learning mmodels used in the classification
"""
import numpy as np
from typing import Tuple
from birdsong.config import config
from tensorflow.keras import Sequential, Model, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def initialize_model(input_shape: tuple, num_classes: int)-> Model:
    """
    Initialize the model
    """
    model = Sequential()


    return model

def compile_model(model: Model, learning_rate=0.0005)-> Model:
    """
    Compile the model
    """
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model compiled")
    return model

def train_model(model: Model,
                train_data: np.ndarray,
                train_labels: np.ndarray,
                batch_size: int,
                epochs: int,
                patience: int,
                validation_data=None,
                validation_split=0.3)-> Tuple[Model, dict]:
    """
    Train the model
    """
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=patience)

    history = model.fit(
        train_data,
        train_labels,
        validation_data=validation_data,
        validation_split = validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1)

    print("Model trained")
    return model, history

def evaluate_model(model: Model,
                   test_data: np.ndarray,
                   test_labels: np.ndarray,
                   batch_size: int)-> Tuple[Model, dict]:
    """
    Evaluate the model
    """
    if model is None:
        raise ValueError(" No model to evaluate.")
        return None, None

    metrics  = model.evaluate(test_data,
                              test_labels,
                              batch_size=batch_size,
                              verbose=0,
                              return_dict = True)
    print("Model evaluated")
    return metrics
