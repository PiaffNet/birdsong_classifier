"""
this file contains the functions / class
describing the Deep Learning mmodels used in the classification
"""
import os
import time
import glob
import numpy as np
from typing import Tuple
from tensorflow import keras
from birdsong.config import config
from birdsong.model.model_collections import get_model_architecture
from birdsong.utils import create_folder_if_not_exists
from birdsong.audiotransform.to_image import AudioPreprocessor
from birdsong import PARENT_BASE_PATH

def initialize_model(model_call_label : str, input_shape: tuple, num_classes: int)-> keras.Model:
    """
    Initialize the model
    """
    model = get_model_architecture(model_call_label, num_classes, input_shape)

    return model

def compile_model(model: keras.Model)-> keras.Model:
    """
    Compile the model
    """
    adam = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("✅ Model compiled")
    return model

def train_model(model: keras.Model,
                train_data,
                validation_data,
                epochs=1000)-> Tuple[keras.Model, dict]:
    """
    Train the model
    """
    print("Training user selected model...")

    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=config.PATIENCE)

    print("✅ Early stopping callback created")

    # Create a callback that saves the model's weights
    checkpoint_dir = os.path.join(PARENT_BASE_PATH,config.CHECKPOINT_FOLDER_PATH)
    create_folder_if_not_exists(checkpoint_dir)

    cp_callback = save_model_checkpoints(checkpoint_dir)
    print("✅ Checkpoint callback created")

    #override epocs number
    try:
        epochs = config.EPOCHS
    except:
        epochs = epochs

    print("epochs number : ", epochs)
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=[es,cp_callback],
        verbose=1)

    print("✅ Model trained")
    return model, history

def continue_training_model(path : str, from_checkpoint : bool , model_name : str):

    '''
    retrain a model either from a model or a checkpoint's weights.

    IF 'from_checkpoint' --> True : 'path' must be to checkpoint folder and
    model_name must be one the available models in model_collections.py.

    ELSE 'from_checkpoint' --> False : 'path' must be to model and model_name can be empty

    '''

    if from_checkpoint:
        try:
            checkpoint_dir = os.path.join(PARENT_BASE_PATH,config.CHECKPOINT_FOLDER_PATH)
            checkpoint_path = os.path.join(checkpoint_dir,"cp-best.ckpt")
            model = compile_model(initialize_model(model_name))
            model.load_weights(checkpoint_path)
            model, history = train_model(model)
        except:
            raise ValueError("❌ No checkpoint found")

    else :
        model = load_model(path)
        model, history = train_model(model)

    return model, history


def evaluate_model(model: keras.Model,
                   test_data)-> Tuple[keras.Model, dict]:
    """
    Evaluate the model
    """
    if model is None:
        raise ValueError("❌ No model to evaluate !")

    metrics  = model.evaluate(test_data,
                              verbose=0,
                              return_dict = True)
    print("✅ Model evaluated")
    return metrics

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """
    model_save_dir = os.path.join(PARENT_BASE_PATH,config.MODEL_SAVE_PATH)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(model_save_dir, f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")
    return None

def load_model()-> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    Return None (but do not Raise) if no model is found

    """
    if config.MODEL_TARGET == "local":

        print(f"\nLoad latest model from local registry...")

        model_save_dir = os.path.join(PARENT_BASE_PATH,config.MODEL_SAVE_PATH)

        # Get the latest model version name by the timestamp on disk
        local_model_paths = glob.glob(f"{model_save_dir}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(f"\nLoad latest model from disk...")

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model
    else:
        return None

def predict_model(model: keras.Model, data_to_predict):
    audio_processor = AudioPreprocessor()
    signal_processed = audio_processor.preprocess_audio_array(data_to_predict)
    signal_tensor = np.expand_dims(signal_processed, axis=0)
    predictions = model.predict(signal_tensor)
    return predictions

def save_model_checkpoints(checkpoint_folder_path, save_freq ='epoch'):

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = os.path.join(checkpoint_folder_path, config.MODEL_NAME,"cp-best.ckpt")

    # Create a callback that saves the model's weights every epoch and keeps only the best one
    cp_callback = keras.callbacks.ModelCheckpoint(
         filepath=checkpoint_path,
         verbose=1,
         save_weights_only=True,
         save_freq= save_freq, save_best_only = True,
         monitor ='val_loss', mode = 'min')

    return cp_callback
