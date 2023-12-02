"""
This file contains methods used to transform the data
before preprocessing.
"""
import os
import numpy as np
from tensorflow import keras
from tensorflow.data.experimental import cardinality
from birdsong.config import config
from birdsong.utils import get_folders_labels, get_classes_labels_dict
from birdsong.audiotransform.to_image import AudioPreprocessor


def get_train_data_set():

    processed_info = AudioPreprocessor()
    IMAGE_SIZE = (64, 376)
    SHUFLE_VALUE = True



    train_ds = keras.utils.image_dataset_from_directory(
                            processed_info.output_folder,
                            validation_split=config.VALIDATION_SPLIT,
                            subset="training",
                            seed=config.RANDOM_SEED,
                            image_size=IMAGE_SIZE,
                            batch_size=config.BATCH_SIZE,
                            color_mode = "grayscale",
                            shuffle = SHUFLE_VALUE)
    return train_ds

def get_validation_test_data_sets():

    processed_info = AudioPreprocessor()
    IMAGE_SIZE = (64, 376)
    SHUFLE_VALUE = True
    val_ds = keras.utils.image_dataset_from_directory(
                            processed_info.output_folder,
                            validation_split=config.VALIDATION_SPLIT,
                            subset="validation",
                            seed=config.RANDOM_SEED,
                            image_size=IMAGE_SIZE,
                            batch_size=config.BATCH_SIZE,
                            color_mode = "grayscale",
                            shuffle = SHUFLE_VALUE)
    val_batches = cardinality(val_ds)
    test_ds = val_ds.take((int(config.TEST_SIZE_PART) * val_batches) // 3)
    val_ds = val_ds.skip((int(config.TEST_SIZE_PART) * val_batches) // 3)
    return val_ds, test_ds


def get_labels(folders_path: str)-> np.ndarray:
    """
    Get the labels
    """
    folders_labels = get_folders_labels(folders_path)
    classes_labels_numeric = list(get_classes_labels_dict(folders_labels).values())
    classes_labels_numeric_clean = list(set(classes_labels_numeric))
    return keras.utils.to_categorical(classes_labels_numeric_clean ,len(classes_labels_numeric_clean))
