"""
This file contains methods used to transform the data
before preprocessing.
"""
import numpy as np
from tensorflow.keras.utils import to_categorical, image_dataset_from_directory
from tensorflow.data.experimental import cardinality
from birdsong.config import config
from birdsong.utils import get_folders_labels, get_classes_labels_dict


def get_train_data_set():
    VALIDATION_SPLIT = 0.3
    BATCH_SIZE = 32
    IMAGE_SIZE = (64, 376)
    SHUFLE_VALUE = True
    RANDOM_SEED = 42
    TEST_SIZE_PART = 1

    train_ds = image_dataset_from_directory(
                            config.OUPTUT_FOLDER_PATH,
                            validation_split=VALIDATION_SPLIT,
                            subset="training",
                            seed=RANDOM_SEED,
                            image_size=IMAGE_SIZE,
                            batch_size=BATCH_SIZE,
                            color_mode = "grayscale",
                            shuffle = SHUFLE_VALUE)
    return train_ds

def get_validation_test_data_sets():
    VALIDATION_SPLIT = 0.3
    BATCH_SIZE = 32
    IMAGE_SIZE = (64, 376)
    SHUFLE_VALUE = True
    RANDOM_SEED = 42
    TEST_SIZE_PART = 1
    val_ds = image_dataset_from_directory(
                            config.OUPTUT_FOLDER_PATH,
                            validation_split=VALIDATION_SPLIT,
                            subset="validation",
                            seed=RANDOM_SEED,
                            image_size=IMAGE_SIZE,
                            batch_size=BATCH_SIZE,
                            color_mode = "grayscale",
                            shuffle = SHUFLE_VALUE)
    val_batches = cardinality(val_ds)
    test_ds = val_ds.take((TEST_SIZE_PART*val_batches) // 3)
    val_ds = val_ds.skip((TEST_SIZE_PART*val_batches) // 3)
    return val_ds, test_ds


def get_labels(folders_path: str)-> np.ndarray:
    """
    Get the labels
    """
    folders_labels = get_folders_labels(folders_path)
    classes_labels_numeric = list(get_classes_labels_dict(folders_labels).values())
    classes_labels_numeric_clean = list(set(classes_labels_numeric))
    return to_categorical(classes_labels_numeric_clean ,len(classes_labels_numeric_clean))
