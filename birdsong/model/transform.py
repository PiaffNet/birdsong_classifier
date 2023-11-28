import numpy as np
from tensorflow.keras.utils import to_categorical
from birdsong.utils import get_folders_labels, get_classes_labels_dict

def transfrom():
    """
    Clean the data
    """
    pass

def get_labels(folders_path: str)-> np.ndarray:
    """
    Get the labels
    """
    folders_labels = get_folders_labels(folders_path)
    classes_labels_numeric = list(get_classes_labels_dict(folders_labels).values())
    classes_labels_numeric_clean = list(set(classes_labels_numeric))
    return to_categorical(classes_labels_numeric_clean ,len(classes_labels_numeric_clean))
