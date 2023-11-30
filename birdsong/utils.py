import os
import numpy as np
from typing import Tuple

def create_folder_if_not_exists(folder_name:str)->None:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder {folder_name} created")

def get_folders_labels(path:str)->list:
    """
    Get the list of folders in a given path, except the ones starting with a dot '.'
    """
    folders_labels = os.listdir(path)
    return [label for label in folders_labels if not label.startswith(".")]

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def get_classes_labels_dict(folders_labels:list)->dict:
    """
    Get the dictionary of classes labels from a list of folders labels
    """
    return {label: i for i, label in enumerate(folders_labels)}


def train_validation_split(data:np.ndarray,
                     labels:np.ndarray,
                     test_size:float)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into train and test sets
    """
    data_train = None
    labels_train = None
    data_val = None
    labels_val = None

    return data_train, labels_train, data_val, labels_val
