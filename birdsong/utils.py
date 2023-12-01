import os
import numpy as np
import pandas as pd
from typing import Tuple

def create_folder_if_not_exists(folder_name:str)->None:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✅ Folder {folder_name} created")

def get_folders_labels(path:str)->list:
    """
    Get the list of folders in a given path, except the ones starting with a dot '.'
    """
    folders_labels = os.listdir(path)
    return [label for label in folders_labels if not label.startswith(".")]


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

def read_prediction(model_predictions: np.ndarray, class_names)->pd.DataFrame:
    (n_rows, n_columns) = model_predictions.shape
    score_pred = [
        [
        np.argmax(model_predictions[n]),
        class_names[np.argmax(model_predictions[n])],
        "this song most likely belongs to {} with a {:.2f} percent confidence."\
        .format(class_names[np.argmax(model_predictions[n])], 100 * np.max(model_predictions[n]))
        ] for n in range(n_rows)
    ]

    return pd.DataFrame(score_pred,
                        columns=["prediction", "class", "description"])
