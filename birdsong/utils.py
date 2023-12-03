import os
import numpy as np
import pandas as pd
import PIL.Image as Image
from typing import Tuple
from tabulate import tabulate


def save_train_history_as_csv(history: dict, path)->None:
    """
    Save the train history as a csv file
    """
    df = pd.DataFrame(history)
    df.to_csv(path, index=False)
    return None


def get_image_shape(image_path:str):
    """simple function to get the image size"""
    image = Image.open(image_path)
    return np.array(image).shape


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



def read_prediction(model_predictions: np.ndarray, class_names)->pd.DataFrame:
    (n_rows, n_columns) = model_predictions.shape
    score_pred = [
        [
        np.argmax(model_predictions[n]),
        class_names[np.argmax(model_predictions[n])],
        "song belongs to {} with a {:.2f}% confidence."\
        .format(class_names[np.argmax(model_predictions[n])], 100 * np.max(model_predictions[n]))
        ] for n in range(n_rows)
    ]
    if n_rows > 10:
        print('10 first predictions:')
        print(tabulate(score_pred[0:10], headers=["prediction", "class", "description"], tablefmt='fancy_grid'))
    else:
        print('here your prediction:')
        print(tabulate(score_pred[0], headers=["prediction", "class", "description"], tablefmt='fancy_grid'))
    return pd.DataFrame(score_pred, columns=["prediction", "class", "description"])
