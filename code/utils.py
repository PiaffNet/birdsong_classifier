import os

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
