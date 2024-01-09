import pytest
import os
import glob
from birdsong.config import config
from birdsong import DATA_TRAIN_AUDIO_PATH, DATA_RAW_PATH

def test_data_output_folder_defined():
    """basic test of config file, birdsong.config should be a valid module.
    Can be adapted to test specific config parameters by user.
    """
    assert config.DATA_OUTPUT_FOLDER != "" and type(config.DATA_OUTPUT_FOLDER ) is str,\
        "DATA_OUTPUT_FOLDER should be defined in _config.yml and should be a string"

def test_model_name():
    assert config.MODEL_NAME in ["baseline_archi_0","baseline_archi_1","BirdNetBlock_a","BirdNetBlock_b"],\
        "MODEL_NAME should be defined in _config.yml and chosen from a list [baseline_archi_0,baseline_archi_1,BirdNetBlock_a,BirdNetBlock_b]"

def test_model_save_path():
    assert (config.MODEL_SAVE_PATH != "") and (type(config.MODEL_SAVE_PATH) is str),\
        "MODEL_SAVE_PATH should be defined in _config.yml and should be a string"

def test_model_checkpoint_path():
    assert config.CHECKPOINT_FOLDER_PATH != "" and type(config.CHECKPOINT_FOLDER_PATH) is str,\
        "CHECKPOINT_FOLDER_PATH should be defined in _config.yml and should be a string"

def test_raw_data_directory():
    """basic test of project data.The raw data directory should be present,
    the directory train audio should be in raw data.
    DATA_RAW_PATH, DATA_TRAIN_AUDIO_PATH are defined in birdsong/__init__.py
    """
    assert os.path.exists(DATA_RAW_PATH) == True, "raw data directory should be present"
    assert os.path.exists(DATA_TRAIN_AUDIO_PATH) == True, "train audio directory should be in raw data"

def test_train_audio_content():
    """basic test of project data train.The raw data directory should be present,
    the directory train audio should be not empty and each folder in train audio directory
    should contains only mp3 files.
    DATA_RAW_PATH, DATA_TRAIN_AUDIO_PATH are defined in birdsong/__init__.py
    """
    from birdsong.utils import get_folders_labels

    if os.path.exists(DATA_TRAIN_AUDIO_PATH):
        # check if train audio directory is not empty
        assert len(get_folders_labels(DATA_TRAIN_AUDIO_PATH)) > 0, "train audio directory is empty"

        # check if train audio directory contains only mp3 files

        all_train_folder_labels  = get_folders_labels(DATA_TRAIN_AUDIO_PATH)
        for folder_label in all_train_folder_labels:
            all_files_nb = len(glob.glob(os.path.join(DATA_TRAIN_AUDIO_PATH,folder_label,"*")))
            mp3_files_nb = len(glob.glob(os.path.join(DATA_TRAIN_AUDIO_PATH,folder_label,"*.mp3")))
            assert all_files_nb == mp3_files_nb, f"folder {folder_label} contains non mp3 files"
