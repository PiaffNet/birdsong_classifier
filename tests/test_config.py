import pytest
import os
import glob


def test_birdsong_config():
    """basic test of config file, birdsong.config should be a valid module.
    Can be adapted to test specific config parameters by user.
    """

    from birdsong.config import config

    assert config.DATA_OUTPUT_FOLDER != "" and type(config.DATA_OUTPUT_FOLDER ) is str

    assert config.SAMPLING_RATE == 48000
    assert config.STFT_NUMBER_SAMPLES == 512
    assert config.HOP_OVERLAP == 0.25
    assert config.N_MELS == 64
    assert config.HTK == 1750
    assert config.F_MIN == 150
    assert config.F_MAX == 15000

    assert config.MODEL_NAME in ["baseline_archi_0","baseline_archi_1","BirdNetBlock_a","BirdNetBlock_b"]
    assert (config.MODEL_SAVE_PATH != "") and (type(config.MODEL_SAVE_PATH) is str)
    assert config.CHECKPOINT_FOLDER_PATH != "" and type(config.CHECKPOINT_FOLDER_PATH) is str
    assert type(config.RANDOM_SEED) is int
    assert config.VALIDATION_SPLIT > 0.2 and config.VALIDATION_SPLIT <= 0.3
    assert config.BATCH_SIZE > 0 and type(config.BATCH_SIZE) is int
    assert config.LEARNING_RATE < 0.1 and config.LEARNING_RATE > 0
    assert type(config.PATIENCE) is int
    if 'EPOCHS' in config.__dict__.keys():
        assert type(config.EPOCHS) is int


def test_raw_data_directory():
    """basic test of project data.The raw data directory should be present,
    the directory train audio should be in raw data.
    DATA_RAW_PATH, DATA_TRAIN_AUDIO_PATH are defined in birdsong/__init__.py
    """

    from birdsong import DATA_TRAIN_AUDIO_PATH, DATA_RAW_PATH

    assert os.path.exists(DATA_RAW_PATH) == True
    assert os.path.exists(DATA_TRAIN_AUDIO_PATH) == True

def test_train_audio_content():
    """basic test of project data train.The raw data directory should be present,
    the directory train audio should be not empty and each folder in train audio directory
    should contains only mp3 files.
    DATA_RAW_PATH, DATA_TRAIN_AUDIO_PATH are defined in birdsong/__init__.py
    """

    from birdsong import DATA_TRAIN_AUDIO_PATH, DATA_RAW_PATH
    from birdsong.utils import get_folders_labels

    if os.path.exists(DATA_TRAIN_AUDIO_PATH):
        # check if train audio directory is not empty
        assert len(get_folders_labels(DATA_TRAIN_AUDIO_PATH)) > 0

        # check if train audio directory contains only mp3 files

        all_train_folder_labels  = get_folders_labels(DATA_TRAIN_AUDIO_PATH)
        for folder_label in all_train_folder_labels:
            all_files_nb = len(glob.glob(os.path.join(DATA_TRAIN_AUDIO_PATH,folder_label,"*")))
            mp3_files_nb = len(glob.glob(os.path.join(DATA_TRAIN_AUDIO_PATH,folder_label,"*.mp3")))
            assert all_files_nb == mp3_files_nb
