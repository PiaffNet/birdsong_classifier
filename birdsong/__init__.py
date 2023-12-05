import os

BASE_PATH = os.path.dirname(__file__)
PARENT_BASE_PATH = os.path.dirname(BASE_PATH)
DATA_RAW_PATH = os.path.join(PARENT_BASE_PATH,'raw_data')
DATA_TRAIN_AUDIO_PATH = os.path.join(DATA_RAW_PATH,"train_audio")
DATA_SPLIT_PATH = os.path.join(DATA_RAW_PATH, 'split_data')
