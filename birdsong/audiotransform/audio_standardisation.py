"""
This file defines the audio preprocessing steps for the birdsong_classifier
project. It contains the following transformation steps:

- Convert all files to single channel (mono)
- Standardize sampling rate (48,000Hz)

This step comes after data splitting and before stft transformation (spectograms).
"""

import pandas as np
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
from pathlib import Path




# """ Set the local file path to access split data set"""

# GITHUB_NAME = github_name
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", GITHUB_NAME, "birdsong_classifier", "data")
# LOCAL_FILE_PATH = os.path.join(LOCAL_DATA_PATH, )


""" Convert audio file to single channel (mono) and standard sample rate (48k)"""

# Note: by default in librosa, all audio is mixed to mono and resampled to 22050 Hz at load time

song_normalized = librosa.load("local_file_path", mono=True, sr=48000)




""" Method to play a song (use load before to get sr)"""

def play_song(song_normalized):
    return Audio(song_normalized, rate=sr)
