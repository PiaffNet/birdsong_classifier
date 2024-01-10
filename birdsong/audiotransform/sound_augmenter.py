import os
import glob
import pydub
import librosa
import numpy as np
from birdsong.config import config
from audiomentations import Compose, PitchShift, Shift, AddGaussianSNR
from birdsong.utils import get_folders_labels
from birdsong import DATA_SPLIT_PATH




class AudioAugmenter():
    """
    This class is used to transform the audio data
    """
    def __init__(self):
        self.data_directory = os.path.join(DATA_SPLIT_PATH)
        self._time_pitch_shift = True
        self._add_SNR_noise = False

    def transform_signal(self, signals: np.ndarray, sample_rate: int)-> np.ndarray:
        """
        This function is used to transform the signals
        """
        pass

    def make_signal_transformations(self):
        folder_lists = get_folders_labels(self.data_directory)
        print(folder_lists)
        for folder in folder_lists:
            folder_path = os.path.join(self.data_directory, folder)
            file_path_list = glob.glob(os.path.join(folder_path,'*.mp3'))
            for file_path in  file_path_list:
                file_label = os.path.splitext(os.path.basename(file_path))[0]
                sample, frame_rate, bytes_per_frame, sample_width = self.load_audio(file_path)
                print(f"sample rate: {frame_rate}, sample shape: {sample.shape}, sample type: {type(sample)}")
                if self._time_pitch_shift:
                    new_sample = self.transform_signal_pitch_shift(sample, frame_rate)
                    self.write_sample_in_mp3(f"{file_label}_time_pitch_shift", new_sample, frame_rate, sample_width)
                if self._add_SNR_noise:
                    new_sample = self.transform_signal_add_SNR_noise(sample, sample_rate)
                    self.write_sample_in_mp3(f"{file_label}_SNR_noise", new_sample, frame_rate, sample_width)

    def load_audio(self, file_path: str)-> tuple:
        """
        Load the audio file
        """
        sample = pydub.AudioSegment.from_mp3(file_path)
        frame_rate = sample.frame_rate
        bytes_per_frame = sample.frame_width
        sample_width = sample.sample_width
        y = np.array(sample.get_array_of_samples(), dtype=np.float32)
        #y, sample_rate = librosa.load(file_path, sr=None)
        return y, frame_rate, bytes_per_frame, sample_width

    def write_sample_in_mp3(self, file_label: str, sample: np.ndarray, frame_rate: int, sample_width: int)-> None:
        """
        Write the sample in mp3 format
        """
        file_path = os.path.join(self.data_directory, file_label + '.mp3')
        song = pydub.AudioSegment(sample.tobytes(), frame_rate=frame_rate,
                                  sample_width=sample_width,
                                  channels=1)
        song.export(file_path, format="mp3")

    def transform_signal_add_SNR_noise(self, signals: np.ndarray, sample_rate: int)-> np.ndarray:
        """
        Add Gaussian noise to the signals
        """
        transform = AddGaussianSNR(min_snr_db=config.MIN_SNR_DB,
                                   max_snr_db=config.MAX_SNR_DB,
                                   p=config.PROBABILITY_SNR
                                   )
        signals = transform(samples=signals, sample_rate=sample_rate)
        return signals

    def transform_signal_pitch_shift(self, signals: np.ndarray, sample_rate: int)-> np.ndarray:
        """
        time shift the signals
        Pitch shift the signals
        """
        transform = Compose([
        PitchShift(min_semitones=config.MIN_SEMITONES,
                   max_semitones=config.MAX_SEMITONES,
                   p=config.PROBABILITY_PITCH),
        Shift(min_shift=config.MIN_SHIFT,
              max_shift=config.MAX_SHIFT,
              shift_unit=config.SHIFT_UNIT,
              p=config.PROBABILITY_SHIFT),
    ])

        signals = transform(samples=signals, sample_rate=sample_rate)
        return signals

    def save_generated_signals(self, signals: np.ndarray, sample_rate: int)-> np.ndarray:
        """
        save the generated signals
        """
        pass
