import numpy as np
from birdsong.config import config
from audiomentations import Compose, PitchShift, Shift, AddGaussianSNR, Reverse




class AudioAugmenter():
    """
    This class is used to transform the audio data
    """
    def __init__(self):
        self._time_pitch_shift = True
        self._add_SNR_noise = True
        self._reverse = True

    def transform_signal_add_SNR_noise(self, signals: np.ndarray, sample_rate: int)-> np.ndarray:
        """
        Add Gaussian noise to the signals
        """
        transform = AddGaussianSNR(min_snr_db=config.MIN_SNR_DB,
                                   max_snr_db=config.MAX_SNR_DB,
                                   p=config.PROBABILITY_SNR
                                   )
        return transform(samples=signals, sample_rate=sample_rate)

    def transform_signal_reverse(self, signals: np.ndarray, sample_rate: int)-> np.ndarray:
        """
        Reverse the signals
        """
        transform = Compose([
            Reverse(p=config.PROBABILITY_REVERSE)
        ])

        return transform(samples=signals, sample_rate=sample_rate)

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

        return transform(samples=signals, sample_rate=sample_rate)
