import numpy as np
from birdsong.config import config
from audiomentations import Compose, PitchShift, Shift, AddGaussianSNR

def transform_signal_add_SNR_noise(signals: np.ndarray, sample_rate: int)-> np.ndarray:
    """
    Add Gaussian noise to the signals
    """
    transform = AddGaussianSNR(min_snr_db=config.MIN_SNR_DB,
                               max_snr_db=config.MAX_SNR_DB,
                               p=config.PROBABILITY_SNR
                               )
    signals = transform(samples=signals, sample_rate=sample_rate)
    return signals

def transform_signal_pitch_shift(signals: np.ndarray, sample_rate: int)-> np.ndarray:
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
