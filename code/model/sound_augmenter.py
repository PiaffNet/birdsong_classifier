import numpy as np
from audiomentations import Compose, PitchShift, Shift, AddGaussianSNR


def transform_signal_add_SNR_noise(signals: np.ndarray(), sr: int)-> np.ndarray():
    """
    Add Gaussian noise to the signals
    """
    transform = AddGaussianSNR(min_snr_db=MIN_SNR_DB,
                               max_snr_db=MAX_SNR_DB,
                               p=PROBABILITY_SNR
                               )
    signals = transform(samples=signals, sample_rate=sr)
    return signals

def transform_signal_pitch_shift(signals: np.ndarray(), sr: int)-> np.ndarray():
    """
    time shift the signals
    Pitch shift the signals
    """
    transform = Compose([
    PitchShift(min_semitones=MIN_SEMITONES,
               max_semitones=MAX_SEMITONES,
               p=PROBABILITY_PITCH),
    Shift(min_shift=MIN_SHIFT,
          max_shift=MAX_SHIFT,
          shift_unit=SHIFT_UNIT,
          p=PROBABILITY_SHIFT),
])

    signals = transform(samples=signals, sample_rate=sr)
    return signals
