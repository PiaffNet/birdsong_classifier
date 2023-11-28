from pandas import pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


USEFUL_FEATS = ['filename','species', 'rating', 'channels', 'sampling_rate' , 'file_type']


"""
Garde seulement les espèces françaises et les colonnes intéressantes. Cela fait
2706 lignes et 7 colonnes. Il reste 257 na dans le playback_used, et 2 dans le bitrate
"""
def clean_data(df):
    species_fr = df[df["country"]== "France"]["species"].unique()
    df_clean = (df[df["species"].isin(species_fr)])[USEFUL_FEATS]
    return df_clean


def sound_to_image(filename):

    #load the file
    y = librosa.load(filename)

    # Compute the mel-spectrogram
    sample_rate = 48000 #window size of 10.7 ms (512 samples at 48 kHz)
    n_fft = 512
    f_min = 150 #frequency range between 150 et 15 kHz
    f_max = 15000
    hop_length = round(0.75*n_fft) # 25 % de overlap, donc 0.75 * n_fft
    n_mels = 64 #mel scale with 64 bands
    htk = 1750 #break frequency

    mel_spectrogram = librosa.feature.melspectrogram(y = y , sr=sample_rate, n_fft = n_fft, n_mels = n_mels, htk= htk, hop_length= hop_length, fmin = f_min, fmax= f_max)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)

    return mel_spectrogram_db
