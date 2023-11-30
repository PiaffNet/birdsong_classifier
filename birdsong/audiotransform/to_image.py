import os
import glob
import numpy as np
from pandas import pd
from pydub import AudioSegment
from IPython.display import Audio
import librosa
import matplotlib.pyplot as plt
from birdsong.config import config
from birdsong.utils import get_folders_labels, create_folder_if_not_exists


USEFUL_FEATS = ['filename','species', 'rating', 'channels', 'sampling_rate' , 'file_type']


"""
Garde seulement les espèces françaises et les colonnes intéressantes. Cela fait
2706 lignes et 7 colonnes. Il reste 257 na dans le playback_used, et 2 dans le bitrate
"""
def clean_data(df):
    species_fr = df[df["country"]== "France"]["species"].unique()
    df_clean = (df[df["species"].isin(species_fr)])[USEFUL_FEATS]
    return df_clean


# def sound_to_image(filename):

#     #load the file
#     y = librosa.load(filename)

#     # Compute the mel-spectrogram
#     sample_rate = 48000 #window size of 10.7 ms (512 samples at 48 kHz)
#     n_fft = 512
#     f_min = 150 #frequency range between 150 et 15 kHz
#     f_max = 15000
#     hop_length = round(0.75*n_fft) # 25 % de overlap, donc 0.75 * n_fft
#     n_mels = 64 #mel scale with 64 bands
#     htk = 1750 #break frequency

#     mel_spectrogram = librosa.feature.melspectrogram(y = y , sr=sample_rate, n_fft = n_fft, n_mels = n_mels, htk= htk, hop_length= hop_length, fmin = f_min, fmax= f_max)
#     mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)

#     return mel_spectrogram_db



 # def create_output_subfolder(self):
    #     subfolder_lists = get_folders_labels(self.input_folder)
    #     for subfolder in subfolder_lists:
    #         subfolder_path = os.path.join(self.input_folder, subfolder)
    #         # if subfolder exists in input_folder, create equivalent subfolder in output_folder
    #         create_folder_if_not_exists(subfolder_path)


""" Convert audio file to single channel (mono) and standard sample rate (48k)
    --> takes input folder containing raw audio files and creates new output folder
    --> specifies target sample rate (48k Hz) and nb of channels (mono by default)
    --> iterates through all audio files, applies preprocessing and saves to new folder
    --> TBD: creates a spectogram for all preprocessed audio files

    - `Input` and `output` folders are paths that need to be specified when method is called
"""

# TODO: - def save_spectogram function to save into new folder
#       - save


# Note: by default in librosa, all audio is mixed to mono and resampled to 22050 Hz at load time

class AudioPreprocessor:
    def __init__(self, input_folder, output_folder, spectogram_type, output_format):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.spectogram_type = spectogram_type # specto type 'ndarray' or '.png'
        self.output_format = output_format


    def create_data(self):
        subfolder_lists = get_folders_labels(self.input_folder)
        for subfolder in subfolder_lists:
            input_subfolder_path = os.path.join(self.input_folder, subfolder)
            target_directory = os.path.join(self.output_folder,subfolder)
            create_folder_if_not_exists(target_directory)
            list_files = glob.glob(str(input_subfolder_path)+'*.mp3')
            for file in list_files:
                self.preprocess_audio(file, target_directory)



    def preprocess_audio(self, file_name, target_directory):
        input_path = os.path.join(self.input_folder, file_name)
        file_label = os.path.splitext(os.path.basename(file_name))[0]
        target_path = os.path.join(target_directory, file_label+self.output_format)

        try:
                # Load the audio file with pydub
                audio = AudioSegment.from_mp3(input_path)

                # Step 1: Resample audio to target sample rate and convert from stereo to mono
                audio = audio.resample(sample_rate_Hz=config.SAMPLING_RATE,
                                    channels=1) # channel=1 for mono

                # Step 2: transform cleaned audio file into spectogram (ndarray)
                spectogram_array = self.get_spectogram(audio)

                # if image wanted, convert format to .png
                self.save_spectogram(spectogram_array, target_path)

                print(f"Preprocessed {file_name}.{self.output_format}")

        except Exception:
            print(f"Error processing {file_name}: {str(Exception)}")



    def get_spectogram(self, audio)->np.ndarray:
         ## load the preprocessed audio file
        if self.spectogram_type == 'MEL':
            spectogram = self.get_mel_spectogram(audio)

        return spectogram

    @staticmethod
    def get_mel_spectogram(audio):
        y = librosa.load(audio)
        ## set mel-spectogram params
        hop_length = round((1 - config.HOP_OVERLAP) * config.STFT_NUMBER_SAMPLES) # 25 % de overlap, donc 0.75 * n_fft

        ## convert to mel-spectogram
        mel_spectogram = librosa.feature.melspectrogram(y=y,
                                                        sr=config.SAMPLING_RATE,
                                                        n_fft=config.STFT_NUMBER_SAMPLES,
                                                        n_mels=config.N_MELS,
                                                        htk=config.HTK,
                                                        hop_length=hop_length,
                                                        fmin=config.F_MIN,
                                                        fmax=config.F_MAX)

        mel_spectogram_db = librosa.power_to_db(mel_spectogram) # returns ndarray type file
        return mel_spectogram_db


    def save_spectogram(self, spectogram_array, target_path):
        if self.output_format == 'png':
            plt.figure(figsize=(10, 5))
            plt.specgram(self,
                         Fs=config.SAMPLING_RATE,
                         cmap="viridis")
            #plt.axis("off")
            png_file_path = f"{output_path}.png"
            plt.savefig(png_file_path,
                        bbox_inches="tight",
                        pad_inches=0)
            plt.close()

        if self.output_format == 'npy':
           # Step 3: save preprocessed file to output (sub)
           np.save(target_path, spectogram_array)
