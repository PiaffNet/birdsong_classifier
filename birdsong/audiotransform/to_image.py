import os
import glob
import numpy as np
import librosa
import skimage as ski
from birdsong.config import config
from birdsong.utils import get_folders_labels, create_folder_if_not_exists, get_image_shape
from birdsong.audiotransform.standardisation import spectrogram_image
from birdsong import PARENT_BASE_PATH


def select_species_per_country(df, country="France"):
    USEFUL_FEATS = ['filename','species', 'rating', 'channels', 'sampling_rate' , 'file_type']
    species_fr = df[df["country"]== country]["species"].unique()
    df_clean = (df[df["species"].isin(species_fr)])[USEFUL_FEATS]
    return df_clean

# Note: by default in librosa, all audio is mixed to mono and resampled to 22050 Hz at load time

class AudioPreprocessor:
    """ Convert audio file to single channel (mono) and standard sample rate (48k)
    --> takes input folder containing raw audio files and creates new output folder
    --> specifies target sample rate (48k Hz) and nb of channels (mono by default)
    --> iterates through all audio files, applies preprocessing and saves to new folder
    --> creates a spectogram for all preprocessed audio files

    - `Input` and `output` folders are paths that need to be specified when method is called
    """
    def __init__(self):
        input_folder_path, output_folder = self.get_data_paths()
        self.input_folder = input_folder_path
        self.output_folder = output_folder
        self.spectogram_type = config.SPECTOGRAM_TYPE # specto type 'ndarray' or '.png'
        self.output_format = config.OUTPUT_FORMAT
        self.image_shape = None
        self.nb_classes = None

    def create_data(self):
        if self.input_folder:
            subfolder_lists = get_folders_labels(self.input_folder)
            for subfolder in subfolder_lists:
                input_subfolder_path = os.path.join(self.input_folder, subfolder)
                if self.output_folder:
                    target_directory = os.path.join(self.output_folder,subfolder)
                    create_folder_if_not_exists(target_directory)
                    file_path_list = glob.glob(os.path.join(input_subfolder_path,'*.mp3'))
                    for file_path in  file_path_list:
                        print(f"prepare processing of {file_path}")
                        self.preprocess_audio(file_path, target_directory)
                else:
                    print("❌ No valid output folder specified by user")

            self.image_shape = self.get_image_sample_shape()
            self.nb_classes = self.get_nb_classes()
        else:
            print("❌ No valid input folder specified by user")

    def get_image_sample_shape(self):
        # grab just an image sample shape
        if self.image_shape == None:

            subfolder_lists = get_folders_labels(self.output_folder)
            sample_dir = subfolder_lists[0]
            target_dir = os.path.join(self.output_folder,sample_dir)
            sample_list = glob.glob(os.path.join(target_dir,'*.png'))
            return get_image_shape(sample_list[0])
        else:
            return self.image_shape

    def get_nb_classes(self):
        if self.nb_classes == None:
            subfolder_lists = get_folders_labels(self.output_folder)
            return len(subfolder_lists)
        else:
            return self.nb_classes

    def preprocess_audio_array(self, audio_signal):
        try:
            # Step 1: transform cleaned audio file into spectogram (ndarray)
            spectogram_array = self.get_spectogram(audio_signal)

            # Step 2: transform spectogram into image array
            image = spectrogram_image(spectogram_array)

            print(f"✅ Preprocessed {audio_signal}")
            return image

        except Exception:
            print(f"❌ Error processing {audio_signal}: {str(Exception)}")

    def preprocess_audio(self, file_path, target_directory):
        #input_path = os.path.join(self.input_folder, file_name)
        file_label = os.path.splitext(os.path.basename(file_path))[0]
        target_path = os.path.join(target_directory, file_label + '.' + self.output_format)

        try:
            # Step 2: transform cleaned audio file into spectogram (ndarray)
            spectogram_array = self.get_spectogram(file_path)

            # if image wanted, convert format to .png
            self.save_spectogram(spectogram_array, target_path)

            print(f"Preprocessed {target_path}")

        except Exception:
            print(f"❌ Error processing {target_path}: {str(Exception)}")



    def get_spectogram(self, audio)->np.ndarray:
         ## load the preprocessed audio file
        if self.spectogram_type == 'MEL':
            spectogram = self.get_mel_spectogram(audio)

        return spectogram

    @staticmethod
    def get_data_paths():
        print('compute data folder paths')
        if config.MODEL_TARGET == 'local':
           input_folder_path = os.path.join(PARENT_BASE_PATH,
                                            config.DATA_PERENT_DIR,
                                            config.DATA_INPUT_FOLDER)
           output_folder = os.path.join(PARENT_BASE_PATH,
                                        config.DATA_PERENT_DIR,
                                        config.DATA_OUTPUT_FOLDER)
           return input_folder_path, output_folder

    @staticmethod
    def get_mel_spectogram(audio):
        y, sr_load = librosa.load(audio, mono=True, sr=config.SAMPLING_RATE)
        ## set mel-spectogram params
        hop_length = round((1 - config.HOP_OVERLAP) * config.STFT_NUMBER_SAMPLES) # 25 % de overlap, donc 0.75 * n_fft

        ## convert to mel-spectogram
        mel_spectogram = librosa.feature.melspectrogram(y=y,
                                                        sr=sr_load,
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
            image = spectrogram_image(spectogram_array)
            ski.io.imsave(target_path, image)

        if self.output_format == 'npy':
           # Step 3: save preprocessed file to output (sub)
           np.save(target_path, spectogram_array)



#mel = sound_to_image(filepath)
#im = spectrogram_image(mel)
#skimage.io.imsave(os.path.join(newdir, dirspecies, file[:-4])+".png", im)
