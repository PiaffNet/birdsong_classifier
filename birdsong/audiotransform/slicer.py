import pandas as pd
import numpy as np
from pydub import AudioSegment
import librosa
import os
from birdsong.utils import create_folder_if_not_exists
from birdsong.config import config
from birdsong import PARENT_BASE_PATH

class AudioSlicer():
    '''
    Slices audio data to 3 seconds mp3s and puts them in the corresponding folder,
    if the slice is too short or too silent it goes into different folders.

    params :
        - input_directory : str  -> directory to copy from, should follow the directory/species_folders format
        - target_directory : str -> path to the copied data, will reconstruct the architecture of the original directory with silence and too_small added
        - silence_intolerance : int -> the higher the number the less likely a sample is to be considered silent
    '''
    def __init__(self, input_directory : str, target_directory : str):
        self.rating_threshold = 2.5
        self.country = "France"
        self.input_directory = input_directory
        self.target_directory = target_directory
        self.silence_intolerance = int(6)
        self.frame_length = int(2048)
        self.duration = int(3000) #ms
        self.silence_path = os.path.join(self.target_directory,"silence")

    @staticmethod
    def make_splits(file_n, sr, frame_length, duration):
        x, sr = librosa.load(file_n, sr=sr)
        X = librosa.stft(x)
        rms_tot = librosa.feature.rms(S=X, frame_length = frame_length)
        try:
            song = AudioSegment.from_file(file_n, "mp3")
            splits = song[::duration]
            return 1, splits, rms_tot
        except:
            return 0, 0, 0

    def slice_audio(self):
        """ Slices the audio files in the input_directory into 3 seconds mp3s
        and puts them in the corresponding folder, if the slice is too short or too silent
        it goes into different folders.
        """
        ref_csv_file_path = os.path.join(PARENT_BASE_PATH,"raw_data","train.csv")
        df = pd.read_csv(ref_csv_file_path)
        create_folder_if_not_exists(self.silence_path)
        _species = list(df.species[df['country'] == self.country].unique())
        df_country_species = df[df['species'].isin(_species)]
        df_country_selection = df_country_species[df_country_species['rating'] > self.rating_threshold]
        bird_code_list = df_country_selection.loc[:,'ebird_code'].unique().tolist()

        for dirspecies in bird_code_list:
            f = os.path.join(self.input_directory, dirspecies)
            # checking if the species input_directory exists and iters through it

            if os.path.isdir(f) :
                split_path = os.path.join(self.target_directory, dirspecies)
                create_folder_if_not_exists(split_path)

                for file in os.listdir(f):
                    sr = int(df['sampling_rate'][df['filename'] == file].values[0][:-5])
                    # charge le fichier dans un format pydub et check le RMS general
                    file_n = os.path.join(f, file)
                    res, splits, rms_tot = self.make_splits(file_n, sr, self.frame_length, self.duration)

                    if (res == 1):
                        for i, split in enumerate(splits):
                            x = np.asarray(split.get_array_of_samples(), dtype=np.float64)
                            if len(x) >= ((sr * 3) - 2):
                                X = librosa.stft(x)
                                rms = librosa.feature.rms(S=X, frame_length=self.frame_length)
                                if rms.mean() < rms_tot.mean() / self.silence_intolerance :
                                    split.export(os.path.join(self.silence_path,f"{file[:-4]}_{i}.mp3"), format="mp3")
                                else:
                                    split.export(os.path.join(split_path ,f"{file[:-4]}_{i}.mp3"), format="mp3")
