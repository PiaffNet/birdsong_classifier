import pandas as pd
import numpy as np
from pydub import AudioSegment
import librosa
import os
from birdsong.config import config


def slicer(directory : str, newdir : str, silence_intolerance : int = 6 ) -> None :
    '''
    Slices audio data to 3 seconds mp3s and puts them in the corresponding folder,
    if the slice is too short or too silent it goes into different folders.

    params :
        - directory : str  -> directory to copy from, should follow the directory/species_folders format
        - newdir : str -> path to the copied data, will reconstruct the architecture of the original directory with silence and too_small added
        - silence_intolerance : int -> the higher the number the less likely a sample is to be considered silent

    TODO:
      - define as a class
      - do not forget delete "too_small" directory after the slicing process
     '''
    df = pd.read_csv("raw_data/train.csv")

    for dirspecies in os.listdir(directory):
        f = os.path.join(directory, dirspecies)
        # checking if the species directory exists and iters through it

        if os.path.isdir(f) :

            for file in os.listdir(f):
                sr = int(df['sampling_rate'][df['filename'] == file].values[0][:-5])
                # charge le fichier dans un format pydub et check le RMS general
                fileN = os.path.join(f, file)
                x, sr = librosa.load(fileN, sr=sr)
                X = librosa.stft(x)
                rms_tot = librosa.feature.rms(S=X, frame_length=2048)
                #song = AudioSegment.from_mp3(fileN)
                try:
                    song = AudioSegment.from_file(fileN, "mp3")
                except:
                    pass

                #split the birdsong audio into 3sec chunks
                splits = song[::3000]  ##comprend pas bien ce que ca fait

                if os.path.isdir(os.path.join(newdir, dirspecies)): #une fois que le fichier est splitté on le copie dans le bon dossier
                    for i,split in enumerate(splits) :
                        split.export(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", format="mp3")
                        x, sr = librosa.load(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", sr=sr)
                        X = librosa.stft(x)
                        rms = librosa.feature.rms(S=X, frame_length=2048)


                        if rms.mean() < rms_tot.mean()/silence_intolerance :  #si dossier silencieux
                            if os.path.isdir(os.path.join(newdir, "silence")):
                                os.rename(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", f"{newdir}/silence/{file[:-4]}_{i}.mp3")
                            else :
                                os.makedirs(os.path.join(newdir, "silence"))
                                os.rename(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", f"{newdir}/silence/{file[:-4]}_{i}.mp3")

                        if len(x) < ((sr*3)-2):
                            if os.path.isdir(os.path.join(newdir, "too_small")) :
                                if os.path.isfile(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3"):
                                    os.rename(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", f"{newdir}/too_small/{file[:-4]}_{i}.mp3")
                                else :
                                    os.rename(f"{newdir}/silence/{file[:-4]}_{i}.mp3", f"{newdir}/too_small/{file[:-4]}_{i}.mp3")
                            else :

                                os.makedirs(os.path.join(newdir, "too_small"))
                                if os.path.isfile(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3"):
                                    os.rename(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", f"{newdir}/too_small/{file[:-4]}_{i}.mp3")
                                else :
                                    os.rename(f"{newdir}/silence/{file[:-4]}_{i}.mp3", f"{newdir}/too_small/{file[:-4]}_{i}.mp3")


                else :  # si le dossier species existe pas on le créée et rebelotte
                    os.makedirs(os.path.join(newdir, dirspecies))

                    for i,split in enumerate(splits) :
                        split.export(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", format="mp3")
                        x, sr = librosa.load(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", sr=sr)
                        X = librosa.stft(x)
                        rms = librosa.feature.rms(S=X, frame_length=2048)


                        if rms.mean() < rms_tot.mean()/silence_intolerance :
                            if os.path.isdir(os.path.join(newdir, "silence")):
                                os.rename(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", f"{newdir}/silence/{file[:-4]}_{i}.mp3")
                            else :
                                os.makedirs(os.path.join(newdir, "silence"))
                                os.rename(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", f"{newdir}/silence/{file[:-4]}_{i}.mp3")

                        if len(x) < ((sr*3)-2):
                            if os.path.isdir(os.path.join(newdir, "too_small")) :
                                if os.path.isfile(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3"):
                                    os.rename(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", f"{newdir}/too_small/{file[:-4]}_{i}.mp3")
                                else :
                                    os.rename(f"{newdir}/silence/{file[:-4]}_{i}.mp3", f"{newdir}/too_small/{file[:-4]}_{i}.mp3")
                            else :

                                os.makedirs(os.path.join(newdir, "too_small"))
                                if os.path.isfile(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3"):
                                    os.rename(f"{newdir}/{dirspecies}/{file[:-4]}_{i}.mp3", f"{newdir}/too_small/{file[:-4]}_{i}.mp3")
                                else :
                                    os.rename(f"{newdir}/silence/{file[:-4]}_{i}.mp3", f"{newdir}/too_small/{file[:-4]}_{i}.mp3")



if __name__ == "__main__" :

    directory = 'raw_data/selected_samples/train_audio'
    newdir = 'raw_data/split_data'
    slicer(directory, newdir, silence_intolerance = 6)
