import pytest
import os
import glob
import shutil
import random
import librosa
from birdsong.utils import get_folders_labels
from birdsong import DATA_TRAIN_AUDIO_PATH, DATA_RAW_PATH

NB_SAMPLE = 1
RANDOM_SEED = 42

@pytest.fixture
def test_audio_slicer():
    from birdsong.audiotransform.slicer import AudioSlicer
    test_audio_slicer = AudioSlicer()

    dst_src = os.path.join(DATA_RAW_PATH,"test_train_data")

    test_audio_slicer.input_directory = dst_src
    test_audio_slicer.target_directory = os.path.join(DATA_RAW_PATH,"test_split_data")
    return test_audio_slicer

@pytest.fixture
def create_test_case_data(test_audio_slicer):

    def copy_dir_content(parent_src, parent_dst, directory_to_copy):
        def ignored_files(adir,filenames):
            return [filename for filename in filenames if filename.startswith(".")]

        src = os.path.join(parent_src,directory_to_copy)
        dst = os.path.join(parent_dst,directory_to_copy)
        shutil.copytree(src, dst, ignore=ignored_files)


    parent_src = DATA_TRAIN_AUDIO_PATH
    dst_src = test_audio_slicer.input_directory

    df, list_test = test_audio_slicer.get_bird_code_list()
    raw_label_test = [list_test[-1]]

    for bird_code in raw_label_test:
        print(f"copying {bird_code} data")
        copy_dir_content(parent_src, dst_src, bird_code)

    yield  dst_src
    shutil.rmtree(dst_src)

@pytest.fixture
def test_slice_audio(test_audio_slicer, create_test_case_data):

    target_directory = test_audio_slicer.target_directory
    test_audio_slicer.slice_audio()

    yield target_directory
    shutil.rmtree(target_directory)



def test_audio_slicer_outputs(create_test_case_data, test_slice_audio, test_audio_slicer):
    """basic test of audio slicer output.
    Slicing is time consuming, so we only test the output on one species.
    At the end of the test, the test data are deleted.
    """

    input_labels = get_folders_labels(test_audio_slicer.input_directory)
    target_labels = get_folders_labels(test_audio_slicer.target_directory)

    assert len(input_labels) == len(target_labels) - 1,\
        "number of folders in input and number of folder-1 (-1 for silence folder) in target should be the same"

    target_sample_dir = input_labels[0]
    file_path_list = glob.glob(os.path.join(test_audio_slicer.target_directory, target_sample_dir, '*.mp3'))
    print(file_path_list)

    if len(file_path_list) > 0:
        file_sample = file_path_list[0]
        file_x, file_sr = librosa.load(file_sample, sr=None)
        assert len(file_x) >= 3 * file_sr - 2, "each slice should be at least 3 seconds long"

    silence_path_list = glob.glob(os.path.join(test_audio_slicer.target_directory, "silence", '*.mp3'))
    if len(silence_path_list) > 0:
        silence_sample = silence_path_list[0]
        silence_x, silence_sr = librosa.load(silence_sample, sr=None)
        assert len(silence_x) >= 3 * silence_sr - 2, "each slice should be at least 3 seconds long"
