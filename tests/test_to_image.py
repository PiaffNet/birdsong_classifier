import glob
import os
import pytest
from birdsong.utils import get_folders_labels

@pytest.fixture
def test_audiopreprocessor():
    from birdsong.audiotransform.to_image import AudioPreprocessor
    test_audiopreprocessor = AudioPreprocessor()
    return test_audiopreprocessor

def test_silence_dir_in_inputs_data(test_audiopreprocessor):

    src_folder = test_audiopreprocessor.input_folder
    all_classes = get_folders_labels(src_folder)
    assert "silence" in all_classes

def test_inputs_data_folder_are_not_empty(test_audiopreprocessor):

    src_folder = test_audiopreprocessor.input_folder
    all_classes = get_folders_labels(src_folder)
    for class_label in all_classes:
        files_nb = len(glob.glob(os.path.join(src_folder, class_label,"*.mp3")))
        assert files_nb > 0

def test_outputs_data_folder_are_not_empty(test_audiopreprocessor):

    dst_folder = test_audiopreprocessor.output_folder
    all_classes = get_folders_labels(dst_folder)
    if all_classes != []:
        for class_label in all_classes:
            files_nb = len(glob.glob(os.path.join(dst_folder, class_label,"*.png")))
            assert files_nb > 0
