"""
This is the main file for the birdsong_classifier project.
"""
from birdsong.config import config
from birdsong.audiotransform.to_image import AudioPreprocessor
from birdsong.audiotransform.slicer import AudioSlicer
from birdsong.model.model import PlayModel
from birdsong.utils import read_prediction

def slice_audio():
    print('Slicing audio...')

    audio_slicer = AudioSlicer()
    audio_slicer.slice_audio()

    print("✅ slicing done")

    return 1

def preprocess_and_train():

    # Transform the data
    # split data

    try:
        res_prep = preprocess()
        if res_prep == 1:
            res, prediction_df = train()
            if res == 1:
                print("✅ preprocess and train done")

    except Exception:
            print(f"Fatal error : {str(Exception)}")


def preprocess():
    print('Preprocessing...')

    audio_processor = AudioPreprocessor()
    audio_processor.create_data()

    print("preprocess done")

    return 1

def train():
    print('Training process...')
    model_to_train = PlayModel()

    predictions_df = model_to_train.get_new_model()

    print("✅ training process done")
    return 1, predictions_df

def continue_training_model(from_checkpoint : bool=False):

    '''
    retrain a model either from a saved model or a checkpoint's weights.

    IF 'from_checkpoint' --> True : 'path' must be to checkpoint folder and
    model_name must be one the available models in model_collections.py.

    ELSE 'from_checkpoint' --> False : 'path' must be to model and model_name can be empty

    '''
    model_to_train = PlayModel()
    if from_checkpoint:
        predictions_df = model_to_train.get_model_from_checkpoint()

    else:
        predictions_df = model_to_train.get_saved_model()

    return 1, predictions_df

def predict(data_to_predict):
    print('Predicting...')

    available_model = PlayModel()
    predictions = available_model.predict_model(data_to_predict)
    return 1, predictions

if __name__ == '__main__':
    try:
        #preprocess()
        #train()
        continue_training_model(from_checkpoint=True)
    except Exception:
        print(f"❌ Error {str(Exception)}")
