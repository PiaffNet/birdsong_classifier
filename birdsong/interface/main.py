"""
This is the main file for the birdsong_classifier project.
"""
from birdsong.audiotransform.to_image import AudioPreprocessor
from birdsong.audiotransform.slicer import AudioSlicer
from birdsong.model.model import PlayModel

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
                print("="*30)

    except Exception:
            print(f"Fatal error : {str(Exception)}")


def preprocess():
    print('Preprocessing...')

    audio_processor = AudioPreprocessor()
    audio_processor.create_data()

    print("preprocess done")
    print("="*30)
    return 1

def train():
    print('Training new model ...')
    model_to_train = PlayModel()

    predictions_df = model_to_train.get_new_model()

    print("✅ training model done")
    print("="*30)
    return 1, predictions_df

def continue_training_model(from_checkpoint : bool=False):

    '''
    retrain a model either from a saved model or a checkpoint's weights.

    IF 'from_checkpoint' --> True : retrain model from the best checkpoint.

    ELSE 'from_checkpoint' --> False : retran the latest saved model.

    '''
    model_to_train = PlayModel()
    if from_checkpoint:
        predictions_df = model_to_train.get_model_from_checkpoint()

    else:
        predictions_df = model_to_train.get_saved_model()
    print("="*30)
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
        continue_training_model(from_checkpoint=False)
    except Exception:
        print(f"❌ Error {str(Exception)}")
