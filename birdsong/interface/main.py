"""
This is the main file for the birdsong_classifier project.
"""
from birdsong.config import config
from birdsong.model.model import initialize_model, compile_model, train_model, evaluate_model
from birdsong.model.transform import get_train_data_set, get_validation_test_data_sets
from birdsong.audiotransform.to_image import AudioPreprocessor
from birdsong.model.transform import get_train_data_set, get_validation_test_data_sets
from birdsong.model.model import initialize_model, compile_model, train_model,\
                                 evaluate_model, save_model,load_model,predict_model, model_get_from_checkpoint
from birdsong.utils import read_prediction

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
    print('Training...')
    train_ds = get_train_data_set()
    val_ds, test_ds = get_validation_test_data_sets()

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print(f"find {num_classes} classes in data set")

    model = initialize_model(model_call_label=config.MODEL_NAME,
                             num_classes=num_classes)
    model = compile_model(model)

    print(model.summary())

    model, history = train_model(model=model,
                                 train_data=train_ds,
                                 validation_data=val_ds)

    save_model(model)
    metrics = evaluate_model(model=model, test_data=test_ds)

    print(f"Training loss {metrics[model.metrics_names[0]]} accuracy {metrics[model.metrics_names[1]]}")


    predictions = model.predict(test_ds)
    predictions_df = read_prediction(predictions, class_names)

    return 1, predictions_df

def continue_training_model(path : str, from_checkpoint : bool , model_name : str):

    '''
    retrain a model either from a model or a checkpoint's weights.

    IF 'from_checkpoint' --> True : 'path' must be to checkpoint folder and
    model_name must be one the available models in model_collections.py.

    ELSE 'from_checkpoint' --> False : 'path' must be to model and model_name can be empty

    '''
    train_ds = get_train_data_set()
    val_ds, test_ds = get_validation_test_data_sets()

    class_names = train_ds.class_names

    if from_checkpoint:
        try:
            model = model_get_from_checkpoint(model_name, len(class_names))
            model, history = train_model(model,train_ds, val_ds)
        except:
            raise ValueError("❌ No checkpoint found")

    else :
        model = load_model(path)
        model, history = train_model(model,train_ds, val_ds)

    return model, history

def predict(data_to_predict):
    print('Predicting...')

    model = load_model()
    predictions = predict_model(model, data_to_predict)
    return 1, predictions

if __name__ == '__main__':
    try:
        #preprocess()
        train()
    except Exception:
        print(f"❌ Error {str(Exception)}")
