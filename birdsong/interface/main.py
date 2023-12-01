"""
This is the main file for the birdsong_classifier project.
"""
from birdsong.model.transform import transfrom, get_labels
from birdsong.audiotransform.preprocessing import preprocess_data
from birdsong.model.model import initialize_model, compile_model, train_model, evaluate_model
from birdsong.model.transform import get_train_data_set, get_validation_test_data_sets
from birdsong.config import config
from birdsong.audiotransform.to_image import AudioPreprocessor
from birdsong.model.transform import get_train_data_set, get_validation_test_data_sets
from birdsong.model.model import initialize_model, compile_model, train_model, evaluate_model, save_model,load_model
from birdsong.utils import read_prediction

def preprocess_and_train():

    # Transform the data
    # split data

    try:
        res_prep = preprocess()
        if res_prep==1:
            res = train()
            print(" preprocess and train done")

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

    print("find {num_classes} in train data set")

    model = initialize_model(model_call_label=config.MODEL_NAME,
                             input_shape=(64, 376, 1),
                             num_classes=num_classes)
    model = compile_model(model)

    model, history = train_model(model=model,
                                 train_data=train_ds,
                                 validation_data=val_ds)

    metrics = evaluate_model(model=model, test_data=test_ds)

    print("Training loss {metrics[0]} accuracy {metrics[1]}")
    save_model(model, config.MODEL_SAVE_PATH)

    return 1

def predict():
    print('Predicting...')

    model = load_model(config.MODEL_SAVE_PATH)
    predictions = model.predict(test_ds)
    return 1

def steps_with_test_split(data, labels):

    X_train, X_test, y_train, y_test = train_validation_split(data,
                                                            labels,
                                                            test_size=config.TEST_SIZE)

    X_train, X_val, y_train, y_val = train_validation_split(X_train,
                                                          y_train,
                                                          test_size=config.TEST_SIZE)

def steps_without_test_split(data, labels):
    X_train, X_val, y_train, y_val = train_validation_split(data,
                                                          labels,
                                                          test_size=config.TEST_SIZE)

    # Preprocess the data
    X_train_processed = preprocess_data(X_train)
    X_val_processed = preprocess_data(X_val)

    model = initialize_model(input_shape=X_train_processed.shape[1:],
                                 num_classes=len(np.unique(y_train)))

    model = compile_model(model)

    model, history = train_model(model,
                                X_train_processed,
                                y_train,
                                batch_size=config.BATCH_SIZE,
                                epochs=config.EPOCHS,
                                patience=config.PATIENCE,
                                validation_data=(X_val_processed, y_val))

    # save the model


if __name__ == '__main__':
    try:
        preprocess()
        train()
    except:
        print("Error")
