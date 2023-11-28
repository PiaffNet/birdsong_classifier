from birdsong.model.transform import transfrom, get_labels
from birdsong.model.preprocessing import preprocess_data
from birdsong.model.model import initialize_model, compile_model, train_model, evaluate_model
from birdsong.utils import train_validation_split
from birdsong.config import config


def preprocess_and_train():

    # Transform the data
    data = transfrom(data)
    labels = get_labels()

    # Train and split the data
    if config.TEST_SPLIT:
        steps_with_test_split(data, labels)
    else:
        steps_without_test_split(data, labels)

    print(" preprocess and train done")


def preprocess():
    print('Preprocessing...')
    return 1

def train():
    print('Training...')
    return 1

def predict():
    print('Predicting...')
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
