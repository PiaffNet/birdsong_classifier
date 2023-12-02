
from tensorflow import keras
from typing import Tuple

def build_archi_0(num_classes : int, input_shape=(64, 376, 1)) -> keras.Model:

    model = keras.Sequential([
            keras.layers.Rescaling(1./255, input_shape=input_shape),
            keras.layers.Conv2D(64, (2,2), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, (2,3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.04),
            keras.layers.Conv2D(64, (3,2), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(1568, activation='relu'),
            keras.layers.Dense(500, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_archi_1(num_classes : int, input_shape=(64, 376, 1)) -> keras.Model:

    model = keras.Sequential([
    keras.layers.Rescaling(1./255, input_shape=input_shape), # (64, 376, 1)
    keras.layers.Conv2D(64, 2, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, (2,3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, (3,2), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.Dropout(rate = 0.24),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(1568, activation='relu', kernel_regularizer='l2'),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(num_classes, activation = 'softmax')
    ])

    return model

def get_model_architecture(model_call : str, num_classes: int, input_shape : tuple) -> keras.Model:
    if model_call == 'baseline_archi_0':
        model = build_archi_0(num_classes, input_shape)

    elif model_call == 'baseline_archi_1':
        model = build_archi_1(num_classes, input_shape)

    else:
        raise NotImplementedError
    return model
