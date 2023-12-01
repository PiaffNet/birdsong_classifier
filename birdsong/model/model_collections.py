from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from typing import Tuple

def build_archi_0(num_classes : int, input_shape=(64, 376, 1)) -> Model:

    model = Sequential([
            layers.Rescaling(1./255, input_shape=input_shape),
            layers.Conv2D(64, (2,2), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (2,3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.04),
            layers.Conv2D(64, (3,2), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, (5,5), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1568, activation='relu'),
            layers.Dense(500, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_archi_1(num_classes : int, input_shape=(64, 376, 1)) -> Model:

    model = Sequential([
    layers.Rescaling(1./255, input_shape=input_shape), # (64, 376, 1)
    layers.Conv2D(64, 2, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (2,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,2), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Dropout(rate = 0.24),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(1568, activation='relu', kernel_regularizer='l2'),
    layers.Dense(500, activation='relu'),
    layers.Dense(num_classes, activation = 'softmax')
    ])

    return model

def get_model_architecture(model_call : str, num_classes: int, input_shape : tuple) -> Model:
    if model_call == 'baseline_archi_0':
        model = build_archi_0(num_classes, input_shape)

    elif model_call == 'baseline_archi_1':
        model = build_archi_1(num_classes, input_shape)

    else:
        raise NotImplementedError
    return model
