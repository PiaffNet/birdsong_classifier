
from tensorflow import keras
from typing import Tuple


class CNNsimple():

        MODEL_NAME = 'CNNsimple'
        MODEL_TYPE = 'functional'

        def __init__(self,num_classes : int, input_shape=(64, 376, 1)):
            self.num_classes = num_classes
            self.input_shape = input_shape

        def build_archi_0(self) -> keras.Model:
            model = keras.Sequential([
                keras.layers.Rescaling(1./255, input_shape=self.input_shape),
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
                keras.layers.Dense(self.num_classes, activation='softmax')
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

class BirdNetBlock():
        """
        Modèles basés sur BirdNet
        """
        MODEL_NAME = 'BirdNetBlock'
        MODEL_TYPE = 'functional'

        def __init__(self,num_classes : int, input_shape=(64, 376, 1)):
            self.num_classes = num_classes
            self.input_shape = input_shape

        # Definition des blocks de downsampling et de ResBlock
        def resblock(self, x, kernelsize, filters):
            fx = keras.layers.Conv2D(filters, kernelsize, activation='relu', padding='same')(x)
            fx = keras.layers.BatchNormalization()(fx)
            fx = keras.layers.Conv2D(filters, kernelsize, padding='same')(fx)
            out = keras.layers.Add()([x,fx])
            out = keras.layers.ReLU()(out)
            out = keras.layers.BatchNormalization()(out)
            return out

        def downsampling(self, x, filters):
            ### Branche 1
            b1= keras.layers.Conv2D(filters, kernel_size = 1, activation = 'relu') (x)
            b1 = keras.layers.Conv2D(filters, kernel_size = 3, strides = 2, padding = 'same',activation = 'relu') (b1)
            b1= keras.layers.Conv2D(filters, kernel_size = 1, activation = 'relu') (b1)

            ### Branche 2
            b2 = keras.layers.AveragePooling2D(pool_size = (2,2), strides = 2, padding = 'same') (x)
            b2= keras.layers.Conv2D(filters, kernel_size = 1, activation = 'relu') (b2)

            out = keras.layers.Add()([b1,b2])
            out = keras.layers.ReLU()(out)
            return out


        def build_archi_a(self) -> keras.Model:
            """"
            Modèle simplifié, avec une seule layer de ResStack
            """

            #preprocessing

            visible = keras.layers.Input(shape=self.input_shape)
            conv1 = keras.layers.Conv2D(32, kernel_size=5, padding ='same', activation='relu')(visible)
            bn1 = keras.layers.BatchNormalization() (conv1)
            relu1 = keras.layers.ReLU() (bn1)
            pool1 = keras.layers.MaxPooling2D(pool_size=(1, 2))(relu1)

            #ResStack1

            ds1= self.downsampling(pool1, 64)
            res1 = self.resblock(ds1, 2, 64)

            # Classification

            conv5= keras.layers.Conv2D(64, kernel_size = 3, strides = 4, padding = 'same', activation = 'relu') (res1)
            bn5 = keras.layers.BatchNormalization() (conv5)
            relu5 = keras.layers.ReLU() (bn5)
            drop5 = keras.layers.Dropout(rate=0.2) (relu5)

            conv6= keras.layers.Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu') (drop5)
            bn6 = keras.layers.BatchNormalization() (conv6)
            relu6 = keras.layers.ReLU() (bn6)
            drop6 = keras.layers.Dropout(rate=0.26) (relu6)

            conv7 = keras.layers.Conv2D(16, kernel_size = 3, padding = 'same', activation = 'relu') (drop6)
            bn7 = keras.layers.BatchNormalization() (conv7)

            flat = keras.Flatten()(bn7)
            dense1 = keras.layers.Dense(256, activation='relu')(flat)

            output = keras.Dense(self.num_classes, activation='softmax')(dense1)
            model = keras.Model(inputs=visible, outputs=output)

            return model

        def build_archi_b(self) -> keras.Model:
            """"
            Modèle fini, plus long à computer
            """
            #preprocessing
            visible = keras.layers.Input(shape=self.input_shape)
            conv1 = keras.layers.Conv2D(32, kernel_size=5, padding ='same', activation='relu')(visible)
            bn1 = keras.layers.BatchNormalization() (conv1)
            relu1 = keras.layers.ReLU() (bn1)
            pool1 = keras.layers.MaxPooling2D(pool_size=(1, 2))(relu1)

            #ResStack1

            ds1 = self.downsampling(pool1, 64)
            res1 = self.resblock(ds1, 2, 64)

            #ResStack2

            ds2 = self.downsampling(res1, 128)
            res2 = self.resblock(ds2, 2, 128)

            #ResStack3

            ds3 = self.downsampling(res2, 256)
            res3 = self.resblock(ds3, 2, 256)

            #ResStack4

            ds4 = self.downsampling(res3, 512)
            res4 = self.resblock(ds4, 2, 512)

            # Classification
            conv5= keras.layers.Conv2D(512, kernel_size = 3, strides = 4, padding = 'same', activation = 'relu') (res4)
            bn5 = keras.layers.BatchNormalization() (conv5)
            relu5 = keras.layers.ReLU() (bn5)
            drop5 = keras.layers.Dropout(rate=0.24) (relu5)

            conv6= keras.layers.Conv2D(256, kernel_size = 3, padding = 'same', activation = 'relu') (drop5)
            bn6 = keras.layers.BatchNormalization() (conv6)
            relu6 = keras.layers.ReLU() (bn6)
            drop6 = keras.layers.Dropout(rate=0.26) (relu6)

            conv7 = keras.layers.Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu') (drop6)
            bn7 = keras.layers.BatchNormalization() (conv7)
            drop7 = keras.layers.Dropout(rate=0.3) (bn7)

            flat = keras.Flatten()(bn7)
            output = keras.Dense(self.num_classes, activation='softmax')(flat)
            model = keras.Model(inputs=visible, outputs=output)

            return model



def get_model_architecture(model_call : str, num_classes: int, input_shape : tuple) -> keras.Model:
    if model_call == 'baseline_archi_0':
        model = build_archi_0(num_classes, input_shape)

    elif model_call == 'baseline_archi_1':
        model = build_archi_1(num_classes, input_shape)

    else:
        raise NotImplementedError
    return model
