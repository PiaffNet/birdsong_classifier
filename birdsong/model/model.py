"""
this file contains the functions / class
describing the Deep Learning mmodels used in the classification
"""
import os
import time
import glob
import numpy as np
import pandas as pd
from typing import Tuple
from tensorflow import config as tf_config
from tensorflow import keras
import plotext as pltxt
from birdsong.config import config
from birdsong.model.model_collections import get_model_architecture
from birdsong.utils import create_folder_if_not_exists
from birdsong.audiotransform.to_image import AudioPreprocessor
from birdsong.model.transform import get_train_data_set, get_validation_test_data_sets
from birdsong import PARENT_BASE_PATH
from birdsong.utils import read_prediction




class PlayModel():
    """This simple class is used to call and interact with the keras model.
    It can be used to train, evaluate, predict and plot the training curves of the
    model. Play attention to the use of private and static methods.
    """
    def __init__(self):
        self.model_call_label = config.MODEL_NAME
        self.nb_classes = None
        self.model = None
        self.metrics = None
        self.call_mode = None
        self.__train_ds = None #private
        self.__val_ds = None #private
        self.__test_ds = None #private


    def get_new_model(self):
        """This method is used to train a new model from scratch"""
        self.call_mode = 'train_from_begining'
        self.model = self.__train_new_model()

        self.metrics = self.evaluate_model(self.model, self.__test_ds)

        predictions = self.model.predict(self.__test_ds)
        predictions_df = read_prediction(predictions, self.__train_ds.class_names)
        return predictions_df

    def get_saved_model(self):
        """This method is used to load a saved model and train it again.
        Here, we call a private method to load and to train the model"""
        self.call_mode = 'train_saved_model'
        try:
            self.model = self.__load_train_model()

            self.metrics = self.evaluate_model(self.model, self.__test_ds)

            predictions = self.model.predict(self.__test_ds)
            predictions_df = read_prediction(predictions, self.__train_ds.class_names)
            return predictions_df
        except:
            raise ValueError("❌ No saved model found. Please train a model first")

    def get_model_from_checkpoint(self):
        """This method is used to train a model from the last checkpoint.
        """
        self.call_mode = 'train_from_checkpoint'
        try:
            self.model = self.__train_model_from_checkpoint()

            self.metrics = self.evaluate_model(self.model, self.__test_ds)

            predictions = self.model.predict(self.__test_ds)
            predictions_df = read_prediction(predictions, self.__train_ds.class_names)
            return predictions_df
        except:
            raise ValueError("❌ No checkpoint found")



    def predict_model(self, data_to_predict):
        """This method is used to predict the class of a new audio file."""
        try:
            self.model = self.load_model()
            audio_processor = AudioPreprocessor()
            signal_processed = audio_processor.preprocess_audio_array(data_to_predict)
            signal_tensor = np.expand_dims(signal_processed, axis=0)
            predictions = self.model.predict(signal_tensor)
            return predictions
        except:
            raise ValueError("❌ No saved model found. Please train a model first")


    def __train_new_model(self):
        """This method is used to train a new model from scratch.
        note that is a private method and can only be called inside the class."""

        self.__train_ds, self.__val_ds, self.__test_ds = self.set_train_data_set()

        class_names = self.__train_ds.class_names
        self.nb_classes = len(class_names)
        print(f"find {self.nb_classes} classes in data set")

        model = self.initialize_model(self.model_call_label, self.nb_classes)
        model = self.compile_model(model)
        print(model.summary())

        model, history = self.train_model(model=model,
                                          train_data=self.__train_ds,
                                          validation_data=self.__val_ds)

        self.plot_train_curves(history)
        timestamp = self.save_model(model)
        self.save_history(history,timestamp)
        return model

    def __train_model_from_checkpoint(self)-> keras.Model:
        """This method is used to train a model from the last checkpoint.
        note that is a private method and can only be called inside the class."""

        self.__train_ds, self.__val_ds, self.__test_ds = self.set_train_data_set()

        class_names = self.__train_ds.class_names
        self.nb_classes = len(class_names)

        model = self.initialize_model(self.model_call_label, self.nb_classes)
        model = self.compile_model(model)

        model = self.set_model_from_checkpoint(model)

        model, history = self.train_model(model, self.__train_ds, self.__val_ds)

        self.plot_train_curves(history)
        timestamp = self.save_model(model)
        self.save_history(history,timestamp)
        return model

    def __load_train_model(self):
        """This method is used to load a saved model and train it again.
        note that is a private method and can only be called inside the class."""

        model = self.load_model()
        self.__train_ds, self.__val_ds, self.__test_ds = self.set_train_data_set()
        class_names = self.__train_ds.class_names
        self.nb_classes = len(class_names)

        model, history = self.train_model(model,self.__train_ds, self.__val_ds)

        self.plot_train_curves(history)
        timestamp = self.save_model(model)
        self.save_history(history,timestamp)
        return model

    @staticmethod
    def load_model()-> keras.Model:
        """
        Return a saved model:
        - locally (latest one in alphabetical order)
        Return None (but do not Raise) if no model is found

        """
        if config.MODEL_TARGET == "local":

            print(f"\nLoad latest model from local registry...")

            model_save_dir = os.path.join(PARENT_BASE_PATH, config.MODEL_SAVE_PATH, config.MODEL_NAME)

            # Get the latest model version name by the timestamp on disk
            local_model_paths = glob.glob(f"{model_save_dir}/*.h5")

            if not local_model_paths:
                return None

            most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

            print( most_recent_model_path_on_disk)

            print(f"\nLoad latest model from disk...")

            latest_model = keras.models.load_model(most_recent_model_path_on_disk)

            print("✅ Model loaded from local disk")
            print("-"*30)
            return latest_model
        else:
            return None

    @staticmethod
    def evaluate_model(model, test_data)-> dict:
        """
        Evaluate the model. The model must be loaded or trained first.
        """
        print("i am here")
        metrics  = model.evaluate(test_data,
                                       verbose=0,
                                       return_dict = True)
        print("✅ Model evaluated")
        print(f"Training loss {metrics[model.metrics_names[0]]} accuracy {metrics[model.metrics_names[1]]}")
        return metrics

    @staticmethod
    def set_train_data_set():
         train_ds = get_train_data_set()
         val_ds, test_ds = get_validation_test_data_sets()
         return train_ds, val_ds, test_ds


    @staticmethod
    def initialize_model(model_call_label, nb_classes):
        """
        Initialize the model
        """
        print('Initializing model...')

        print('model data sets configaration done')
        processed_info = AudioPreprocessor()
        (n_rows,n_columns) = processed_info.get_image_sample_shape()

        print(f"model input shape : {(n_rows,n_columns,1)}")
        print(f"model output is {nb_classes} classes")

        model = get_model_architecture(model_call_label, nb_classes,
                                    input_shape=(n_rows,n_columns,1))

        print(f"✅ Model {model_call_label} initialized")
        print("-"*30)
        return model

    @staticmethod
    def compile_model(model: keras.Model)-> keras.Model:
        """
        Compile the model
        """
        adam = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        model.compile(optimizer=adam,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        print("✅ Model compiled")
        print("-"*30)
        return model

    @staticmethod
    def train_model(model,
                    train_data,
                    validation_data,
                    epochs=1000)-> Tuple[keras.Model, dict]:
        """
        Train the model
        """
        print("Training user selected model...")

        print("Num GPUs Available: ", len(tf_config.list_physical_devices('GPU')))

        es = keras.callbacks.EarlyStopping(monitor='val_loss',
                        mode='min',
                        verbose=1,
                        patience=config.PATIENCE)

        print("✅ Early stopping callback created")

        # Create a callback that saves the model's weights
        checkpoint_dir = os.path.join(PARENT_BASE_PATH, config.CHECKPOINT_FOLDER_PATH)
        create_folder_if_not_exists(checkpoint_dir)
        create_folder_if_not_exists(os.path.join(checkpoint_dir, config.MODEL_NAME))

        # Include the epoch in the file name (uses `str.format`)
        checkpoint_path = os.path.join(checkpoint_dir, config.MODEL_NAME,"cp-best.ckpt")

        # Create a callback that saves the model's weights every epoch and keeps only the best one
        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq= 'epoch',
            save_best_only = True,
            monitor ='val_loss',
            mode = 'min')
        print("✅ Checkpoint callback created")

        #override epocs number
        try:
            epochs = config.EPOCHS
        except:
            epochs = epochs

        print("epochs number : ", epochs)
        history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=[es, cp_callback],
            verbose=1)


        print("✅ User selected model trained")
        print("-"*30)
        return model, history

    @staticmethod
    def save_model(model: keras.Model = None) -> None:
        """
        Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
        """
        model_save_dir = os.path.join(PARENT_BASE_PATH, config.MODEL_SAVE_PATH, config.MODEL_NAME)
        create_folder_if_not_exists(model_save_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Save model locally
        model_path = os.path.join(model_save_dir, f"{timestamp}.h5")
        model.save(model_path)

        print("✅ Model saved locally")
        print("-"*30)
        return timestamp

    @staticmethod
    def set_model_from_checkpoint(model : keras.Model)-> keras.Model:
        try:
            checkpoint_dir = os.path.join(PARENT_BASE_PATH,config.CHECKPOINT_FOLDER_PATH,config.MODEL_NAME)
            checkpoint_path = os.path.join(checkpoint_dir,"cp-best.ckpt")
            model.load_weights(checkpoint_path)
            return model
        except:
            raise ValueError("❌ No checkpoint found")

    @staticmethod
    def plot_train_curves(history, metrics_keys = ['accuracy']):

        pltxt.clf()
        pltxt.plotsize(100, 30)
        pltxt.subplots(1, 2)
        pltxt.subplot(1, 1)
        pltxt.subplot(1, 2)

        pltxt.subplot(1, 1)
        pltxt.theme('pro')
        pltxt.plot(history.history['loss'], marker = "hd", label = "loss")
        pltxt.plot(history.history['val_loss'], marker = "hd", label = "val loss")
        pltxt.xlabel('epochs')
        pltxt.ylabel('loss')
        pltxt.title('Model loss')

        pltxt.subplot(1, 2)
        pltxt.theme('pro')
        pltxt.plot(history.history[metrics_keys[0]], marker = "hd", label = metrics_keys[0])
        pltxt.plot(history.history['val_' + metrics_keys[0]], marker = "hd", label = "val " + metrics_keys[0])
        pltxt.xlabel('epochs')
        pltxt.ylabel(metrics_keys[0])
        pltxt.title('Model '+ metrics_keys[0])

        pltxt.show()

    @staticmethod
    def save_history(object_history, timestamp):
        """Save the train history as a csv file
        """
        print('Saving model train history...')
        create_folder_if_not_exists(os.path.join(PARENT_BASE_PATH,
                                    config.MODEL_SAVE_PATH,
                                    config.MODEL_NAME))
        hist_csv_file = os.path.join(PARENT_BASE_PATH,
                                    config.MODEL_SAVE_PATH,
                                    config.MODEL_NAME,
                                    f"{timestamp}_history_log.csv")
        hist_df = pd.DataFrame(object_history.history)
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
        print(f"✅ Model train history saved in {hist_csv_file}")
        return None
