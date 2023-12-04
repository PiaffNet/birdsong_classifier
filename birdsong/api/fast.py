# $WIPE_BEGIN

from birdsong.model.model import load_model, predict_model
from birdsong.audiotransform.to_image import AudioPreprocessor
import numpy as np

# $WIPE_END

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app.state.model = load_model()


@app.get("/predict")
def predict(
        path: str  #path to the directory where the test file is
    ):      # 1
    """
    Predict the bird, with the pre chosen model in load_model
    """

    model = app.state.model
    assert model is not None

    prediction = predict_model(model, path)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    max = np.max(prediction)
    class_id = np.where(prediction == max)[1]

    return dict(bird = int(class_id))


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello birdies ! ;) ")
    # $CHA_END
