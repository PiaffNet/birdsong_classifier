# $WIPE_BEGIN

from birdsong.model.model import load_model, predict_model
from birdsong.audiotransform.to_image import AudioPreprocessor

import numpy as np
import os

# $WIPE_END

from typing import Annotated
from fastapi import FastAPI, File, UploadFile
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

classes = ['Sand Martin',
 'Barn Swallow',
 'California Quail',
 'Canada Goose',
 'Caspian Tern',
 'Common Loon',
 'Northern Raven',
 'Common Redpoll',
 'Common Tern',
 'Black-necked Grebe',
 'Eurasian Collared Dove',
 'Common Starling',
 'Gadwall',
 'Eurasian Teal',
 'Golden Eagle',
 'Great Egret',
 'European Herring Gull',
 'Horned Lark',
 'House Sparrow',
 'Mallard',
 'Merlin',
 'Northern Shoveler',
 'Western Osprey',
 'Pectoral Sandpiper',
 'Peregrine Falcon',
 'Red Crossbill',
 'Ring-billed Gull',
 'Ring-necked Duck',
 'Rock Dove',
 'Ruddy Duck',
 'Short-eared Owl',
 'silence',
 'Snow Bunting',
 'Tundra Swan']


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello birdies ! ;) ")
    # $CHA_END


@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_sound(sound: UploadFile | None = None):
    return {'filename': sound.filename, 'content': sound.content_type}


@app.get("/predict")
def predict(file):
    """
    Predict the bird, with the pre chosen model in load_model
    """

    model = app.state.model
    assert model is not None

    path = create_upload_file(file)["filename"]
    prediction = predict_model(model, path)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    max = np.max(prediction)
    class_id = np.where(prediction == max)[1]

    return dict(bird = classes[int(class_id)], confidence = float(max))
