from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
import numpy as np
from tensorflow import keras
from PIL import Image
import cv2
import uvicorn

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_breast_novel = keras.models.load_model('model/breast_novel_model')
model_breast_pre = keras.models.load_model('model/breast_pre_train_model')
model_blood_novel = keras.models.load_model('model/blood_novel_model')
model_blood_pre = keras.models.load_model('model/blood_pre_train_model')

CLASS_NAMES_BREAST = ['Malignant', 'Normal']
CLASS_NAMES_BLOOD = ['basophil', 'eosinophil', 'erythroblast', 'immature graulocytes', 'lymphocyte', 'monocyte',
                     'neutrophil', 'platelet']


def read_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/", tags=["Root"])
async def root():
    return {"message": "Hello World, From Shai :D ... use /docs to proceed"}


@app.post("/predict_novel_breast")
async def create_upload_file(file: UploadFile = File(...)):
    image = read_image(await file.read())
    image = np.expand_dims(image, 0)
    prediction = model_breast_novel.predict(image)
    confidence = prediction[0] if prediction[0] > 0.5 else 1.0 - prediction[0]
    predicted_class = CLASS_NAMES_BREAST[0] if prediction[0] <= 0.5 else CLASS_NAMES_BREAST[1]

    return {'confidence': confidence[0].item(), 'class': predicted_class}


@app.post("/predict_pre_train_breast")
async def create_upload_file(file: UploadFile = File(...)):
    image = read_image(await file.read())
    image = cv2.resize(image, (75, 75), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = np.expand_dims(image, 0)
    prediction = model_breast_pre.predict(image/255)
    print((prediction))
    confidence = prediction[0] if prediction[0] > 0.5 else 1.0 - prediction[0]
    predicted_class = CLASS_NAMES_BREAST[0] if prediction[0] <= 0.5 else CLASS_NAMES_BREAST[1]

    print(type(confidence[0]))
    return {'confidence': confidence[0].item(), 'class': predicted_class}


@app.post("/predict_novel_blood")
async def create_upload_file(file: UploadFile = File(...)):
    image = read_image(await file.read())
    #image = np.expand_dims(image, 0)

    image = image.reshape(-1, 28, 28, 3)
    image=image/255
    model_blood_novel.summary()
    confidence = model_blood_novel.predict(image)
    print(confidence)
    prediction = np.rint(confidence)
    print(prediction)
    y = int(max(prediction[0] * [0, 1, 2, 3, 4, 5, 6, 7]))
    print(y)
    predicted_class = CLASS_NAMES_BLOOD[y]
    return {'confidence': max(confidence[0]).item(), 'class': predicted_class}


#
@app.post("/predict_pre_train_blood")
async def create_upload_file(file: UploadFile = File(...)):
    image = read_image(await file.read())
    image = np.expand_dims(image, 0)

    # image = image.reshape(-1, 28, 28, 3)
    confidence = model_blood_novel.predict(image/255)
    prediction = np.rint(confidence)
    y = int(max(prediction[0] * [0, 1, 2, 3, 4, 5, 6, 7]))
    print(y)
    predicted_class = CLASS_NAMES_BLOOD[y]
    return {'confidence': max(confidence[0]).item(), 'class': predicted_class}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
