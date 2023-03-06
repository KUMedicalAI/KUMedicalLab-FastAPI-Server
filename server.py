from io import BytesIO

from fastapi import FastAPI, UploadFile, File
import numpy as np
from tensorflow import keras
from PIL import Image
import uvicorn

app = FastAPI()

model = keras.models.load_model('model/breast_self_model')

CLASS_NAMES = ['Malignant', 'Normal']

def read_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/", tags=["Root"])
async def root():
    return {"message": "Hello World, From Shai :D ... use /docs to proceed"}


@app.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):
    image = read_image(await file.read())
    image = np.expand_dims(image, 0)
    prediction = model.predict(image)
    print((prediction))
    confidence = prediction[0] if prediction[0] > 0.5 else 1.0-prediction[0]
    predicted_class = CLASS_NAMES[0] if prediction[0] <= 0.5 else CLASS_NAMES[1]

    print(type(confidence[0]))
    return {'confidence': f"{confidence[0].item():.2%}", 'class': predicted_class}

if __name__ == "__main__":
  uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)