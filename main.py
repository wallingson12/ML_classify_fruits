from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# ---- Modelo ----
model = tf.keras.models.load_model("modelo_frutas.h5")  # mesmo diretório

# ---- Classes ----
train_dir = "train_variacoes"  # mesmo diretório, pastas de frutas
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# ---- Preprocessamento ----
def preprocess_image(image_bytes, target_size=(128,128)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)/255.0
    return np.expand_dims(image_array, axis=0)

# ---- Rota principal serve HTML ----
@app.get("/")
def home():
    return FileResponse("index.html")  # mesmo diretório

# ---- Endpoint de previsão ----
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)
        preds = model.predict(img)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        return {"class": class_names[class_idx], "confidence": confidence}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# ---- Rodar API ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)