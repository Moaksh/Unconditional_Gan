from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from model_load.lightweight_gan.lightweight_gan import load_lightweight_model
from IPython.display import Image
from pydantic import BaseModel
from io import BytesIO
import base64
import os

app = FastAPI()


generation_type = "default"
model = load_lightweight_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def generate():
    image_saved_path, generated_image = model.generate_app(
    nrow=1,
    checkpoint=-1,
    types=generation_type
    )
    buffer = BytesIO()
    Image(buffer, format="JPG")
    imgstr = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return Response(content = imgstr, media_type = "image/jpg")
