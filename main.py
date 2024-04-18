from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from model_load.lightweight_gan.lightweight_gan import load_lightweight_model
from IPython.display import Image
from fastapi.responses import FileResponse
from pathlib import Path
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


@app.get("/get_image")
async def get_image():
    image_path = Path("./results/default-generated--1/0.jpg")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(image_path)
