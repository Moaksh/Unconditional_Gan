
collection_name = "cyberkongz"
nrows = 8
generation_type = "default"
from model_load.lightweight_gan.lightweight_gan import load_lightweight_model
from IPython.display import Image

model = load_lightweight_model()
image_saved_path, generated_image = model.generate_app(
    nrow=1,
    checkpoint=-1,
    types=generation_type
)
Image(image_saved_path)
