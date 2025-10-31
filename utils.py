# utils.py
from PIL import Image
import io

def load_image(uploaded_file) -> Image.Image:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    return image

