from aura_sr import AuraSR
import requests
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import os

os.makedirs("test", exist_ok=True)

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        return Image.open(image_data)
    except (requests.HTTPError, requests.RequestException) as e:
        print(f"Failed to download the image. Error: {e}")
    except UnidentifiedImageError:
        print("Failed to identify and open the image.")
    return None

aura_sr = AuraSR.from_pretrained()

image = load_image_from_url("https://mingukkang.github.io/GigaGAN/static/images/iguana_output.jpg")
if image:
    image = image.resize((256, 256))

    upscaled_image = aura_sr.upscale_4x(image)
    upscaled_image.save(os.path.join("test", "output.png"))

    upscaled_image = aura_sr.upscale_4x_overlapped(image, weight_type='constant')
    upscaled_image.save(os.path.join("test", "output_overlapped_constant.png"))

    upscaled_image = aura_sr.upscale_4x_overlapped(image, weight_type='checkboard')
    upscaled_image.save(os.path.join("test", "output_checkerboard.png"))
else:
    print("No image to process.")
