from aura_sr import AuraSR
import requests
from io import BytesIO
from PIL import Image
import os

os.makedirs("test", exist_ok=True)


def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if request was successful
        image_data = BytesIO(response.content)
        return Image.open(image_data)
    except requests.RequestException as e:
        print(f"Failed to fetch image from url {url}: {e}")
        return None
    except IOError as e:
        print(f"Failed to open image: {e}")
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

