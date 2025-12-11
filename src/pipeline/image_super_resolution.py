import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


class ImageSuperResolution:
    """Apply ESRGAN super-resolution to an image."""

    def __init__(self, image_path: str, saved_model_path: str) -> None:
        print(">>> [IMAGE SUPER RESOLUTION] running...")
        os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
        self._image_path = image_path
        self._saved_model_path = saved_model_path

    def exec(self) -> str:
        hr_image = self.preprocess_image(self._image_path)
        model = hub.load(self._saved_model_path)

        start = time.time()
        fake_image = model(hr_image)
        fake_image = tf.squeeze(fake_image)
        print("Time Taken: %f" % (time.time() - start))

        output_file = self.save_image(tf.squeeze(fake_image), filename="super_resolution")
        return output_file

    def preprocess_image(self, image_path: str):
        """Load image from path and preprocess for the model."""
        hr_image = tf.image.decode_image(tf.io.read_file(image_path))
        if hr_image.shape[-1] == 4:
            hr_image = hr_image[..., :-1]
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
        hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, 0)

    def save_image(self, image, filename: str):
        """Save tensor image to disk."""
        if not isinstance(image, Image.Image):
            image = tf.clip_by_value(image, 0, 255)
            image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        output_path = f"{filename}.jpg"
        image.save(output_path)
        print(f"Saved as {output_path}")
        return output_path

    def plot_image(self, image, title: str = ""):
        image = np.asarray(image)
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        plt.imshow(image)
        plt.axis("off")
        plt.title(title)
