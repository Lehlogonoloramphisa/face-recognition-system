import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from .image_super_resolution import ImageSuperResolution
import os


class ImagePreProcessing:
    def __init__(self, taskName: str = "Denoising", imageURL: str = "", className: str = ""):
        print(">>> [IMAGE PRE-PROCESSING] running...")
        self.__IMAGE_URL = imageURL
        self.__CLASS_NAME = className

        defaultTaskList = [
            "Denoising",
            "Dehazing_Indoor",
            "Dehazing_Outdoor",
            "Deblurring",
            "Deraining",
            "Enhancement",
            "Retouching",
            "Super_Resolution",
        ]

        self.__TASK_NAME = taskName if taskName in defaultTaskList else "Denoising"

        # TF-Hub model mapping
        self.model_handle_map = {
            "Denoising": "https://tfhub.dev/sayakpaul/maxim_s-3_denoising_sidd/1",
            "Dehazing_Indoor": "https://tfhub.dev/sayakpaul/maxim_s-2_dehazing_sots-indoor/1",
            "Dehazing_Outdoor": "https://tfhub.dev/sayakpaul/maxim_s-2_dehazing_sots-outdoor/1",
            "Deblurring": "https://tfhub.dev/sayakpaul/maxim_s-3_deblurring_gopro/1",
            "Deraining": "https://tfhub.dev/sayakpaul/maxim_s-2_deraining_raindrop/1",
            "Enhancement": "https://tfhub.dev/sayakpaul/maxim_s-2_enhancement_lol/1",
            "Retouching": "https://tfhub.dev/sayakpaul/maxim_s-2_enhancement_fivek/1",
            "Super_Resolution": "https://tfhub.dev/captain-pool/esrgan-tf2/1",
        }

    def exec(self):
        if self.__TASK_NAME == "Super_Resolution":
            # Call the super-resolution module separately
            ImageSuperResolution(self.__CLASS_NAME + ".png", "./services/models/esrgan-super-resolution-model").exec()
            return self.__CLASS_NAME + ".png"

        # Run preprocessing model eagerly and get numpy output
        final_image = self.get_model(self.model_handle_map[self.__TASK_NAME])
        final_image = np.clip(final_image[0], 0.0, 1.0)
        final_image_pil = Image.fromarray((final_image * 255).astype(np.uint8))
        output_path = self.__CLASS_NAME + ".png"
        final_image_pil.save(output_path)

        print(f">>> Preprocessed image saved to {output_path}")
        return output_path

    def process_image(self, image_path, target_dim=256):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((target_dim, target_dim))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
        return tf.convert_to_tensor(img_array)

    def get_model(self, model_url):
        """Load TF-Hub layer and run inference eagerly on the configured image."""
        hub_layer = hub.KerasLayer(model_url, trainable=False)

        # Load image as numpy array
        img = Image.open(self.__IMAGE_URL).convert("RGB")
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # shape: (1, 256, 256, 3)

        # Evaluate the TF Hub layer eagerly
        outputs = hub_layer(tf.convert_to_tensor(img_array, dtype=tf.float32))
        return outputs.numpy()  # now it’s a real NumPy array
