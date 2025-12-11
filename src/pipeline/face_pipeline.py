import os
import shutil
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


class FacePipeline:
    def __init__(
        self,
        image_path: str,
        class_name: str,
        task_name: str = "Denoising",
        preview_dir: str = "preview",
        num_augmented: int = 50,
        output_size: int = 256,
    ):
        """
        Full preprocessing + augmentation pipeline for a single image.
        """
        self.image_path = image_path
        self.class_name = class_name
        self.task_name = task_name
        self.preview_dir = preview_dir
        self.num_augmented = num_augmented
        self.output_size = output_size

        # Set up augmented image directory
        self.augmented_dir = os.path.join(self.preview_dir, self.class_name)
        if os.path.exists(self.augmented_dir):
            shutil.rmtree(self.augmented_dir)
        os.makedirs(self.augmented_dir, exist_ok=True)

    def preprocess_image(self):
        """
        Apply a TF-Hub preprocessing model (denoising, deblurring, etc.)
        """
        model_map = {
            "Denoising": "https://tfhub.dev/sayakpaul/maxim_s-3_denoising_sidd/1",
            "Dehazing_Indoor": "https://tfhub.dev/sayakpaul/maxim_s-2_dehazing_sots-indoor/1",
            "Dehazing_Outdoor": "https://tfhub.dev/sayakpaul/maxim_s-2_dehazing_sots-outdoor/1",
            "Deblurring": "https://tfhub.dev/sayakpaul/maxim_s-3_deblurring_gopro/1",
            "Deraining": "https://tfhub.dev/sayakpaul/maxim_s-2_deraining_raindrop/1",
            "Enhancement": "https://tfhub.dev/sayakpaul/maxim_s-2_enhancement_lol/1",
            "Retouching": "https://tfhub.dev/sayakpaul/maxim_s-2_enhancement_fivek/1",
        }

        model_url = model_map.get(self.task_name, model_map["Denoising"])
        print(f">>> [Preprocessing] Task: {self.task_name}, Model: {model_url}")

        inputs = tf.keras.Input(shape=(self.output_size, self.output_size, 3))
        hub_layer = hub.KerasLayer(model_url)
        outputs = hub_layer(inputs)
        model = tf.keras.Model(inputs, outputs)

        img = np.asarray(Image.open(self.image_path).convert("RGB"), np.float32) / 255.0
        img = tf.image.resize(img, (self.output_size, self.output_size))
        img = np.expand_dims(img, axis=0)

        pred = model(img, training=False)
        pred = np.clip(pred.numpy()[0], 0.0, 1.0)
        preprocessed_file = os.path.join(self.augmented_dir, f"{self.class_name}_preprocessed.png")
        Image.fromarray((pred * 255).astype(np.uint8)).save(preprocessed_file)
        print(f">>> Preprocessed image saved to {preprocessed_file}")

        return preprocessed_file

    def augment_image(self, img_path):
        """
        Generate augmented images from the preprocessed image.
        """
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for _ in datagen.flow(
            x,
            batch_size=1,
            save_to_dir=self.augmented_dir,
            save_prefix=self.class_name,
            save_format="jpeg",
        ):
            i += 1
            if i >= self.num_augmented:
                break

        print(f">>> {self.num_augmented} augmented images saved to {self.augmented_dir}")

    def exec(self):
        preprocessed_file = self.preprocess_image()
        self.augment_image(preprocessed_file)
        return self.augmented_dir
