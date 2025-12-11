import os
import shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image


class ImageAugmented:
    def __init__(self, imgPath: str, saveToDir: str = "preview", class_name: str = "", num_augmented: int = 50):
        print(">>> [IMAGE AUGMENTATION] running...")
        self.imgPath = imgPath
        self.saveToDir = saveToDir
        self.class_name = class_name
        self.num_augmented = num_augmented

        # Setup output directory
        self.output_dir = os.path.join(self.saveToDir, self.class_name)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

    def exec(self):
        # Create the data generator
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        # Load image
        img = load_img(self.imgPath)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=self.output_dir,
                                  save_prefix=self.class_name, save_format='jpeg'):
            i += 1
            if i >= self.num_augmented:
                break

        print(f">>> Saved {self.num_augmented} augmented images to {self.output_dir}")
        return self.output_dir
