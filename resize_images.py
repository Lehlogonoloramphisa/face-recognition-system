import os
from PIL import Image

input_folder = "data/folder_of_images"
output_folder = "data/folder_resized"

os.makedirs(output_folder, exist_ok=True)

MAX_SIZE = 1024

for file in os.listdir(input_folder):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(input_folder, file)
    img = Image.open(path)

    img.thumbnail((MAX_SIZE, MAX_SIZE))  # keep aspect ratio
    img.save(os.path.join(output_folder, file))

    print("Resized:", file)
