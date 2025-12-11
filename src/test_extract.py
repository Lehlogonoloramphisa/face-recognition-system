from src.your_module import extract_face  # Adjust import
face_pixels = extract_face("data/folder_resized/picture-eight.jpg")
print("Face shape:", face_pixels.shape)
