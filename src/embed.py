import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
from PIL import Image
import threading
import os

# Thread-safe singleton pattern for models
_model_lock = threading.Lock()
_detector = None
_model = None
_weights_loaded = False

def get_detector():
    """Lazy-load MTCNN detector (thread-safe)."""
    global _detector
    with _model_lock:
        if _detector is None:
            _detector = MTCNN()
    return _detector

def get_model():
    """Lazy-load FaceNet model (thread-safe)."""
    global _model, _weights_loaded
    with _model_lock:
        if _model is None:
            weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "facenet_keras_weights.h5"))
            _model = FaceNet()
            if os.path.exists(weights_path) and not _weights_loaded:
                try:
                    print(f"Loading FaceNet weights from {weights_path}")
                    _model.model.load_weights(weights_path)
                    _weights_loaded = True
                except Exception as exc:
                    print(f"Warning: failed to load custom weights ({exc}); using default keras-facenet weights.")
            elif not os.path.exists(weights_path):
                print(f"Warning: weights file not found at {weights_path}, using default keras-facenet weights.")
    return _model

def extract_face(filename, required_size=(160, 160)):
    """Detects the largest face and returns a resized RGB array."""
    detector = get_detector()
    image = Image.open(filename).convert("RGB")
    pixels = np.asarray(image)
    
    # Use minimal MTCNN settings for speed
    results = detector.detect_faces(pixels)
    print(f"Detected faces in {filename}: {results}")
    
    if not results:
        raise ValueError(f"No face detected in {filename}")
    
    # Pick largest face by area
    x1, y1, w, h = sorted(results, key=lambda r: r["box"][2] * r["box"][3], reverse=True)[0]["box"]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = x1 + w, y1 + h
    
    face = pixels[y1:y2, x1:x2]
    
    # Use faster resizing
    face_image = Image.fromarray(face)
    face_image = face_image.resize(required_size, Image.Resampling.BILINEAR)
    
    return np.asarray(face_image)

def get_embedding(face_pixels):
    """Returns L2-normalized FaceNet embedding for a face array."""
    model = get_model()

    # Convert to float32; let keras-facenet handle its own prewhitening internally
    face_pixels = face_pixels.astype("float32")
    face_pixels = np.expand_dims(face_pixels, axis=0)

    # Get embedding
    embedding = model.embeddings(face_pixels)[0]

    # L2 normalize for safety
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding
