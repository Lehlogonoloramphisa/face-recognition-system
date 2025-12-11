# Face Recognition System

End-to-end face processing tools built around MTCNN detection and FaceNet embeddings, plus a preprocessing/augmentation pipeline.

## Project Layout
- `data/` � place input images here (e.g., folders of faces).
- `models/` � optional FaceNet weights (`facenet_keras_weights.h5`). If incompatible, the built-in keras-facenet weights are used.
- `preview/` � pipeline outputs (preprocessed/augmented images).
- `src/` � source code:
  - `compare.py` � compare faces (pairwise or batch) with optional visualization.
  - `embed.py` � MTCNN face detection + FaceNet embedding utilities.
  - `detect_faces.py` � extract and save a detected face.
  - `pipeline_main.py` � CLI wrapper for the unified preprocessing + augmentation pipeline.
  - `pipeline/` � preprocessing (TF-Hub), augmentation, unified `face_pipeline.py`, plus legacy training scaffolds.
  - `full_pipeline.py` � sample script wiring preprocess/augment/embed steps.

## Setup
1) Create/activate a virtual environment.
2) Install dependencies: `pip install -r requirements.txt`.
3) (Optional) Place compatible FaceNet weights at `models/facenet_keras_weights.h5`. If loading fails, defaults are used.

## Step-by-Step Instructions
1) Set up environment
   - Create/activate a virtualenv.
   - Install dependencies: `pip install -r requirements.txt`.

2) (Optional) Add FaceNet weights
   - If you have compatible weights, place them at `models/facenet_keras_weights.h5`.
   - If they don�t match the keras-facenet model, the defaults will be used.

3) Prepare input images
   - Put your images under `data/` (e.g., `data/folder_resized/`).

4) Run preprocessing + augmentation (unified pipeline)
   ```bash
   python src/pipeline_main.py \
     --input_image_path data/folder_resized/picture-eight.jpg \
     --class_name sample \
     --task_name Denoising \
     --num_augmented 50 \
     --preview_dir preview
   ```
   - Outputs to `preview/<class_name>/` (preprocessed PNG + augmented JPEGs).

5) Compare faces
   - Pairwise (with optional viz):
     ```bash
     python src/compare.py data/ref.jpg --candidate data/test.jpg --threshold 0.95 --viz outputs/ref_test_viz.jpg
     ```
   - Batch all pairs in a folder (optional interactive viz):
     ```bash
     python src/compare.py data/folder_resized --threshold 0.95 --visualize
     ```

6) Extract a face crop
   ```bash
   python src/detect_faces.py data/input.jpg data/face.jpg --size 160 160
   ```

## Core Usage
- Single pair compare with optional visualization:
  ```bash
  python src/compare.py data/ref.jpg --candidate data/test.jpg --threshold 0.95 --viz outputs/ref_test_viz.jpg
  ```
- Batch compare a folder (all pairs), optional per-pair display:
  ```bash
  python src/compare.py data/folder_resized --threshold 0.95 --visualize
  ```
- Extract and save a detected face crop:
  ```bash
  python src/detect_faces.py data/input.jpg data/face.jpg --size 160 160
  ```

## Preprocess + Augment Pipeline
- Unified pipeline (preprocess via TF-Hub task, then augment `N` images):
  ```bash
  python src/pipeline_main.py \
    --input_image_path data/folder_resized/picture-eight.jpg \
    --class_name sample \
    --task_name Denoising \
    --num_augmented 50 \
    --preview_dir preview
  ```
  Outputs to `preview/<class_name>/` with a preprocessed PNG and augmented JPEGs.

## Notes
- MTCNN selects the largest detected face; embeddings are L2-normalized.
- Typical FaceNet L2 thresholds range ~0.8�1.2; tune `--threshold` for your data.
- If you see weight shape mismatches, remove/replace `models/facenet_keras_weights.h5` to let the default weights load.
#
