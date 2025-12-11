import os
import sys
import argparse
import timeit
from src.pipeline.image_preprocessing import ImagePreProcessing
from src.pipeline.image_augmented import ImageAugmented
from src.embed import get_embedding  # assumes you have this function


def run_pipeline(input_image_path: str, class_name: str, output_dir: str = "preview"):
    print(">>> PIPELINE STARTED")
    start = timeit.default_timer()

    # 1️⃣ Preprocess the image
    preprocessed_file = ImagePreProcessing(
        imageURL=input_image_path,
        taskName="Denoising",
        className=class_name
    ).exec()
    print(f">>> Preprocessed image saved at: {preprocessed_file}")

    # 2️⃣ Augment the preprocessed image
    augmented_files = ImageAugmented(
        imgPath=preprocessed_file,
        saveToDir=output_dir,
        class_name=class_name
    ).exec()
    print(f">>> Augmented {len(augmented_files)} images into directory: {os.path.join(output_dir, class_name)}")

    # 3️⃣ Generate embeddings for all augmented images
    embeddings = {}
    for f in augmented_files:
        emb = get_embedding(f)
        embeddings[f] = emb
    print(f">>> Generated embeddings for {len(embeddings)} images")

    stop = timeit.default_timer()
    print(f">>> PIPELINE FINISHED in {stop - start:.2f} seconds")
    return embeddings


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Full FaceNet Pipeline: preprocess, augment, embeddings")
    parser.add_argument("--input_image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--class_name", type=str, required=True, help="Class name for outputs")
    parser.add_argument("--output_dir", type=str, default="preview", help="Directory to save augmented images")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    embeddings = run_pipeline(
        input_image_path=args.input_image_path,
        class_name=args.class_name,
        output_dir=args.output_dir
    )
