import sys
import argparse
import time
from pipeline.face_pipeline import FacePipeline


def main(args):
    print(">>> STARTED FACE PIPELINE")
    start_time = time.time()

    pipeline = FacePipeline(
        image_path=args.input_image_path,
        class_name=args.class_name,
        task_name=args.task_name,
        preview_dir=args.preview_dir,
        num_augmented=args.num_augmented,
    )

    output_dir = pipeline.exec()

    end_time = time.time()
    print(f">>> FINISHED in {end_time - start_time:.2f} seconds")
    print(f">>> Augmented images are saved in: {output_dir}")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Preprocess and augment a single image using FacePipeline")

    parser.add_argument(
        "--input_image_path", type=str, required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--class_name", type=str, required=True,
        help="Class name for output images (used as folder + prefix)"
    )
    parser.add_argument(
        "--task_name", type=str, default="Denoising",
        help="Preprocessing task: Denoising, Dehazing_Indoor, Dehazing_Outdoor, Deblurring, Deraining, Enhancement, Retouching"
    )
    parser.add_argument(
        "--preview_dir", type=str, default="preview",
        help="Directory where preprocessed + augmented images will be saved"
    )
    parser.add_argument(
        "--num_augmented", type=int, default=50,
        help="Number of augmented images to generate"
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
