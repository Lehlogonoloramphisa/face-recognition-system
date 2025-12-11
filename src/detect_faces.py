from embed import extract_face

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and save detected face.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("output", help="Output face image path")
    parser.add_argument("--size", type=int, nargs=2, default=[160, 160], metavar=("W", "H"), help="Resize width height")
    args = parser.parse_args()

    face = extract_face(args.image, required_size=tuple(args.size))
    from PIL import Image
    Image.fromarray(face).save(args.output)
    print(f"Saved face to {args.output}")
