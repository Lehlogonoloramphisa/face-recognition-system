import argparse
import os
import numpy as np
from embed import extract_face, get_embedding
from PIL import Image
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
import pickle
import hashlib

# Cache for embeddings to avoid recomputation
_embedding_cache = {}

def get_image_hash(filepath):
    """Generate hash for image file to use as cache key."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def show_pair(img1, img2, distance, verdict):
    """Display two images and wait for user key press."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(img1)
    axes[0].set_title("Image 1")
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    plt.suptitle(f"Distance: {distance:.4f} | {verdict}")
    plt.tight_layout()

    # Show the plot without blocking
    plt.show(block=False)

    print("\n➡ Press ENTER to continue, or 'q' then ENTER to quit.")
    user_input = input()

    plt.close(fig)

    if user_input.lower() == "q":
        return False
    return True

def get_cached_embedding(img_path, use_cache=True):
    """Get embedding from cache or compute it."""
    if not use_cache:
        face = extract_face(img_path)
        return get_embedding(face)
    
    img_hash = get_image_hash(img_path)
    if img_hash in _embedding_cache:
        return _embedding_cache[img_hash]
    
    face = extract_face(img_path)
    embedding = get_embedding(face)
    _embedding_cache[img_hash] = embedding
    return embedding

def compare_faces(img1_path, img2_path, threshold=0.95, visualize=False, use_cache=True):
    """Compare two face images and optionally visualize them."""
    start_time = time.time()
    
    try:
        # Get embeddings (from cache if available)
        emb1 = get_cached_embedding(img1_path, use_cache)
        emb2 = get_cached_embedding(img2_path, use_cache)

        distance = np.linalg.norm(emb1 - emb2)
        verdict = "Same Person" if distance < threshold else "Different People"
        elapsed = time.time() - start_time

        print(f"\nComparing:\n  {os.path.basename(img1_path)}\n  {os.path.basename(img2_path)}")
        print(f"Distance: {distance:.4f} | Threshold: {threshold} | Verdict: {verdict} | Time: {elapsed:.2f}s")

        if visualize:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
            continue_flag = show_pair(img1, img2, distance, verdict)
            return verdict, distance, continue_flag

        return verdict, distance, True
    except Exception as e:
        print(f"❌ Error comparing {img1_path} and {img2_path}: {e}")
        return "Error", float('inf'), True

def precompute_all_embeddings(image_paths, max_workers=4):
    """Precompute embeddings for all images in parallel."""
    print(f"Precomputing embeddings for {len(image_paths)} images...")
    
    def process_image(path):
        try:
            return path, get_cached_embedding(path, use_cache=False)
        except Exception as e:
            print(f"⚠️ Skipping {os.path.basename(path)}: {e}")
            return path, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image, image_paths))
    
    # Store successful embeddings in cache
    for path, emb in results:
        if emb is not None:
            img_hash = get_image_hash(path)
            _embedding_cache[img_hash] = emb
    
    return [path for path, emb in results if emb is not None]

def batch_compare(folder_path, threshold=0.95, visualize=False):
    """Compare all images in a folder pairwise with step-by-step visualization."""
    # Get all valid image files
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if os.path.isfile(os.path.join(folder_path, f)) and 
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if len(image_files) < 2:
        print("⚠️ Need at least 2 images for comparison")
        return []
    
    print(f"Found {len(image_files)} images")
    
    # Precompute all embeddings first (parallel)
    valid_images = precompute_all_embeddings(image_files)
    
    if len(valid_images) < 2:
        print("⚠️ Not enough images with detected faces")
        return []
    
    print(f"Successfully processed {len(valid_images)} images with faces")
    results = []
    
    # Now compare all pairs (fast since embeddings are cached)
    total_pairs = len(valid_images) * (len(valid_images) - 1) // 2
    print(f"Comparing {total_pairs} pairs...")
    
    for i in range(len(valid_images)):
        for j in range(i + 1, len(valid_images)):
            img1, img2 = valid_images[i], valid_images[j]
            
            # Get embeddings from cache
            img1_hash = get_image_hash(img1)
            img2_hash = get_image_hash(img2)
            
            emb1 = _embedding_cache[img1_hash]
            emb2 = _embedding_cache[img2_hash]
            
            distance = np.linalg.norm(emb1 - emb2)
            verdict = "Same Person" if distance < threshold else "Different People"
            
            print(f"\nPair {len(results)+1}/{total_pairs}:")
            print(f"  {os.path.basename(img1)} vs {os.path.basename(img2)}")
            print(f"  Distance: {distance:.4f} | Verdict: {verdict}")
            
            if visualize:
                img1_pil = Image.open(img1).convert("RGB")
                img2_pil = Image.open(img2).convert("RGB")
                continue_flag = show_pair(img1_pil, img2_pil, distance, verdict)
                if not continue_flag:
                    print("⛔ Comparison stopped by user.")
                    return results
            
            results.append((img1, img2, verdict, distance))
    
    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"Total comparisons: {len(results)}")
    same_count = sum(1 for r in results if r[2] == "Same Person")
    print(f"Same person pairs: {same_count}")
    print(f"Different people pairs: {len(results) - same_count}")
    print(f"{'='*50}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face comparison using FaceNet embeddings")
    parser.add_argument("img1", type=str, help="Reference image OR folder for batch comparison")
    parser.add_argument("img2", type=str, nargs="?", default=None, help="Second image OR folder (optional)")
    parser.add_argument("--threshold", type=float, default=0.95, help="L2 distance threshold")
    parser.add_argument("--visualize", action="store_true", help="Visualize comparison results")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding cache (slower)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for preprocessing")
    
    args = parser.parse_args()
    
    # Clear cache if disabled
    if args.no_cache:
        _embedding_cache.clear()
    
    # Case 1: Compare two images
    if args.img2 and os.path.isfile(args.img2):
        compare_faces(args.img1, args.img2, 
                     threshold=args.threshold, 
                     visualize=args.visualize,
                     use_cache=not args.no_cache)
    
    # Case 2: Compare reference image against all images in folder
    elif args.img2 and os.path.isdir(args.img2):
        # This needs special handling
        print("Reference vs folder mode not fully optimized yet")
        # You could implement similar caching here
    
    # Case 3: Full batch inside a single folder (optimized)
    elif os.path.isdir(args.img1) and not args.img2:
        results = batch_compare(args.img1, 
                               threshold=args.threshold, 
                               visualize=args.visualize)
        
        # Save results to file
        if results:
            output_file = f"comparison_results_{int(time.time())}.txt"
            with open(output_file, 'w') as f:
                f.write("Image1,Image2,Verdict,Distance\n")
                for img1, img2, verdict, distance in results:
                    f.write(f"{os.path.basename(img1)},{os.path.basename(img2)},{verdict},{distance:.4f}\n")
            print(f"\nResults saved to: {output_file}")
    
    else:
        print("Error: Provide a second image or a folder path for batch evaluation.")








        