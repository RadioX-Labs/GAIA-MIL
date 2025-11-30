import os
import cv2
import numpy as np
from shutil import copy2

def is_grayscale(img, threshold=30):
    # img: numpy array (BGR or RGB)
    if len(img.shape) != 3 or img.shape[2] != 3:
        return True  # Already grayscale
    diff = np.max(np.abs(img[:,:,0].astype(int) - img[:,:,1].astype(int)))
    diff += np.max(np.abs(img[:,:,1].astype(int) - img[:,:,2].astype(int)))
    diff += np.max(np.abs(img[:,:,2].astype(int) - img[:,:,0].astype(int)))
    return diff < threshold

def is_noisy(img, std_threshold=70):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    return std > std_threshold

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    kept_count = 0
    color_removed = 0
    noise_removed = 0
    failed_count = 0
    total = 0

    for root, _, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)
        out_rel = os.path.join(output_dir, rel)
        os.makedirs(out_rel, exist_ok=True)
        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')) or fname.startswith('._'):
                continue
            total += 1
            fpath = os.path.join(root, fname)
            img = cv2.imread(fpath)
            if img is None:
                failed_count += 1
                continue
            if not is_grayscale(img):
                color_removed += 1
                continue  # SKIP color images
            if is_noisy(img):
                noise_removed += 1
                continue  # SKIP noisy
            # If passed, copy
            copy2(fpath, os.path.join(out_rel, fname))
            kept_count += 1
            print("Kept", fpath)

    print(f"\nSummary:")
    print(f"Total images checked: {total}")
    print(f"Images kept: {kept_count}")
    print(f"Images removed due to color: {color_removed}")
    print(f"Images removed due to noise: {noise_removed}")
    print(f"Images failed to load: {failed_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir)