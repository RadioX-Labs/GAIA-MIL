import cv2
import os
import numpy as np
from PIL import Image

def crop_to_largest_contour(img, margin=10):
    """
    Crops the image to the bounding box of the largest contour,
    with an optional margin (in pixels).
    img: numpy array (BGR)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + margin * 2, img.shape[1] - x)
    h = min(h + margin * 2, img.shape[0] - y)
    cropped = img[y:y+h, x:x+w]
    return cropped

def central_crop_cv2(img, output_size):
    """
    Centrally crops the input image (numpy array) to the desired output_size.
    output_size: (height, width)
    """
    img_h, img_w = img.shape[:2]
    crop_h, crop_w = output_size
    start_x = max((img_w - crop_w) // 2, 0)
    start_y = max((img_h - crop_h) // 2, 0)
    end_x = start_x + crop_w
    end_y = start_y + crop_h
    return img[start_y:end_y, start_x:end_x]

def crop_ultrasound_image_pil(pil_img, target_size=(600,600), margin=10):
    """
    Takes a PIL image, runs ultrasound cone crop + central crop, returns a PIL image.
    """
    img = np.array(pil_img)
    if img.shape[-1] == 4:  # RGBA
        img = img[...,:3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cropped = crop_to_largest_contour(img, margin)
    if cropped.shape[0] < target_size[0] or cropped.shape[1] < target_size[1]:
        cropped = cv2.resize(cropped, target_size[::-1], interpolation=cv2.INTER_AREA)
    else:
        cropped = central_crop_cv2(cropped, target_size)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped)

def process_folder(input_folder, output_folder, target_size=(600,600), margin=10, image_extensions=('.png', '.jpg', '.jpeg')):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(image_extensions) or fname.startswith('._'):
            continue
        input_path = os.path.join(input_folder, fname)
        output_path = os.path.join(output_folder, fname)
        try:
            pil_img = Image.open(input_path).convert('RGB')
            cropped_img = crop_ultrasound_image_pil(pil_img, target_size, margin)
            cropped_img.save(output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Failed {input_path}: {e}")

def process_folders_recursively(base_input_dir, base_output_dir, target_size=(600,600), margin=10):
    """
    Recursively processes all images in all subfolders, preserving directory structure.
    """
    for root, dirs, files in os.walk(base_input_dir):
        rel_path = os.path.relpath(root, base_input_dir)
        cur_output_dir = os.path.join(base_output_dir, rel_path)
        if not os.path.exists(cur_output_dir):
            os.makedirs(cur_output_dir)
        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')) or fname.startswith('._'):
                continue
            input_path = os.path.join(root, fname)
            output_path = os.path.join(cur_output_dir, fname)
            try:
                pil_img = Image.open(input_path).convert('RGB')
                cropped_img = crop_ultrasound_image_pil(pil_img, target_size, margin)
                cropped_img.save(output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Failed {input_path}: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Crop all ultrasound images in (possibly nested) folders')
    parser.add_argument('--input_dir', type=str, required=True, help='Input folder (can be nested)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder (will preserve structure)')
    parser.add_argument('--size', type=int, default=600, help='Central crop size (default 600)')
    parser.add_argument('--margin', type=int, default=10, help='Contour margin (default 10)')

    args = parser.parse_args()

    process_folders_recursively(
        base_input_dir=args.input_dir,
        base_output_dir=args.output_dir,
        target_size=(args.size, args.size),
        margin=args.margin
    )