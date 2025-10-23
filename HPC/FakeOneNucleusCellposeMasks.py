import os
import glob
import argparse
import numpy as np
import cv2
from cellpose.io import imread

def create_fake_nucleus_mask(img_shape):
    """Create a 16-bit mask with one fake nucleus in the top-left corner."""
    mask = np.zeros(img_shape, dtype=np.uint16)
    center = (20, 20)  # (x, y)
    radius = 15
    cv2.circle(mask, center, radius, color=1, thickness=-1)  # filled circle with label "1"
    return mask

def create_fake_nucleus_png_masks(args):
    os.makedirs(args.output_segpath, exist_ok=True)

    dapi_files = glob.glob(os.path.join(args.input_path, '*_DAPI.tiff'))
    dapi_files.sort()

    exclude_substrings = ("r01c04", "r03c04", "r04c04", "r05c04", "r06c04", "r07c04")

    filtered_files = [f for f in dapi_files if any(sub in f for sub in exclude_substrings)]

    print(f"Found {len(filtered_files)} files requiring fake nucleus masks.")

    for f in filtered_files:
        try:
            img = imread(f)**args.dapi_factor
            height, width = img.shape
            mask = create_fake_nucleus_mask((height, width))

            image_name = os.path.basename(f).split('_DAPI.tiff')[0]
            mask_filename = os.path.join(args.output_segpath, f'{image_name}_Nuclei_Segmentation.png')

            success = cv2.imwrite(mask_filename, mask)
            if success:
                print(f"Saved fake nucleus mask for: {f}")
            else:
                print(f"Failed to save PNG for: {f}")

        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake nucleus masks (16-bit PNG) for excluded DAPI images.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to *_DAPI.tiff images.")
    parser.add_argument("--output_segpath", type=str, required=True, help="Where to save PNG mask files.")
    parser.add_argument("--dapi_factor", default=0.25, type=float, help="DAPI image adjustment factor (not used in mask).")

    args = parser.parse_args()
    create_fake_nucleus_png_masks(args)
