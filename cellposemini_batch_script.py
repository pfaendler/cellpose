import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
from cellpose import models, io
from cellpose.io import imread
import random
import time

def main(args):
    # Set random seed for reproducibility
    random.seed(args.seed)

    # Ensure output directory exists
    os.makedirs(args.output_segpath, exist_ok=True)

    # Initialize Cellpose model for nuclei segmentation
    nucleus_model = models.CellposeModel(gpu=True, model_type='nuclei')
    print("Initialized Cellpose nuclei model.")

    # Search for all *_DAPI.tiff files
    dapi_files = glob.glob(os.path.join(args.input_path, '*_DAPI.tiff'))
    batch_size = args.batch_size  # Adjust based on your system's memory capacity


    start_time = time.time()

    # Process the DAPI images in batches
    for i in range(0, len(dapi_files), batch_size):
        batch_time_s = time.time()
        # Select the current batch of files
        batch_files = dapi_files[i:i + batch_size]

        try:
            imgs = [imread(f) ** 0.25 for f in batch_files]

            # Perform segmentation
            masks, flows, styles = nucleus_model.eval(imgs, diameter=None, flow_threshold=0, batch_size=16)

            # Save segmentation masks for each file in the batch
            for img, mask, file_path in zip(imgs, masks, batch_files):
                image_name = os.path.basename(file_path).split('_DAPI.tiff')[0]
                mask_filename = os.path.join(args.output_segpath, f'{image_name}_Nuclei_Segmentation')
                io.save_masks(img, mask, flows, file_names=mask_filename)

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_time_s
            print(f"Finished processing batch {i//batch_size + 1} of {len(dapi_files)//batch_size + 1} in {batch_duration:.2f} seconds.", flush=True)

        except Exception as e:
            print(f"  Error processing batch starting with image {batch_files[0]}: {e}\n")

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"All DAPI images have been segmented in {total_duration:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch segmentation of DAPI images using Cellpose.")

    # Arguments
    parser.add_argument('--input_path', type=str, help="Path to the directory containing the DAPI images.")
    parser.add_argument('--output_segpath', type=str, help="Path to the directory where segmentation results will be saved.")
    parser.add_argument('--batch_size', type=int, default=200, help="Number of images to process in each batch. Default is 200.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility. Default is 42.")

    args = parser.parse_args()
    main(args)
