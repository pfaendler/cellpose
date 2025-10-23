import os
import glob
from cellpose import models, io
from cellpose.io import imread
import random
import time
import argparse
import torch
import sys

def init_gpu():
    if torch.cuda.is_available():
        gpu = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(gpu)
        print(f"Using GPU: {gpu}")
    else:
        print("No GPU available. Exiting.")
        sys.exit(1)

def main(args):
    random.seed(args.seed)
    init_gpu()

    basepath = '/nfs/nas22.ethz.ch/fs2202/biol_imsb_snijder_1/Data/Ramon/Code/Cellpose/cpose_env/models/'
    modelofinterest = args.cp_model
    modelpath = os.path.join(basepath, modelofinterest)

    os.makedirs(args.output_segpath, exist_ok=True)

    # Initialize Cellpose model
    nucleus_model = models.CellposeModel(gpu=True, pretrained_model=modelpath)
    print("Initialized Cellpose nuclei model.")

    # All *_DAPI.tiff files
    all_dapi_files = glob.glob(os.path.join(args.input_path, '*_DAPI.tiff'))

    # Skip files that have already been segmented
    dapi_files = []
    for f in all_dapi_files:
        image_name = os.path.basename(f).split('_DAPI.tiff')[0]
        mask_filename = os.path.join(args.output_segpath, f'{image_name}_Nuclei_Segmentation_cp_masks.png')
        if not os.path.exists(mask_filename):
            dapi_files.append(f)

    print(f"Found {len(all_dapi_files)} total images, {len(dapi_files)} need segmentation.")

    batch_size = args.batch_size

    start_time = time.time()

    for i in range(0, len(dapi_files), batch_size):
        batch_time_s = time.time()
        batch_files = dapi_files[i:i + batch_size]

        try:
            imgs = [imread(f)**args.dapi_factor for f in batch_files]

            masks, flows, styles = nucleus_model.eval(
                imgs, diameter=args.diameter, flow_threshold=0, batch_size=args.batch_size_gpu
            )

            for img, mask, flow, file_path in zip(imgs, masks, flows, batch_files):
                image_name = os.path.basename(file_path).split('_DAPI.tiff')[0]
                mask_filename = os.path.join(args.output_segpath, f'{image_name}_Nuclei_Segmentation')
                io.save_masks(img, mask, flow, file_names=mask_filename)

            batch_end_time = time.time()
            print(f"Finished batch {i//batch_size + 1} of {len(dapi_files)//batch_size + 1} in {batch_end_time - batch_time_s:.2f} sec.")

        except Exception as e:
            print(f"  Error processing batch starting with {batch_files[0]}: {e}\n")

    end_time = time.time()
    print(f"All required images segmented in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment DAPI images using Cellpose.")
    parser.add_argument("--input_path", type=str, help="Path to the input directory containing *_DAPI.tiff files.")
    parser.add_argument("--output_segpath", type=str, help="Path to the directory where segmentation masks will be saved.")
    parser.add_argument("--batch_size", default=200, type=int, help="Number of images processed per batch.")
    parser.add_argument("--batch_size_gpu", default=50, type=int, help="Number of images processed on GPU at a time.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--dapi_factor", default=0.25, type=float, help="Exponent factor applied to DAPI images (e.g., 0.25 = sqrt).")
    parser.add_argument("--diameter", default=0, type=int, help="Estimated diameter of nuclei. 0 = auto.")
    parser.add_argument("--cp_model", default='nucleitorch_0', type=str, help="Name of the pretrained Cellpose model.")

    args = parser.parse_args()
    main(args)
