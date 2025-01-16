import os
import glob
from cellpose import models, io
from cellpose.io import imread
import random
import time
import argparse
import torch
import os



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
    
    # Ensure output directory exists
    os.makedirs(args.output_segpath, exist_ok=True)

    # Initialize Cellpose model for nuclei segmentation
    nucleus_model = models.CellposeModel(gpu=True, pretrained_model=modelpath)
    print("Initialized Cellpose nuclei model.")

    # Search for all *_DAPI.tiff files
    dapi_files = glob.glob(os.path.join(args.input_path, '*_DAPI.tiff'))

    # Number of images to process in each batch
    batch_size = args.batch_size  # Adjust based on your system's memory capacity

    start_time = time.time()

    # Process the DAPI images in batches
    for i in range(0, len(dapi_files), batch_size):
        batch_time_s = time.time()
        # Select the current batch of files
        batch_files = dapi_files[i:i + batch_size]

        try:
            imgs = [imread(f)**args.dapi_factor for f in batch_files]

            # Perform segmentation
            masks, flows, styles = nucleus_model.eval(imgs, diameter=args.diameter, flow_threshold=0, batch_size=args.batch_size_gpu)

            # Save segmentation masks for each file in the batch
            for img, mask, file_path in zip(imgs, masks, batch_files):
                image_name = os.path.basename(file_path).split('_DAPI.tiff')[0]
                mask_filename = os.path.join(args.output_segpath, f'{image_name}_Nuclei_Segmentation')
                io.save_masks(img, mask, flows, file_names=mask_filename)

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_time_s
            print(f"Finished processing batch {i//batch_size + 1} of {len(dapi_files)//batch_size + 1} in {batch_duration:.2f} seconds.")

        except Exception as e:
            print(f"  Error processing batch starting with image {batch_files[0]}: {e}\n")

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"All DAPI images have been segmented in {total_duration:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment DAPI images using Cellpose.")
    parser.add_argument("--input_path", type=str, help="Path to the input directory containing *_DAPI.tiff files.")
    parser.add_argument("--output_segpath", type=str, help="Path to the directory where segmentation masks will be saved.")
    parser.add_argument("--batch_size",default=200, type=int, help="Batch size - number of images loaded into an array at the same time.")
    parser.add_argument("--batch_size_gpu",default=50, type=int, help="Batch size - number of images loaded by gpu at the same time.")
    parser.add_argument("--seed",default=42, type=int, help="For reproducibility.")
    parser.add_argument("--dapi_factor", default=0.25, type=float, help="Multiplication factor for dapi image - sqrt of image helps to get better segmentation for me")
    parser.add_argument("--diameter", default=0, type=int, help="Diameter - if 0 then estimated for each batch.")
    parser.add_argument("--cp_model", default='nucleitorch_0', type=str, help="Pretrained cellpose model of interest")


    args = parser.parse_args()
    main(args)
