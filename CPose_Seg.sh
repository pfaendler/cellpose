#!/bin/bash
#SBATCH --output=output_log2.txt
#SBATCH --error=error_log2.txt
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH -n 10
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10000

#load runname to name the ouput file
module load stack/2024-04 python_cuda
source cpose_env/bin/activate
which python

nvidia-smi

python NucleiSegmentationCellpose.py \
--input_path '/nfs/nas22.ethz.ch/fs2202/biol_imsb_snijder_1/Data/Ramon/TestStains/TestStain_PSC1010_RNAFISH/Images_Corrected/' \
--output_segpath '/nfs/nas22.ethz.ch/fs2202/biol_imsb_snijder_1/Data/Ramon/TestStains/TestStain_PSC1010_RNAFISH/CellPose_Segmentation_005_D_0/' \
--batch_size 208 \
--batch_size_gpu 16 \
--seed 42 \
--cp_model 'nucleitorch_1' \
--dapi_factor 0.05 \
--diameter 0