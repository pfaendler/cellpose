#!/bin/bash

source Scripts/activate

python cellposemini_batch_script.py --input_path "Y:/Data/Lucie/MEN/MEN030/MEN030_PDC_MVP001/Images_Corrected/" --output_segpath "Y:/Data/Lucie/MEN/MEN030/MEN030_PDC_MVP001/CellPose_Segmentation/" --batch_size 200 --seed 42