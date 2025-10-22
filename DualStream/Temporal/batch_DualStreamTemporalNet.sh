#!/bin/sh
### ------------- specify queue name ----------------
#BSUB -q c02516
### ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
### ------------- specify job name ----------------
#BSUB -J DualStreamTemporalNet
### ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"
### ------------- specify wall-clock time (max allowed is 12:00)---------------- #BSUB -W 12:00
#BSUB -o OUTPUT_FILE%J.out
#BSUB -e OUTPUT_FILE%J.err
source /zhome/f2/a/224066/IDLCV/bin/activate
python /zhome/f2/a/224066/Project2/DualStream/Temporal/DualStreamTemporalNet.py
