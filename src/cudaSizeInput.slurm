#!/bin/bash
#SBATCH --nodelist=udc-ba30-7
#SBATCH --mem-per-cpu=1096
#SBATCH --time=01:15:00 
#SBATCH --part=training
#SBATCH --account=parallelcomputing
#SBATCH --output=cuda-output


module load cuda
nvcc -O3 mmm_template.cu 
./a.out $1
