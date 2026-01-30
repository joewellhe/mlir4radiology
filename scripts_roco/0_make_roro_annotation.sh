#!/bin/bash

#SBATCH --job-name=make_rocov2_annotation
#SBATCH --partition=shared-cpu
#SBATCH --nodelist=cpu302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/make_rocov2_annotation_%j.log

echo "=========================================="
echo "任务开始时间: $(date)"
echo "运行节点: $(hostname)"
echo "任务ID: $SLURM_JOB_ID"
echo "=========================================="

echo "load module"
module load GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
echo "activate python"
source ~/henv/bin/activate
echo "run make rorov2 annotation"

python dataset/make_rocov2_annotation.py 

echo "completed!"

