#!/bin/bash

#SBATCH --job-name=scmlir_visualization
#SBATCH --partition=shared-gpu
#SBATCH --nodelist=gpu027
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/scmlir_visualization_%j.log

echo "=========================================="
echo "任务开始时间: $(date)"
echo "运行节点: $(hostname)"
echo "任务ID: $SLURM_JOB_ID"
echo "=========================================="

echo "load module"
module load GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
echo "activate python"
source ~/henv/bin/activate
echo "show GPU"
nvidia-smi
echo "run training"

dataset="iu_xray"
test_image="./test_case/CXR3348_IM-1605"
save_path="/home/users/h/hej/project/mlir4radiology/new_models"
version="scmlir_v1"
save_base="./save/$dataset/$version"
checkpoint="$save_base/checkpoints/scmlir_model.pth"
savepath="$save_base/CXR3348_IM-1605"

python model/visualization.py \
    --checkpoint ${checkpoint} \
    --img_dir ${test_image} \
    --save_path ${savepath}
