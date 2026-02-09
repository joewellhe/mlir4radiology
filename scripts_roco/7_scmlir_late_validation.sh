#!/bin/bash

#SBATCH --job-name=scmlir_late_validation
#SBATCH --partition=shared-gpu
#SBATCH --nodelist=gpu048
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/scmlir_late_validation_%j.log

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

dataset="roco"
version="scmlir_v2"
savepath="./save/$dataset/$version"
checkpoint="$savepath/checkpoints/scmlir_model.pth"
raw_csv="$savepath/result/test_backbone_result.csv"
rag_csv="$savepath/result/test_scmlir_result.csv"
gt_csv="$savepath/result/test_refs.csv"
annotation="/home/users/h/hej/scratch/dataset/rocov2/annotation.json"
base_dir="/home/users/h/hej/scratch/dataset/rocov2"
output_csv="$savepath/result/merge.csv"

python -u model/late_validation.py \
    --checkpoint ${checkpoint} \
    --raw_csv ${raw_csv} \
    --rag_csv ${rag_csv} \
    --gt_csv ${gt_csv} \
    --annotation  ${annotation}\
    --output_csv ${output_csv} \
    --image_root ${base_dir} \
