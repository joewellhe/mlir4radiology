#!/bin/bash

#SBATCH --job-name=scmlir_backbone_test_mimic
#SBATCH --partition=shared-gpu
#SBATCH --nodelist=gpu003
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/scmlir_backbone_test_mimic_%j.log

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

dataset="mimic_cxr"
base_dir="/home/users/h/hej/scratch/dataset/mimic-cxr/files"
annotation="/home/users/h/hej/scratch/dataset/mimic-cxr/mimic_annotation_all.json"


version="scmlir_v2"
savepath="./save/$dataset/$version"
delta_file="$savepath/checkpoints/deep_checkpoint_step42310.pth"

python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --delta_file ${delta_file} \
    --base_dir ${base_dir} \
    --test_batch_size 16 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --num_workers 12 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt

