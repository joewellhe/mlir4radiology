#!/bin/bash

#SBATCH --job-name=scmlir_backbone_test_iu
#SBATCH --partition=shared-gpu
#SBATCH --nodelist=gpu003
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/scmlir_backbone_test_iu_%j.log

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
annotation="/home/users/h/hej/scratch/dataset/iu_xray/annotation.json"
base_dir="/home/users/h/hej/scratch/dataset/iu_xray/images"

version="scmlir_v2"
savepath="./save/$dataset/$version"
delta_file="$savepath/checkpoints/scmlir_model.pth"

python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --delta_file ${delta_file} \
    --base_dir ${base_dir} \
    --batch_size 8 \
    --val_batch_size 12 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 2 \
    --devices 1 \
    --max_epochs 15 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 2 \
    2>&1 |tee -a ${savepath}/log.txt
