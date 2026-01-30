#!/bin/bash

#SBATCH --job-name=scmlir_retriever_train_ro
#SBATCH --partition=shared-gpu
#SBATCH --nodelist=gpu033
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/scmlir_retriever_train_ro_%j.log

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
base_dir="/home/users/h/hej/scratch/dataset/rocov2"
annotation="/home/users/h/hej/scratch/dataset/rocov2/annotation.json"


version="scmlir_v1"
savepath="./save/$dataset/$version"
delta_file="$savepath/checkpoints/scmlir_model.pth"

python -u train.py \
    --retrieval_only True \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --batch_size 32 \
    --val_batch_size 32 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \
    --max_epochs 50 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 2 \
    --learning_rate 5e-4 \
    2>&1 |tee -a ${savepath}/log.txt
