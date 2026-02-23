#!/bin/bash

#SBATCH --job-name=scmlir_rag_test
#SBATCH --partition=shared-gpu
#SBATCH --nodelist=gpu006
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=./logs/scmlir_rag_test_%j.log

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
delta_file="$savepath/checkpoints/scmlir_model.pth"
similar_cases_file="$savepath/index/test_similar_cases.json"

# similar_cases_file="$savepath/index/test_similar_cases.json"

python -u train.py \
    --test \
    --dataset ${dataset} \
    --similar_cases_file ${similar_cases_file} \
    --RAG_prompt True \
    --delta_file ${delta_file} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 16 \
    --val_batch_size 8 \
    --savedmodel_path ${savepath} \
    --max_length 120 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2 \
    --length_penalty 2\
    --num_workers 4 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt
