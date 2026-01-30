#!/bin/bash

#SBATCH --job-name=scmlir_index_build_roco
#SBATCH --partition=shared-gpu
#SBATCH --nodelist=gpu033
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/scmlir_index_build_roco_%j.log

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
test_image="/home/users/h/hej/project/mlir4radiology/test_case/ROCOv2_2023_test_004699"

version="scmlir_v1"
save_base="./save/$dataset/$version"
checkpoint="$save_base/checkpoints/scmlir_model.pth"
savepath="$save_base/index"

# parser.add_argument("--mode", type=str, default="build", choices=["build", "test", "simlar_case_creat", "create_test_similar"])
python -u model/two_stage_retrieval.py \
    --checkpoint ${checkpoint} \
    --annotation_file ${annotation} \
    --data_base_dir ${base_dir} \
    --test_image  ${test_image}\
    --save_path ${savepath} \
    --mode "build" \
