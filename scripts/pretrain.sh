#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=h100:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=24
#SBATCH --job-name=ViT-S-16
#SBATCH --partition short-unkillable
#SBATCH -o /network/scratch/s/sarvjeet-singh.ghotra/git/open_clip/src/logs/ViT-S-16/%j_%t.out
#SBATCH -e /network/scratch/s/sarvjeet-singh.ghotra/git/open_clip/src/logs/ViT-S-16/%j_%t.err

# SBATCH --nodelist=cn-g[008,011-013,015-016,021,023,025,026,028-029]



source ~/.bashrc
mamba activate openclip
module load cuda/11.8

cd src
export PYTHONPATH="$PYTHONPATH:$PWD/src"
# wandb offline
export WANDB_MODE=offline

EXP_NAME=$1
# EXP_NAME=dbg

# TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --save-frequency 1 \
    --train-data="/home/mila/s/sarvjeet-singh.ghotra/scratch/data/laion400m/laion400m_40M/{00000..04941}.tar" \
    --batch-size=4096 \
    --workers=6 \
    --model ViT-S-16 \
    --seed 63611133 \
    --dataset-type webdataset \
    --imagenet-val /network/datasets/imagenet.var/imagenet_torchvision/val \
    --epochs 50 \
    --precision 'amp_bf16' \
    --warmup 200 \
    --report-to "wandb" \
    --wandb-project-name "openclip" \
    --train-num-samples 26000000 \
    --dataset-resampled \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --accum-freq 2 \
    --zeroshot-frequency 1 \
    --log-every-n-steps 32 \
    --name $EXP_NAME \
    --lr=0.0005 \
    --resume latest \



    # --dataset-resampled \
    # --non_strict_weight_load \
    # --warmup 500 \
    # --train-only-sem \
    # --skip-scheduler \
    # --prev-run-steps 9775 \


    # --logs OUTPUT_PARENT_DIR




# --torchcompile \
# --lr=0.00003 \
# --skip-scheduler \
# --batch-size=256 \
# --name 'cnt_ViT-B-32_laion' \
# --logs /home/mila/s/sarvjeet-singh.ghotra/scratch/models/latent_mllm_reason/openclip \
# --warmup 2000 \
