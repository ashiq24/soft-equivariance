#!/bin/bash

# ============================================================
# ImageNet Training Script for Soft-Equivariant ViT
# ============================================================


# ============================================================
# ImageNet Training Script for Soft-Equivariant ViT
# ============================================================


CONFIG_PATH="config/imagenet_configs.yaml"
NUM_GPUS=1  # Number of GPUs to use
CONFIG_NAME=""

# ============================================================
# Parse command-line arguments for soft-thresholding and group type
# ============================================================
SOFT_THRESHOLDING=""
SOFT_THRESHOLDING_POS=""
GROUP_TYPE=""



# Parse all command-line arguments, extracting known ones and collecting extras
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config_name)
            CONFIG_NAME="$2"
            shift; shift
            ;;
        --soft_thresholding)
            SOFT_THRESHOLDING="$2"
            EXTRA_ARGS="${EXTRA_ARGS} --soft_thresholding ${SOFT_THRESHOLDING}"
            shift; shift
            ;;
        --soft_thresholding_pos)
            SOFT_THRESHOLDING_POS="$2"
            EXTRA_ARGS="${EXTRA_ARGS} --soft_thresholding_pos ${SOFT_THRESHOLDING_POS}"
            shift; shift
            ;;
        --group_type)
            GROUP_TYPE="$2"
            EXTRA_ARGS="${EXTRA_ARGS} --group_type ${GROUP_TYPE}"
            shift; shift
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# ============================================================
# Training Configuration
# ============================================================


if [ ${NUM_GPUS} -eq 1 ]; then
    echo "Running single GPU training..."
    # Single GPU: Explicitly disable distributed training
    unset WORLD_SIZE
    unset RANK
    unset LOCAL_RANK
    unset MASTER_ADDR
    unset MASTER_PORT
    unset SLURM_NTASKS
    unset SLURM_PROCID
    unset SLURM_LOCALID

    python main_imagenet.py \
        --softeq_config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --amp \
        --amp-dtype bfloat16 \
        --grad-accum-steps 4 \
        --clip-grad 1.0 \
        --layer-decay 0.65 \
        --drop-path 0.1 \
        --model-ema \
        --model-ema-decay 0.9999 \
        ${EXTRA_ARGS}
else
    echo "Running multi-GPU training with ${NUM_GPUS} GPUs..."
    # Multi-GPU: Use torchrun for DDP
    torchrun --nproc_per_node=${NUM_GPUS} main_imagenet.py \
        --softeq_config_path ${CONFIG_PATH} \
        --config_name ${CONFIG_NAME} \
        --amp \
        --amp-dtype bfloat16 \
        --grad-accum-steps 1 \
        --clip-grad 1.0 \
        --layer-decay 0.65 \
        --drop-path 0.1 \
        --model-ema \
        --model-ema-decay 0.9999 \
        ${EXTRA_ARGS}
fi
