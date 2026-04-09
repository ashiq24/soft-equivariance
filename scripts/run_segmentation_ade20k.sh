#!/bin/bash

# ADE20K Segmentation Experiments
# Using config/segmentation_ade.yaml with 150 classes, 512x512 resolution
########################################################### segformer

# Baseline experiments (soft_thresholding 1.0)
python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_c18 --run_name segformer_c18_ade_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 500 --evaluate_only --seed 42
python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_c18 --run_name segformer_c18_ade_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 500 --evaluate_only --seed 420
python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_c18 --run_name segformer_c18_ade_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 500 --evaluate_only --seed 4200


# Canonicalization baseline experiments
python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_canon_c18 --run_name segformer_canon_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 500  --seed 42
python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_canon_c18 --run_name segformer_canon_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 500 --seed 420
python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_canon_c18 --run_name segformer_canon_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 500  --seed 4200

# Our approach with soft thresholding 0.7
python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_c18 --run_name segformer_c18_ade_ours_180_8_n --soft_thresholding 0.8 --soft_thresholding_pos 1.0 --lr 0.00005 --backbone_lr 0.00005 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --weight_decay 0.02 --seed 42
python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_c18 --run_name segformer_c18_ade_ours_180_8_n --soft_thresholding 0.8 --soft_thresholding_pos 1.0 --lr 0.00005 --backbone_lr 0.00005 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --weight_decay 0.02 --seed 420
python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_c18 --run_name segformer_c18_ade_ours_180_8_n --soft_thresholding 0.8 --soft_thresholding_pos 1.0 --lr 0.00005 --backbone_lr 0.00005 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --weight_decay 0.02 --seed 4200

python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_c18 --run_name segformer_c18_ade_ours_180_7 --soft_thresholding 0.7 --soft_thresholding_pos 1.0 --lr 0.00005 --backbone_lr 0.00005 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --seed 42
python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_c18 --run_name segformer_c18_ade_ours_180_7 --soft_thresholding 0.7 --soft_thresholding_pos 1.0 --lr 0.00005 --backbone_lr 0.00005 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --seed 420
python seg_main.py --config config/segmentation_ade.yaml --config_name segformer_c18 --run_name segformer_c18_ade_ours_180_7 --soft_thresholding 0.7 --soft_thresholding_pos 1.0 --lr 0.00005 --backbone_lr 0.00005 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --seed 4200



########################################################### vit

# # Baseline experiments (soft_thresholding 1.0)
python seg_main.py --config config/segmentation_ade.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ade_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00005 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 42
python seg_main.py --config config/segmentation_ade.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ade_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00005 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 420
python seg_main.py --config config/segmentation_ade.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ade_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00005 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 4200

########################################################### canonicalization

python seg_main.py --config config/segmentation_ade.yaml --config_name vit_seg_canon_c18 --run_name vit_seg_canon_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00005 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 42
python seg_main.py --config config/segmentation_ade.yaml --config_name vit_seg_canon_c18 --run_name vit_seg_canon_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00005 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 420
python seg_main.py --config config/segmentation_ade.yaml --config_name vit_seg_canon_c18 --run_name vit_seg_canon_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00005 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 4200

### ours 

python seg_main.py --config config/segmentation_ade.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ade_ours_180_8_8_n --soft_thresholding 0.8 --soft_thresholding_pos 0.8 --lr 0.001 --backbone_lr 0.00005 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --hard_mask --seed 42 --weight_decay 0.1
python seg_main.py --config config/segmentation_ade.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ade_ours_180_8_8_n --soft_thresholding 0.8 --soft_thresholding_pos 0.8 --lr 0.001 --backbone_lr 0.00005 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --hard_mask --seed 420 --weight_decay 0.1
python seg_main.py --config config/segmentation_ade.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ade_ours_180_8_8_n --soft_thresholding 0.8 --soft_thresholding_pos 0.8 --lr 0.001 --backbone_lr 0.00005 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --hard_mask --seed 4200 --weight_decay 0.1


########################################################### dinov2

# # Baseline experiments (soft_thresholding 1.0)
python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ade_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 42
python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ade_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 420
python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ade_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 4200

### canonicalization

python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_canon_c18 --run_name dinov2_seg_canon_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 42
python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_canon_c18 --run_name dinov2_seg_canon_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 420
python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_canon_c18 --run_name dinov2_seg_canon_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 18 --eval_rot 30 --warmup_steps 500 --seed 4200



# # Our approach with soft thresholding 0.8
python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ade_ours_8_1_n --soft_thresholding 0.8 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --hard_mask --seed 42
python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ade_ours_8_1_n --soft_thresholding 0.8 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --hard_mask --seed 420
python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ade_ours_8_1_n --soft_thresholding 0.8 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --hard_mask --seed 4200

# # Our approach with soft thresholding 0.9
python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ade_ours_360_9 --soft_thresholding 0.9 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 360 --eval_rot 30 --warmup_steps 500 --hard_mask --seed 42
python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ade_ours_180_9 --soft_thresholding 0.9 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --hard_mask --seed 420
python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ade_ours_180_9 --soft_thresholding 0.9 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --hard_mask --seed 4200


