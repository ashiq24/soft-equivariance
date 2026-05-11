# python seg_main.py --config config/segmentation_ade.yaml --config_name dinov3_seg --run_name dinov3_seg_c36_ade_ours_8_8 --soft_thresholding 0.8 --soft_thresholding_pos 0.8 --lr 0.001 --backbone_lr 0.000005 --n_rotations 360 --eval_rot 30 --warmup_steps 1000 --hard_mask --seed 42

# python seg_main.py --config config/segmentation_ade.yaml --config_name dinov3_seg --run_name dinov3_seg_c36_ade_ours_7_7 --soft_thresholding 0.7 --soft_thresholding_pos 0.7 --lr 0.001 --backbone_lr 0.000005 --n_rotations 360 --eval_rot 30 --warmup_steps 1000 --hard_mask --seed 42

python seg_main.py --config config/segmentation_ade.yaml --config_name dinov3_seg --run_name dinov3_seg_c36_ade_ours_8_1 --soft_thresholding 0.8 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.000005 --n_rotations 360 --eval_rot 30 --warmup_steps 1000 --hard_mask --seed 42 --evaluate_only --checkpoint_path ./logs/ade20k/wandb/run-20260510_012413-zzhwacvo/files/best.pt

# python seg_main.py --config config/segmentation_ade.yaml --config_name dinov3_seg --run_name dinov3_seg_c36_ade_ours_7_1 --soft_thresholding 0.7 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.000005 --n_rotations 360 --eval_rot 30 --warmup_steps 1000 --hard_mask --seed 42


python seg_main.py --config config/segmentation_ade.yaml --config_name dinov3_seg --run_name dinov3_seg_c4_ade_ours_8_8 --soft_thresholding 0.8 --soft_thresholding_pos 0.8 --lr 0.001 --backbone_lr 0.000005 --n_rotations 4 --eval_rot 30 --warmup_steps 1000 --hard_mask --seed 42 --evaluate_only --checkpoint_path ./logs/ade20k/wandb/run-20260510_012413-n6o3nw4j/files/best.pt

# python seg_main.py --config config/segmentation_ade.yaml --config_name dinov3_seg --run_name dinov3_seg_c4_ade_ours_7_7 --soft_thresholding 0.7 --soft_thresholding_pos 0.7 --lr 0.001 --backbone_lr 0.000005 --n_rotations 4 --eval_rot 30 --warmup_steps 1000 --hard_mask --seed 42

# python seg_main.py --config config/segmentation_ade.yaml --config_name dinov3_seg --run_name dinov3_seg_c4_ade_ours_8_1 --soft_thresholding 0.8 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.000005 --n_rotations 4 --eval_rot 30 --warmup_steps 1000 --hard_mask --seed 42

# python seg_main.py --config config/segmentation_ade.yaml --config_name dinov3_seg --run_name dinov3_seg_c4_ade_ours_7_1 --soft_thresholding 0.7 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.000005 --n_rotations 4 --eval_rot 30 --warmup_steps 1000 --hard_mask --seed 42


python seg_main.py --config config/segmentation_ade.yaml --config_name dinov3_seg --run_name dinov3_seg_c4_ade_ours_4_4 --soft_thresholding 0.4 --soft_thresholding_pos 0.4 --lr 0.001 --backbone_lr 0.000005 --n_rotations 4 --eval_rot 30 --warmup_steps 1000 --hard_mask --seed 42 --evaluate_only --checkpoint_path ./logs/ade20k/wandb/run-20260510_012414-g8tiefn0/files/best.pt

python seg_main.py --config config/segmentation_ade.yaml --config_name dinov3_seg --run_name dinov3_seg_c4_ade_ours_1_1 --soft_thresholding 0.1 --soft_thresholding_pos 0.1 --lr 0.001 --backbone_lr 0.000005 --n_rotations 4 --eval_rot 30 --warmup_steps 1000 --hard_mask --seed 42 --evaluate_only --checkpoint_path ./logs/ade20k/wandb/run-20260510_012400-ij53cvp0/files/best.pt