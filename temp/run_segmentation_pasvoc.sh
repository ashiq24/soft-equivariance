python seg_main.py --config config/segmentation_ade.yaml --config_name dinov3_seg --run_name dinov3_seg_c18_ade_ours_8_8 --soft_thresholding 0.8 --soft_thresholding_pos 0.8 --lr 0.001 --backbone_lr 0.00001 --n_rotations 180 --eval_rot 30 --warmup_steps 500 --hard_mask --seed 42


