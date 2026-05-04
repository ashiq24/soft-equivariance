bash hugging_face_releases/auto_release.sh \
      --command "python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ours_4_9_1 --soft_thresholding 0.9 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 4 --eval_rot 30 --warmup_steps 250 --seed 420000 --hard_mask --weight_decay 0.01" \
      --checkpoint ./logs/segmentation/wandb/run-20260504_001020-ff4v517g/files/best.pt

huggingface-cli upload ashiq24/softeq-dinov2-base-voc-seg-c4-s0.9-sp1.0 hugging_face_releases/filtered-dinov2-base-voc-seg-c4-s1.0