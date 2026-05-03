bash hugging_face_releases/auto_release.sh \
      --command "python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ours_180_9 --soft_thresholding 0.9 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 180 --eval_rot 30 --warmup_steps 250 --seed 42000 --hard_mask --weight_decay 0.01" \
      --checkpoint ./logs/segmentation/wandb/run-20260502_035654-l9ub5rbv/files/best.pt

huggingface-cli upload ashiq24/softeq-dinov2-base-voc-seg-c180-s1.0-sp0.9 hugging_face_releases/filtered-dinov2-base-voc-seg-c180-s1.0