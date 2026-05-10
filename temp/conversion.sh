bash hugging_face_releases/auto_release.sh \
      --command "python seg_main.py --config config/segmentation_ade.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ade_ours_8_1 --soft_thresholding 0.8 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 4 --eval_rot 30 --warmup_steps 500 --hard_mask --seed 42000" \
      --checkpoint ./logs/ade20k/wandb/run-20260505_190418-jm4dlevc/files/best.pt



huggingface-cli upload ashiq24/softeq-dinov2-base-ade20k-seg-c4-s0.8-sp1.0  ./hugging_face_releases/filtered-dinov2-base-ade-seg-c4-s1.0