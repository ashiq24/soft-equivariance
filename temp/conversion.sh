bash hugging_face_releases/auto_release.sh \
      --command "python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ours --soft_thresholding 1.0 --soft_thresholding_pos 0.90 --lr 0.001 --backbone_lr 0.00005 --n_rotations 720 --eval_rot 30 --warmup_steps 250 --seed 420000 --hard_mask --weight_decay 0.07" \
      --checkpoint ./logs/segmentation/wandb/run-20260501_163709-qwi3n34z/files/best.pt

huggingface-cli upload ashiq24/softeq-vit-base-patch16-224-voc-seg-c720-s0.90 hugging_face_releases/filtered-vit-base-patch16-224-voc-seg-c720-s0.90