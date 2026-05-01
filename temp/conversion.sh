python hugging_face_releases/package_model.py \
    --safetensors temp/converted_vit_c720_s9/model.safetensors \
    --output_dir  hugging_face_releases/filtered-vit-base-patch16-224-voc-seg-c720-s0.9 \
    --model_arch  filtered_vit_seg \
    --pretrained_model google/vit-base-patch16-224 \
    --num_labels  21 \
    --n_rotations 720 \
    --soft_thresholding 1.0 \
    --soft_thresholding_pos 0.9 \
    --hard_mask

python hugging_face_releases/test_release.py \
    --config      config/segmentation.yaml \
    --config_name vit_seg_c18 \
    --checkpoint  logs/segmentation/wandb/run-20260429_164823-9jwht0p9/files/best.pt \
    --hf_dir      hugging_face_releases/filtered-vit-base-patch16-224-voc-seg-c720-s0.9 \
    --n_rotations 720 \
    --soft_thresholding 1.0 \
    --soft_thresholding_pos 0.9 \
    --hard_mask
