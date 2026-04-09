bash evaluate_imagenet.sh --config_name imagenet1k_vit_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0
bash evaluate_imagenet.sh --config_name imagenet1k_dinov2_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0
bash evaluate_imagenet.sh --config_name imagenet1k_dinov2_reg_c18 --soft_thresholding 1.0 --soft_thresholding_pos 1.0
bash evaluate_imagenet.sh --config_name imagenet1k_resnet_c18 --soft_thresholding 1.0

bash train_imagenet.sh --config_name imagenet1k_vit_canon_c18 --batch-size 128 --lr 0.00001
bash train_imagenet.sh --config_name imagenet1k_dinov2_canon_c18 --batch-size 128 --lr 0.00001
bash train_imagenet.sh --config_name imagenet1k_resnet_canon_c18 --lr 0.00001


# coarse rotations
bash train_imagenet.sh --config_name imagenet1k_vit --soft_thresholding 0.9 --soft_thresholding_pos 0.8 --lr 0.00001
bash train_imagenet.sh --config_name imagenet1k_dinov2_c18 --soft_thresholding 0.8 --soft_thresholding_pos 0.5 --lr 0.00001
bash train_imagenet.sh --config_name imagenet1k_resnet_c18 --soft_thresholding 0.95 --lr 0.00001 

# finer rotations
bash train_imagenet.sh --config_name imagenet1k_vit --soft_thresholding 1.0 --soft_thresholding_pos 0.8 --lr 0.00001 --n_rotations 72
bash train_imagenet.sh --config_name imagenet1k_dinov2_c18 --soft_thresholding 0.8 --soft_thresholding_pos 1.0 --lr 0.00001 --n_rotations 72
bash train_imagenet.sh --config_name imagenet1k_resnet_c18 --soft_thresholding 0.95 --lr 0.00001 --n_rotations 180 --n_rotations 72