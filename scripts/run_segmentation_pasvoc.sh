# base models

python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 --run_name vit_seg_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00005  --eval_rot 30 --warmup_steps 250 --seed 42
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 --run_name vit_seg_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00005  --eval_rot 30 --warmup_steps 250 --seed 420
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 --run_name vit_seg_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00005  --eval_rot 30 --warmup_steps 250 --seed 4200
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 --run_name vit_seg_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00005  --eval_rot 30 --warmup_steps 250 --seed 42000
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 --run_name vit_seg_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00005  --eval_rot 30 --warmup_steps 250 --seed 420000


python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 18 --eval_rot 30 --warmup_steps 250 --seed 42
seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 18 --eval_rot 30 --warmup_steps 250 --seed 420
seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 18 --eval_rot 30 --warmup_steps 250 --seed 4200
seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 18 --eval_rot 30 --warmup_steps 250 --seed 42000
seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 18 --eval_rot 30 --warmup_steps 250 --seed 420000


python seg_main.py --config config/segmentation.yaml --config_name segformer_c18 --run_name segformer_c18_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 250 --seed 42
python seg_main.py --config config/segmentation.yaml --config_name segformer_c18 --run_name segformer_c18_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 250 --seed 420
python seg_main.py --config config/segmentation.yaml --config_name segformer_c18 --run_name segformer_c18_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 250 --seed 420
seg_main.py --config config/segmentation.yaml --config_name segformer_c18 --run_name segformer_c18_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 250 --seed 42000
seg_main.py --config config/segmentation.yaml --config_name segformer_c18 --run_name segformer_c18_b --soft_thresholding 1.0 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 250 --seed 420000


# our models (soft equivariance)
python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ours --soft_thresholding 0.95 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 720 --eval_rot 30 --warmup_steps 250 --seed 42 --hard_mask --weight_decay 0.01
python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ours --soft_thresholding 0.95 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 720 --eval_rot 30 --warmup_steps 250 --seed 420 --hard_mask --weight_decay 0.01
python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ours --soft_thresholding 0.95 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 720 --eval_rot 30 --warmup_steps 250 --seed 4200 --hard_mask --weight_decay 0.01
python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ours --soft_thresholding 0.95 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 720 --eval_rot 30 --warmup_steps 250 --seed 42000 --hard_mask --weight_decay 0.01
python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_c18 --run_name dinov2_seg_c18_ours --soft_thresholding 0.95 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.00001 --n_rotations 720 --eval_rot 30 --warmup_steps 250 --seed 420000 --hard_mask --weight_decay 0.01


python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ours --soft_thresholding 1.0 --soft_thresholding_pos 0.90 --lr 0.001 --backbone_lr 0.00005 --n_rotations 720 --eval_rot 30 --warmup_steps 250 --seed 42 --hard_mask --weight_decay 0.07
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ours --soft_thresholding 1.0 --soft_thresholding_pos 0.90 --lr 0.001 --backbone_lr 0.00005 --n_rotations 720 --eval_rot 30 --warmup_steps 250 --seed 420 --hard_mask --weight_decay 0.07
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ours --soft_thresholding 1.0 --soft_thresholding_pos 0.90 --lr 0.001 --backbone_lr 0.00005 --n_rotations 720 --eval_rot 30 --warmup_steps 250 --seed 4200 --hard_mask --weight_decay 0.07
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ours --soft_thresholding 1.0 --soft_thresholding_pos 0.90 --lr 0.001 --backbone_lr 0.00005 --n_rotations 720 --eval_rot 30 --warmup_steps 250 --seed 42000 --hard_mask --weight_decay 0.07
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_c18 --run_name vit_seg_c18_ours --soft_thresholding 1.0 --soft_thresholding_pos 0.90 --lr 0.001 --backbone_lr 0.00005 --n_rotations 720 --eval_rot 30 --warmup_steps 250 --seed 420000 --hard_mask --weight_decay 0.07

python seg_main.py --config config/segmentation.yaml --config_name segformer_c18 --run_name segformer_c18_ours --soft_thresholding 0.95 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 250 --seed 42 --min_filter_size 5 --weight_decay 0.01 --hard_mask
python seg_main.py --config config/segmentation.yaml --config_name segformer_c18 --run_name segformer_c18_ours --soft_thresholding 0.95 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 250 --seed 420 --min_filter_size 5 --weight_decay 0.01 --hard_mask
python seg_main.py --config config/segmentation.yaml --config_name segformer_c18 --run_name segformer_c18_ours --soft_thresholding 0.95 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 250 --seed 4200 --min_filter_size 5 --weight_decay 0.01 --hard_mask
python seg_main.py --config config/segmentation.yaml --config_name segformer_c18 --run_name segformer_c18_ours --soft_thresholding 0.95 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 250 --seed 42000 --min_filter_size 5 --weight_decay 0.01 --hard_mask
python seg_main.py --config config/segmentation.yaml --config_name segformer_c18 --run_name segformer_c18_ours --soft_thresholding 0.95 --soft_thresholding_pos 1.0 --lr 0.001 --backbone_lr 0.0001 --n_rotations 36 --eval_rot 30 --warmup_steps 250 --seed 420000 --min_filter_size 5 --weight_decay 0.01 --hard_mask

# canonicalizers
# batch size is reduced to make room for the additional memory cost of the canonicalization module. 
python seg_main.py --config config/segmentation.yaml --config_name segformer_canon_c18 --run_name segformer_canon_c18 --lr 0.001 --backbone_lr 0.0001 --batch_size 16 --seed 42
python seg_main.py --config config/segmentation.yaml --config_name segformer_canon_c18 --run_name segformer_canon_c18 --lr 0.001 --backbone_lr 0.0001 --batch_size 16 --seed 420
python seg_main.py --config config/segmentation.yaml --config_name segformer_canon_c18 --run_name segformer_canon_c18 --lr 0.001 --backbone_lr 0.0001 --batch_size 16 --seed 4200
python seg_main.py --config config/segmentation.yaml --config_name segformer_canon_c18 --run_name segformer_canon_c18 --lr 0.001 --backbone_lr 0.0001 --batch_size 16 --seed 42000
python seg_main.py --config config/segmentation.yaml --config_name segformer_canon_c18 --run_name segformer_canon_c18 --lr 0.001 --backbone_lr 0.0001 --batch_size 16 --seed 420000

python seg_main.py --config config/segmentation.yaml --config_name vit_seg_canon_c18 --run_name vit_seg_canon_c18 --lr 0.001 --backbone_lr 0.00005 --batch_size 16 --seed 42
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_canon_c18 --run_name vit_seg_canon_c18 --lr 0.001 --backbone_lr 0.00005 --batch_size 16 --seed 420
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_canon_c18 --run_name vit_seg_canon_c18 --lr 0.001 --backbone_lr 0.00005 --batch_size 16 --seed 4200
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_canon_c18 --run_name vit_seg_canon_c18 --lr 0.001 --backbone_lr 0.00005 --batch_size 16 --seed 42000
python seg_main.py --config config/segmentation.yaml --config_name vit_seg_canon_c18 --run_name vit_seg_canon_c18 --lr 0.001 --backbone_lr 0.00005 --batch_size 16 --seed 420000

python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_canon_c18 --run_name dinov2_seg_canon_c18 --lr 0.001 --backbone_lr 0.00001 --batch_size 16 --seed 42
python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_canon_c18 --run_name dinov2_seg_canon_c18 --lr 0.001 --backbone_lr 0.00001 --batch_size 16 --seed 420
python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_canon_c18 --run_name dinov2_seg_canon_c18 --lr 0.001 --backbone_lr 0.00001 --batch_size 16 --seed 4200
python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_canon_c18 --run_name dinov2_seg_canon_c18 --lr 0.001 --backbone_lr 0.00001 --batch_size 16 --seed 42000
python seg_main.py --config config/segmentation.yaml --config_name dinov2_seg_canon_c18 --run_name dinov2_seg_canon_c18 --lr 0.001 --backbone_lr 0.00001 --batch_size 16 --seed 420000