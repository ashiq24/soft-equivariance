
python misc_main.py --config config/misc_o5.yaml --name "mlp_rpp"  --config_name rpp_emlp_o5 --device cpu --noise_level 0.3
python misc_main.py --config config/misc_o5.yaml --name "filtered_ours"  --config_name base_o5 --noise_level 0.3 --soft_thresholding 0.07 --weight_decay 0.00001 --hard_mask
python misc_main.py --config config/misc_o5.yaml --name "filtered_ours"  --config_name base_o5 --noise_level 0.3 --soft_thresholding 0.05 --weight_decay 0.00001 --hard_mask




python misc_main.py --config config/misc_o5.yaml --name "mlp_rpp"  --config_name rpp_emlp_o5 --device cpu --noise_level 0.4
# filter
python misc_main.py --config config/misc_o5.yaml --name "filtered_ours"  --config_name base_o5 --noise_level 0.4 --soft_thresholding 0.07 --weight_decay 0.00001 --hard_mask
python misc_main.py --config config/misc_o5.yaml --name "filtered_ours"  --config_name base_o5 --noise_level 0.4 --soft_thresholding 0.05 --weight_decay 0.00001 --hard_mask


python misc_main.py --config config/misc_o5.yaml --name "mlp_rpp"  --config_name rpp_emlp_o5 --device cpu --noise_level 0.5
# filter
python misc_main.py --config config/misc_o5.yaml --name "filtered_ours"  --config_name base_o5 --noise_level 0.5 --soft_thresholding 0.07 --weight_decay 0.00001 --hard_mask
python misc_main.py --config config/misc_o5.yaml --name "filtered_ours"  --config_name base_o5 --noise_level 0.5 --soft_thresholding 0.05 --weight_decay 0.00001 --hard_mask
