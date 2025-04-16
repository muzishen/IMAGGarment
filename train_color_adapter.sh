  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4  --mixed_precision "fp16"  --num_machines 1  --dynamo_backend "no" --main_process_port 29500\
  train_color_adapter.py \
  --pretrained_model_name_or_path="/path_to/stable-diffusion-v1-5/" \
  --pretrained_ip_adapter_path='/path_to/ip-adapter_sd15.bin' \
  --image_encoder_path="/path_to/ipa_encoder" \
  --data_json_file="/path_to/GarmentBench/sketch_pair.json" \
  --data_root_path="/path_to/GarmentBench " \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=15 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/save_path" \
  --save_steps=3000  \
  --num_train_epochs 1000 
